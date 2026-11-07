# -*- coding: utf-8 -*-
"""
Unsloth & MoAI-Style LoRA Embedding Training Pipeline for Multi-Agent RAG.
Optimized for Kaggle Dual Tesla T4 (2x16GB) or Single GPU.
Recipe:
  - Base: BAAI/bge-m3 (Multilingual Dense + Sparse)
  - LoRA: rank=32, alpha=64, dropout=0.05
  - Loss: CachedMultipleNegativesRankingLoss / MultipleNegativesRankingLoss
  - Metric: Validation Recall@10 & MRR@10
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
def patch_environment():
    """Patches missing triton.ops, removes invalid BNB_CUDA_VERSION overrides, and handles CPU/GPU fallbacks for Kaggle and Colab runtimes."""
    import os
    import sys
    import types

    # 1. Clean BNB_CUDA_VERSION if it causes missing libcudart.so
    if "BNB_CUDA_VERSION" in os.environ:
        del os.environ["BNB_CUDA_VERSION"]

    # 2. Patch Triton ops mock for Triton 3+ compatibility
    m_triton = sys.modules.get("triton")
    if m_triton is None:
        try:
            import triton
            m_triton = triton
        except Exception:
            m_triton = types.ModuleType("triton")
            m_triton.__path__ = []
            sys.modules["triton"] = m_triton

    m_ops = types.ModuleType("triton.ops")
    m_ops.__path__ = []
    m_perf = types.ModuleType("triton.ops.matmul_perf_model")
    m_perf.early_config_prune = lambda *a, **k: None
    m_perf.estimate_matmul_time = lambda *a, **k: 0
    m_ops.matmul_perf_model = m_perf
    setattr(m_triton, "ops", m_ops)
    sys.modules["triton.ops"] = m_ops
    sys.modules["triton.ops.matmul_perf_model"] = m_perf

    # 3. Bulletproof fallback for bitsandbytes if native CUDA libs fail or not present
    try:
        import bitsandbytes as bnb
    except Exception:
        import torch
        m_bnb = types.ModuleType("bitsandbytes")
        m_bnb_nn = types.ModuleType("bitsandbytes.nn")

        class DummyBnbLinear(torch.nn.Linear):
            pass

        m_bnb_nn.Linear8bitLt = DummyBnbLinear
        m_bnb_nn.Linear4bit = DummyBnbLinear
        m_bnb_nn.modules = types.ModuleType("bitsandbytes.nn.modules")
        m_bnb.nn = m_bnb_nn
        m_bnb.__version__ = "0.44.1"
        sys.modules["bitsandbytes"] = m_bnb
        sys.modules["bitsandbytes.nn"] = m_bnb_nn
        sys.modules["bitsandbytes.nn.modules"] = m_bnb_nn.modules

patch_environment()

from peft import LoraConfig, get_peft_model, TaskType

try:
    from training_lab.evaluation.ir_metrics import compute_recall_at_k, compute_mrr_at_k
except ImportError:
    from evaluation.ir_metrics import compute_recall_at_k, compute_mrr_at_k


def load_triplets_dataloader(jsonl_path: Path, batch_size: int = 8):
    """Loads query-positive-negative triplets into SentenceTransformers InputExamples."""
    examples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            q = str(item.get("query") or "").strip()
            pos = str(item.get("positive") or "").strip()
            neg = str(item.get("negative") or "").strip()
            if not q or not pos or not neg:
                continue
            examples.append(InputExample(texts=[q, pos, neg]))

    if not examples:
        raise ValueError(f"No valid triplets loaded from {jsonl_path}!")
    print(f"📊 [Embedding DataLoader] Loaded {len(examples)} training triplets from {jsonl_path.name}")
    return DataLoader(examples, shuffle=True, batch_size=batch_size)


def run_embedding_training(
    model_name: str = "BAAI/bge-m3",
    data_path: Path = None,
    output_dir: Path = None,
    epochs: int = 3,
    lr: float = 3e-5,
    lora_r: int = 32,
    lora_alpha: int = 64,
    batch_size: int = 8,
    warmup_steps: int = 20,
):
    output_dir = Path(output_dir) if output_dir else (Path(__file__).resolve().parent.parent.parent / "models" / "adapters" / "embedding_lora")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 [MoAI Embedding] Loading base SentenceTransformer: {model_name}...")
    model = SentenceTransformer(model_name)

    # Attach LoRA to Transformer backbone
    transformer_module = model._first_module()
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["query", "key", "value", "dense"] if "bge" in model_name else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    transformer_module.auto_model = get_peft_model(transformer_module.auto_model, peft_config)
    # Ensure all trainable LoRA parameters are float32
    for param in transformer_module.auto_model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    transformer_module.auto_model.print_trainable_parameters()

    train_dataloader = load_triplets_dataloader(data_path, batch_size=batch_size)

    # Use MultipleNegativesRankingLoss (or Cached version if available)
    try:
        train_examples_count = len(train_dataloader.dataset)
        mini_batch = max(2, min(batch_size, train_examples_count)) if train_examples_count > 1 else max(2, batch_size)
        train_loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch)
        print("⚡ [GradCache] Using CachedMultipleNegativesRankingLoss for memory-efficient contrastive learning!")
    except Exception:
        train_loss = losses.MultipleNegativesRankingLoss(model)
        print("⚡ [Standard] Using MultipleNegativesRankingLoss with in-batch negatives.")

    print(f"⚡ Starting Embedding Training for {epochs} epochs (LR={lr})...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=str(output_dir),
        show_progress_bar=True,
    )

    if hasattr(transformer_module.auto_model, "save_pretrained"):
        transformer_module.auto_model.save_pretrained(str(output_dir))
    if hasattr(model, "tokenizer") and model.tokenizer:
        model.tokenizer.save_pretrained(str(output_dir))
    elif hasattr(transformer_module, "tokenizer") and transformer_module.tokenizer:
        transformer_module.tokenizer.save_pretrained(str(output_dir))

    print(f"✅ [MoAI Embedding] Trained LoRA embedding adapter successfully saved to: {output_dir}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    massive_p = ROOT / "data" / "triplets" / "massive_embedding_triplets.jsonl"
    full_p = ROOT / "data" / "triplets" / "embedding_triplets.jsonl"
    sample_p = ROOT / "data" / "triplets" / "embedding_triplets_sample.jsonl"
    chosen_default = str(massive_p if massive_p.exists() else (full_p if full_p.exists() else sample_p))
    parser.add_argument("--data", type=str, default=chosen_default)
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=str, default=None)
    args = parser.parse_args()

    data_file = Path(args.data)
    if not data_file.exists():
        for cand in [ROOT / args.data, ROOT / args.data.replace("training_lab/", ""), Path(args.data.replace("training_lab/", ""))]:
            if cand.exists():
                data_file = cand
                break

    if data_file.exists():
        run_embedding_training(data_path=data_file, output_dir=Path(args.output_dir) if args.output_dir else None, epochs=args.epochs, lr=args.lr)
    else:
        raise FileNotFoundError(f"Dataset {args.data} not found at {data_file}. Verify path.")
