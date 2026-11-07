#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

# Chạy SFT Router với hỗ trợ phân bổ tự động Dual GPU T4x2:

import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Agent1TrainingConfig


def sanitize_torchvision():
    """Tự động vô hiệu hóa cờ torchvision trong transformers nếu torchvision bị lỗi C++ operator."""
    try:
        import torchvision
        _ = getattr(torchvision.ops, "nms", None)
    except Exception:
        try:
            import transformers.utils.import_utils as tu
            tu._torchvision_available = False
            tu.is_torchvision_available = lambda: False
        except Exception:
            pass


def main():
    sanitize_torchvision()
    parser = argparse.ArgumentParser(description="Fine-tune Agent 1 Retrieval Router on Kaggle T4x2")
    parser.add_argument("--epochs", type=int, default=3, help="Number of SFT training epochs")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Peak learning rate")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA/DoRA rank r")
    parser.add_argument("--data", type=str, default=None, help="Custom dataset path")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    args = parser.parse_args()

    from src.trainer import run_router_sft_training
    from src.evaluator import evaluate_router_adapter

    kwargs = {"epochs": args.epochs, "learning_rate": args.lr, "lora_r": args.lora_r, "lora_alpha": args.lora_r * 2}
    if args.data: kwargs["dataset_path"] = Path(args.data)
    if args.output_dir: kwargs["output_dir"] = Path(args.output_dir)
    cfg = Agent1TrainingConfig(**kwargs)
    if args.eval_only:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, str(cfg.output_dir))
        evaluate_router_adapter(model, tokenizer)
    else:
        trainer = run_router_sft_training(cfg)
        tok = getattr(trainer, "processing_class", getattr(trainer, "tokenizer", None))
        if tok is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(cfg.base_model_id)
        evaluate_router_adapter(trainer.model, tok)


if __name__ == "__main__":
    main()
