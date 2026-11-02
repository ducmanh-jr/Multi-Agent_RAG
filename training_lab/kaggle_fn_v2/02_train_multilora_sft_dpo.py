"""
Step 2 of Kaggle FN v2 Fine-Tuning Pipeline:
Trains the 3 QLoRA Specialists ('router', 'generator' SFT+DPO, 'critic' SFT) on Qwen/Qwen2.5-7B-Instruct
and the 'agent4_nli' Verifier on MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
using Kaggle Dual Tesla T4 (2 x 16 GB VRAM, float16 + NF4 4-bit).

Usage on Kaggle:
  python 02_train_multilora_sft_dpo.py --adapter all --data_dir ./datasets --output_dir ./kaggle_fn_v2_weights
"""

from __future__ import annotations

import argparse
import os


def train_qlora_sft(
    base_model: str,
    train_jsonl: str,
    output_dir: str,
    lora_r: int = 32,
    lora_alpha: int = 64,
    epochs: int = 3,
    lr: float = 2e-4,
):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # T4 (sm_75) uses float16
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)

    dataset = load_dataset("json", data_files=train_jsonl, split="train")

    def _format_chat(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = dataset.map(_format_chat)

    sft_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=3072,
        neftune_noise_alpha=5.0,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_args,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _compress_safetensors_fp16(output_dir)
    del trainer, model, tokenizer
    _free_vram()
    print(f"[OK] Saved QLoRA SFT adapter (FP16) to: {output_dir}")


def _free_vram():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _compress_safetensors_fp16(folder: str):
    """Ensure saved adapter_model.safetensors is stored in FP16 (<80MB) for fast Drive & GitHub sync."""
    import torch
    from safetensors.torch import load_file, save_file
    sf_path = os.path.join(folder, "adapter_model.safetensors")
    if os.path.exists(sf_path):
        tensors = load_file(sf_path)
        save_file({k: v.to(torch.float16) for k, v in tensors.items()}, sf_path)


def train_generator_dpo(
    base_model: str,
    sft_adapter_dir: str,
    dpo_jsonl: str,
    output_dir: str,
    beta: float = 0.1,
    lr: float = 5e-6,
):
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, sft_adapter_dir, is_trainable=True)
    dataset = load_dataset("json", data_files=dpo_jsonl, split="train")

    dpo_args = DPOConfig(
        output_dir=output_dir,
        beta=beta,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=2560,
        max_prompt_length=1792,
        report_to="none",
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    _compress_safetensors_fp16(output_dir)
    del trainer, model, tokenizer
    _free_vram()
    print(f"[OK] Saved DPO-aligned Generator adapter (FP16) to: {output_dir}")


def train_nli_verifier(
    data_path: str,
    output_dir: str,
    base_nli_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    epochs: int = 2,
):
    """Stage 4: Fine-tune Agent 4 Fact-Checker NLI Cross-Encoder on nli_verifier_train.jsonl."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    from datasets import load_dataset

    label2id = {"SUPPORT": 0, "CONTRADICT": 1, "NOT_ENOUGH_INFO": 2}
    id2label = {0: "SUPPORT", 1: "CONTRADICT", 2: "NOT_ENOUGH_INFO"}

    tokenizer = AutoTokenizer.from_pretrained(base_nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_nli_model,
        num_labels=3,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    raw_ds = load_dataset("json", data_files=data_path, split="train")

    def _preprocess(batch):
        enc = tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        enc["labels"] = [label2id.get(str(lbl).upper(), 2) for lbl in batch["label"]]
        return enc

    tok_ds = raw_ds.map(_preprocess, batched=True)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=tok_ds, processing_class=tokenizer)
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[OK] Saved Agent 4 NLI Verifier to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", "--base-model", dest="base_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--adapter",
        choices=["router", "generator", "critic", "nli", "all"],
        default="all",
        help="Chọn Agent cần huấn luyện: router (Agent 1), generator (Agent 3 Gen), critic (Agent 3 Critic), nli (Agent 4 FactChecker), hoặc all",
    )
    parser.add_argument("--data_dir", "--dataset-dir", dest="data_dir", default="training_lab/kaggle_fn_v2/datasets")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="training_lab/weights_drive_bundle")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.adapter in ("router", "all"):
        train_qlora_sft(
            args.base_model,
            os.path.join(args.data_dir, "router_sft.jsonl"),
            os.path.join(args.output_dir, "agent1_router"),
            epochs=args.epochs,
        )
    if args.adapter in ("generator", "all"):
        sft_dir = os.path.join(args.output_dir, "generator_qlora_sft")
        final_gen_dir = os.path.join(args.output_dir, "generator_qlora", "final")
        train_qlora_sft(
            args.base_model,
            os.path.join(args.data_dir, "generator_sft.jsonl"),
            sft_dir,
            epochs=args.epochs,
        )
        train_generator_dpo(
            args.base_model,
            sft_dir,
            os.path.join(args.data_dir, "generator_dpo.jsonl"),
            final_gen_dir,
        )
    if args.adapter in ("critic", "all"):
        train_qlora_sft(
            args.base_model,
            os.path.join(args.data_dir, "critic_sft.jsonl"),
            os.path.join(args.output_dir, "critic_qlora", "final"),
            epochs=args.epochs,
        )
    if args.adapter in ("nli", "all"):
        nli_file = os.path.join(args.data_dir, "nli_verifier_train.jsonl")
        if os.path.exists(nli_file):
            train_nli_verifier(
                nli_file,
                os.path.join(args.output_dir, "fact_checker_nli"),
                epochs=args.epochs,
            )
