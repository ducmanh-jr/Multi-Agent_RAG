# -*- coding: utf-8 -*-
"""
Agent 1 (Retrieval Router & Supervisor Planner) — Kaggle Dual-T4 (32GB VRAM) Fine-Tuning Script.
Trains `Qwen/Qwen2.5-7B-Instruct` with QLoRA 4-bit NF4 + DoRA (r=64, lora_alpha=128) + NEFTune
on `training_lab/agent1_retrieval/dataset/router_sft.jsonl`.
Outputs FP16 weights (<80MB) to `training_lab/agent1_retrieval/weights/`.
"""

import argparse
import gc
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def train_agent1_router(
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    dataset_path: str = "training_lab/agent1_retrieval/dataset/router_sft.jsonl",
    output_dir: str = "training_lab/agent1_retrieval/weights",
    epochs: int = 3,
    lr: float = 2e-4,
):
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    is_t4x2 = gpu_count >= 2
    print(f"🚀 [Agent 1 Router Trainer] GPUs detected: {gpu_count} (T4x2 Mode: {is_t4x2})")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "balanced" if is_t4x2 else "auto",
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }
    if is_t4x2:
        model_kwargs["max_memory"] = {0: "14.5GiB", 1: "14.5GiB"}

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=64 if is_t4x2 else 32,
        lora_alpha=128 if is_t4x2 else 64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=is_t4x2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    sft_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.08,
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        max_seq_length=2048,
        neftune_noise_alpha=5.0,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_args,
    )
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    sf_path = os.path.join(output_dir, "adapter_model.safetensors")
    if os.path.exists(sf_path):
        tensors = load_file(sf_path)
        save_file({k: v.to(torch.float16) for k, v in tensors.items()}, sf_path)

    del trainer, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"✅ [Agent 1 Router Complete] Saved FP16 weights to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", default="training_lab/agent1_retrieval/dataset/router_sft.jsonl")
    parser.add_argument("--output-dir", default="training_lab/agent1_retrieval/weights")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    train_agent1_router(args.base_model, args.dataset, args.output_dir, args.epochs)
