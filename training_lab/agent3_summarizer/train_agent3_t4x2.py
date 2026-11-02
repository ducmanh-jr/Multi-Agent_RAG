# -*- coding: utf-8 -*-
"""
Agent 3 (Summarizer: Generator SFT+DPO & Critic Hard-Negatives) — Kaggle Dual-T4 (32GB VRAM) Trainer.
Trains two specialist QLoRA adapters on `Qwen/Qwen2.5-7B-Instruct`:
  1. `weights/generator_qlora` : Stage 1A SFT (r=64, max_seq_length=3072) -> Stage 1B DPO Alignment (beta=0.1)
  2. `weights/critic_qlora`    : Stage 2 Hard-Negative SFT + DoRA (r=64, lora_alpha=128)
"""

import argparse
import gc
import os
import shutil
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer


def _free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _compress_fp16(folder: str):
    sf_path = os.path.join(folder, "adapter_model.safetensors")
    if os.path.exists(sf_path):
        tensors = load_file(sf_path)
        save_file({k: v.to(torch.float16) for k, v in tensors.items()}, sf_path)


def _load_base_t4x2(base_model: str):
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    is_t4x2 = gpu_count >= 2
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "quantization_config": bnb_config,
        "device_map": "balanced" if is_t4x2 else "auto",
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }
    if is_t4x2:
        kwargs["max_memory"] = {0: "14.5GiB", 1: "14.5GiB"}
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    return model, tokenizer, is_t4x2


def train_generator_sft_and_dpo(
    base_model: str,
    sft_jsonl: str,
    dpo_jsonl: str,
    output_dir: str,
    epochs: int = 2,
):
    print("🚀 [Agent 3 Generator] Stage 1A: Grounded Synthesis SFT on Dual T4...")
    sft_tmp = output_dir + "_tmp_sft"
    model, tokenizer, is_t4x2 = _load_base_t4x2(base_model)
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=64 if is_t4x2 else 32,
        lora_alpha=128 if is_t4x2 else 64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    sft_ds = load_dataset("json", data_files=sft_jsonl, split="train")
    sft_args = SFTConfig(
        output_dir=sft_tmp,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.08,
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        max_seq_length=3072 if is_t4x2 else 2048,
        neftune_noise_alpha=5.0,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )
    trainer = SFTTrainer(model=model, train_dataset=sft_ds, peft_config=peft_config, processing_class=tokenizer, args=sft_args)
    trainer.train()
    os.makedirs(sft_tmp, exist_ok=True)
    trainer.model.save_pretrained(sft_tmp)
    tokenizer.save_pretrained(sft_tmp)
    del trainer, model, tokenizer
    _free_vram()

    print("🚀 [Agent 3 Generator] Stage 1B: DPO Preference Alignment (beta=0.1) on Dual T4...")
    model, tokenizer, is_t4x2 = _load_base_t4x2(base_model)
    model = PeftModel.from_pretrained(model, sft_tmp, is_trainable=True)
    dpo_ds = load_dataset("json", data_files=dpo_jsonl, split="train")
    dpo_args = DPOConfig(
        output_dir=output_dir,
        beta=0.1,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        max_length=3072 if is_t4x2 else 2048,
        max_prompt_length=2048 if is_t4x2 else 1536,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )
    dpo_trainer = DPOTrainer(model=model, ref_model=None, args=dpo_args, train_dataset=dpo_ds, processing_class=tokenizer)
    dpo_trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _compress_fp16(output_dir)
    del dpo_trainer, model, tokenizer
    _free_vram()
    shutil.rmtree(sft_tmp, ignore_errors=True)
    print(f"✅ [Agent 3 Generator Complete] Saved SFT+DPO FP16 weights to: {output_dir}")


def train_critic_sft(
    base_model: str,
    critic_jsonl: str,
    output_dir: str,
    epochs: int = 2,
):
    print("🚀 [Agent 3 Critic] Training Hard-Negative Inner Critic QLoRA on Dual T4...")
    model, tokenizer, is_t4x2 = _load_base_t4x2(base_model)
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
    ds = load_dataset("json", data_files=critic_jsonl, split="train")
    sft_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.08,
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        max_seq_length=3072 if is_t4x2 else 2048,
        neftune_noise_alpha=5.0,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )
    trainer = SFTTrainer(model=model, train_dataset=ds, peft_config=peft_config, processing_class=tokenizer, args=sft_args)
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _compress_fp16(output_dir)
    del trainer, model, tokenizer
    _free_vram()
    print(f"✅ [Agent 3 Critic Complete] Saved FP16 weights to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--part", choices=["generator", "critic", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    base_dir = "training_lab/agent3_summarizer"
    if args.part in ("generator", "both"):
        train_generator_sft_and_dpo(
            args.base_model,
            f"{base_dir}/dataset/generator_sft.jsonl",
            f"{base_dir}/dataset/generator_dpo.jsonl",
            f"{base_dir}/weights/generator_qlora",
            epochs=args.epochs,
        )
    if args.part in ("critic", "both"):
        train_critic_sft(
            args.base_model,
            f"{base_dir}/dataset/critic_sft.jsonl",
            f"{base_dir}/weights/critic_qlora",
            epochs=args.epochs,
        )
