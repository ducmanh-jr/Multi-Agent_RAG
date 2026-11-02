# -*- coding: utf-8 -*-
"""
SFT Trainer & FP16 Compact Weight Exporter for Agent 1 (Retrieval Router).
Ensures `adapter_model.safetensors` is cast to `torch.float16` (< 80MB) for direct GitHub storage.
"""

import torch
from safetensors.torch import load_file, save_file
from trl import SFTTrainer, SFTConfig
from .config import Agent1TrainingConfig
from .data_loader import load_router_dataset
from .model_builder import build_router_model_and_tokenizer


def export_fp16_safetensors(output_dir):
    """Casts saved adapter weights to FP16 so file size stays < 80MB (GitHub 100MB limit compliant)."""
    sf_path = output_dir / "adapter_model.safetensors"
    if sf_path.exists():
        tensors = load_file(str(sf_path))
        fp16_tensors = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in tensors.items()}
        save_file(fp16_tensors, str(sf_path))
        size_mb = sf_path.stat().st_size / (1024 * 1024)
        print(f"📦 [Agent 1 Exporter] Exported FP16 adapter: {sf_path} ({size_mb:.2f} MB)")


def run_router_sft_training(cfg: Agent1TrainingConfig):
    """Executes full SFT training loop on Kaggle Dual T4 and exports FP16 adapter weights."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, peft_config = build_router_model_and_tokenizer(cfg)
    train_ds, val_ds = load_router_dataset(cfg.dataset_path, tokenizer, cfg.val_split_ratio, cfg.seed)

    sft_args = SFTConfig(
        output_dir=str(cfg.output_dir / "checkpoints"),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        max_seq_length=cfg.max_seq_length,
        dataset_text_field="text",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        neftune_noise_alpha=cfg.neftune_noise_alpha,
        gradient_checkpointing=True,
        report_to="none",
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_args,
    )
    trainer.train()

    trainer.model.save_pretrained(str(cfg.output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(cfg.output_dir))
    export_fp16_safetensors(cfg.output_dir)
    return trainer
