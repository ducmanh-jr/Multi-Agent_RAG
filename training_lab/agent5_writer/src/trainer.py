# -*- coding: utf-8 -*-
"""
Two-Stage SFT + RPO/DPO Trainer for Agent 5 (Academic Report Writer) on Kaggle Dual T4.
Exports compact FP16 `adapter_model.safetensors` (< 80MB) for direct GitHub storage and Multi-LoRA serving.
"""

import torch
from pathlib import Path
from safetensors.torch import load_file, save_file
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
from .config import Agent5TrainingConfig
from .data_loader import load_writer_sft_dataset, load_writer_dpo_dataset
from .model_builder import build_writer_model_and_tokenizer


def export_fp16_adapter(out_dir: Path):
    sf_path = out_dir / "adapter_model.safetensors"
    if sf_path.exists():
        tensors = load_file(str(sf_path))
        fp16_tensors = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in tensors.items()}
        save_file(fp16_tensors, str(sf_path))
        mb = sf_path.stat().st_size / (1024 * 1024)
        print(f"📦 [Agent 5 Exporter] Saved FP16 Writer adapter: {sf_path} ({mb:.2f} MB)")


def run_writer_sft_and_rpo_training(cfg: Agent5TrainingConfig):
    """Runs Stage 1 SFT followed by Stage 2 RPO/DPO Preference Alignment for Academic Report Synthesis."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, peft_config = build_writer_model_and_tokenizer(cfg)
    train_sft, val_sft = load_writer_sft_dataset(cfg.dataset_dir / "writer_sft.jsonl", tokenizer, seed=cfg.seed)

    sft_args = SFTConfig(
        output_dir=str(cfg.output_dir / "sft_checkpoints"),
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr_sft,
        num_train_epochs=cfg.epochs_sft,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        max_seq_length=cfg.max_seq_length,
        dataset_text_field="text",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        neftune_noise_alpha=cfg.neftune_noise_alpha,
        gradient_checkpointing=True,
        report_to="none",
    )
    sft_trainer = SFTTrainer(
        model=model, train_dataset=train_sft, eval_dataset=val_sft, peft_config=peft_config, processing_class=tokenizer, args=sft_args
    )
    sft_trainer.train()

    dpo_file = cfg.dataset_dir / "writer_dpo.jsonl"
    if dpo_file.exists():
        train_dpo, val_dpo = load_writer_dpo_dataset(dpo_file, seed=cfg.seed)
        dpo_args = DPOConfig(
            output_dir=str(cfg.output_dir / "dpo_checkpoints"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=cfg.lr_dpo,
            num_train_epochs=cfg.epochs_dpo,
            beta=cfg.dpo_beta,
            rpo_alpha=cfg.rpo_alpha,
            fp16=True,
            logging_steps=10,
            max_length=2048,
            max_prompt_length=1024,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            gradient_checkpointing=True,
            report_to="none",
        )
        dpo_trainer = DPOTrainer(
            model=sft_trainer.model, ref_model=None, args=dpo_args, train_dataset=train_dpo, eval_dataset=val_dpo, processing_class=tokenizer
        )
        dpo_trainer.train()
        final_model = dpo_trainer.model
    else:
        final_model = sft_trainer.model

    final_model.save_pretrained(str(cfg.output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(cfg.output_dir))
    export_fp16_adapter(cfg.output_dir)
    return final_model, tokenizer
