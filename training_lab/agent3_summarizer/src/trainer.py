# -*- coding: utf-8 -*-
"""
Two-Stage SFT + DPO Trainer for Generator (`generator_qlora`) & SFT Trainer for Critic (`critic_qlora`).
Exports FP16 `adapter_model.safetensors` (< 80MB) directly to `weights/`.
"""

import torch
from pathlib import Path
from safetensors.torch import load_file, save_file
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
from .config import Agent3TrainingConfig
from .data_loader import load_sft_jsonl_dataset, load_dpo_jsonl_dataset
from .model_builder import build_summarizer_model_and_tokenizer


def cast_adapter_to_fp16(out_dir: Path):
    sf_path = out_dir / "adapter_model.safetensors"
    if sf_path.exists():
        tensors = load_file(str(sf_path))
        fp16_tensors = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in tensors.items()}
        save_file(fp16_tensors, str(sf_path))
        mb = sf_path.stat().st_size / (1024 * 1024)
        print(f"📦 [Agent 3 Exporter] Saved FP16 adapter: {sf_path} ({mb:.2f} MB)")


def train_generator_sft_and_dpo(cfg: Agent3TrainingConfig):
    """Runs Stage 1 SFT followed by Stage 2 Direct Preference Optimization (DPO) for Generator."""
    out_dir = cfg.weights_dir / "generator_qlora"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, peft_config = build_summarizer_model_and_tokenizer(
        cfg.base_model_id, cfg.gen_lora_r, cfg.gen_lora_alpha, cfg.lora_dropout, cfg.use_dora, cfg.max_memory_per_gpu
    )
    train_sft, val_sft = load_sft_jsonl_dataset(cfg.dataset_dir / "generator_sft.jsonl", tokenizer, seed=cfg.seed)

    sft_args = SFTConfig(
        output_dir=str(out_dir / "sft_checkpoints"),
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.gen_lr_sft,
        num_train_epochs=cfg.gen_epochs_sft,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
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

    # Stage 2: DPO Alignment
    dpo_file = cfg.dataset_dir / "generator_dpo.jsonl"
    if dpo_file.exists():
        train_dpo, val_dpo = load_dpo_jsonl_dataset(dpo_file, seed=cfg.seed)
        dpo_args = DPOConfig(
            output_dir=str(out_dir / "dpo_checkpoints"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=cfg.gen_lr_dpo,
            num_train_epochs=cfg.gen_epochs_dpo,
            beta=cfg.dpo_beta,
            rpo_alpha=0.1,  # RPO (Regularized Preference Optimization: DPO + 0.1 * NLL loss)
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

    final_model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    cast_adapter_to_fp16(out_dir)
    del model
    torch.cuda.empty_cache()


def train_critic_sft(cfg: Agent3TrainingConfig):
    """Runs Hard-Negative Adversarial SFT for Critic (`critic_qlora`)."""
    out_dir = cfg.weights_dir / "critic_qlora"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, peft_config = build_summarizer_model_and_tokenizer(
        cfg.base_model_id, cfg.critic_lora_r, cfg.critic_lora_alpha, cfg.lora_dropout, cfg.use_dora, cfg.max_memory_per_gpu
    )
    train_ds, val_ds = load_sft_jsonl_dataset(cfg.dataset_dir / "critic_sft.jsonl", tokenizer, seed=cfg.seed)

    sft_args = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.critic_lr,
        num_train_epochs=cfg.critic_epochs,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        max_seq_length=cfg.critic_max_seq_length,
        dataset_text_field="text",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        neftune_noise_alpha=cfg.neftune_noise_alpha,
        gradient_checkpointing=True,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model, train_dataset=train_ds, eval_dataset=val_ds, peft_config=peft_config, processing_class=tokenizer, args=sft_args
    )
    trainer.train()

    trainer.model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    cast_adapter_to_fp16(out_dir)
    del model
    torch.cuda.empty_cache()
