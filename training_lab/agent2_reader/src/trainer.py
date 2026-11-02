# -*- coding: utf-8 -*-
"""
SOTA Training Pipeline for Agent 2 (LayoutLMv3 Document Reader) on Kaggle Dual T4.
Includes Multi-Class Focal Loss (gamma=2.0) to boost rare scientific layout classes (`Formula`, `Table`, `Footnote`).
"""

import json
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from .config import Agent2TrainingConfig, DOCLAYNET_LABELS
from .data_loader import load_doclaynet_dataset
from .model_builder import build_layoutlmv3_model


class FocalLossLayoutTrainer(Trainer):
    """Custom Trainer using Multi-Class Focal Loss (gamma=2.0) for imbalanced document layout tokens."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if labels is not None:
            ce_loss = F.cross_entropy(logits.view(-1, model.config.num_labels), labels.view(-1), reduction="none", ignore_index=-100)
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** 2.0 * ce_loss).mean()
            return (focal_loss, outputs) if return_outputs else focal_loss
        return (outputs.loss, outputs) if return_outputs else outputs.loss


def run_reader_training(cfg: Agent2TrainingConfig):
    """Executes full Multi-Class Focal Loss training on LayoutLMv3 and exports weights + label_map.json."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = build_layoutlmv3_model(cfg)
    train_ds, val_ds = load_doclaynet_dataset(cfg.dataset_path, tokenizer, cfg.max_seq_length, cfg.val_split_ratio, cfg.seed)

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=10,
        report_to="none",
        seed=cfg.seed,
    )

    trainer = FocalLossLayoutTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()

    trainer.model.save_pretrained(str(cfg.output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(cfg.output_dir))
    label_map = {
        "id2label": {str(i): l for i, l in enumerate(DOCLAYNET_LABELS)},
        "label2id": {l: i for i, l in enumerate(DOCLAYNET_LABELS)},
        "loss_function": "MultiClassFocalLoss(gamma=2.0)",
    }
    (cfg.output_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    print(f"✅ [Agent 2 Trainer] Saved LayoutLMv3 SOTA Focal-Loss weights & label_map.json to {cfg.output_dir}")
    return trainer.model, tokenizer
