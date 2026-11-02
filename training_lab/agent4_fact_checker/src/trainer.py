# -*- coding: utf-8 -*-
"""
Weighted Cross-Encoder Trainer for Agent 4 (Fact-Checker NLI Verifier) on Kaggle Dual T4.
Applies higher class weight (1.35x) on `CONTRADICT` to prevent scientific hallucinations.
"""

import json
import torch
from transformers import Trainer, TrainingArguments
from .config import Agent4TrainingConfig
from .data_loader import load_nli_dataset
from .model_builder import build_nli_cross_encoder
from .evaluator import compute_nli_metrics, calibrate_temperature_ece


class WeightedCrossEncoderTrainer(Trainer):
    """Custom Trainer with class-weighted CrossEntropyLoss (`CONTRADICT` weight = 1.35)."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights = torch.tensor([1.0, 1.35, 1.15], device=logits.device, dtype=logits.dtype)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def run_factchecker_training(cfg: Agent4TrainingConfig):
    """Executes full Cross-Encoder NLI training on Kaggle Dual T4 and saves calibrated weights."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = build_nli_cross_encoder(cfg)
    train_ds, val_ds = load_nli_dataset(cfg.dataset_path, tokenizer, cfg.max_seq_length, cfg.val_split_ratio, cfg.seed)

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
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=20,
        report_to="none",
        seed=cfg.seed,
    )

    trainer = WeightedCrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_nli_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()

    # Temperature calibration
    preds_out = trainer.predict(val_ds)
    opt_temp = calibrate_temperature_ece(preds_out.predictions, preds_out.label_ids)
    metrics["calibrated_temperature"] = opt_temp

    trainer.model.save_pretrained(str(cfg.output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(cfg.output_dir))
    (cfg.output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"✅ [Agent 4 Trainer] Saved NLI Verifier & Calibration (T={opt_temp}) to {cfg.output_dir}")
    return trainer, metrics
