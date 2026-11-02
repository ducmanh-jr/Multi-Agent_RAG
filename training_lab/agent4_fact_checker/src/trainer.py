# -*- coding: utf-8 -*-
"""
SOTA Weighted Focal + R-Drop Cross-Encoder Trainer for Agent 4 (Fact-Checker NLI Verifier) on Kaggle Dual T4.
Combines:
1. Class-Weighted Focal Loss (gamma=2.0, CONTRADICT weight = 1.35x)
2. R-Drop Regularization (Bidirectional KL-divergence consistency between two forward passes)
3. Post-training Temperature Scaling Calibration (ECE minimization)
"""

import json
import torch
import inspect
import dataclasses
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from .config import Agent4TrainingConfig
from .data_loader import load_nli_dataset
from .model_builder import build_nli_cross_encoder
from .evaluator import compute_nli_metrics, calibrate_temperature_ece


def safe_build_config(config_cls, **kwargs):
    """Safely instantiates TrainingArguments across different transformers versions."""
    try:
        valid_fields = {f.name for f in dataclasses.fields(config_cls)}
    except TypeError:
        valid_fields = set(inspect.signature(config_cls.__init__).parameters.keys())

    if "warmup_steps" in valid_fields and "warmup_ratio" in kwargs and "warmup_ratio" not in valid_fields:
        ratio = kwargs.pop("warmup_ratio")
        if "warmup_steps" not in kwargs:
            kwargs["warmup_steps"] = max(1, int(ratio * 100))

    if "eval_strategy" in valid_fields and "evaluation_strategy" in kwargs and "evaluation_strategy" not in valid_fields:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    elif "evaluation_strategy" in valid_fields and "eval_strategy" in kwargs and "eval_strategy" not in valid_fields:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
    return config_cls(**filtered)


class WeightedCrossEncoderTrainer(Trainer):
    """SOTA Trainer with Class-Weighted Focal Loss (gamma=2.0) + R-Drop KL Consistency Regularization."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs1 = model(**inputs)
        logits1 = outputs1.get("logits")

        weights = torch.tensor([1.0, 1.35, 1.15], device=logits1.device, dtype=logits1.dtype)
        ce1 = F.cross_entropy(logits1.view(-1, self.model.config.num_labels), labels.view(-1), weight=weights, reduction="none")
        pt1 = torch.exp(-ce1)
        focal1 = ((1.0 - pt1) ** 2.0 * ce1).mean()

        if model.training:
            # R-Drop second forward pass for KL consistency regularization
            outputs2 = model(**inputs)
            logits2 = outputs2.get("logits")
            ce2 = F.cross_entropy(logits2.view(-1, self.model.config.num_labels), labels.view(-1), weight=weights, reduction="none")
            pt2 = torch.exp(-ce2)
            focal2 = ((1.0 - pt2) ** 2.0 * ce2).mean()

            p1 = F.log_softmax(logits1, dim=-1)
            p2 = F.log_softmax(logits2, dim=-1)
            kl_loss = 0.5 * (F.kl_div(p1, p2.exp(), reduction="batchmean") + F.kl_div(p2, p1.exp(), reduction="batchmean"))
            loss = 0.5 * (focal1 + focal2) + 0.3 * kl_loss
        else:
            loss = focal1

        return (loss, outputs1) if return_outputs else loss


def run_factchecker_training(cfg: Agent4TrainingConfig):
    """Executes full Cross-Encoder NLI training with R-Drop + Focal Loss on Kaggle Dual T4."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = build_nli_cross_encoder(cfg)
    train_ds, val_ds = load_nli_dataset(cfg.dataset_path, tokenizer, cfg.max_seq_length, cfg.val_split_ratio, cfg.seed)

    training_args = safe_build_config(
        TrainingArguments,
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
        warmup_steps=10,
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
    print("🚀 [Agent 4 Trainer] Starting R-Drop + Focal Loss NLI training...")
    trainer.train()
    metrics = trainer.evaluate()

    preds_out = trainer.predict(val_ds)
    opt_temp = calibrate_temperature_ece(preds_out.predictions, preds_out.label_ids)
    metrics["calibrated_temperature"] = opt_temp

    trainer.model.save_pretrained(str(cfg.output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(cfg.output_dir))
    (cfg.output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"✅ [Agent 4 Trainer] Saved SOTA R-Drop + Focal-Loss NLI Verifier (T={opt_temp}) to {cfg.output_dir}")
    return trainer, metrics
