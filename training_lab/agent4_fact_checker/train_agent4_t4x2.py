# -*- coding: utf-8 -*-
"""
Agent 4 (Fact-Checker 3-Class Cross-Encoder NLI Verifier) — Kaggle Dual-T4 Trainer.
Trains a 3-label Sequence Classification Cross-Encoder (`SUPPORT`, `CONTRADICT`, `NOT_ENOUGH_INFO`)
on `training_lab/agent4_fact_checker/dataset/nli_verifier.jsonl`.
Outputs FP16 weights to `training_lab/agent4_fact_checker/weights/`.
"""

import argparse
import gc
import os
import torch
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def train_agent4_nli(
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    dataset_path: str = "training_lab/agent4_fact_checker/dataset/nli_verifier.jsonl",
    output_dir: str = "training_lab/agent4_fact_checker/weights",
    epochs: int = 4,
    lr: float = 2e-5,
):
    print(f"🚀 [Agent 4 NLI Trainer] Fine-tuning {base_model} on {dataset_path}...")
    label2id = {"SUPPORT": 0, "CONTRADICT": 1, "NOT_ENOUGH_INFO": 2}
    id2label = {0: "SUPPORT", 1: "CONTRADICT", 2: "NOT_ENOUGH_INFO"}

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=3,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    raw_ds = load_dataset("json", data_files=dataset_path, split="train")

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
        per_device_train_batch_size=16,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        save_strategy="no",
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=tok_ds, processing_class=tokenizer)
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    sf_path = os.path.join(output_dir, "model.safetensors")
    if os.path.exists(sf_path):
        tensors = load_file(sf_path)
        save_file({k: v.to(torch.float16) for k, v in tensors.items()}, sf_path)

    del trainer, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"✅ [Agent 4 NLI Complete] Saved FP16 NLI Verifier weights to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--dataset", default="training_lab/agent4_fact_checker/dataset/nli_verifier.jsonl")
    parser.add_argument("--output-dir", default="training_lab/agent4_fact_checker/weights")
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    train_agent4_nli(args.base_model, args.dataset, args.output_dir, args.epochs)
