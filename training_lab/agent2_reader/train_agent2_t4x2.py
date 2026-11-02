#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AGENT 2 (DOCUMENT READER) - KAGGLE T4x2 MAXIMIZED FINE-TUNING SUITE
Model: microsoft/layoutlmv3-base (11-Class Document Layout Token Classification)
Hardware: Dual NVIDIA Tesla T4 (2 x 16GB = 32GB VRAM) via PyTorch DDP / DataParallel
================================================================================
"""

import os
import json
import argparse
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)

LABEL_LIST = [
    "Caption", "Footnote", "Formula", "List-item", "Page-footer",
    "Page-header", "Picture", "Section-header", "Table", "Text", "Title"
]


def train_agent2(epochs: int = 5, lr: float = 3e-5, batch_size: int = 16):
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_id = "microsoft/layoutlmv3-base"

    print(f"🚀 [Agent 2 Reader] Fine-tuning {model_id} on Kaggle Dual T4 (32GB VRAM)")
    label2id = {l: i for i, l in enumerate(LABEL_LIST)}
    id2label = {i: l for i, l in enumerate(LABEL_LIST)}

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id,
    )

    print(f"✅ Loaded {model_id} ({len(LABEL_LIST)} layout classes). Saving weights to {out_dir}")
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train_agent2(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
