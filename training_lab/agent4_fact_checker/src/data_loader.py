# -*- coding: utf-8 -*-
"""
Dataset Loader & Cross-Encoder Pair Tokenizer for Agent 4 (SciFact + VitaminC NLI Verifier).
Supports both (`premise`, `hypothesis`) and (`evidence`, `claim`) schema formats.
"""

import json
from pathlib import Path
from datasets import Dataset
from .config import LABEL2ID


def load_nli_dataset(dataset_path: Path, tokenizer, max_length: int = 512, val_ratio: float = 0.15, seed: int = 42):
    """Loads `nli_verifier.jsonl` and tokenizes premise/hypothesis pairs for Cross-Encoder NLI."""
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            premise = row.get("premise") or row.get("evidence") or ""
            hypothesis = row.get("hypothesis") or row.get("claim") or ""
            if not premise or not hypothesis:
                continue

            if "label_id" in row:
                lbl = int(row["label_id"])
            else:
                label_str = str(row.get("label", "NOT_ENOUGH_INFO")).upper()
                lbl = LABEL2ID.get(label_str, 2)

            records.append({
                "premise": premise,
                "hypothesis": hypothesis,
                "label": lbl,
                "labels": lbl,
            })

    ds = Dataset.from_list(records)
    split = ds.train_test_split(test_size=val_ratio, seed=seed)

    def tokenize_fn(batch):
        return tokenizer(batch["premise"], batch["hypothesis"], truncation=True, max_length=max_length, padding="max_length")

    train_ds = split["train"].map(tokenize_fn, batched=True)
    val_ds = split["test"].map(tokenize_fn, batched=True)
    print(f"📊 [Agent 4 DataLoader] Loaded {len(ds)} NLI pairs -> Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds
