# -*- coding: utf-8 -*-
"""
Dataset Loader & Cross-Encoder Pair Tokenizer for Agent 4 (SciFact + VitaminC NLI Verifier).
"""

import json
from pathlib import Path
from datasets import Dataset
from .config import LABEL2ID


def load_nli_dataset(dataset_path: Path, tokenizer, max_length: int = 512, val_ratio: float = 0.15, seed: int = 42):
    """Loads `nli_verifier.jsonl` (`claim`, `evidence`, `label`) and tokenizes pairs for Cross-Encoder NLI."""
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            label_str = row.get("label", "NOT_ENOUGH_INFO").upper()
            records.append({
                "claim": row["claim"],
                "evidence": row["evidence"],
                "label": LABEL2ID.get(label_str, 2),
            })

    ds = Dataset.from_list(records)
    split = ds.train_test_split(test_size=val_ratio, seed=seed)

    def tokenize_fn(batch):
        return tokenizer(batch["claim"], batch["evidence"], truncation=True, max_length=max_length, padding="max_length")

    train_ds = split["train"].map(tokenize_fn, batched=True)
    val_ds = split["test"].map(tokenize_fn, batched=True)
    print(f"📊 [Agent 4 DataLoader] Loaded {len(ds)} NLI pairs -> Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds
