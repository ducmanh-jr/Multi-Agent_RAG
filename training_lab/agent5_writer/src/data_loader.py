# -*- coding: utf-8 -*-
"""
Dataset Loader for Agent 5 (Academic Report Writer SFT & RPO/DPO Preference Alignment).
"""

import json
from pathlib import Path
from datasets import Dataset


def load_writer_sft_dataset(jsonl_path: Path, tokenizer, val_ratio: float = 0.1, seed: int = 42) -> tuple[Dataset, Dataset]:
    """Loads `writer_sft.jsonl` scholarly synthesis samples and applies ChatML formatting."""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
            rows.append({"text": text})
    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=val_ratio, seed=seed)
    print(f"📊 [Agent 5 SFT DataLoader] Loaded {len(ds)} scholarly synthesis samples -> Train: {len(split['train'])}, Val: {len(split['test'])}")
    return split["train"], split["test"]


def load_writer_dpo_dataset(jsonl_path: Path, val_ratio: float = 0.1, seed: int = 42) -> tuple[Dataset, Dataset]:
    """Loads `writer_dpo.jsonl` preference pairs (structured IEEE/Markdown with exact citations vs uncited text)."""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=val_ratio, seed=seed)
    print(f"📊 [Agent 5 DPO DataLoader] Loaded {len(ds)} citation-alignment pairs -> Train: {len(split['train'])}, Val: {len(split['test'])}")
    return split["train"], split["test"]
