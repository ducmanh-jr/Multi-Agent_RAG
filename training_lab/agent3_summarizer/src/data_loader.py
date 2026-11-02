# -*- coding: utf-8 -*-
"""
Dataset Loader for Agent 3 (Generator SFT, Generator DPO Preference Pairs, and Critic Hard-Negatives).
"""

import json
from pathlib import Path
from datasets import Dataset


def load_sft_jsonl_dataset(jsonl_path: Path, tokenizer, val_ratio: float = 0.1, seed: int = 42) -> tuple[Dataset, Dataset]:
    """Loads ChatML JSONL (`generator_sft.jsonl` or `critic_sft.jsonl`) and splits into Train/Val."""
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
    print(f"📊 [Agent 3 SFT DataLoader] {jsonl_path.name}: {len(ds)} total -> Train: {len(split['train'])}, Val: {len(split['test'])}")
    return split["train"], split["test"]


def load_dpo_jsonl_dataset(jsonl_path: Path, val_ratio: float = 0.1, seed: int = 42) -> tuple[Dataset, Dataset]:
    """Loads DPO Preference Pairs (`prompt`, `chosen`, `rejected`) from `generator_dpo.jsonl`."""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=val_ratio, seed=seed)
    print(f"📊 [Agent 3 DPO DataLoader] {jsonl_path.name}: {len(ds)} pairs -> Train: {len(split['train'])}, Val: {len(split['test'])}")
    return split["train"], split["test"]
