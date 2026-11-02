# -*- coding: utf-8 -*-
"""
Dataset Loader & Spatial Bounding-Box Tokenizer for Agent 2 (LayoutLMv3 Document Reader).
"""

import json
from pathlib import Path
from datasets import Dataset


def load_doclaynet_dataset(dataset_path: Path, tokenizer, max_length: int = 512, val_split_ratio: float = 0.1, seed: int = 42):
    """Loads DocLayNet JSONL samples (tokens, bboxes, ner_tags) and encodes for LayoutLMv3."""
    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            tokens = item.get("tokens", [])
            bboxes = item.get("bboxes", [])
            tags = item.get("ner_tags", [])
            if not tokens:
                continue
            rows.append({"tokens": tokens, "bboxes": bboxes, "ner_tags": tags})

    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=val_split_ratio, seed=seed)
    print(f"📊 [Agent 2 DataLoader] Loaded {len(ds)} LayoutLMv3 pages -> Train: {len(split['train'])}, Val: {len(split['test'])}")
    return split["train"], split["test"]
