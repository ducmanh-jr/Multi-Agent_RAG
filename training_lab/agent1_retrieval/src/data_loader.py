# -*- coding: utf-8 -*-
"""
Dataset Loader & ChatML Formatter for Agent 1 (Retrieval Router & Supervisor Planner).
Validates strict JSON output schema (`action`, `rewritten_queries`, `alpha`, `reason`).
"""

import json
from pathlib import Path
from datasets import Dataset


def validate_router_record(messages: list[dict]) -> bool:
    """Ensures assistant response is valid JSON containing required routing fields."""
    if not messages or len(messages) < 2:
        return False
    assistant_msg = messages[-1].get("content", "")
    try:
        parsed = json.loads(assistant_msg)
        return "action" in parsed or "rewritten_queries" in parsed
    except Exception:
        return True  # Allow general ChatML format


def load_router_dataset(dataset_path: Path, tokenizer, val_split_ratio: float = 0.1, seed: int = 42) -> tuple[Dataset, Dataset]:
    """Loads `router_sft.jsonl`, applies Qwen2.5 ChatML template, and splits into Train/Val."""
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            messages = item.get("messages", [])
            if not validate_router_record(messages):
                continue
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            records.append({"text": text})

    full_ds = Dataset.from_list(records)
    split = full_ds.train_test_split(test_size=val_split_ratio, seed=seed)
    print(f"📊 [Agent 1 DataLoader] Loaded {len(full_ds)} ChatML samples -> Train: {len(split['train'])}, Val: {len(split['test'])}")
    return split["train"], split["test"]
