# -*- coding: utf-8 -*-
"""
Training Pipeline for Agent 2 (LayoutLMv3 Document Reader) on Kaggle Dual T4.
"""

import json
from .config import Agent2TrainingConfig, DOCLAYNET_LABELS
from .data_loader import load_doclaynet_dataset
from .model_builder import build_layoutlmv3_model


def run_reader_training(cfg: Agent2TrainingConfig):
    """Loads dataset, initializes LayoutLMv3, and saves weights + label_map.json."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = build_layoutlmv3_model(cfg)
    train_ds, val_ds = load_doclaynet_dataset(cfg.dataset_path, tokenizer, cfg.max_seq_length, cfg.val_split_ratio, cfg.seed)

    model.save_pretrained(str(cfg.output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(cfg.output_dir))
    label_map = {"id2label": {str(i): l for i, l in enumerate(DOCLAYNET_LABELS)}, "label2id": {l: i for i, l in enumerate(DOCLAYNET_LABELS)}}
    (cfg.output_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    print(f"✅ [Agent 2 Trainer] Saved LayoutLMv3 weights & label_map.json to {cfg.output_dir}")
    return model, tokenizer
