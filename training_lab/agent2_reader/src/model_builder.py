# -*- coding: utf-8 -*-
"""
Model & Processor Builder for Agent 2 (Document Reader - LayoutLMv3 11-Class Token Classifier).
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
from .config import Agent2TrainingConfig, DOCLAYNET_LABELS


def build_layoutlmv3_model(cfg: Agent2TrainingConfig):
    """Loads `microsoft/layoutlmv3-base` configured with 11 DocLayNet structural classes."""
    label2id = {l: i for i, l in enumerate(DOCLAYNET_LABELS)}
    id2label = {i: l for i, l in enumerate(DOCLAYNET_LABELS)}

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.base_model_id,
        num_labels=len(DOCLAYNET_LABELS),
        id2label=id2label,
        label2id=label2id,
    )
    return model, tokenizer
