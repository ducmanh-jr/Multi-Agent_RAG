# -*- coding: utf-8 -*-
"""
Evaluator & Temperature Scaling Calibration (ECE) for Agent 4 (Fact-Checker NLI Verifier).
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support


def compute_nli_metrics(eval_pred):
    """Computes Accuracy, Macro-F1, and Per-Class F1 (`SUPPORT`, `CONTRADICT`, `NOT_ENOUGH_INFO`)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    _, _, f1_per_class, _ = precision_recall_fscore_support(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "f1_support": round(float(f1_per_class[0]), 4),
        "f1_contradict": round(float(f1_per_class[1]), 4),
        "f1_nei": round(float(f1_per_class[2]), 4),
    }


def calibrate_temperature_ece(logits: np.ndarray, labels: np.ndarray) -> float:
    """Finds optimal Temperature T in [0.5, 3.0] minimizing Negative Log-Likelihood for calibrated confidence."""
    best_t = 1.0
    best_nll = float("inf")
    for t in np.linspace(0.5, 2.5, 41):
        scaled = logits / t
        exp_s = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        nll = -np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-9, 1.0)))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return round(best_t, 3)
