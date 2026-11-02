#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI Entrypoint for Agent 4 (Fact-Checker 3-Class NLI Verifier) Fine-Tuning on Kaggle Dual T4.
Delegates to modular package `src/` (`config`, `data_loader`, `model_builder`, `trainer`, `evaluator`).
"""

import argparse
from src.config import Agent4TrainingConfig
from src.trainer import run_factchecker_training


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Agent 4 NLI Verifier on Kaggle T4x2")
    parser.add_argument("--model-id", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    cfg = Agent4TrainingConfig(
        base_model_id=args.model_id,
        epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    run_factchecker_training(cfg)


if __name__ == "__main__":
    main()
