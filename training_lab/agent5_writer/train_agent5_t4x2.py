#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI Entrypoint for Agent 5 (Academic Report Writer & Citation Synthesizer) Fine-Tuning on Kaggle Dual T4.
Delegates to modular package `src/` (`config`, `data_loader`, `model_builder`, `trainer`, `evaluator`).
"""

import argparse
from src.config import Agent5TrainingConfig
from src.trainer import run_writer_sft_and_rpo_training
from src.evaluator import evaluate_academic_writer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Agent 5 Academic Writer on Kaggle T4x2")
    parser.add_argument("--epochs-sft", type=int, default=3)
    parser.add_argument("--epochs-dpo", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    args = parser.parse_args()

    cfg = Agent5TrainingConfig(epochs_sft=args.epochs_sft, epochs_dpo=args.epochs_dpo, lr_sft=args.lr)
    run_writer_sft_and_rpo_training(cfg)
    evaluate_academic_writer(cfg.output_dir)


if __name__ == "__main__":
    main()
