#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI Entrypoint for Agent 1 (Retrieval Router) Fine-Tuning on Kaggle Dual T4 (2 x 16GB VRAM).
Delegates to modular package `src/` (`config`, `data_loader`, `model_builder`, `trainer`, `evaluator`).
"""

import argparse
from src.config import Agent1TrainingConfig
from src.trainer import run_router_sft_training
from src.evaluator import evaluate_router_adapter


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Agent 1 Retrieval Router on Kaggle T4x2")
    parser.add_argument("--epochs", type=int, default=4, help="Number of SFT training epochs")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Peak learning rate")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA/DoRA rank r")
    parser.add_argument("--eval-only", action="store_true", help="Run post-training evaluation after SFT")
    args = parser.parse_args()

    cfg = Agent1TrainingConfig(epochs=args.epochs, learning_rate=args.lr, lora_r=args.lora_r, lora_alpha=args.lora_r * 2)
    trainer = run_router_sft_training(cfg)
    evaluate_router_adapter(trainer.model, trainer.processing_class)


if __name__ == "__main__":
    main()
