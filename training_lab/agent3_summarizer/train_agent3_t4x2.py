#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys


import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Agent3TrainingConfig


def sanitize_torchvision():
    """Tự động vô hiệu hóa cờ torchvision trong transformers nếu torchvision bị lỗi C++ operator."""
    try:
        import torchvision
        _ = getattr(torchvision.ops, "nms", None)
    except Exception:
        try:
            import transformers.utils.import_utils as tu
            tu._torchvision_available = False
            tu.is_torchvision_available = lambda: False
        except Exception:
            pass


def main():
    sanitize_torchvision()
    parser = argparse.ArgumentParser(description="Fine-tune Agent 3 Generator & Critic on Kaggle T4x2")
    parser.add_argument("--role", choices=["generator", "critic", "both"], default="both")
    parser.add_argument("--epochs-sft", type=int, default=3)
    parser.add_argument("--epochs-dpo", type=int, default=2)
    
    parser.add_argument("--weights-dir", "--output-dir", dest="weights_dir", type=str, default=None)
    parser.add_argument("--dataset-dir", "--data", dest="dataset_dir", type=str, default=None)
    args = parser.parse_args()

    from src.trainer import train_generator_sft_and_dpo, train_critic_sft
    from src.evaluator import evaluate_generator_and_critic

    kwargs = {"gen_epochs_sft": args.epochs_sft, "gen_epochs_dpo": args.epochs_dpo, "critic_epochs": args.epochs_sft}
    if args.dataset_dir: kwargs["dataset_dir"] = Path(args.dataset_dir)
    if args.weights_dir: kwargs["weights_dir"] = Path(args.weights_dir)
    cfg = Agent3TrainingConfig(**kwargs)
    if args.role in ("generator", "both"):
        train_generator_sft_and_dpo(cfg)
    if args.role in ("critic", "both"):
        train_critic_sft(cfg)
    evaluate_generator_and_critic(cfg.weights_dir)


if __name__ == "__main__":
    main()
