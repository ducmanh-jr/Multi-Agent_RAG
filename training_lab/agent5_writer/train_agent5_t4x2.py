#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys


import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Agent5TrainingConfig


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
    parser = argparse.ArgumentParser(description="Fine-tune Agent 5 Academic Writer on Kaggle T4x2")
    parser.add_argument("--epochs-sft", type=int, default=3)
    parser.add_argument("--epochs-dpo", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--dataset-dir", "--data", dest="dataset_dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    from src.trainer import run_writer_sft_and_rpo_training
    from src.evaluator import evaluate_academic_writer

    kwargs = {"epochs_sft": args.epochs_sft, "epochs_dpo": args.epochs_dpo, "lr_sft": args.lr}
    if args.dataset_dir: kwargs["dataset_dir"] = Path(args.dataset_dir)
    if args.output_dir: kwargs["output_dir"] = Path(args.output_dir)
    cfg = Agent5TrainingConfig(**kwargs)
    run_writer_sft_and_rpo_training(cfg)
    evaluate_academic_writer(cfg.output_dir)


if __name__ == "__main__":
    main()
