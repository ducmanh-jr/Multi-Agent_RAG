#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Agent5TrainingConfig


def sanitize_torchvision():
    """Tự động vô hiệu hóa/gỡ bỏ torchvision nếu bị lỗi C++ operator 'torchvision::nms does not exist'."""
    try:
        import torchvision
        _ = getattr(torchvision.ops, "nms", None)
    except Exception:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for k in list(sys.modules.keys()):
            if "torchvision" in k:
                del sys.modules[k]
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
    args = parser.parse_args()

    from src.trainer import run_writer_sft_and_rpo_training
    from src.evaluator import evaluate_academic_writer

    cfg = Agent5TrainingConfig(epochs_sft=args.epochs_sft, epochs_dpo=args.epochs_dpo, lr_sft=args.lr)
    run_writer_sft_and_rpo_training(cfg)
    evaluate_academic_writer(cfg.output_dir)


if __name__ == "__main__":
    main()
