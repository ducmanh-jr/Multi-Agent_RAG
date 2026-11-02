#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Agent2TrainingConfig


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
    parser = argparse.ArgumentParser(description="Fine-tune Agent 2 LayoutLMv3 Reader on Kaggle T4x2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    from src.trainer import run_reader_training
    from src.evaluator import evaluate_layout_parser

    cfg = Agent2TrainingConfig(epochs=args.epochs, learning_rate=args.lr, per_device_train_batch_size=args.batch_size)
    model, tokenizer = run_reader_training(cfg)
    evaluate_layout_parser(model, tokenizer)


if __name__ == "__main__":
    main()
