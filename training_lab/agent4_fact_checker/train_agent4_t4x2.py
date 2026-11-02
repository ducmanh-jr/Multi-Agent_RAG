#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Agent4TrainingConfig


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
    parser = argparse.ArgumentParser(description="Fine-tune Agent 4 NLI Verifier on Kaggle T4x2")
    parser.add_argument("--model-id", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    from src.trainer import run_factchecker_training

    cfg = Agent4TrainingConfig(
        base_model_id=args.model_id,
        epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    run_factchecker_training(cfg)


if __name__ == "__main__":
    main()
