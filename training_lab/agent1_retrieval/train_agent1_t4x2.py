#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

# Khóa 1 GPU duy nhất cho tiến trình đơn để tránh lỗi SFTTrainer nn.DataParallel:
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Agent1TrainingConfig


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
    parser = argparse.ArgumentParser(description="Fine-tune Agent 1 Retrieval Router on Kaggle T4x2")
    parser.add_argument("--epochs", type=int, default=3, help="Number of SFT training epochs")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Peak learning rate")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA/DoRA rank r")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    args = parser.parse_args()

    from src.trainer import run_router_sft_training
    from src.evaluator import evaluate_router_adapter

    cfg = Agent1TrainingConfig(epochs=args.epochs, learning_rate=args.lr, lora_r=args.lora_r, lora_alpha=args.lora_r * 2)
    trainer = run_router_sft_training(cfg)
    evaluate_router_adapter(trainer.model, trainer.processing_class)


if __name__ == "__main__":
    main()
