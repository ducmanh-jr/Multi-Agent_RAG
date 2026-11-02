#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Agent3TrainingConfig


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
    parser = argparse.ArgumentParser(description="Fine-tune Agent 3 Generator & Critic on Kaggle T4x2")
    parser.add_argument("--role", choices=["generator", "critic", "both"], default="both")
    parser.add_argument("--epochs-sft", type=int, default=3)
    parser.add_argument("--epochs-dpo", type=int, default=2)
    args = parser.parse_args()

    from src.trainer import train_generator_sft_and_dpo, train_critic_sft
    from src.evaluator import evaluate_generator_and_critic

    cfg = Agent3TrainingConfig(gen_epochs_sft=args.epochs_sft, gen_epochs_dpo=args.epochs_dpo, critic_epochs=args.epochs_sft)
    if args.role in ("generator", "both"):
        train_generator_sft_and_dpo(cfg)
    if args.role in ("critic", "both"):
        train_critic_sft(cfg)
    evaluate_generator_and_critic(cfg.weights_dir)


if __name__ == "__main__":
    main()
