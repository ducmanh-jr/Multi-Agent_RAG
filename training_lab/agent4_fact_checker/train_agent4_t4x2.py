def patch_environment():
    """Patches missing triton.ops and bitsandbytes CUDA version mismatch on newer Python/PyTorch runtimes."""
    import os
    import sys
    from unittest.mock import MagicMock

    try:
        import triton.ops.matmul_perf_model
    except (ImportError, ModuleNotFoundError, AttributeError):
        if "triton" not in sys.modules:
            sys.modules["triton"] = MagicMock()
        mock_ops = MagicMock()
        mock_perf = MagicMock()
        mock_perf.early_config_prune = lambda *args, **kwargs: None
        mock_perf.estimate_matmul_time = lambda *args, **kwargs: 0
        mock_ops.matmul_perf_model = mock_perf
        sys.modules["triton.ops"] = mock_ops
        sys.modules["triton.ops.matmul_perf_model"] = mock_perf

    if "BNB_CUDA_VERSION" not in os.environ:
        os.environ["BNB_CUDA_VERSION"] = "124"

patch_environment()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys


import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Agent4TrainingConfig


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
    parser = argparse.ArgumentParser(description="Fine-tune Agent 4 NLI Verifier on Kaggle T4x2")
    parser.add_argument("--model-id", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    from src.trainer import run_factchecker_training

    kwargs = {
        "base_model_id": args.model_id,
        "epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "learning_rate": args.lr,
    }
    if args.data: kwargs["dataset_path"] = Path(args.data)
    if args.output_dir: kwargs["output_dir"] = Path(args.output_dir)
    cfg = Agent4TrainingConfig(**kwargs)
    trainer, metrics = run_factchecker_training(cfg)
    print(f'✅ [Agent 4] Final Macro F1: {metrics.get("macro_f1", metrics.get("eval_macro_f1", 0.0)):.4f}')


if __name__ == "__main__":
    main()
