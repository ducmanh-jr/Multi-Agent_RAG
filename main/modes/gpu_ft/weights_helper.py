"""
Helper to locate local LoRA adapters inside modes/gpu_ft/weights/.
Provides direct local paths for fallback or offline local GPU/CPU execution.
"""

from pathlib import Path
from typing import Optional

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def get_adapter_path(adapter_name: str) -> Optional[Path]:
    """Find local path for given adapter inside modes/gpu_ft/weights/."""
    candidates = [
        WEIGHTS_DIR / adapter_name,
        WEIGHTS_DIR / adapter_name / "final",
    ]
    for c in candidates:
        if c.exists() and (c / "adapter_config.json").exists():
            return c
        if c.exists() and (c / "config.json").exists():
            return c
    return None
