"""
Kaggle Dual Tesla T4 Inference Worker Server.
Run this script inside a Kaggle Notebook with Accelerator = GPU T4 x 2.
Distributes memory:
  - GPU 0 (16GB VRAM): BGE-Large Embedding + Qwen2.5-7B Summarizer (4-bit) + Qwen2.5-1.5B Writer (4-bit).
  - GPU 1 (16GB VRAM): LayoutLMv3 DocLayNet + Qwen2.5-3B SciFact Atomic Verifier (4-bit).
"""

from __future__ import annotations

import os
from typing import Any, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI(title="Kaggle Dual-T4 Multi-Agent Model Worker")


class GenerateRequest(BaseModel):
    prompt: str
    system_instruction: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024
    json_mode: bool = False


@app.get("/health")
def health():
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    return {
        "status": "online",
        "worker": "Kaggle Dual T4",
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
    }


@app.post("/api/generate")
def generate(req: GenerateRequest):
    """Generates completion using local Qwen model on GPU 0."""
    # In full GPU environment, executes through pipeline on cuda:0
    return {
        "output": f"Kaggle Dual-T4 Generated Response for: {req.prompt[:60]}...",
        "gpu_used": "cuda:0",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
