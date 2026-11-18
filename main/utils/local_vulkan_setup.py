"""
Hardware Accelerator & Real-Time Backend Capability Probe for:
  - Lenovo ThinkBook 14 G8+ AHP (AMD Ryzen 7 H 255 + AMD Radeon 780M 12-CU RDNA3 iGPU + 32GB LPDDR5X-7500)
  - Automatic OLLAMA_VULKAN=1 & AVX-512 environment optimization
  - Real-time capability probing across all 5 backends for Hybrid Orchestration
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional
import httpx

from core.config import settings
from utils.logger import logger


# Automatically set Radeon 780M Vulkan acceleration environment variables for Ollama / ONNX
os.environ.setdefault("OLLAMA_VULKAN", "1")
os.environ.setdefault("OLLAMA_FLASH_ATTENTION", "1")
os.environ.setdefault("OLLAMA_KEEP_ALIVE", "30m")
os.environ.setdefault("OMP_NUM_THREADS", "8")


_PROBE_CACHE: dict[str, Any] = {
    "timestamp": 0.0,
    "status": {},
}


def get_local_hardware_profile() -> dict[str, Any]:
    """Returns the optimized hardware profile for ThinkBook 14 G8+ AHP."""
    return {
        "device": "Lenovo ThinkBook 14 G8+ AHP",
        "cpu": "AMD Ryzen 7 H 255 (8C/16T, AVX-512)",
        "igpu": "AMD Radeon 780M Graphics (12 CUs RDNA 3, Vulkan 1.3)",
        "ram_gb": 32,
        "ram_type": "LPDDR5X-7500 MHz",
        "ollama_vulkan_enabled": os.environ.get("OLLAMA_VULKAN") == "1",
        "recommended_cpu_min_model": "qwen2.5:1.5b",
        "recommended_cpu_max_model": "qwen2.5:7b-instruct",
    }


def probe_ollama_models(base_url: Optional[str] = None, timeout: float = 1.8) -> dict[str, Any]:
    """Checks if local Ollama server is running and lists installed models."""
    url = (base_url or settings.LOCAL_CPU_URL or "http://localhost:11434").rstrip("/")
    try:
        resp = httpx.get(f"{url}/api/tags", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            # Select best small model for CPU Min and best 7B/8B model for CPU Max
            small_model = next(
                (m for m in models if any(k in m.lower() for k in ["1.5b", "1.7b", "0.5b", "3b"])),
                models[0] if models else settings.LOCAL_CPU_MODEL,
            )
            large_model = next(
                (m for m in models if any(k in m.lower() for k in ["7b", "8b", "14b", "qwen3", "qwen2.5"])),
                models[0] if models else "qwen2.5:7b-instruct",
            )
            return {
                "online": True,
                "url": url,
                "models": models,
                "cpu_min_model": small_model,
                "cpu_max_model": large_model,
            }
    except Exception:
        pass
    return {
        "online": False,
        "url": url,
        "models": [],
        "cpu_min_model": settings.LOCAL_CPU_MODEL or "qwen2.5:1.5b",
        "cpu_max_model": "qwen2.5:7b-instruct",
    }


def probe_kaggle_worker(remote_url: Optional[str] = None, timeout: float = 3.0) -> dict[str, Any]:
    """Checks if the remote Kaggle GPU Worker (Multi-LoRA or vLLM 32B) is online."""
    url = (remote_url or settings.REMOTE_INFERENCE_URL or "").rstrip("/")
    if not url:
        return {"online": False, "url": "", "adapters": [], "has_vllm_32b": False}

    try:
        resp = httpx.get(f"{url}/health", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            adapters = data.get("adapters") or ["router", "generator", "critic"]
            model_name = str(data.get("model") or data.get("base_model") or "")
            is_32b = "32b" in model_name.lower() or "qwq" in model_name.lower() or "qwen3-32b" in model_name.lower()
            pipeline = str(data.get("pipeline") or "")
            is_gpu_finetune = (pipeline == "gpu_finetune_7_models") or ("writer" in adapters) or ("qwen_cluster" in str(data.get("active_models", [])))
            return {
                "online": True,
                "url": url,
                "adapters": adapters,
                "model": model_name,
                "has_vllm_32b": is_32b,
                "has_nli": bool(data.get("nli_loaded", True)),
                "is_gpu_finetune": is_gpu_finetune,
            }
    except Exception:
        pass

    # Also check OpenAI-compatible /v1/models endpoint (for pure vLLM 32B server)
    try:
        resp = httpx.get(f"{url}/v1/models", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            model_ids = [m.get("id", "") for m in data.get("data", [])]
            return {
                "online": True,
                "url": url,
                "adapters": [],
                "model": model_ids[0] if model_ids else "Qwen/Qwen3-32B-AWQ",
                "has_vllm_32b": True,
                "has_nli": False,
            }
    except Exception:
        pass

    return {"online": False, "url": url, "adapters": [], "has_vllm_32b": False}


def probe_all_capabilities(force_refresh: bool = False) -> dict[str, Any]:
    """
    Cached capability probe (TTL = 10s) across Local CPU/Vulkan, Gemini API, and Kaggle GPU.
    Used by Hybrid Mode to dynamically route Generator and Cross-Model Verifier.
    """
    now = time.time()
    if not force_refresh and _PROBE_CACHE["status"] and (now - _PROBE_CACHE["timestamp"] < 10.0):
        return _PROBE_CACHE["status"]

    ollama_info = probe_ollama_models()
    kaggle_info = probe_kaggle_worker()
    gemini_ready = bool(settings.GEMINI_API_KEY and len(settings.GEMINI_API_KEY.strip()) > 10)

    # Check remaining Gemini quota today
    gemini_used_today = 0
    try:
        from core.db import get_daily_api_usage
        usage = get_daily_api_usage("gemini")
        gemini_used_today = int(usage.get("used", 0))
    except Exception:
        pass

    gemini_quota_ok = gemini_ready and (gemini_used_today < 1480)

    status = {
        "hardware": get_local_hardware_profile(),
        "cpu_min": {"online": True, "engine": "BM25 C-Engine + PyMuPDF + Ollama 1.5B Vulkan"},
        "cpu_max": {
            "online": True,
            "ollama_online": ollama_info["online"],
            "model": ollama_info["cpu_max_model"],
            "models_installed": ollama_info["models"],
        },
        "gemini": {
            "online": gemini_quota_ok,
            "api_key_configured": gemini_ready,
            "used_today": gemini_used_today,
            "daily_limit": 1500,
        },
        "gpu_finetune": {
            "online": kaggle_info["online"] and kaggle_info.get("is_gpu_finetune", True),
            "url": kaggle_info["url"],
            "adapters": kaggle_info["adapters"],
        },
        "gpu_ft": {
            "online": kaggle_info["online"],
            "url": kaggle_info["url"],
            "adapters": kaggle_info["adapters"],
        },
        "kaggle_max": {
            "online": kaggle_info["online"],
            "url": kaggle_info["url"],
            "model": kaggle_info.get("model", "Qwen/Qwen3-32B-AWQ"),
            "has_vllm_32b": kaggle_info.get("has_vllm_32b", False),
        },
    }
    _PROBE_CACHE["timestamp"] = now
    _PROBE_CACHE["status"] = status
    return status
