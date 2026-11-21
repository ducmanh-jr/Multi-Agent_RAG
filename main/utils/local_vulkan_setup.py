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

from core.config import get_run_settings
from utils.logger import logger


# Automatically set Radeon 780M Vulkan acceleration environment variables for Ollama / ONNX
os.environ.setdefault("OLLAMA_VULKAN", "1")
os.environ.setdefault("OLLAMA_FLASH_ATTENTION", "1")
os.environ.setdefault("OLLAMA_KEEP_ALIVE", "30m")
os.environ.setdefault("OMP_NUM_THREADS", "8")


_PROBE_CACHE: tuple[tuple[str, str, str, bool], float, dict[str, Any]] | None = None


def get_local_hardware_profile() -> dict[str, Any]:
    """Report host hardware and include psutil details when available."""
    import platform

    try:
        import psutil
    except ImportError:
        psutil = None

    cpu_name = platform.processor() or platform.machine() or "Host Processor"
    cores_physical = psutil.cpu_count(logical=False) if psutil else None
    cores_logical = os.cpu_count() or 1
    core_count = f"{cores_physical}C/{cores_logical}T" if cores_physical else f"{cores_logical}T"

    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3)) if psutil else None

    node_name = platform.node() or "Local Host"
    system_os = f"{platform.system()} {platform.release()}"

    igpu_desc = "Auto-detected Accelerator"
    if "AMD" in cpu_name.upper():
        igpu_desc = "AMD Radeon Graphics (Vulkan / ROCm compatible)"
    elif "INTEL" in cpu_name.upper():
        igpu_desc = "Intel Iris / Arc Graphics (OpenVINO compatible)"
    elif "APPLE" in cpu_name.upper() or platform.system() == "Darwin":
        igpu_desc = "Apple Silicon Metal Acceleration"

    return {
        "device": node_name,
        "os": system_os,
        "cpu": f"{cpu_name} ({core_count})",
        "igpu": igpu_desc,
        "ram_gb": ram_gb,
        "ollama_vulkan_enabled": os.environ.get("OLLAMA_VULKAN") == "1",
        "recommended_cpu_min_model": "qwen2.5:1.5b",
        "recommended_cpu_max_model": "qwen2.5:7b-instruct",
    }


def probe_ollama_models(base_url: Optional[str] = None, timeout: float = 1.8) -> dict[str, Any]:
    """Checks if local Ollama server is running and lists installed models."""
    settings = get_run_settings()
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


def probe_kaggle_worker(remote_url: Optional[str] = None, timeout: float = 5.0) -> dict[str, Any]:
    """Checks if the remote Kaggle GPU Worker (Multi-LoRA or vLLM 32B) is online."""
    url = (remote_url or get_run_settings().REMOTE_INFERENCE_URL or "").rstrip("/")
    if not url:
        return {"online": False, "url": "", "adapters": [], "has_vllm_32b": False}

    try:
        resp = httpx.get(f"{url}/health", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            worker_health = data.get("worker_health") if isinstance(data.get("worker_health"), dict) else {}
            adapters = data.get("adapters") or worker_health.get("loaded_adapters") or ["router", "generator", "critic"]
            model_name = str(data.get("model") or data.get("base_model") or worker_health.get("model") or "")
            is_32b = "32b" in model_name.lower() or "qwq" in model_name.lower() or "qwen3-32b" in model_name.lower()
            pipeline = str(data.get("pipeline") or worker_health.get("pipeline") or data.get("active_mode") or "")
            is_gpu_finetune = (
                pipeline in ("gpu_finetune_7_models", "gpu_finetune")
                or ("writer" in adapters)
                or ("qwen_cluster" in str(data.get("active_models", [])))
            )
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
    global _PROBE_CACHE
    settings = get_run_settings()
    gemini_ready = bool(settings.GEMINI_API_KEY and len(settings.GEMINI_API_KEY.strip()) > 10)
    key = (settings.LOCAL_CPU_URL, settings.LOCAL_CPU_MODEL, settings.REMOTE_INFERENCE_URL, gemini_ready)
    now = time.monotonic()
    cached = _PROBE_CACHE
    if not force_refresh and cached and cached[0] == key and now - cached[1] < 10.0:
        return cached[2]

    ollama_info = probe_ollama_models()
    kaggle_info = probe_kaggle_worker()

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
    _PROBE_CACHE = (key, now, status)
    return status
