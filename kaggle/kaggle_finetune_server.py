"""
Kaggle Dual Tesla T4 Inference Worker Server for 'gpu_finetune' SOTA Mode.
Hosts all fine-tuned models across 2x GPU T4 (32GB VRAM) with Dual-Engine support:

1. Primary Engine ('vllm', default when vLLM + Dual-T4 available):
   - Runs vLLM OpenAI server on 127.0.0.1:8101 with Qwen2.5-7B-Instruct (FP16, TP=2)
   - Multi-LoRA serving ('router', 'generator', 'critic', 'writer') with PagedAttention,
     CUDA Graphs, Prefix Caching, and guided_json structured output.
   - BGE-Reranker-v2-m3 NLI Head hosted on GPU 1 alongside TP shard.

2. Fallback Engine ('hf', automatic if vLLM unavailable or FINETUNE_ENGINE=hf):
   - Dual-GPU NF4 4-bit split: GPU 0 ('router', 'generator') + GPU 1 ('critic', 'writer', 'factcheck')
     with per-GPU locks, SDPA attention, and KV caching.
"""

from __future__ import annotations

import atexit
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import torch
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

app = FastAPI(title="Kaggle Dual-T4 GPU Finetune Worker (7-Model SOTA Pipeline)")

# Global state
_models: dict[str, Any] = {}
_tokenizers: dict[str, Any] = {}
_loaded_adapters: list[str] = []
_adapter_paths: dict[str, str] = {}
_adapter_gpu_map: dict[str, int] = {}
_gpu_locks: dict[int, threading.Lock] = {0: threading.Lock(), 1: threading.Lock()}

_active_engine: str = "hf"  # "vllm" or "hf"
_vllm_proc: Optional[subprocess.Popen] = None
_vllm_port: int = int(os.environ.get("VLLM_FINETUNE_PORT", "8101"))
_vllm_base_url: str = f"http://127.0.0.1:{_vllm_port}"
_async_http: Optional[httpx.AsyncClient] = None

ROUTER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "sub_queries": {"type": "array", "items": {"type": "string"}},
        "alpha": {"type": "number"},
        "top_k": {"type": "integer"},
        "reasoning": {"type": "string"},
    },
    "required": ["intent", "sub_queries", "alpha", "top_k"],
}

GENERATOR_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "claims": {"type": "array", "items": {"type": "string"}},
        "evidence_spans": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "claims", "evidence_spans"],
}

CRITIC_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "critique": {"type": "string"},
        "action": {"type": "string", "enum": ["accept", "rewrite", "request_more_context"]},
        "severity": {"type": "string", "enum": ["none", "minor", "major"]},
        "issues_found": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["critique", "action", "severity", "issues_found"],
}


def find_weight_path(sub_path: str) -> Optional[str]:
    """Search authoritative PROJECT_DIR, common Kaggle and local paths for weights directories."""
    candidates = []
    proj_dir = os.environ.get("PROJECT_DIR")
    if proj_dir:
        candidates.extend([
            Path(proj_dir) / "training_lab" / "final_weights" / sub_path,
            Path(proj_dir) / "training_lab" / "final_weights" / "final_weights" / sub_path,
        ])
    candidates.extend([
        Path("/kaggle/working/Multi-Agent_RAG/training_lab/final_weights") / sub_path,
        Path("/kaggle/working/Multi-Agent_RAG/training_lab/final_weights/final_weights") / sub_path,
        Path("/kaggle/working/training_lab/final_weights") / sub_path,
        Path("/kaggle/working/training_lab/final_weights/final_weights") / sub_path,
        Path("/kaggle/working/final_weights") / sub_path,
        Path("/kaggle/working/drive_download") / sub_path,
        Path.cwd() / "training_lab" / "final_weights" / sub_path,
        Path("./training_lab/final_weights") / sub_path,
        Path("../training_lab/final_weights") / sub_path,
        Path("/kaggle/input") / sub_path,
    ])
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    working_dir = Path("/kaggle/working")
    if working_dir.exists():
        for match in working_dir.glob(f"**/{sub_path}"):
            if match.exists():
                return str(match.resolve())
    return None


def _discover_adapter_paths() -> dict[str, str]:
    specs = {
        "generator": "03_agent3_summarizer/generator_qlora",
        "router": "01_agent1_router",
        "critic": "03_agent3_summarizer/critic_qlora",
        "writer": "05_agent5_writer",
    }
    found: dict[str, str] = {}
    for name, sub in specs.items():
        path = find_weight_path(sub)
        if path and os.path.exists(path):
            found[name] = path
        else:
            print(f"[Worker WARNING] Checkpoint for adapter '{name}' not found at '{sub}'.")
    return found


def _stop_vllm_subprocess() -> None:
    """Cleanly terminate the internal vLLM subprocess group if running."""
    global _vllm_proc
    if _vllm_proc is None:
        return
    try:
        pgid = os.getpgid(_vllm_proc.pid) if hasattr(os, "getpgid") else None
        if pgid and hasattr(os, "killpg"):
            os.killpg(pgid, signal.SIGTERM)
        else:
            _vllm_proc.terminate()
        _vllm_proc.wait(timeout=10)
    except Exception:
        try:
            pgid = os.getpgid(_vllm_proc.pid) if hasattr(os, "getpgid") else None
            if pgid and hasattr(os, "killpg"):
                os.killpg(pgid, signal.SIGKILL)
            else:
                _vllm_proc.kill()
        except Exception:
            pass
    finally:
        _vllm_proc = None


atexit.register(_stop_vllm_subprocess)


# =====================================================================
# Engine A: vLLM Multi-LoRA Server (FP16 TP=2 across Dual Tesla T4)
# =====================================================================

def _start_vllm_engine(adapter_paths: dict[str, str]) -> bool:
    """Start vLLM OpenAI-compatible server with Multi-LoRA on Dual T4. Returns True if ready."""
    global _vllm_proc, _loaded_adapters, _active_engine

    if importlib.util.find_spec("vllm") is None:
        print("[Worker] 'vllm' package not found. Skipping vLLM engine.")
        return False

    required = ["router", "generator", "critic", "writer"]
    if not all(k in adapter_paths for k in required):
        print(f"[Worker] Missing adapters for vLLM Multi-LoRA: {[k for k in required if k not in adapter_paths]}")
        return False

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count < 1:
        return False

    strategy = os.environ.get("VLLM_FINETUNE_STRATEGY", "awq_tp1").strip().lower()
    if strategy == "fp16_tp2" and gpu_count >= 2:
        model_id = os.getenv("BASE_QWEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
        tp_size = 2
        gpu_util = os.environ.get("VLLM_FINETUNE_GPU_UTIL", "0.78")
        quant_args: list[str] = []
        sub_env = os.environ.copy()
    else:
        # Default 'awq_tp1': Run Qwen2.5-7B-Instruct-AWQ on GPU 0 (TP=1) to eliminate PCIe All-Reduce bottleneck on T4x2!
        model_id = os.getenv("BASE_QWEN_AWQ_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct-AWQ")
        tp_size = 1
        gpu_util = os.environ.get("VLLM_FINETUNE_GPU_UTIL", "0.86")
        quant_args = ["--quantization", "awq"]
        sub_env = os.environ.copy()
        sub_env["CUDA_VISIBLE_DEVICES"] = "0"

    lora_args = [f"{name}={path}" for name, path in adapter_paths.items()]

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--dtype", "half",
        *quant_args,
        "--tensor-parallel-size", str(tp_size),
        "--gpu-memory-utilization", str(gpu_util),
        "--max-model-len", os.environ.get("VLLM_MAX_MODEL_LEN", "4096"),
        "--enable-lora",
        "--max-lora-rank", "64",
        "--max-loras", "4",
        "--enable-prefix-caching",
        "--disable-log-requests",
        "--host", "127.0.0.1",
        "--port", str(_vllm_port),
        "--lora-modules", *lora_args,
    ]
    if os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1":
        cmd.append("--enforce-eager")

    print(f"[Worker] Starting vLLM Multi-LoRA Engine (strategy={strategy}, model={model_id}, TP={tp_size}, gpu_util={gpu_util}) on :{_vllm_port}...")
    log_path = Path("/kaggle/working/vllm_finetune.log") if Path("/kaggle/working").exists() else Path("vllm_finetune.log")
    log_file = open(log_path, "w", encoding="utf-8")

    popen_kwargs: dict[str, Any] = {
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        "env": sub_env,
    }
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True

    _vllm_proc = subprocess.Popen(cmd, **popen_kwargs)

    boot_timeout = int(os.environ.get("VLLM_BOOT_TIMEOUT", "360"))
    deadline = time.time() + boot_timeout
    with httpx.Client(timeout=5.0) as client:
        while time.time() < deadline:
            if _vllm_proc.poll() is not None:
                print(f"[Worker ERROR] vLLM process exited early with code {_vllm_proc.returncode}. Check {log_path}")
                _stop_vllm_subprocess()
                return False
            try:
                resp = client.get(f"{_vllm_base_url}/v1/models")
                if resp.status_code == 200:
                    data = resp.json()
                    ids = {m.get("id") for m in data.get("data", []) if isinstance(m, dict)}
                    if all(ad in ids for ad in required):
                        _loaded_adapters = list(required)
                        _models["qwen_cluster"] = f"vllm:{model_id}:tp{tp_size}"
                        _active_engine = "vllm"
                        print(f"[Worker] vLLM Multi-LoRA Engine ONLINE with adapters: {_loaded_adapters}")
                        return True
            except Exception:
                pass
            time.sleep(2.0)

    print(f"[Worker ERROR] vLLM timed out after {boot_timeout}s. Falling back to HF Dual-GPU engine.")
    _stop_vllm_subprocess()
    return False


async def _call_vllm_chat(
    prompt: str,
    adapter: str,
    max_tokens: int,
    temperature: float,
    json_schema: Optional[dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    """Call internal vLLM OpenAI chat completions endpoint with optional guided_json."""
    global _async_http
    if _async_http is None:
        _async_http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    body: dict[str, Any] = {
        "model": adapter,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_schema is not None:
        body["extra_body"] = {"guided_json": json_schema}

    t0 = time.time()
    resp = await _async_http.post(f"{_vllm_base_url}/v1/chat/completions", json=body)

    # Compatibility fallback if installed vLLM version rejects extra_body/guided_json
    if resp.status_code == 400 and "extra_body" in body:
        body.pop("extra_body", None)
        body["response_format"] = {"type": "json_object"}
        resp = await _async_http.post(f"{_vllm_base_url}/v1/chat/completions", json=body)
        if resp.status_code == 400:
            body.pop("response_format", None)
            resp = await _async_http.post(f"{_vllm_base_url}/v1/chat/completions", json=body)

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"vLLM error ({resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})
    elapsed = max(0.01, time.time() - t0)
    comp_toks = int(usage.get("completion_tokens") or 0)
    usage_meta = {
        "engine": "vllm",
        "adapter": adapter,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": comp_toks,
        "latency_sec": round(elapsed, 2),
        "tok_per_sec": round(comp_toks / elapsed, 1) if comp_toks else 0.0,
    }
    return text, usage_meta


# =====================================================================
# Engine B: Optimized Dual-GPU HuggingFace + PEFT Fallback
# =====================================================================

def _get_bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def _load_hf_instance(gpu_idx: int, adapters: dict[str, str], holder_key: str) -> None:
    """Load one Qwen2.5-7B 4-bit instance on the specified GPU and attach assigned adapters."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = os.getenv("BASE_QWEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    bnb_config = _get_bnb_config()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "quantization_config": bnb_config,
        "device_map": {"": gpu_idx} if torch.cuda.is_available() else "auto",
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa", **load_kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    first = True
    for name, path in adapters.items():
        if first:
            model = PeftModel.from_pretrained(model, path, adapter_name=name)
            first = False
        else:
            model.load_adapter(path, adapter_name=name)
        if name not in _loaded_adapters:
            _loaded_adapters.append(name)
        _adapter_gpu_map[name] = gpu_idx
        print(f"[Worker HF] GPU {gpu_idx}: attached adapter '{name}' from {path}")

    model.eval()
    _models[holder_key] = model
    _tokenizers[holder_key] = tokenizer


def load_hf_dual_gpu_cluster() -> None:
    """Split adapters across GPU 0 (router, generator) and GPU 1 (critic, writer) when 2 GPUs exist."""
    global _active_engine
    if "qwen_cluster" in _models:
        return
    if not torch.cuda.is_available():
        print("[Worker WARNING] CUDA not available.")
        return

    gpu_count = torch.cuda.device_count()
    paths = _adapter_paths or _discover_adapter_paths()

    if gpu_count >= 2:
        gpu0_adapters = {k: paths[k] for k in ("generator", "router") if k in paths}
        gpu1_adapters = {k: paths[k] for k in ("critic", "writer") if k in paths}
        if gpu0_adapters:
            _load_hf_instance(0, gpu0_adapters, "qwen_cluster")
        if gpu1_adapters:
            try:
                _load_hf_instance(1, gpu1_adapters, "qwen_cluster_gpu1")
            except Exception as exc:
                print(f"[Worker HF] GPU 1 secondary load failed ({exc}), attaching all adapters to GPU 0.")
                model0 = _models["qwen_cluster"]
                for name, path in gpu1_adapters.items():
                    model0.load_adapter(path, adapter_name=name)
                    _loaded_adapters.append(name)
                    _adapter_gpu_map[name] = 0
    else:
        _load_hf_instance(0, paths, "qwen_cluster")

    _active_engine = "hf"


def _generate_with_hf(
    prompt: str,
    adapter: str,
    max_new_tokens: int,
    temperature: float,
    system_prompt: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    gpu_idx = _adapter_gpu_map.get(adapter, 0)
    holder_key = "qwen_cluster_gpu1" if gpu_idx == 1 and "qwen_cluster_gpu1" in _models else "qwen_cluster"
    model = _models.get(holder_key)
    tokenizer = _tokenizers.get(holder_key)
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Qwen cluster base model not loaded on GPU.")
    if adapter not in _loaded_adapters:
        raise HTTPException(status_code=503, detail=f"Adapter '{adapter}' is not loaded on this worker.")

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    lock = _gpu_locks.get(gpu_idx, _gpu_locks[0])

    t0 = time.time()
    with lock:
        if hasattr(model, "set_adapter"):
            model.set_adapter(adapter)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        do_sample = temperature > 0.2
        gen_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)

    prompt_len = int(inputs.input_ids.shape[1])
    out_ids = outputs[0][prompt_len:]
    response = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
    elapsed = max(0.01, time.time() - t0)
    comp_toks = int(len(out_ids))
    usage_meta = {
        "engine": f"hf_gpu{gpu_idx}",
        "adapter": adapter,
        "prompt_tokens": prompt_len,
        "completion_tokens": comp_toks,
        "latency_sec": round(elapsed, 2),
        "tok_per_sec": round(comp_toks / elapsed, 1) if comp_toks else 0.0,
    }
    return response, usage_meta


async def run_llm(
    prompt: str,
    adapter: str,
    max_tokens: int,
    temperature: float,
    json_schema: Optional[dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    """Unified async dispatch to vLLM (primary) or HF Dual-GPU (fallback)."""
    if _active_engine == "vllm":
        return await _call_vllm_chat(
            prompt=prompt,
            adapter=adapter,
            max_tokens=max_tokens,
            temperature=temperature,
            json_schema=json_schema,
            system_prompt=system_prompt,
        )
    return await run_in_threadpool(
        _generate_with_hf,
        prompt,
        adapter,
        max_tokens,
        temperature,
        system_prompt,
    )


# =====================================================================
# GPU 1: Fact Checker NLI Head (Agent 4)
# =====================================================================

def load_factcheck_model():
    if "factcheck" in _models:
        return _models["factcheck"], _tokenizers["factcheck"]

    fc_path = find_weight_path("04_agent4_fact_checker")
    if not fc_path:
        print("[Worker WARNING] 04_agent4_fact_checker path not found.")
        return None, None

    print(f"[Worker] Loading Agent 4 Fact-Checker NLI Head from {fc_path} on cuda:1...")
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        target_device = 1 if torch.cuda.device_count() >= 2 else 0
        tokenizer = AutoTokenizer.from_pretrained(fc_path, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            fc_path,
            torch_dtype=torch.float16,
            device_map={"": target_device},
            trust_remote_code=True,
        )
        model.eval()
        _models["factcheck"] = model
        _tokenizers["factcheck"] = tokenizer
        return model, tokenizer
    except Exception as e:
        print(f"[Worker ERROR] Failed loading Fact-Checker model: {e}")
        return None, None


# =====================================================================
# Pydantic Schemas & Lifecycle
# =====================================================================

class RouterRequest(BaseModel):
    query: str
    adapter: str = "router"

class SummarizeRequest(BaseModel):
    query: str
    context: str
    feedback: Optional[str] = ""
    adapter: str = "generator"
    max_tokens: int = 600
    language: str = "vi"

class CriticRequest(BaseModel):
    query: str
    answer: str
    context: str
    claims: list[str] = []
    adapter: str = "critic"
    max_tokens: int = 256

class FactCheckRequest(BaseModel):
    claims: list[str]
    context: str = ""
    adapter: str = "factcheck"

class WriterRequest(BaseModel):
    query: str
    answer: str
    claims: list[dict[str, Any]] = []
    adapter: str = "writer"
    max_tokens: int = 1100
    language: str = "vi"

class RerankRequest(BaseModel):
    query: str
    candidates: list[str]

class LayoutRequest(BaseModel):
    pdf_path: str
    pages: list[int] = []

ReaderLayoutRequest = LayoutRequest

class GenerateRequest(BaseModel):
    prompt: str
    system_instruction: Optional[str] = None
    adapter: str = "generator"
    max_tokens: int = 768
    temperature: float = 0.2
    json_mode: bool = False


@app.on_event("startup")
def warmup_all_gpus():
    global _adapter_paths
    print("\n" + "=" * 65)
    print("[STARTUP] KHOI DONG DUAL TESLA T4 WORKER (OPTIMIZED T4x2)...")
    print("=" * 65)

    _adapter_paths = _discover_adapter_paths()
    preferred_engine = os.environ.get("FINETUNE_ENGINE", "vllm").strip().lower()

    started_vllm = False
    if preferred_engine == "vllm":
        started_vllm = _start_vllm_engine(_adapter_paths)

    if not started_vllm:
        load_hf_dual_gpu_cluster()

    load_factcheck_model()

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            vram = torch.cuda.memory_allocated(i) / (1024 ** 3)
            print(f"[GPU {i}] Active engine={_active_engine}, torch_allocated={vram:.2f} GB")

    print("=" * 65 + "\n")


@app.on_event("shutdown")
async def shutdown_worker():
    global _async_http
    if _async_http is not None:
        await _async_http.aclose()
        _async_http = None
    _stop_vllm_subprocess()


# =====================================================================
# Helpers
# =====================================================================

def _parse_json_object(raw: str) -> Optional[dict[str, Any]]:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        block = m.group(0)
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            try:
                import ast
                parsed = ast.literal_eval(block)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
    return None


# =====================================================================
# Endpoints
# =====================================================================

@app.get("/")
@app.get("/health")
def healthcheck():
    gpu_info = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info[f"gpu_{i}"] = {
                "name": torch.cuda.get_device_name(i),
                "vram_used_gb": round(torch.cuda.memory_allocated(i) / (1024 ** 3), 2),
                "vram_total_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3), 2),
            }

    mandatory_adapters = ["router", "generator", "critic", "writer"]
    missing_adapters = [ad for ad in mandatory_adapters if ad not in _loaded_adapters]
    mandatory_models = ["qwen_cluster", "factcheck"]
    missing_models = [m for m in mandatory_models if m not in _models]
    is_ready = (len(missing_adapters) == 0 and len(missing_models) == 0)

    capabilities = {
        "router": ("router" in _loaded_adapters) and ("qwen_cluster" in _models),
        "generator": ("generator" in _loaded_adapters) and ("qwen_cluster" in _models),
        "critic": ("critic" in _loaded_adapters) and ("qwen_cluster" in _models),
        "writer": ("writer" in _loaded_adapters) and ("qwen_cluster" in _models),
        "fact_checker_nli": "factcheck" in _models,
        "layout_reader": "reader" in _models,
        "embedding": "embedding" in _models,
        "reranker": "reranker" in _models,
    }

    return {
        "status": "online" if is_ready else "degraded",
        "ready": is_ready,
        "worker_status": "online" if is_ready else "degraded",
        "mandatory_ready": is_ready,
        "pipeline": "gpu_finetune_7_models",
        "engine": _active_engine,
        "gpus_available": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "loaded_adapters": _loaded_adapters,
        "missing_adapters": missing_adapters,
        "active_models": list(_models.keys()),
        "missing_models": missing_models,
        "capabilities": capabilities,
        "gpus": gpu_info,
    }


@app.post("/api/router")
async def router_endpoint(req: RouterRequest):
    prompt = (
        "You are an intelligent routing agent for an Academic RAG system. "
        "Analyze the user query and output a JSON object with: intent, sub_queries, alpha (0.0 to 1.0), top_k (1 to 15), and reasoning.\n\n"
        f"Query: {req.query}"
    )
    raw, usage = await run_llm(
        prompt=prompt,
        adapter="router",
        max_tokens=256,
        temperature=0.1,
        json_schema=ROUTER_JSON_SCHEMA,
    )
    parsed = _parse_json_object(raw)
    if parsed:
        parsed["usage"] = usage
        return parsed
    return {"intent": "academic_research", "sub_queries": [req.query], "alpha": 0.5, "top_k": 6, "raw": raw, "usage": usage}


@app.post("/api/summarize/generator")
async def summarize_generator(req: SummarizeRequest):
    lang_hint = (
        "Hãy viết trường 'answer' và 'claims' bằng TIẾNG VIỆT học thuật chuẩn xác, mạch lạc, giữ nguyên thuật ngữ kỹ thuật nếu cần và gắn trích dẫn [1], [2]."
        if req.language == "vi"
        else "Write the 'answer' and 'claims' in clear academic English with inline citations [1], [2]."
    )
    # Keep Context as the fixed prefix for vLLM automatic prefix caching across retries
    prompt = (
        f"Context:\n{req.context}\n\n"
        f"Question: {req.query}\n"
        f"Language Instruction: {lang_hint}\n"
    )
    if req.feedback:
        prompt += f"[Critic Feedback]: {req.feedback}\n"
    prompt += "Output valid JSON with keys: 'answer' (string), 'claims' (list of atomic statements), 'evidence_spans' (list of verbatim quotes)."

    max_tok = min(max(int(req.max_tokens or 600), 256), 900)
    raw, usage = await run_llm(
        prompt=prompt,
        adapter="generator",
        max_tokens=max_tok,
        temperature=0.15,
        json_schema=GENERATOR_JSON_SCHEMA,
    )
    parsed = _parse_json_object(raw)
    if parsed and "answer" in parsed:
        parsed.setdefault("claims", [])
        parsed.setdefault("evidence_spans", [])
        parsed["usage"] = usage
        return parsed
    return {"answer": raw, "claims": [], "evidence_spans": [], "usage": usage}


@app.post("/api/summarize/critic")
async def summarize_critic(req: CriticRequest):
    # Context first so vLLM reuses KV cache from the generator call
    prompt = (
        f"Context:\n{req.context}\n\n"
        f"Question: {req.query}\n"
        f"Proposed Answer:\n{req.answer}\n"
        f"Claims:\n{json.dumps(req.claims, ensure_ascii=False)}\n\n"
        "Audit whether the answer/claims contain hallucinations unsupported by Context.\n"
        "Set severity='none' and action='accept' if facts are grounded (ignore minor punctuation/style differences).\n"
        "Set severity='major' and action='rewrite' ONLY for clear factual contradictions or fabricated numbers/claims.\n"
        "Output valid JSON with keys: critique, action ('accept' | 'rewrite' | 'request_more_context'), severity ('none' | 'minor' | 'major'), issues_found (list)."
    )
    max_tok = min(max(int(req.max_tokens or 256), 128), 384)
    raw, usage = await run_llm(
        prompt=prompt,
        adapter="critic",
        max_tokens=max_tok,
        temperature=0.1,
        json_schema=CRITIC_JSON_SCHEMA,
    )
    parsed = _parse_json_object(raw)
    if parsed and "action" in parsed:
        parsed.setdefault("severity", "none" if parsed.get("action") == "accept" else "major")
        parsed.setdefault("issues_found", [])
        parsed["usage"] = usage
        return parsed

    action_match = re.search(
        r"['\"]?action['\"]?\s*[:=]\s*['\"]?(accept|approve|rewrite|request_more_context)['\"]?",
        raw,
        re.IGNORECASE,
    )
    if action_match:
        val = action_match.group(1).lower()
        act = "accept" if val in ("accept", "approve") else ("request_more_context" if val == "request_more_context" else "rewrite")
        crit_match = re.search(r"['\"]?critique['\"]?\s*[:=]\s*(.*?)(?=['\"]?action|\Z)", raw, re.DOTALL | re.IGNORECASE)
        critique = crit_match.group(1).strip(" \t\n\r'\",") if crit_match else raw[:300]
        return {"action": act, "severity": "none" if act == "accept" else "minor", "critique": critique, "issues_found": [], "usage": usage}

    return {
        "critique": f"Không thể phân tích phản hồi JSON từ Critic: {raw[:250]}",
        "action": "rewrite",
        "severity": "minor",
        "issues_found": ["critic_malformed_json_response"],
        "usage": usage,
    }


@app.post("/api/factcheck")
def factcheck_endpoint(req: FactCheckRequest):
    model, tokenizer = load_factcheck_model()
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Fact-Checker NLI model is not loaded on GPU. Cannot verify claims.",
        )

    evaluations = []
    labels = ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]
    for c in req.claims:
        inputs = tokenizer(req.context[:1800], c, truncation=True, max_length=512, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()
            pred_idx = int(torch.argmax(logits, dim=-1).item())
            pred_label = labels[pred_idx] if pred_idx < len(labels) else "NOT_ENOUGH_INFO"
            evaluations.append({
                "claim": c,
                "verdict": pred_label,
                "confidence": float(probs[pred_idx]),
                "reason": f"BGE NLI Head: {pred_label} (p={probs[pred_idx]:.2f})",
            })

    return {"evaluations": evaluations}


@app.post("/api/writer")
async def writer_endpoint(req: WriterRequest):
    v_text = "\n".join([f"- {c.get('claim_text') or c.get('text') or str(c)}" for c in req.claims])
    lang_directive = (
        "Hãy trình bày toàn bộ báo cáo (tiêu đề, đề mục, phân tích, kết luận) bằng TIẾNG VIỆT học thuật chuẩn mực, súc tích, có trích dẫn [1], [2]."
        if req.language == "vi"
        else "Write the full academic report in formal Markdown with inline citations [1], [2]."
    )
    prompt = (
        "You are an elite scientific academic paper author. Synthesize a structured, accurate Markdown report strictly grounded in the verified evidence below.\n\n"
        f"Research Question: {req.query}\n\n"
        f"Executive Summary:\n{req.answer}\n\n"
        f"Verified Scientific Evidence:\n{v_text}\n\n"
        f"Language & Style: {lang_directive}"
    )
    max_tok = min(max(int(req.max_tokens or 1100), 512), 1400)
    report, usage = await run_llm(
        prompt=prompt,
        adapter="writer",
        max_tokens=max_tok,
        temperature=0.2,
    )
    return {
        "report": report,
        "metadata": {"model": "qwen2.5-7b-lora-rslora", "engine": _active_engine},
        "usage": usage,
    }


class EmbedRequest(BaseModel):
    texts: list[str]


@app.post("/api/rerank")
def rerank_endpoint(req: RerankRequest):
    if "reranker" not in _models:
        raise HTTPException(
            status_code=501,
            detail="Capability 'reranker' (07_reranker_bge_large) chưa được nạp trên worker này.",
        )
    return {"ranked_indices": list(range(len(req.candidates)))}


@app.post("/api/embed")
def embed_endpoint(req: EmbedRequest):
    if "embedding" not in _models:
        raise HTTPException(
            status_code=501,
            detail="Capability 'embedding' (06_embedding_bge_m3) chưa được nạp trên worker này.",
        )
    return {"embeddings": []}


@app.post("/api/reader/layout")
def reader_layout_endpoint(req: ReaderLayoutRequest):
    if "reader" not in _models:
        raise HTTPException(
            status_code=501,
            detail="Capability 'layout_reader' (02_agent2_reader LayoutLMv3) chưa được nạp trên worker này.",
        )
    return {"layouts": []}


@app.post("/generate")
@app.post("/api/generate")
async def generic_generate(req: GenerateRequest):
    out, usage = await run_llm(
        prompt=req.prompt,
        adapter=req.adapter,
        max_tokens=min(int(req.max_tokens or 768), 1536),
        temperature=req.temperature,
        system_prompt=req.system_instruction,
    )
    return {"text": out, "adapter": req.adapter, "usage": usage}
