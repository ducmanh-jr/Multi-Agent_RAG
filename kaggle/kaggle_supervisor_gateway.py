"""
Kaggle Master Process Supervisor Gateway (Port 8000).
Zero-VRAM Orchestrator managing dynamic Hot-Swapping between:
  1. Worker A (gpu_finetune / 7 SOTA models) on internal port 8001.
  2. Worker B (gpu_max / Qwen 32B AWQ TP=2) on internal port 8002.

Provides seamless remote switching via /api/admin/switch_mode without restarting the Kaggle notebook.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
        sys.stderr.reconfigure(encoding="utf-8", errors="ignore")
    except Exception:
        pass
import time
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

app = FastAPI(title="Kaggle Master Process Supervisor Gateway", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_DIR = os.environ.get("PROJECT_DIR", "/kaggle/working/Multi-Agent_RAG")
if not os.path.exists(PROJECT_DIR):
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# State Tracking
_state = {
    "active_mode": "gpu_finetune",  # "gpu_finetune" | "gpu_max"
    "is_switching": False,
    "last_switch_time": time.time(),
    "switch_message": "Hệ thống hoạt động bình thường",
}

_worker_process: Optional[subprocess.Popen] = None
_vllm_process: Optional[subprocess.Popen] = None
_http_client = httpx.AsyncClient(timeout=180.0)


def get_active_worker_url() -> str:
    if _state["active_mode"] == "gpu_max":
        return "http://127.0.0.1:8002"
    return "http://127.0.0.1:8001"


def terminate_all_workers():
    """Kill worker subprocesses so Linux OS immediately recovers 100% VRAM."""
    global _worker_process, _vllm_process
    print("[Supervisor] Terminating existing GPU workers to free 100% VRAM...")

    for proc, name in [(_worker_process, "Worker A (7-Model)"), (_vllm_process, "Worker B (vLLM 32B)")]:
        if proc and proc.poll() is None:
            try:
                print(f"[Supervisor] Killing {name} (PID: {proc.pid})...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception as e:
                print(f"[Supervisor] Warning during termination: {e}")

    _worker_process = None
    _vllm_process = None

    # Force kill any lingering worker or vLLM tensor-parallel child processes on Linux
    try:
        os.system("pkill -9 -f 'vllm' 2>/dev/null")
        os.system("pkill -9 -f 'kaggle_finetune_server' 2>/dev/null")
    except Exception:
        pass

    # Force OS garbage collection and CUDA flush if torch exists
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("[Supervisor] VRAM reclamation complete.")


def spawn_worker_finetune():
    """Starts Worker A: 7-Model SOTA Server on internal port 8001."""
    global _worker_process
    terminate_all_workers()
    print("[Supervisor] Starting Worker A (kaggle_finetune_server) on internal port 8001...")

    cmd = [
        sys.executable, "-m", "uvicorn",
        "kaggle_finetune_server:app",
        "--host", "127.0.0.1",
        "--port", "8001",
        "--app-dir", os.path.join(PROJECT_DIR, "kaggle"),
    ]
    log_file_path = os.path.join(PROJECT_DIR, "kaggle", "worker_finetune.log")
    worker_log = open(log_file_path, "a", encoding="utf-8", errors="ignore")
    _worker_process = subprocess.Popen(cmd, stdout=worker_log, stderr=subprocess.STDOUT)


def spawn_worker_max():
    """Starts Worker B: vLLM 32B AWQ TP=2 on internal port 8002."""
    global _vllm_process
    terminate_all_workers()
    print("[Supervisor] Starting Worker B (vLLM 32B AWQ TP=2) on internal port 8002...")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "--tensor-parallel-size", "2",
        "--dtype", "float16",
        "--quantization", "awq",
        "--enforce-eager",
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "8192",
        "--port", "8002",
    ]
    log_file_path = os.path.join(PROJECT_DIR, "kaggle", "worker_max.log")
    vllm_log = open(log_file_path, "a", encoding="utf-8", errors="ignore")
    _vllm_process = subprocess.Popen(cmd, stdout=vllm_log, stderr=subprocess.STDOUT)


async def is_worker_healthy(url: str) -> bool:
    """Check if worker responds 200 OK immediately."""
    try:
        r = await _http_client.get(f"{url}/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


async def wait_for_worker_health(url: str, max_retries: int = 60) -> bool:
    """Poll internal worker until it responds 200 OK."""
    health_url = f"{url}/health"
    for i in range(max_retries):
        try:
            r = await _http_client.get(health_url, timeout=2.5)
            if r.status_code == 200:
                print(f"[Supervisor] Worker at {url} is ONLINE!")
                return True
        except Exception:
            pass
        await asyncio.sleep(2)
    return False


@app.on_event("startup")
async def startup_event():
    init_mode = os.environ.get("INITIAL_MODE", "gpu_finetune").lower().strip()
    _state["active_mode"] = init_mode
    print("=" * 65)
    print(f"🚀 [Kaggle Master Supervisor] Starting Gateway on Port 8000 (Initial: {init_mode.upper()})...")
    print("=" * 65)
    if init_mode == "gpu_max":
        spawn_worker_max()
    else:
        spawn_worker_finetune()


class SwitchModeRequest(BaseModel):
    target_mode: str  # "gpu_finetune" | "gpu_max"


@app.get("/")
@app.get("/health")
async def health():
    active_url = get_active_worker_url()
    worker_health = {}
    try:
        r = await _http_client.get(f"{active_url}/health", timeout=2.0)
        if r.status_code == 200:
            try:
                worker_health = r.json()
            except Exception:
                worker_health = {"worker_status": "online"}
            if not worker_health.get("worker_status"):
                worker_health["worker_status"] = "online"
    except Exception as e:
        worker_health = {"worker_status": "starting_or_offline", "error": str(e)}

    return {
        "status": "online",
        "supervisor": "Kaggle Master Supervisor Gateway v2.0",
        "active_mode": _state["active_mode"],
        "is_switching": _state["is_switching"],
        "switch_message": _state["switch_message"],
        "worker_health": worker_health,
    }


@app.get("/api/admin/logs")
async def get_logs():
    res = {}
    for log_name in ["supervisor.log", "worker_finetune.log", "worker_max.log"]:
        lp = os.path.join(PROJECT_DIR, "kaggle", log_name)
        if os.path.exists(lp):
            try:
                with open(lp, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    res[log_name] = "".join(lines[-50:])
            except Exception as e:
                res[log_name] = f"Error reading log: {e}"
        else:
            res[log_name] = "Not found"
    return res


@app.post("/api/admin/switch_mode")
async def switch_mode(req: SwitchModeRequest):
    target = req.target_mode.lower().strip()
    if target not in ["gpu_finetune", "gpu_max", "gpu_ft"]:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ chuyển sang 'gpu_finetune' hoặc 'gpu_max'")

    # Normalize gpu_ft to gpu_finetune since 7-model covers it
    if target == "gpu_ft":
        target = "gpu_finetune"

    if _state["active_mode"] == target:
        # Check if the active worker is actually alive and healthy!
        alive = await is_worker_healthy(get_active_worker_url())
        if alive:
            return {"status": "already_active", "active_mode": target, "message": f"Chế độ {target} đang hoạt động."}
        print(f"[Supervisor] Mode {target} was active but worker is dead! Respawning...")

    if _state["is_switching"]:
        raise HTTPException(status_code=429, detail="Hệ thống đang trong quá trình tráo đổi VRAM, xin vui lòng chờ.")

    _state["is_switching"] = True
    _state["switch_message"] = f"Đang tráo đổi VRAM sang chế độ {target.upper()}..."

    try:
        if target == "gpu_max":
            spawn_worker_max()
            online = await wait_for_worker_health("http://127.0.0.1:8002", max_retries=90)
        else:
            spawn_worker_finetune()
            online = await wait_for_worker_health("http://127.0.0.1:8001", max_retries=75)

        if not online:
            _state["is_switching"] = False
            _state["switch_message"] = f"Lỗi khởi động mô hình {target} sau 120 giây."
            # Fallback: if gpu_max failed, restore gpu_finetune so cluster doesn't stay dead!
            if target == "gpu_max":
                print("[Supervisor] Fallback: Respawning Worker A (gpu_finetune)...")
                spawn_worker_finetune()
                _state["active_mode"] = "gpu_finetune"
            raise HTTPException(status_code=500, detail=_state["switch_message"])

        _state["active_mode"] = target
        _state["is_switching"] = False
        _state["last_switch_time"] = time.time()
        _state["switch_message"] = f"Đã kích hoạt thành công chế độ {target.upper()}!"

        return {
            "status": "success",
            "active_mode": target,
            "message": _state["switch_message"],
        }
    except Exception as e:
        _state["is_switching"] = False
        _state["switch_message"] = f"Lỗi tráo đổi: {e}"
        # Ensure worker A is restored if dead
        if not await is_worker_healthy("http://127.0.0.1:8001"):
            print("[Supervisor] Recovery: Respawning Worker A...")
            spawn_worker_finetune()
            _state["active_mode"] = "gpu_finetune"
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# REVERSE PROXY ROUTER (Tự động chuyển tiếp yêu cầu đến Worker đang kích hoạt)
# ==============================================================================

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_all(request: Request, path: str):
    if _state["is_switching"]:
        return JSONResponse(
            status_code=503,
            content={"error": "GPU đang trong quá trình tráo đổi nóng VRAM. Vui lòng thử lại sau 30 giây."}
        )

    active_url = get_active_worker_url()
    method = request.method
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)

    # Nếu đang ở mode gpu_max (vLLM) nhưng client gửi /generate hoặc /api/router /api/summarize...
    if _state["active_mode"] == "gpu_max":
        # vLLM expects OpenAI format at /v1/chat/completions
        if path in ["generate", "api/generate", "api/router", "api/summarize/generator", "api/summarize/critic", "api/writer"]:
            try:
                body_json = await request.json() if body else {}
                prompt = body_json.get("prompt") or body_json.get("query") or ""
                sys_inst = body_json.get("system_instruction") or "Bạn là Đại Mô Hình Nghiên Cứu Khoa Học 32B."
                vllm_payload = {
                    "model": "Qwen/Qwen2.5-32B-Instruct-AWQ",
                    "messages": [
                        {"role": "system", "content": sys_inst},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": body_json.get("temperature", 0.2),
                    "max_tokens": body_json.get("max_tokens", 2048),
                }
                resp = await _http_client.post(f"{active_url}/v1/chat/completions", json=vllm_payload)
                data = resp.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return JSONResponse(content={"text": text, "output": text, "model": "Qwen2.5-32B-Instruct-AWQ"})
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": f"vLLM Proxy error: {e}"})

    # Normal reverse proxy for 7-Model Suite or default endpoints
    target_url = f"{active_url}/{path}"
    try:
        resp = await _http_client.request(
            method=method,
            url=target_url,
            content=body,
            headers=headers,
            params=request.query_params,
        )
        return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))
    except Exception as e:
        # If worker is unreachable, auto-respawn if dead
        alive = await is_worker_healthy(active_url)
        if not alive and not _state["is_switching"]:
            print(f"[Supervisor] Worker {active_url} is dead on request to /{path}! Respawning Worker A...")
            spawn_worker_finetune()
            _state["active_mode"] = "gpu_finetune"
        return JSONResponse(
            status_code=503,
            content={"error": f"Máy chủ GPU Worker đang khởi động hoặc nạp lại ({e}). Vui lòng thử lại sau vài giây."}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
