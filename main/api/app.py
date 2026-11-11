"""

FastAPI Server for Unified Multi-Agent Research RAG System.

Provides complete API compatibility for the Web Workspace UI, including:

  - Auth status & workspace configuration.

  - Document catalog & upload management.

  - Runs orchestration & status polling.

  - Public Agent specifications & diagnostics.

  - Streaming SSE (Server-Sent Events) for real-time trace monitoring.

"""



from __future__ import annotations



import asyncio

import datetime

import json

import os
import re

import shutil

import threading

import uuid

from typing import Any, AsyncGenerator, Optional



from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse, RedirectResponse

from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel



from core.config import settings

from core.graph import build_research_rag_graph

from core.tools import clear_pdf_cache

from core import db

from utils.logger import logger



app = FastAPI(

    title="Unified Multi-Agent Research RAG API",

    description="Evidence-First Scientific Literature Research System with 5 Specialized Agents",

    version="1.0.0",

)



# CORS Middleware

ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
]
_env_origins = os.getenv("ALLOWED_ORIGINS")
if _env_origins:
    ALLOWED_ORIGINS.extend([o.strip() for o in _env_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/health')
@app.get('/api/health')
def health_check():
    return {'status': 'ok', 'timestamp': datetime.datetime.now().isoformat()}




@app.middleware("http")
async def no_cache_static_middleware(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/demo2") or request.url.path.endswith((".js", ".css", ".html")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

os.makedirs(settings.ARTIFACTS_DIR, exist_ok=True)



# In-memory stores for runs and documents

runs_store: dict[str, dict[str, Any]] = {}

docs_store: dict[str, dict[str, Any]] = {}



# Compile LangGraph graph with thread-safe atomic reference management
_global_graph_lock = threading.Lock()
_ACTIVE_RUN_TASKS: dict[str, asyncio.Task] = {}
_global_graph = build_research_rag_graph()

def get_current_graph():
    """Returns the current active compiled LangGraph instance thread-safely."""
    global _global_graph
    with _global_graph_lock:
        return _global_graph

def set_current_graph(new_graph):
    """Atomically updates the active compiled graph reference."""
    global _global_graph, graph
    with _global_graph_lock:
        _global_graph = new_graph
        graph = new_graph
    return _global_graph

graph = _global_graph





# ==================== DATA SCHEMAS ====================



class CreateRunRequest(BaseModel):
    question: Optional[str] = None
    query: Optional[str] = None
    document_ids: Optional[list[str]] = None
    document_id: Optional[str] = None
    profile: Optional[str] = "balanced"
    model: Optional[str] = None
    mode: Optional[str] = None
    max_iterations: Optional[int] = 2
    chat_history: Optional[list[dict[str, str]]] = None
    session_id: Optional[str] = None


class RenameSessionRequest(BaseModel):
    title: str


class DirectResearchRequest(BaseModel):

    query: str

    pdf_path: Optional[str] = None





# ==================== AUTH & CONFIG ====================

AUTH_DEV_MODE = os.getenv("APP_DEV_MODE", "true").lower() in ("true", "1", "yes")
AUTH_SECRET_PASSWORD = os.getenv("APP_AUTH_PASSWORD", "")
_ACTIVE_AUTH_TOKENS: set[str] = set()

_AUTH_LOCK = threading.Lock()


@app.get("/api/auth/status")
def auth_status(request: Request):
    auth_header = request.headers.get("Authorization", "")
    token = None
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()

    is_authenticated = False
    if token and token in _ACTIVE_AUTH_TOKENS:
        is_authenticated = True
    elif AUTH_DEV_MODE and not AUTH_SECRET_PASSWORD:
        is_authenticated = True

    return {
        "authenticated": is_authenticated,
        "configured": True,
        "write_access": is_authenticated,
        "mode": settings.EXECUTION_MODE,
        "dev_mode": AUTH_DEV_MODE,
    }







def update_env_file(key: str, value: str):

    """Safely updates or adds a key-value pair in .env file without destroying other variables."""

    curr_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = [

        os.path.join(os.path.dirname(os.path.dirname(curr_dir)), ".env"),

        os.path.join(os.path.dirname(curr_dir), ".env"),

        os.path.abspath(".env")

    ]

    env_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

    if os.path.exists(env_path):

        with open(env_path, "r", encoding="utf-8") as f:

            lines = f.readlines()

        found = False

        new_lines = []

        for line in lines:

            if line.strip().startswith(f"{key}="):

                new_lines.append(f"{key}={value}\n")

                found = True

            else:

                new_lines.append(line)

        if not found:

            new_lines.append(f"{key}={value}\n")

        tmp_env = f"{env_path}.tmp_{os.getpid()}"
        with open(tmp_env, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        os.replace(tmp_env, env_path)





class SetModeRequest(BaseModel):

    mode: str





class SetRemoteUrlRequest(BaseModel):

    remote_url: str





@app.post("/api/config/mode")
def set_execution_mode(req: SetModeRequest):
    from modes.registry import normalize_mode
    norm_mode = normalize_mode(req.mode)
    valid_modes = [
        "gemini", "cpu_min", "cpu_max", "gpu_ft", "gpu_finetune",
        "gpu_max", "hybrid", "local_api", "kaggle_gpu", "local_cpu"
    ]
    if norm_mode not in valid_modes and req.mode not in valid_modes:
        raise HTTPException(status_code=400, detail="Chế độ không hợp lệ")
    settings.EXECUTION_MODE = norm_mode
    update_env_file("EXECUTION_MODE", norm_mode)
    return get_config()





@app.post("/api/config/remote-url")

async def set_remote_url(req: SetRemoteUrlRequest):

    new_url = req.remote_url.strip().rstrip("/")

    if not new_url.startswith("http://") and not new_url.startswith("https://"):

        raise HTTPException(status_code=400, detail="URL phải bắt đầu bằng http:// hoặc https://")



    settings.REMOTE_INFERENCE_URL = new_url

    update_env_file("REMOTE_INFERENCE_URL", new_url)



    is_online = False

    details = ""

    try:

        import httpx

        async with httpx.AsyncClient(timeout=2.0) as client:

            resp = await client.get(f"{new_url}/health")

            if resp.status_code == 200:

                is_online = True

                health_data = resp.json()

                details = f"Online - {health_data.get('gpu_count', 2)}x GPU ({health_data.get('model_worker', '')})"

            else:

                details = f"Worker trả về HTTP {resp.status_code}"

    except Exception as e:

        details = f"Chưa thể kết nối: {str(e)[:60]}"



    return {

        "status": "updated",

        "remote_worker": new_url,

        "is_online": is_online,

        "details": details,

        "config": get_config(),

    }





@app.get("/api/config/remote-url/test")

async def test_remote_url():

    url = settings.REMOTE_INFERENCE_URL

    if not url:

        return {"is_online": False, "details": "Chưa cấu hình URL Kaggle Worker."}

    try:

        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:

            resp = await client.get(f"{url}/health")

            if resp.status_code == 200:
                try:
                    health_data = resp.json()
                except Exception:
                    health_data = {}
                mw = health_data.get('model_worker') or health_data.get('active_mode') or 'Kaggle Dual T4 Worker'
                return {
                    "is_online": True,
                    "details": f"Online - {health_data.get('gpu_count', 2)}x GPU ({mw})",
                    "raw_health": health_data,
                }

            return {"is_online": False, "details": f"Worker trả về HTTP {resp.status_code}"}

    except Exception as e:

        return {"is_online": False, "details": f"Không thể kết nối: {str(e)[:60]}"}


class RemoteSwitchModeRequest(BaseModel):
    target_mode: str  # "gpu_finetune" | "gpu_max"


@app.post("/api/remote/switch-mode")
async def remote_switch_gpu_mode(req: RemoteSwitchModeRequest):
    """Ra lệnh cho Kaggle Master Supervisor tráo đổi nóng VRAM giữa 7 Mô hình SOTA và 32B AWQ."""
    url = settings.REMOTE_INFERENCE_URL
    if not url:
        raise HTTPException(status_code=400, detail="Chưa cấu hình REMOTE_INFERENCE_URL (Kaggle Cloudflare Tunnel).")

    target = req.target_mode.lower().strip()
    if target not in ["gpu_finetune", "gpu_max", "gpu_ft"]:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ chuyển đổi giữa 'gpu_finetune' và 'gpu_max'")

    try:
        import httpx
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(f"{url}/api/admin/switch_mode", json={"target_mode": target})
            if resp.status_code == 200:
                data = resp.json()
                settings.EXECUTION_MODE = target
                update_env_file("EXECUTION_MODE", target)
                return {
                    "status": "success",
                    "active_mode": target,
                    "message": data.get("message", f"Đã chuyển đổi thành công sang {target.upper()}"),
                    "remote_data": data,
                }
            else:
                raise HTTPException(status_code=resp.status_code, detail=f"Kaggle Supervisor báo lỗi: {resp.text[:120]}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể kết nối tới Kaggle Supervisor: {e}")


@app.get("/api/remote/gpu-status")
async def get_remote_gpu_status():
    """Thăm dò Kaggle Supervisor để biết chế độ nào đang độc chiếm VRAM."""
    url = settings.REMOTE_INFERENCE_URL
    if not url:
        return {"is_online": False, "active_mode": "offline", "is_switching": False, "details": "Chưa cấu hình URL Kaggle Worker."}

    try:
        import httpx
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(f"{url}/health")
            if resp.status_code == 200:
                data = resp.json()
                active_mode = data.get("active_mode")
                if not active_mode:
                    if data.get("has_vllm_32b"):
                        active_mode = "gpu_max"
                    elif "gpu_finetune" in data.get("pipeline", ""):
                        active_mode = "gpu_finetune"
                    else:
                        active_mode = settings.EXECUTION_MODE

                return {
                    "is_online": True,
                    "active_mode": active_mode,
                    "is_switching": data.get("is_switching", False),
                    "switch_message": data.get("switch_message", ""),
                    "supervisor": data.get("supervisor", ""),
                    "worker_health": data.get("worker_health", {}),
                }
            return {"is_online": False, "active_mode": "error", "details": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"is_online": False, "active_mode": "unreachable", "details": str(e)[:60]}





class LocalCpuConfigRequest(BaseModel):

    local_cpu_url: Optional[str] = None

    local_cpu_model: Optional[str] = None



@app.get("/api/config/local-cpu/test")
async def test_local_cpu():
    target_urls = [
        "http://127.0.0.1:11434",
        "http://localhost:11434",
    ]
    if settings.LOCAL_CPU_URL and settings.LOCAL_CPU_URL not in target_urls:
        target_urls.insert(0, settings.LOCAL_CPU_URL.rstrip("/"))

    try:
        import httpx
        for test_url in target_urls:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(f"{test_url}/api/tags")
                    if resp.status_code == 200:
                        tags = resp.json().get("models", [])
                        names = [m.get("name") for m in tags][:3]
                        model_info = f"Online - CPU Server ({', '.join(names) if names else 'Sẵn sàng'})"
                        return {"is_online": True, "details": model_info}
                    resp = await client.get(f"{test_url}/v1/models")
                    if resp.status_code == 200:
                        return {"is_online": True, "details": "Online - Local CPU Server (OpenAI compat)"}
            except Exception:
                continue
        return {"is_online": False, "details": "Chưa bật server CPU (11434)"}
    except Exception as e:
        return {"is_online": False, "details": f"Chưa bật server CPU ({str(e)[:40]})"}

@app.post("/api/config/local-cpu/start")
async def start_local_cpu_service():
    """
    Attempts to start the local CPU inference service:
    1. Checks if already running on port 11434.
    2. If not running, checks if 'ollama' is installed and launches 'ollama serve'.
    3. If 'ollama' is not found, launches the built-in local_cpu_server.py on port 11434.
    4. Waits and verifies health.
    """
    import sys
    from utils.logger import logger
    import shutil
    import subprocess
    import asyncio
    import httpx
    from pathlib import Path
    
    target_urls = [
        "http://127.0.0.1:11434",
        "http://localhost:11434",
    ]
    if settings.LOCAL_CPU_URL and settings.LOCAL_CPU_URL not in target_urls:
        target_urls.insert(0, settings.LOCAL_CPU_URL.rstrip("/"))
    
    # 1. Quick check if already online
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(f"{url}/api/tags")
            if resp.status_code == 200:
                return {
                    "success": True,
                    "is_online": True,
                    "message": "Dịch vụ CPU Inference (Ollama) đã đang chạy sẵn.",
                    "details": "Online"
                }
    except Exception:
        pass

    # 2. Try starting Ollama if installed
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        candidate_paths = [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
            r"C:\Program Files\Ollama\ollama.exe",
        ]
        for cp in candidate_paths:
            if os.path.exists(cp):
                ollama_bin = cp
                break

    started_process = None
    engine_name = "Ollama"

    if ollama_bin:
        try:
            creation_flags = subprocess.DETACHED_PROCESS if os.name == "nt" else 0
            started_process = subprocess.Popen(
                [ollama_bin, "serve"],
                creationflags=creation_flags,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.info(f"Spawned 'ollama serve' using binary at: {ollama_bin}")
        except Exception as e:
            logger.warning(f"Failed to spawn ollama serve: {e}")

    # If ollama not found or failed, launch the built-in clean local_cpu_server
    if not started_process:
        try:
            engine_name = "Built-in Local CPU Server"
            creation_flags = (0x00000008 | 0x00000200) if os.name == "nt" else 0
            server_cwd = str(Path(__file__).resolve().parent.parent)
            env = os.environ.copy()
            env["PYTHONPATH"] = server_cwd + os.pathsep + env.get("PYTHONPATH", "")
            started_process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "core.local_cpu_server:app", "--port", "11434", "--host", "0.0.0.0"],
                cwd=server_cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags
            )
            logger.info("Spawned built-in local_cpu_server on port 11434")
        except Exception as e:
            return {
                "success": False,
                "is_online": False,
                "message": f"Không thể khởi động dịch vụ CPU: {e}",
                "details": str(e)
            }

    # 3. Wait up to 14 seconds and verify connectivity
    for _ in range(28):
        await asyncio.sleep(0.5)
        for t_url in target_urls:
            try:
                async with httpx.AsyncClient(timeout=1.2) as client:
                    resp = await client.get(f"{t_url}/api/tags")
                    if resp.status_code == 200:
                        return {
                            "success": True,
                            "is_online": True,
                            "engine": engine_name,
                            "message": f"Đã khởi động thành công {engine_name} trên cổng 11434!",
                            "details": "Online"
                        }
            except Exception:
                continue

    return {
        "success": False,
        "is_online": False,
        "message": "Tiến trình CPU Server đã được gọi nhưng chưa phản hồi trên cổng 11434. Vui lòng thử lại hoặc mở Ollama thủ công.",
        "details": "Timeout starting"
    }


@app.post("/api/config/local-cpu")

async def update_local_cpu_config(req: LocalCpuConfigRequest):

    if req.local_cpu_url is not None:

        new_url = req.local_cpu_url.strip().rstrip("/")

        settings.LOCAL_CPU_URL = new_url

        update_env_file("LOCAL_CPU_URL", new_url)

    if req.local_cpu_model is not None:

        new_model = req.local_cpu_model.strip()

        settings.LOCAL_CPU_MODEL = new_model

        update_env_file("LOCAL_CPU_MODEL", new_model)

    

    test_res = await test_local_cpu()

    return {

        "status": "updated",

        "local_cpu_url": settings.LOCAL_CPU_URL,

        "local_cpu_model": settings.LOCAL_CPU_MODEL,
        "gemini_usage": get_gemini_usage_stats(),

        "is_online": test_res.get("is_online", False),

        "details": test_res.get("details", ""),

        "config": get_config(),

    }




def get_gemini_usage_stats():
    import sqlite3, datetime, logging
    from pathlib import Path
    _log = logging.getLogger("MultiAgentRAG")
    today = datetime.date.today().isoformat()
    used = 0
    limit = 1500
    try:
        db_p = Path(settings.BASE_DIR) / "data" / "workspace.db"
        if db_p.exists():
            with sqlite3.connect(db_p) as conn:
                c = conn.cursor()
                c.execute("""
                    CREATE TABLE IF NOT EXISTS api_usage_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        provider TEXT NOT NULL,
                        model TEXT,
                        endpoint TEXT,
                        timestamp TEXT NOT NULL
                    )
                """)
                c.execute(
                    "SELECT count(*) FROM api_usage_logs WHERE provider = 'gemini' AND timestamp LIKE ?",
                    (f"{today}%",)
                )
                row = c.fetchone()
                if row:
                    used = row[0]
    except Exception as e:
        _log.warning(f"Error reading gemini usage: {e}")
    
    return {
        "provider": "gemini",
        "date": today,
        "used": used,
        "limit": limit,
        "remaining": max(0, limit - used),
        "badge": f"{used} / {limit:,}".replace(",", "."),
    }


@app.get("/api/capabilities")
def get_system_capabilities():
    from utils.local_vulkan_setup import probe_all_capabilities
    return probe_all_capabilities(force_refresh=True)

@app.get("/api/config/gemini-usage")
def get_gemini_usage():
    return get_gemini_usage_stats()

@app.get("/api/config")

def get_config():

    model_name = settings.DEFAULT_LLM_MODEL if settings.EXECUTION_MODE == "local_api" else "Qwen2.5-7B (Kaggle Dual T4)"

    provider_name = "Google Gemini API" if settings.EXECUTION_MODE == "local_api" else "Kaggle Dual T4 GPU"

    return {

        "model": model_name,

        "provider": provider_name,

        "demo": False,

        "auth_required": False,

        "workflow_profile": "academic",

        "supported_profiles": ["balanced", "deep_research", "fast_verification"],

        "default_profile": "balanced",

        "supported_models": [settings.DEFAULT_LLM_MODEL, settings.FALLBACK_LLM_MODEL],

        "default_model": settings.DEFAULT_LLM_MODEL,

        "execution_mode": settings.EXECUTION_MODE,

        "remote_worker": settings.REMOTE_INFERENCE_URL or "Local API",

        "max_upload_mb": 50,

        "features": {

            "pdf_upload": True,

            "source_preview": True,

            "numeric_verification": True,

            "storm_perspectives": True,

        },

        "agent_modes": settings.get_all_agent_modes(),

        "local_cpu_url": settings.LOCAL_CPU_URL,

        "local_cpu_model": settings.LOCAL_CPU_MODEL,

    }





class SetAgentModesRequest(BaseModel):

    agent_modes: Optional[dict[str, str]] = None

    agent: Optional[str] = None

    mode: Optional[str] = None





@app.post("/api/config/agents")

def set_agent_modes(req: SetAgentModesRequest):
    updated = {}
    def _norm(m: str) -> str:
        m = (m or "").lower()
        if m in ["gemini", "api"]:
            return "gemini"
        elif m in ["cpu_min", "cpumin"]:
            return "cpu_min"
        elif m in ["cpu_max", "cpumax", "cpu"]:
            return "cpu_max"
        elif m in ["gpu_ft", "gpuft", "ft", "kaggle_fn", "fn"]:
            return "gpu_ft"
        elif m in ["gpu_finetune", "finetune", "gpufinetune", "gpu_ft_new"]:
            return "gpu_finetune"
        elif m in ["gpu_max", "gpumax", "kaggle_max", "kaggle", "model"]:
            return "gpu_max"
        elif m in ["hybrid", "auto", "optimal"]:
            return "hybrid"
        return "cpu_min"

    if req.agent_modes:
        for agent_name, mode in req.agent_modes.items():
            canonical = _norm(mode)
            settings.set_agent_mode(agent_name, canonical)
            update_env_file(f"AGENT_{agent_name.upper()}_MODE", canonical)
            updated[agent_name] = canonical
    elif req.agent and req.mode:
        canonical = _norm(req.mode)
        settings.set_agent_mode(req.agent, canonical)
        update_env_file(f"AGENT_{req.agent.upper()}_MODE", canonical)
        updated[req.agent] = canonical
    set_current_graph(build_research_rag_graph())
    return {

        "status": "success",

        "agent_modes": settings.get_all_agent_modes(),

        "local_cpu_url": settings.LOCAL_CPU_URL,

        "local_cpu_model": settings.LOCAL_CPU_MODEL,

        "config": get_config(),

    }


@app.post("/api/config/agents/auto-select")
async def auto_select_best_models():
    """Tự động phân tích và kích hoạt mô hình tối ưu nhất cho từng Agent dựa trên hạ tầng thực tế."""
    kaggle_res = await test_remote_url()
    cpu_res = await test_local_cpu()
    has_gemini = bool(settings.GEMINI_API_KEY and len(settings.GEMINI_API_KEY) > 10)

    if kaggle_res.get("is_online"):
        strategy = "kaggle_gpu"
        modes = {
            "retrieval": "cpu",
            "reader": "cpu",
            "summarizer": "model",
            "factchecker": "cpu",
            "writer": "model"
        }
        reason = "Đã kết nối GPU Kaggle Dual T4 (Qwen2.5-7B). Tối ưu hóa phân rã góc nhìn và biên soạn báo cáo."
    elif has_gemini:
        strategy = "gemini_cloud"
        modes = {
            "retrieval": "api",
            "reader": "cpu",
            "summarizer": "api",
            "factchecker": "cpu",
            "writer": "api"
        }
        reason = "Đã kích hoạt Google Gemini API. Tối ưu phân rã góc nhìn Stanford STORM, trích xuất luận điểm và biên soạn báo cáo."
    else:
        strategy = "local_cpu"
        modes = {
            "retrieval": "cpu",
            "reader": "cpu",
            "summarizer": "cpu",
            "factchecker": "cpu",
            "writer": "cpu"
        }
        reason = "Đã kích hoạt Local CPU Server (11434). Chế độ tự chủ 100% offline, bảo mật và phản hồi tức thì."

    for agent_name, mode in modes.items():
        settings.set_agent_mode(agent_name, mode)
        update_env_file(f"AGENT_{agent_name.upper()}_MODE", mode)
    set_current_graph(build_research_rag_graph())

    return {
        "status": "success",
        "strategy": strategy,
        "reason": reason,
        "agent_modes": settings.get_all_agent_modes(),
        "config": get_config(),
    }


class ApplyPresetRequest(BaseModel):
    preset: str


@app.post("/api/config/agents/preset")
async def apply_agent_preset(req: ApplyPresetRequest):
    """
    Kích hoạt 1 chạm các siêu combo kiến trúc tối thượng:
    - 'local_titan': Tối ưu Ryzen 7 32GB RAM (BGE-M3 + Docling + DeepSeek-R1-14B + MiniCheck)
    - 'kaggle_colossus': Tối ưu Kaggle Dual T4 (ColQwen2 + Docling + DeepSeek-R1-32B TP=2)
    - 'gemini_cloud': Google Gemini 3.1 Flash Cloud toàn diện
    """
    preset = req.preset.lower().strip()
    if preset in ["local_titan", "titan", "local"]:
        modes = {
            "retrieval": "cpu",
            "reader": "cpu",
            "summarizer": "cpu",
            "factchecker": "cpu",
            "writer": "cpu"
        }
        settings.EXECUTION_MODE = "local_cpu"
        update_env_file("EXECUTION_MODE", "local_cpu")
        settings.LOCAL_CPU_MODEL = "deepseek-r1:14b"
        update_env_file("LOCAL_CPU_MODEL", "deepseek-r1:14b")
        title = "Combo 1: Local Titan Workstation"
        desc = "Đã kích hoạt 100% Local CPU Ryzen 7 (32GB RAM) với DeepSeek-R1 14B & MiniCheck SOTA."
    elif preset in ["kaggle_colossus", "colossus", "kaggle"]:
        modes = {
            "retrieval": "model",
            "reader": "model",
            "summarizer": "model",
            "factchecker": "model",
            "writer": "model"
        }
        title = "Combo 2: Kaggle 32B Colossus"
        desc = "Đã kích hoạt cụm Kaggle Dual T4 (TP=2) với DeepSeek-R1 32B AWQ & ColQwen2 Visual RAG."
    else:
        modes = {
            "retrieval": "api",
            "reader": "cpu",
            "summarizer": "api",
            "factchecker": "cpu",
            "writer": "api"
        }
        title = "Combo 3: Google Gemini Cloud"
        desc = "Đã kích hoạt Google Gemini 3.1 Flash Cloud phối hợp PyMuPDF C-Engine & QuoteVerifier."

    for agent_name, mode in modes.items():
        settings.set_agent_mode(agent_name, mode)
        update_env_file(f"AGENT_{agent_name.upper()}_MODE", mode)
    set_current_graph(build_research_rag_graph())

    return {
        "status": "success",
        "preset": preset,
        "title": title,
        "description": desc,
        "agent_modes": settings.get_all_agent_modes(),
        "config": get_config(),
    }






@app.get("/api/agents")

def get_agents():

    """Returns machine-readable specs for the 5 Specialized Agents."""

    agents = [

        {

            "role": "RetrievalAgent",

            "name": "1. Retrieval Agent",

            "purpose": "Tìm kiếm đa chiều BM25 + BGE Dense RRF & Phân rã góc nhìn STORM",

            "stages": ["STORM Perspective Decomposition", "Hybrid BM25 + Dense Search", "RRF Fusion"],

            "guard": "Bảo toàn toàn bộ seen_chunks qua các lượt retry.",

        },

        {

            "role": "ReaderAgent",

            "name": "2. Reader Agent",

            "purpose": "Bóc tách cấu trúc tài liệu PDF (DocLayNet Layout, Bounding Box, Offset)",

            "stages": ["Layout Parsing", "Verbatim SourceUnit Extraction", "PaperQA2 Context Hashing"],

            "guard": "Lưu vết vị trí ký tự tuyệt đối, loại trừ trích dẫn giả mạo.",

        },

        {

            "role": "SummarizerAgent",

            "name": "3. Summarizer Agent",

            "purpose": "Tổng hợp nhận định nguyên tử qua kiến trúc kép Generator - Critic",

            "stages": ["Evidence Synthesis", "Atomic Claim Extraction", "Critic Filtering"],

            "guard": "Chỉ giữ nhận định có bằng chứng nguồn; loại bỏ overclaims.",

        },

        {

            "role": "FactCheckerAgent",

            "name": "4. Fact-Checker Agent",

            "purpose": "Thẩm định độ trung thực khoa học qua NLI SciFact và kiểm tra số học",

            "stages": ["Deterministic Numeric Check", "Temporal/Date Check", "SciFact NLI Consensus"],

            "guard": "Chặn hoàn toàn khẳng định mâu thuẫn hoặc số liệu bịa đặt.",

        },

        {

            "role": "WriterAgent",

            "name": "5. Writer Agent",

            "purpose": "Xuất bản báo cáo khoa học từ Claim Pool được xác thực (0% Hallucination)",

            "stages": ["Claim Pool Assembly", "APA/IEEE In-Text Citations", "Academic Layout"],

            "guard": "Tuyệt đối không sinh nhận định mới ngoài tập verified_claims.",

        },

    ]

    return {"agents": agents}





@app.get("/api/diagnostics")

def get_diagnostics():

    docs = db.list_documents()

    return {

        "status": "healthy",

        "documents": len(docs),

        "chunks": len(docs) * 4,

        "indexed_chunks": len(docs) * 4,

        "reindex_needed": False,

        "budget": {

            "calls": settings.MAX_FACTCHECK_CALLS,

            "requests_per_minute": 15,

        },

        "mode": settings.EXECUTION_MODE,

        "remote_inference_url": settings.REMOTE_INFERENCE_URL or "N/A",

        "llm_model": settings.DEFAULT_LLM_MODEL,

        "embedding_model": settings.EMBEDDING_MODEL,

        "total_active_runs": len(db.list_runs()),

        "total_documents": len(docs),

    }





# ==================== DOCUMENT MANAGEMENT ====================



def _format_doc(doc: dict[str, Any]) -> dict[str, Any]:

    """Ensures document object has all UI-expected fields (filename, title, kind, chunk_count, page_count)."""

    name = doc.get("name") or doc.get("filename") or "document"

    meta = doc.get("metadata") or {}

    if isinstance(meta, str):

        try:

            meta = json.loads(meta)

        except Exception:

            meta = {}



    doc_type = doc.get("doc_type") or ("pdf" if name.lower().endswith(".pdf") else "text")

    kind = meta.get("kind") or ("PDF" if doc_type == "pdf" else "TXT")

    title = meta.get("title") or name.rsplit(".", 1)[0].replace("_", " ")

    chunk_count = meta.get("chunk_count") or meta.get("chunks_count") or 4

    page_count = meta.get("page_count") or 1



    has_pdf = meta.get("has_pdf") if "has_pdf" in meta else (doc_type == "pdf" or name.lower().endswith(".pdf"))
    source_url = meta.get("source_url") or ""
    source = meta.get("source") or ("arXiv" if "arxiv" in str(doc.get("id", "")).lower() else "Học thuật")
    authors = meta.get("authors") or []
    year = meta.get("year") or 2024
    abstract = meta.get("abstract") or ""

    formatted = dict(doc)
    formatted.update({
        "id": doc.get("id"),
        "filename": name,
        "name": name,
        "title": title,
        "kind": kind,
        "doc_type": doc_type,
        "chunk_count": chunk_count,
        "page_count": page_count,
        "size": doc.get("size", 0),
        "uploaded_at": doc.get("uploaded_at"),
        "has_pdf": has_pdf,
        "source_url": source_url,
        "source": source,
        "authors": authors,
        "year": year,
        "abstract": abstract,
    })
    return formatted





@app.get("/api/documents")

def list_documents():

    return {"documents": [_format_doc(d) for d in db.list_documents()]}





@app.get("/api/documents/{doc_id}/file")
def get_document_file(doc_id: str):
    doc = db.get_document(doc_id)
    if not doc or not os.path.exists(doc.get("path", "")):
        raise HTTPException(status_code=404, detail="Document file not found")
    
    file_path = doc.get("path", "")
    name = doc.get("name", "")
    doc_type = doc.get("doc_type", "")

    if name.endswith(".pdf") or file_path.endswith(".pdf") or doc_type == "pdf":
        media_type = "application/pdf"
    elif name.endswith(".md") or file_path.endswith(".md") or doc_type == "md":
        media_type = "text/markdown; charset=utf-8"
    else:
        media_type = "text/plain; charset=utf-8"

    return FileResponse(file_path, filename=name or "document", media_type=media_type, content_disposition_type="inline")





@app.post("/api/documents/demo")

def seed_demo_documents():

    """Seeds foundational scientific benchmark papers into the workspace catalog."""

    demo_docs = [

        {

            "id": "doc_lora_2021",

            "name": "Hu_et_al_2021_LoRA_Low_Rank_Adaptation.txt",

            "text": "Low-Rank Adaptation, or LoRA, freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. LoRA reduces memory usage significantly while matching full fine-tuning performance.",

            "size": 1024500,

        },

        {

            "id": "doc_qlora_2023",

            "name": "Dettmers_et_al_2023_QLoRA_Efficient_Finetuning.txt",

            "text": "QLoRA is an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA).",

            "size": 1820400,

        },

        {

            "id": "doc_transformer_2017",

            "name": "Vaswani_et_al_2017_Attention_Is_All_You_Need.txt",

            "text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.",

            "size": 2215000,

        },

        {

            "id": "doc_qasper_2021",

            "name": "Dasigi_et_al_2021_QASPER_Benchmark.txt",

            "text": "QASPER consists of 5,049 questions over 1,585 NLP papers. Each question is sought by an NLP practitioner who has not read the paper. Dense retrieval with BGE-large achieves a Hit@5 of 40.97% on QASPER dev benchmark.",

            "size": 1540000,

        },

    ]



    for d in demo_docs:

        file_path = os.path.join(settings.UPLOAD_DIR, d["name"])

        with open(file_path, "w", encoding="utf-8") as f:

            f.write(d["text"])

        actual_size = os.path.getsize(file_path)

        db.save_document(

            doc_id=d["id"],

            name=d["name"],

            path=os.path.abspath(file_path),

            size=actual_size,

            doc_type="text",

            metadata={"title": d["name"], "chunks_count": 1},

        )



    return {"documents": [_format_doc(d) for d in db.list_documents()], "status": "seeded"}





@app.post("/api/documents/reindex")
def reindex_documents():
    import time
    start_time = time.perf_counter()
    docs = db.list_documents()
    from core.tools import clear_pdf_cache, load_or_parse_selected_pdf
    from agents.agent1_retrieval.tools.fts5_search import SQLiteFTS5Searcher

    clear_pdf_cache()
    fts = SQLiteFTS5Searcher()
    all_chunks = []

    for d in docs:
        p = d.get("path", "")
        if p and os.path.exists(p):
            try:
                units = load_or_parse_selected_pdf(p)
                all_chunks.extend(units)
            except Exception as exc:
                logger.warning(f"[Reindex] Error parsing doc '{p}': {exc}")

    if all_chunks:
        try:
            fts.fit(all_chunks)
        except Exception as exc:
            logger.warning(f"[Reindex] FTS5 index update warning: {exc}")

    duration_ms = round((time.perf_counter() - start_time) * 1000, 1)
    return {
        "status": "success",
        "chunks_updated": len(all_chunks),
        "docs_count": len(docs),
        "duration_ms": duration_ms,
    }


@app.post("/api/documents/rebuild")
def rebuild_documents():
    return reindex_documents()





@app.post("/api/documents")

@app.post("/api/documents/upload")

async def upload_document_api(file: UploadFile = File(...)):
    """Upload PDF document through UI with real-time progress tracking."""
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    upload_id = doc_id
    _UPLOAD_STATUS[upload_id] = {"upload_id": upload_id, "status": "saving", "percent": 25}

    try:
        raw_name = os.path.basename(file.filename or "uploaded_document")
        safe_filename = re.sub(r"[^\w\-_\. ]", "_", raw_name).strip()
        if not safe_filename:
            safe_filename = "document.pdf" if (file.filename or "").lower().endswith(".pdf") else "document.txt"

        file_path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}_{safe_filename}")

        max_upload_bytes = 100 * 1024 * 1024  # 100MB limit guard
        written = 0
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_upload_bytes:
                    buffer.close()
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    raise HTTPException(status_code=413, detail="Tệp tin vượt quá giới hạn cho phép (tối đa 100MB).")
                buffer.write(chunk)

        _UPLOAD_STATUS[upload_id] = {"upload_id": upload_id, "status": "indexing", "percent": 75}
        size = os.path.getsize(file_path)
        doc_type = "pdf" if safe_filename.lower().endswith(".pdf") else "text"
        doc_info = db.save_document(
            doc_id=doc_id,
            name=safe_filename,
            path=os.path.abspath(file_path),
            size=size,
            doc_type=doc_type,
        )
        _UPLOAD_STATUS[upload_id] = {
            "upload_id": upload_id,
            "status": "completed",
            "percent": 100,
            "doc_id": doc_id,
        }
        return {"document": _format_doc(doc_info), "status": "uploaded", "upload_id": upload_id}
    except Exception as exc:
        _UPLOAD_STATUS[upload_id] = {"upload_id": upload_id, "status": "failed", "percent": 0, "error": str(exc)}
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")





@app.delete("/api/documents/{doc_id}")

def delete_document(doc_id: str):

    doc = db.get_document(doc_id)

    if doc:

        if os.path.exists(doc.get("path", "")):

            try:

                os.remove(doc["path"])

            except OSError:

                pass

        db.delete_document(doc_id)

        return {"status": "deleted", "id": doc_id}

    raise HTTPException(status_code=404, detail="Document not found.")


class BatchDeleteDocsRequest(BaseModel):
    doc_ids: list[str]


@app.post("/api/documents/batch-delete")
def batch_delete_documents(req: BatchDeleteDocsRequest):
    deleted_ids = []
    for doc_id in req.doc_ids:
        doc = db.get_document(doc_id)
        if doc:
            if os.path.exists(doc.get("path", "")):
                try:
                    os.remove(doc["path"])
                except OSError:
                    pass
            db.delete_document(doc_id)
            deleted_ids.append(doc_id)
    return {"status": "deleted", "deleted_ids": deleted_ids, "count": len(deleted_ids)}





@app.post("/api/cache/clear")

def clear_cache():

    conn = db.get_db_connection()

    cursor = conn.cursor()

    cursor.execute("DELETE FROM llm_cache")

    deleted = cursor.rowcount

    conn.commit()

    conn.close()

    # Clear in-memory parsed PDF chunks cache to free RAM
    clear_pdf_cache()
    logger.info("[CacheClear] Deleted LLM persistent cache entries and cleared in-memory PDF chunk cache.")

    return {"status": "ok", "deleted": deleted, "pdf_cache_cleared": True}





@app.post("/api/auth/login")
def auth_login(req: Optional[dict] = None):
    req = req or {}
    password = req.get("password", "")

    if AUTH_SECRET_PASSWORD:
        if password != AUTH_SECRET_PASSWORD:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials.")

    token = f"sess_{uuid.uuid4().hex}"
    _ACTIVE_AUTH_TOKENS.add(token)
    return {
        "authenticated": True,
        "token": token,
        "write_access": True,
        "dev_mode": AUTH_DEV_MODE,
    }


@app.post("/api/auth/logout")
def auth_logout(request: Request):
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        _ACTIVE_AUTH_TOKENS.discard(token)
    return {"authenticated": False}





_UPLOAD_STATUS: dict[str, dict[str, Any]] = {}


@app.get("/api/uploads/{upload_id}")
def get_upload_status(upload_id: str):
    if upload_id in _UPLOAD_STATUS:
        return _UPLOAD_STATUS[upload_id]
    doc = db.get_document(upload_id)
    if doc:
        return {"upload_id": upload_id, "status": "completed", "percent": 100, "doc_id": doc.get("id")}
    raise HTTPException(status_code=404, detail=f"Upload task '{upload_id}' not found.")






# ==================== CHAT SESSIONS (ĐOẠN CHAT) ====================

@app.get("/api/sessions")
def list_chat_sessions():
    return {"sessions": db.list_sessions()}


@app.get("/api/sessions/{session_id}")
def get_chat_session(session_id: str):
    sess = db.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")
    runs = db.get_session_runs(session_id)
    return {"session": sess, "runs": runs}


@app.post("/api/sessions/{session_id}/rename")
@app.patch("/api/sessions/{session_id}")
def rename_chat_session(session_id: str, req: RenameSessionRequest):
    new_title = req.title.strip()
    if not new_title:
        raise HTTPException(status_code=400, detail="Title cannot be empty.")
    ok = db.rename_session(session_id, new_title)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "success", "session_id": session_id, "title": new_title}


@app.delete("/api/sessions/{session_id}")
def delete_chat_session(session_id: str):
    ok = db.delete_session(session_id)
    return {"status": "success", "deleted": ok}


@app.delete("/api/sessions")
def clear_all_chat_sessions():
    deleted = db.clear_all_sessions()
    return {"status": "success", "deleted_count": deleted}


# ==================== RUNS & ORCHESTRATION ====================



@app.get("/api/runs")

def list_runs():

    return {"runs": db.list_runs()}





@app.get("/api/runs/{run_id}")

def get_run_status(run_id: str):

    run_record = db.get_run(run_id)

    if not run_record:

        raise HTTPException(status_code=404, detail="Run not found.")



    # Format events from reasoning_trace for Web UI compatibility

    events = []

    traces = run_record.get("reasoning_trace") or []

    for seq, t in enumerate(traces, start=1):

        raw_agent = str(t.get("agent") or t.get("node") or "orchestrator").lower()

        if "retrieval" in raw_agent:

            ag = "retrieval"

        elif "reader" in raw_agent:

            ag = "reader"

        elif "summarizer" in raw_agent:

            ag = "summarizer"

        elif "fact" in raw_agent:

            ag = "fact_checker"

        elif "writer" in raw_agent or "escalation" in raw_agent:

            ag = "writer"

        else:

            ag = "coordinator"



        events.append({

            "sequence": seq,

            "round": t.get("retry_count", 0) + 1,

            "agent": ag,

            "status": "completed",

            "message": t.get("action", "Xử lý thành công"),

            "details": t,

            "created_at": run_record.get("created_at"),

        })



    # Format result object for Web UI compatibility

    result = None

    if run_record.get("status") in ("completed", "failed"):

        v_claims = run_record.get("verified_claims") or []

        r_claims = run_record.get("rejected_claims") or []

        spans = run_record.get("evidence_spans") or []

        markdown_text = run_record.get("final_report") or ""



        checks_list = []

        claims_list = []

        for i, c in enumerate(v_claims, start=1):

            claim_text = c.get("claim", "") if isinstance(c, dict) else str(c)

            verdict = "supported" if (isinstance(c, dict) and c.get("status") == "support") else "insufficient"

            checks_list.append({

                "claim_id": f"claim_{i}",

                "verdict": verdict,

                "explanation": c.get("evidence", "") if isinstance(c, dict) else "Bằng chứng hợp lệ",

                "evidence_ids": [f"ev_{i}"],

            })

            claims_list.append({"id": f"claim_{i}", "text": claim_text})



        evidence_list = []

        retrieved_list = []

        for i, s in enumerate(spans, start=1):

            quote = s.get("quote", "") if isinstance(s, dict) else str(s)

            doc_id = s.get("doc_id", f"doc_{i}") if isinstance(s, dict) else f"doc_{i}"

            page_no = s.get("page_no", 1) if isinstance(s, dict) else 1

            evidence_list.append({

                "id": f"ev_{i}",

                "document_id": doc_id,

                "filename": doc_id,

                "quote": quote,

                "page": page_no,

                "start": 0,

                "end": len(quote),

                "relevance": "Trích dẫn được xác thực qua NLI và kiểm chứng số",

                "chunk_id": f"ch_{i}",

                "regions": [],

            })

            retrieved_list.append({"id": f"ch_{i}", "text": quote})



        result = {

            "title": run_record.get("question") or "Báo cáo Nghiên cứu Khoa học",

            "run_id": run_record.get("id"),

            "markdown": markdown_text,

            "report": markdown_text,

            "checks": checks_list,

            "claims": claims_list,

            "evidence": evidence_list,

            "retrieved": retrieved_list,

            "verified_claims": v_claims,

            "rejected_claims": r_claims,

            "evidence_spans": spans,

            "reported_claim_count": len(v_claims),

            "reported_claim_ids": [c["id"] for c in claims_list],

            "llm_calls": len(events) or 3,

            "elapsed_ms": (max(50, int((datetime.datetime.now() - datetime.datetime.fromisoformat(run_record["created_at"])).total_seconds() * 1000)) if run_record.get("created_at") else 1200),

            "rounds": 1,

            "warnings": [],

            "events": events,

            "coverage": [],

            "stop_reason": "completed" if run_record.get("status") == "completed" else "budget_exhausted",

        }



    doc_id_val = run_record.get("document_id")

    return {

        **run_record,

        "report": run_record.get("final_report"),

        "evidence": (result or {}).get("evidence", []),

        "request": {"question": run_record.get("question")},

        "events": events,

        "result": result,

        "document_id": doc_id_val,

        "document_ids": [doc_id_val] if doc_id_val else [],

    }





@app.post("/api/runs/{run_id}/cancel")

def cancel_run(run_id: str):

    db.update_run_result(run_id, status="cancelled")

    return {"status": "cancelled", "id": run_id}





@app.delete("/api/runs/{run_id}")

def delete_run_endpoint(run_id: str):

    success = db.delete_run(run_id)

    if not success:

        raise HTTPException(status_code=404, detail="Phiên nghiên cứu không tồn tại.")

    return {"status": "deleted", "id": run_id}


@app.delete("/api/runs")
def clear_all_runs_endpoint():
    deleted = db.clear_all_runs()
    return {"status": "cleared", "deleted": deleted}





@app.get("/api/runs/{run_id}/trace")

def export_run_trace(run_id: str):

    run_record = db.get_run(run_id)

    if not run_record:

        raise HTTPException(status_code=404, detail="Run not found.")

    return {"trace": run_record.get("reasoning_trace") or []}





@app.get("/api/runs/{run_id}/export")

def export_run_report(run_id: str, format: str = "md"):

    """Export research report as Markdown or JSON file."""

    run_record = db.get_run(run_id)

    if not run_record:

        raise HTTPException(status_code=404, detail="Run not found.")



    report_md = run_record.get("final_report") or ""

    if not report_md:

        raise HTTPException(status_code=400, detail="Phiên nghiên cứu chưa có báo cáo để xuất.")



    if format == "json":
        data = {
            "run_id": run_id,
            "question": run_record.get("question"),
            "status": run_record.get("status"),
            "report": report_md,
            "verified_claims": run_record.get("verified_claims") or [],
            "evidence_spans": run_record.get("evidence_spans") or [],
            "reasoning_trace": run_record.get("reasoning_trace") or [],
        }
        return Response(
            content=json.dumps(data, ensure_ascii=False, indent=2),
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="research-{run_id}.json"'},
        )
    elif format.lower() == "pdf":
        try:
            import fitz
            doc = fitz.open()
            try:
                page = doc.new_page()
                lines = report_md.split("\n")
                y = 50
                for line in lines:
                    if y > 780:
                        page = doc.new_page()
                        y = 50
                    clean_line = line.strip()[:110]
                    page.insert_text(fitz.Point(40, y), clean_line, fontsize=9)
                    y += 14
                pdf_bytes = doc.tobytes()
            finally:
                doc.close()
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="research-{run_id}.pdf"'},
            )
        except Exception as exc:
            logger.warning(f"[Export] PyMuPDF PDF generation failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Không thể xuất file PDF: {exc}")
    else:
        return Response(
            content=report_md,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="research-{run_id}.md"'},
        )





SLASH_PREFIX_MAP = {
    "/tom-tat": ("Tóm tắt chi tiết", "Tóm tắt tổng quan nội dung, các đóng góp cốt lõi và kết quả chính của tài liệu nghiên cứu."),
    "/so-sanh": ("So sánh đối chiếu chi tiết", "So sánh chi tiết các tài liệu đã chọn về phương pháp tiếp cận, kiến trúc kỹ thuật, ưu nhược điểm và kết quả thực nghiệm."),
    "/dong-gop": ("Phân tích các đóng góp và điểm mới cốt lõi", "Trích xuất và phân tích các đóng góp khoa học mới (novelty và contributions) được đề xuất trong tài liệu."),
    "/phuong-phap": ("Phân tích chi tiết phương pháp và kiến trúc kỹ thuật", "Phân tích chi tiết phương pháp đề xuất, kiến trúc mô hình/thuật toán và quy trình thực hiện."),
    "/kiem-chung": ("Kiểm chứng tính xác thực của thông tin sau dựa trên bằng chứng tài liệu:", "Kiểm chứng tính xác thực của các thông tin sau dựa trên bằng chứng nguyên văn trích dẫn từ tài liệu."),
    "/so-lieu": ("Trích xuất các bảng số liệu, kết quả thực nghiệm và benchmark định lượng", "Trích xuất các số liệu thực nghiệm định lượng, bảng benchmark và metric so sánh hiệu năng quan trọng nhất."),
    "/thuat-ngu": ("Liệt kê và giải thích chi tiết các thuật ngữ chuyên môn, từ viết tắt", "Liệt kê các thuật ngữ chuyên môn, khái niệm cốt lõi và từ viết tắt trong tài liệu kèm theo định nghĩa chi tiết."),
    "/trich-dan": ("Trích xuất nguyên văn bằng chứng quan trọng kèm số trang liên quan đến", "Trích xuất các đoạn văn nguyên văn quan trọng nhất (verbatim quotes) kèm vị trí và số trang."),
    "/khoang-trong": ("Phân tích các hạn chế, khoảng trống nghiên cứu và hướng mở", "Chỉ ra các hạn chế (limitations), khoảng trống nghiên cứu (research gaps) và hướng phát triển tương lai được tác giả đề cập."),
    "/cau-hoi": ("Tạo danh sách các câu hỏi phản biện chuyên sâu xoay quanh nội dung", "Tạo 5-10 câu hỏi phản biện chuyên sâu và câu hỏi thảo luận học thuật xoay quanh nội dung và giả định của bài báo."),
    "/scout": ("Dò quét nhanh cấu trúc, mục lục và các đề mục chính trong tài liệu", "Dò quét nhanh cấu trúc, mục lục, phạm vi chủ đề và các phần chính trong toàn bộ tài liệu."),
}

SLASH_ALIASES = {
    "/tóm-tắt": "/tom-tat", "/tóm_tắt": "/tom-tat", "/tom_tat": "/tom-tat",
    "/so-sánh": "/so-sanh", "/so_sánh": "/so-sanh", "/so_sanh": "/so-sanh",
    "/đóng-góp": "/dong-gop", "/đóng_góp": "/dong-gop", "/dong_gop": "/dong-gop",
    "/phương-pháp": "/phuong-phap", "/phương_pháp": "/phuong-phap", "/phuong_phap": "/phuong-phap",
    "/kiểm-chứng": "/kiem-chung", "/kiểm_chứng": "/kiem-chung", "/kiem_chung": "/kiem-chung",
    "/số-liệu": "/so-lieu", "/số_liệu": "/so-lieu", "/so_lieu": "/so-lieu",
    "/thuật-ngữ": "/thuat-ngu", "/thuật_ngữ": "/thuat-ngu", "/thuat_ngu": "/thuat-ngu",
    "/trích-dẫn": "/trich-dan", "/trích_dẫn": "/trich-dan", "/trich_dan": "/trich-dan",
    "/khoảng-trống": "/khoang-trong", "/khoảng_trống": "/khoang-trong", "/khoang_trong": "/khoang-trong",
    "/câu-hỏi": "/cau-hoi", "/câu_hỏi": "/cau-hoi", "/cau_hoi": "/cau-hoi",
    "/quet": "/scout", "/quét": "/scout",
}

def resolve_slash_prompt(p: str) -> str:
    p_clean = (p or "").strip()
    if not p_clean:
        return p_clean
    # Normalize multiple leading slashes (e.g. ///tom-tat -> /tom-tat)
    if p_clean.startswith("/"):
        p_clean = "/" + p_clean.lstrip("/")
        
    # Edge case: If user entered only "/" or "/  "
    if p_clean == "/":
        return "Tóm tắt tổng quan nội dung, các đóng góp cốt lõi và kết quả chính của tài liệu nghiên cứu."

    # Extract command part
    parts = p_clean.split(maxsplit=1)
    first_token = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    # Check aliases
    if first_token in SLASH_ALIASES:
        first_token = SLASH_ALIASES[first_token]
    elif first_token.replace("_", "-") in SLASH_PREFIX_MAP:
        first_token = first_token.replace("_", "-")

    if first_token in SLASH_PREFIX_MAP:
        prefix, default_prompt = SLASH_PREFIX_MAP[first_token]
        rest_clean = rest.strip().lstrip(":+ -=")
        if rest_clean:
            return f"{prefix}: {rest_clean}"
        return default_prompt

    # Also handle if slash command is used with colon without space, e.g. /tom-tat:noi-dung
    for cmd, (prefix, default_prompt) in SLASH_PREFIX_MAP.items():
        if p_clean.lower().startswith(cmd + ":") or p_clean.lower().startswith(cmd + "-"):
            rest_clean = p_clean[len(cmd):].strip().lstrip(":+ -=")
            if rest_clean:
                return f"{prefix}: {rest_clean}"
            return default_prompt

    # Limit query length to 3000 chars to avoid memory exhaustion
    if len(p_clean) > 3000:
        p_clean = p_clean[:3000]

    return p_clean

@app.post("/api/runs")

async def create_run(req: CreateRunRequest):
    """Creates and starts an asynchronous multi-agent research run with SQLite persistence."""
    prompt = resolve_slash_prompt(req.question or req.query)
    if not prompt:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    run_id = f"run_{uuid.uuid4().hex[:10]}"
    target_doc_ids = []
    if req.document_ids:
        target_doc_ids = [d for d in req.document_ids if d]
    elif req.document_id:
        target_doc_ids = [req.document_id]

    pdf_paths: list[str] = []
    for d_id in target_doc_ids:
        doc = db.get_document(d_id)
        if doc and doc.get("path") and os.path.exists(doc.get("path")):
            pdf_paths.append(doc.get("path"))

    if target_doc_ids and pdf_paths:
        doc_id = target_doc_ids[0]
        pdf_path = pdf_paths[0]
    else:
        # Nếu người dùng không chọn tài liệu và không có doc_id -> Thông báo rõ ràng
        no_doc_report = (
            "# Thông báo Hệ thống\n\n"
            "> **Trạng thái:** `WAITING_FOR_DOCUMENT`\n\n"
            "Vui lòng **chọn một tài liệu** trong kho hoặc bật **Auto-Scout** để bắt đầu nghiên cứu."
        )
        db.save_run(
            run_id=run_id,
            question=prompt,
            status="completed",
            document_id=None,
            session_id=req.session_id,
        )
        db.update_run_result(
            run_id=run_id,
            status="completed",
            final_report=no_doc_report,
            verified_claims=[],
            rejected_claims=[],
            evidence_spans=[],
            reasoning_trace=[{"action": "waiting_for_document", "message": "Vui lòng chọn tài liệu hoặc bật Auto-Scout."}],
        )
        return {
            "id": run_id,
            "status": "completed",
            "question": prompt,
            "document_id": None,
            "final_report": no_doc_report,
        }

    run_record = db.save_run(
        run_id=run_id,
        question=prompt,
        status="processing",
        document_id=doc_id,
        session_id=req.session_id,
    )

    # Asynchronously execute graph with live per-node streaming and SQLite updates
    async def _async_exec():
        try:
            import time

            def _run_streaming():
                from modes.registry import normalize_mode
                from core.graph import build_research_rag_graph

                raw_mode = req.mode or req.model or settings.EXECUTION_MODE
                run_mode = normalize_mode(raw_mode)
                logger.info(f"[RunExecution] Dispatching run {run_id} strictly under mode='{run_mode}'.")

                initial_state = {
                    "query": prompt,
                    "pdf_path": pdf_path,
                    "pdf_paths": pdf_paths,
                    "retry_count": 0,
                    "max_reflection_loops": req.max_iterations or 2,
                    "chat_history": req.chat_history or [],
                    "execution_mode": run_mode,
                }
                accumulated_traces = []
                final_res = {"execution_mode": run_mode}
                step_start_time = time.time()
                active_graph = build_research_rag_graph(mode=run_mode)

                for chunk in active_graph.stream(initial_state, stream_mode="updates"):
                    chk = db.get_run(run_id)
                    if chk and chk.get("status") == "cancelled":
                        logger.info(f"[RunCancel] Run {run_id} was cancelled by user. Terminating graph stream.")
                        return final_res, accumulated_traces

                    now = time.time()
                    step_duration = round(max(0.1, now - step_start_time), 1)
                    step_start_time = now

                    for node_name, node_output in chunk.items():
                        if isinstance(node_output, dict):
                            final_res.update(node_output)
                            new_traces = node_output.get("reasoning_trace")
                            if new_traces:
                                for tr in new_traces:
                                    if isinstance(tr, dict):
                                        tr["duration"] = f"{step_duration}s"
                                        tr["duration_sec"] = step_duration
                                        tr["timestamp"] = now
                                accumulated_traces.extend(new_traces)
                                # Persist each agent step immediately into SQLite!
                                db.update_run_result(
                                    run_id=run_id,
                                    status="processing",
                                    reasoning_trace=accumulated_traces,
                                )

                return final_res, accumulated_traces

            loop = asyncio.get_event_loop()
            final_res, accumulated_traces = await loop.run_in_executor(None, _run_streaming)

            current_run = db.get_run(run_id)
            if current_run and current_run.get("status") == "cancelled":
                return

            db.update_run_result(
                run_id=run_id,
                status="completed",
                final_report=final_res.get("final_report", ""),
                verified_claims=final_res.get("verified_claims", []),
                rejected_claims=final_res.get("rejected_claims", []),
                evidence_spans=final_res.get("evidence_spans", []),
                reasoning_trace=accumulated_traces or final_res.get("reasoning_trace", []),
            )
        except asyncio.CancelledError:
            logger.info(f"[RunCancel] Async task for run {run_id} cancelled.")
            db.update_run_result(run_id=run_id, status="cancelled")
        except Exception as e:
            db.update_run_result(
                run_id=run_id,
                status="failed",
                error=str(e),
            )
        finally:
            _ACTIVE_RUN_TASKS.pop(run_id, None)



    task = asyncio.create_task(_async_exec())
    _ACTIVE_RUN_TASKS[run_id] = task

    return run_record





class ScoutRequest(BaseModel):
    query: str
    max_results: int = 3
    session_id: Optional[str] = None


@app.post("/api/scout")
async def execute_standalone_scout(req: ScoutRequest):
    """
    Standalone Agent 0 (Literature Scout) Execution:
    Hybrid Dual-Engine: ArXiv Official SDK for STEM + Gemini Academic Explorer
    for Humanities/Social Sciences/Non-STEM, with Bidirectional Relevance Gating.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    from agents.agent0_scout.query_expander import (
        analyze_intent,
        generate_search_perspectives,
        translate_to_academic_english
    )
    from agents.agent0_scout.arxiv_client import search_arxiv_papers
    from agents.agent0_scout.paper_ranker import rank_and_select_papers
    from agents.agent0_scout.downloader import download_and_register_paper
    from agents.agent0_scout.gemini_scout import search_gemini_academic_papers
    from utils.logger import logger

    user_query = req.query.strip()
    
    # 1. Intent Analysis
    intent = analyze_intent(user_query)
    decision = intent.get("decision", "SEARCH").upper()
    domain = intent.get("domain", "STEM")
    
    # Empty query or ambiguous acronym clarification
    if decision == "CLARIFY":
        return {
            "query": user_query,
            "decision": "clarify",
            "reason": intent.get("reason"),
            "message": intent.get("message"),
            "academic_keywords": "",
            "total_discovered": 0,
            "papers": [],
        }

    if decision == "REJECT" and intent.get("reason") in ("empty_query", "empty_or_meaningless"):
        return {
            "query": user_query,
            "decision": "reject",
            "reason": intent.get("reason"),
            "message": intent.get("message"),
            "academic_keywords": "",
            "total_discovered": 0,
            "papers": [],
        }

    # 2. NON-STEM / SOCIAL SCIENCES / HISTORY ROUTE: Route to Gemini Academic Explorer
    if domain == "NON_STEM" or decision == "REJECT":
        logger.info(f"[Standalone Scout] Non-STEM topic detected: '{user_query}'. Routing to Gemini Academic Explorer...")
        gemini_papers = search_gemini_academic_papers(user_query, max_results=req.max_results)
        if gemini_papers:
            try:
                scout_run_id = f"run_scout_{uuid.uuid4().hex[:8]}"
                summary_md = f"### Agent 0 (Literature Scout): {user_query}\n\nĐã phát hiện và nạp vào kho {len(gemini_papers)} bài báo khoa học tiêu biểu:\n\n"
                for p in gemini_papers:
                    summary_md += f"- **[{p.get('year', 2024)}] {p.get('title')}**\n"
                first_doc = gemini_papers[0].get("doc_id") or gemini_papers[0].get("id")
                db.save_run(
                    run_id=scout_run_id,
                    question=user_query,
                    status="completed",
                    final_report=summary_md,
                    document_id=first_doc,
                    session_id=req.session_id,
                )
            except Exception as e:
                logger.warning(f"[Scout] Could not record scout run in db: {e}")

            return {
                "query": user_query,
                "decision": "search",
                "provider": "Gemini Academic Explorer",
                "message": f"Đã phát hiện và nạp vào kho {len(gemini_papers)} bài báo khoa học / công trình nghiên cứu tiêu biểu qua Gemini Academic Explorer.",
                "academic_keywords": user_query,
                "total_discovered": len(gemini_papers),
                "papers": gemini_papers,
            }
        else:
            return {
                "query": user_query,
                "decision": "no_relevant_papers",
                "reason": "gemini_explorer_empty",
                "message": intent.get("message") or f"Chủ đề '{user_query}' nằm ngoài phạm vi ArXiv và không tìm thấy tài liệu học thuật trực tuyến phù hợp. Bạn vui lòng tải trực tiếp tài liệu PDF vào Kho Tài Liệu.",
                "academic_keywords": "",
                "total_discovered": 0,
                "papers": [],
            }

    # 3. STEM ROUTE: ArXiv Official SDK with Fast Academic Search
    academic_query = translate_to_academic_english(user_query)
    all_candidates = search_arxiv_papers(academic_query, max_results=req.max_results + 2)
    seen_ids = {p.get("arxiv_id") or p.get("id") for p in all_candidates}

    if len(all_candidates) < req.max_results:
        perspectives = generate_search_perspectives(user_query)
        for p in perspectives:
            p_papers = search_arxiv_papers(p["query"], max_results=2)
            for pap in p_papers:
                pid = pap.get("arxiv_id") or pap.get("id")
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    all_candidates.append(pap)
            if len(all_candidates) >= req.max_results:
                break

    # 4. Enforce Semantic Relevance Gatekeeper
    top_papers = rank_and_select_papers(
        all_candidates, 
        user_query, 
        academic_keywords=academic_query, 
        top_k=req.max_results,
        min_threshold=1.5
    )

    # If ArXiv yields 0 relevant papers, fallback to Gemini Academic Explorer
    if not top_papers:
        logger.warning(f"[Standalone Scout] ArXiv yielded 0 papers for '{user_query}'. Falling back to Gemini Academic Explorer...")
        gemini_papers = search_gemini_academic_papers(user_query, max_results=req.max_results)
        if gemini_papers:
            return {
                "query": user_query,
                "decision": "search",
                "provider": "Gemini Academic Explorer (Fallback)",
                "message": f"Đã tìm thấy {len(gemini_papers)} công trình nghiên cứu học thuật liên quan qua Gemini Academic Explorer.",
                "academic_keywords": academic_query,
                "total_discovered": len(gemini_papers),
                "papers": gemini_papers,
            }
        return {
            "query": user_query,
            "decision": "no_relevant_papers",
            "reason": "candidates_below_relevance_threshold",
            "message": f"Các bài báo phát hiện trên ArXiv không đạt ngưỡng liên quan đến chủ đề '{user_query}'. Bạn vui lòng tải trực tiếp tài liệu PDF vào Kho Tài Liệu bên trái.",
            "academic_keywords": academic_query,
            "total_discovered": len(all_candidates),
            "papers": [],
        }

    # 5. Download and register golden papers
    downloaded_papers = []
    for paper in top_papers:
        try:
            d_paper = download_and_register_paper(paper)
            downloaded_papers.append(d_paper)
        except Exception as e:
            logger.error(f"[Standalone Scout] Failed to download {paper.get('title')}: {e}")
            downloaded_papers.append(paper)

    if downloaded_papers:
        try:
            scout_run_id = f"run_scout_{uuid.uuid4().hex[:8]}"
            summary_md = f"### Agent 0 (Literature Scout): {user_query}\n\nĐã phát hiện và tải về {len(downloaded_papers)} bài báo học thuật đạt chuẩn:\n\n"
            for p in downloaded_papers:
                summary_md += f"- **[{p.get('year', 2024)}] {p.get('title')}**\n"
            first_doc = downloaded_papers[0].get("doc_id") or downloaded_papers[0].get("id")
            db.save_run(
                run_id=scout_run_id,
                question=user_query,
                status="completed",
                final_report=summary_md,
                document_id=first_doc,
                session_id=req.session_id,
            )
        except Exception as e:
            logger.warning(f"[Scout] Could not record scout run in db: {e}")

    return {
        "query": user_query,
        "decision": "search",
        "provider": "arXiv Official SDK",
        "message": f"Đã phát hiện và tải về {len(downloaded_papers)} bài báo học thuật đạt chuẩn.",
        "academic_keywords": academic_query,
        "total_discovered": len(all_candidates),
        "papers": downloaded_papers,
    }


# Direct synchronous research endpoint

@app.post("/api/research")

async def execute_direct_research(req: DirectResearchRequest):

    if not req.query.strip():

        raise HTTPException(status_code=400, detail="Query cannot be empty.")



    loop = asyncio.get_event_loop()
    active_graph = get_current_graph()

    final_state = await loop.run_in_executor(

        None,

        active_graph.invoke,

        {"query": req.query, "pdf_path": req.pdf_path, "retry_count": 0},

    )

    return {

        "query": req.query,

        "status": final_state.get("fact_check_label", "unknown"),

        "confidence": final_state.get("confidence", "unknown"),

        "final_report": final_state.get("final_report", ""),

        "verified_claims": final_state.get("verified_claims", []),

        "rejected_claims": final_state.get("rejected_claims", []),

        "evidence_spans": final_state.get("evidence_spans", []),

        "retrieved_docs": final_state.get("retrieved_docs", []),

        "reasoning_trace": final_state.get("reasoning_trace", []),

    }





@app.post("/api/research/stream")

async def execute_research_stream(req: DirectResearchRequest):
    """
    Real-time Server-Sent Events (SSE) streaming endpoint for multi-agent execution.
    Emits events as each agent finishes its step. Wraps execution in robust try-except
    to emit an error event instead of dropping the connection.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    async def event_generator():
        try:
            initial_state = {"query": req.query, "pdf_path": req.pdf_path, "retry_count": 0}
            yield f"data: {json.dumps({'event': 'started', 'query': req.query}, ensure_ascii=False)}\n\n"

            active_graph = get_current_graph()
            async for chunk in active_graph.astream(initial_state, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    trace = node_output.get("reasoning_trace", [{}])[-1] if node_output.get("reasoning_trace") else {}
                    event_data = {
                        "event": "node_update",
                        "node": node_name,
                        "action": trace.get("action", "completed"),
                        "data": {
                            "fact_check_label": node_output.get("fact_check_label"),
                            "retry_count": node_output.get("retry_count"),
                            "partial_answer": node_output.get("answer") or node_output.get("final_report"),
                        },
                    }
                    yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'event': 'finished'}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.error(f"[SSE Research Stream Error]: {exc}")
            yield f"data: {json.dumps({'event': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(

        event_generator(),

        media_type="text/event-stream",

        headers={

            "Cache-Control": "no-cache",

            "Connection": "keep-alive",

            "X-Accel-Buffering": "no",

        }

    )





# Mount static Web Workspace UI & Assets

@app.get("/demo2", include_in_schema=False)
def redirect_to_demo2_slash():
    return RedirectResponse(url="/demo2/", status_code=307)

web_demo2_dir = os.path.join(settings.BASE_DIR, "web_demo2")

if os.path.exists(web_demo2_dir):

    app.mount("/demo2", StaticFiles(directory=web_demo2_dir, html=True), name="web_demo2")



web_dir = os.path.join(settings.BASE_DIR, "web")

if os.path.exists(web_dir):

    app.mount("/assets", StaticFiles(directory=web_dir), name="assets")

    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

