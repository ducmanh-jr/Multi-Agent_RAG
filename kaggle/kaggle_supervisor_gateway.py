"""
Kaggle Master Process Supervisor Gateway (Port 8000).
Zero-VRAM Orchestrator managing dynamic Hot-Swapping between:
  1. Worker A (gpu_finetune / 7 SOTA models) on internal port 8001.
  2. Worker B (gpu_max / Qwen 32B AWQ TP=2) on internal port 8002.
  3. Hybrid Orchestration Policy.

Fully integrated with WorkerManager for:
- Process group isolation (start_new_session=True)
- Guaranteed process termination and parent reaping
- Dual-GPU OS-level VRAM probe and baseline reclamation verification
- Request lease guarding (acquire_worker_lease)
- Truthful health checks, readiness polling, and status contracts
- Bearer token authentication guard for admin endpoints
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
from collections import defaultdict
from typing import Any, Optional
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

try:
    from kaggle.worker_manager import WorkerManager, WorkerSwitchingError
except ImportError:
    from worker_manager import WorkerManager, WorkerSwitchingError


PROJECT_DIR = os.environ.get("PROJECT_DIR", "/kaggle/working/Multi-Agent_RAG")
if not os.path.exists(PROJECT_DIR):
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

worker_manager = WorkerManager(project_dir=PROJECT_DIR)
_http_client = httpx.AsyncClient(timeout=180.0)

# Authentication and Security Configurations
ADMIN_TOKEN = os.environ.get("SUPERVISOR_ADMIN_TOKEN") or os.environ.get("ADMIN_API_KEY")
INFERENCE_API_KEY = os.environ.get("INFERENCE_API_KEY") or os.environ.get("CLIENT_API_KEY")

# Rate Limiting Configuration (Per-IP sliding window, default: 60 req/min)
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW = float(os.environ.get("RATE_LIMIT_WINDOW", "60.0"))
_rate_limit_lock = asyncio.Lock()
_client_request_history: dict[str, list[float]] = defaultdict(list)


def get_client_ip(request: Request) -> str:
    """
    Extracts the true client IP, prioritizing Cloudflare CF-Connecting-IP,
    then X-Forwarded-For, falling back to client.host.
    """
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip and cf_ip.strip():
        return cf_ip.strip()
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for and x_forwarded_for.strip():
        return x_forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


async def check_inference_rate_limit(request: Request):
    """
    In-memory sliding window rate limiter per client IP.
    Protects GPU against inference spam and denial of service.
    """
    if RATE_LIMIT_REQUESTS <= 0:
        return
    client_ip = get_client_ip(request)
    now = time.time()
    async with _rate_limit_lock:
        timestamps = _client_request_history[client_ip]
        cutoff = now - RATE_LIMIT_WINDOW
        active_timestamps = [t for t in timestamps if t > cutoff]
        if len(active_timestamps) >= RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail=f"Too Many Requests: Vượt quá giới hạn tần suất ({RATE_LIMIT_REQUESTS} requests/{int(RATE_LIMIT_WINDOW)}s). Vui lòng thử lại sau."
            )
        active_timestamps.append(now)
        _client_request_history[client_ip] = active_timestamps


def verify_admin_auth(request: Request):
    """
    Guards admin endpoints with Bearer Token auth.
    Fail-closed: strictly requires SUPERVISOR_ADMIN_TOKEN to be configured and matched.
    """
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="Lỗi cấu hình Gateway: SUPERVISOR_ADMIN_TOKEN chưa được thiết lập. Từ chối quyền truy cập quản trị (Fail-Closed)."
        )
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Thiếu hoặc sai định dạng header Authorization. Yêu cầu 'Bearer <token>'."
        )
    token = auth_header.split(" ", 1)[1].strip()
    if token != ADMIN_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Admin token không hợp lệ."
        )


def verify_inference_auth(request: Request):
    """
    Separates Client Inference Auth from Admin Auth.
    When INFERENCE_API_KEY is configured, clients must provide X-API-Key or Bearer token.
    If INFERENCE_API_KEY is empty/unset, allows open inference (e.g. for local testbench).
    Admin token is also accepted for authorized operations.
    """
    if not INFERENCE_API_KEY:
        return

    # Check X-API-Key header
    api_key = request.headers.get("x-api-key", "").strip()
    if api_key and (api_key == INFERENCE_API_KEY or (ADMIN_TOKEN and api_key == ADMIN_TOKEN)):
        return

    # Check Bearer authorization
    auth_header = request.headers.get("authorization", "").strip()
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        if token and (token == INFERENCE_API_KEY or (ADMIN_TOKEN and token == ADMIN_TOKEN)):
            return

    # Check query parameter ?api_key=...
    q_key = request.query_params.get("api_key", "").strip()
    if q_key and (q_key == INFERENCE_API_KEY or (ADMIN_TOKEN and q_key == ADMIN_TOKEN)):
        return

    raise HTTPException(
        status_code=401,
        detail="Unauthorized: Endpoint suy luận yêu cầu xác thực qua 'X-API-Key' hoặc 'Bearer <token>'."
    )


app = FastAPI(title="Kaggle Master Process Supervisor Gateway", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    init_mode = os.environ.get("INITIAL_MODE", "gpu_finetune").lower().strip()
    auto_spawn = os.environ.get("AUTO_SPAWN_WORKER", "true").lower() in ("true", "1")
    print("=" * 70)
    print(f"🚀 [Kaggle Supervisor Gateway v2.2] Port 8000 (Initial Mode: {init_mode.upper()})...")
    print("=" * 70)

    # In test environments or when auto_spawn is disabled, do not launch background GPU process
    if auto_spawn and "pytest" not in sys.modules:
        asyncio.create_task(worker_manager.request_switch_mode(init_mode))


@app.on_event("shutdown")
async def shutdown_event():
    print("[Supervisor] Gateway đang tắt, dọn dẹp GPU worker hiện tại...")
    try:
        await asyncio.wait_for(worker_manager._terminate_current_worker(), timeout=35.0)
    except Exception as e:
        print(f"[Supervisor] Lỗi khi dọn dẹp worker lúc tắt: {e}")
    try:
        await _http_client.aclose()
    except Exception:
        pass


class SwitchModeRequest(BaseModel):
    target_mode: str  # "gpu_finetune" | "gpu_max" | "hybrid"
    wait: bool = False  # If True: await completion (200 OK); If False: 202 Accepted


@app.get("/")
@app.get("/health")
async def health():
    """
    Truthful Health & Status Contract Endpoint.
    Directly reflects WorkerManager status contract, VRAM probe, and active capabilities.
    """
    payload = worker_manager.get_status_payload()

    # Maintain backward compatibility fields for clients (run_kaggle.py, main/api/app.py)
    is_ready = payload["transition_state"] == "ready" and payload["active_worker"] is not None
    payload["status"] = "online" if payload["runtime_status"] == "running" else payload["runtime_status"]
    payload["supervisor"] = "Kaggle Master Supervisor Gateway v2.2.0"
    payload["active_mode"] = payload["selected_mode"]
    payload["switch_message"] = payload["transition_message"]
    payload["worker_health"] = {
        "status": "online" if is_ready else "starting_or_offline",
        "worker_status": "online" if is_ready else "starting_or_offline",
        "active_worker": payload["active_worker"],
        "capabilities": payload["active_worker_capabilities"],
    }
    return payload


@app.get("/api/admin/status", dependencies=[Depends(verify_admin_auth)])
async def get_admin_status():
    """Full operational status contract as defined in architecture specification."""
    return worker_manager.get_status_payload()


@app.post("/api/admin/switch_mode", dependencies=[Depends(verify_admin_auth)])
async def switch_mode(req: SwitchModeRequest):
    """
    Admin Endpoint for Dynamic Mode Switching:
    - Supports 'gpu_finetune', 'gpu_max', 'hybrid' (and 'gpu_ft' normalized to 'gpu_finetune').
    - Returns 202 Accepted for async mode switches (dispatched in background).
    - If wait=True: awaits transition completion before responding.
    """
    target = req.target_mode.lower().strip()
    if target == "gpu_ft":
        target = "gpu_finetune"

    if target not in ["gpu_finetune", "gpu_max", "hybrid"]:
        raise HTTPException(
            status_code=400,
            detail="Chế độ không hợp lệ. Chỉ hỗ trợ 'gpu_finetune', 'gpu_max', hoặc 'hybrid'."
        )

    # Check for in-flight transitions
    if worker_manager.lock.locked() or worker_manager.transition_state in (
        "draining", "stopping", "reclaiming_vram", "spawning"
    ):
        raise HTTPException(
            status_code=429,
            detail="Hệ thống đang trong quá trình tráo đổi VRAM, xin vui lòng chờ."
        )

    # If wait is True: wait for transition synchronously
    if req.wait:
        res = await worker_manager.request_switch_mode(target)
        status_val = res.get("status")
        if status_val in ("success", "already_active", "accepted"):
            return res
        elif status_val == "aborted":
            return JSONResponse(status_code=409, content=res)
        else:
            return JSONResponse(status_code=500, content=res)

    # Async default: dispatch in background task and return 202 Accepted immediately
    asyncio.create_task(worker_manager.request_switch_mode(target))
    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "target_mode": target,
            "transition_state": "switching",
            "message": f"Yêu cầu chuyển sang chế độ {target.upper()} đã được tiếp nhận và đang thực hiện dưới background."
        }
    )


@app.get("/api/admin/logs", dependencies=[Depends(verify_admin_auth)])
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


# ==============================================================================
# REVERSE PROXY ROUTER (Guarded by Worker Lease & vLLM Adapter)
# ==============================================================================

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_all(request: Request, path: str):
    # 0. Client Inference Authentication & Rate Limiting
    verify_inference_auth(request)
    await check_inference_rate_limit(request)

    # 1. State check: fail closed if actively transitioning
    if worker_manager.transition_state in ("draining", "stopping", "reclaiming_vram", "spawning"):
        return JSONResponse(
            status_code=503,
            content={
                "error": f"GPU đang trong quá trình chuyển chế độ [{worker_manager.transition_state}]. Vui lòng thử lại sau.",
                "transition_state": worker_manager.transition_state,
                "transition_id": worker_manager.transition_id,
            }
        )

    active_url = worker_manager.get_active_worker_url()
    if not active_url or worker_manager.active_worker is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Chưa có GPU Worker nào đang hoạt động hoặc sẵn sàng phục vụ.",
                "transition_state": worker_manager.transition_state,
            }
        )

    # 2. Acquire worker lease to protect active request against mid-flight worker termination
    try:
        async with worker_manager.acquire_worker_lease():
            method = request.method
            body = await request.body()
            headers = dict(request.headers)
            headers.pop("host", None)

            # Adapt generation/summarization/routing payloads for vLLM when active worker is gpu_max
            if worker_manager.active_worker == "gpu_max":
                if path in (
                    "generate", "api/generate", "api/router", "router",
                    "api/summarize/generator", "summarize/generator",
                    "api/summarize/critic", "summarize/critic",
                    "api/writer", "writer"
                ):
                    try:
                        import json as py_json
                        import re
                        body_json = await request.json() if body else {}

                        # Tailored prompts preserving 100% of input data:
                        if path in ("api/router", "router"):
                            q = body_json.get("query", "") or body_json.get("prompt", "")
                            sys_inst = "You are an intelligent routing agent for an Academic RAG system. Output ONLY valid JSON."
                            prompt = (
                                "Analyze the user query and output a JSON object with: intent, sub_queries, alpha (0.0 to 1.0), top_k (1 to 15), and reasoning.\n\n"
                                f"Query: {q}\nOutput JSON:"
                            )
                        elif path in ("api/summarize/generator", "summarize/generator"):
                            q = body_json.get("query", "")
                            ctx = body_json.get("context", "")
                            fb = body_json.get("feedback", "")
                            sys_inst = "You are a rigorous scientific synthesis agent. Output valid JSON with 'answer', 'claims', and 'evidence_spans'."
                            prompt = (
                                f"Question: {q}\n\nContext:\n{ctx}\n\n"
                            )
                            if fb:
                                prompt += f"[Critic Feedback]: {fb}\n\n"
                            prompt += "Output format JSON with 'answer', 'claims' (list of atomic statements), and 'evidence_spans'."
                        elif path in ("api/summarize/critic", "summarize/critic"):
                            q = body_json.get("query", "")
                            ans = body_json.get("answer", "")
                            ctx = body_json.get("context", "")
                            claims = body_json.get("claims", [])
                            sys_inst = "You are an adversarial scientific peer reviewer. Output valid JSON with 'action' ('accept' or 'rewrite'), 'critique', and 'issues_found'."
                            prompt = (
                                f"Query: {q}\n"
                                f"Generated Answer: {ans}\n"
                                f"Source Context: {ctx}\n"
                                f"Extracted Claims: {py_json.dumps(claims, ensure_ascii=False)}\n\n"
                                "Evaluate whether each claim is fully supported by the source context. "
                                "Output JSON with 'action' ('accept' or 'rewrite'), 'critique', and 'issues_found'."
                            )
                        elif path in ("api/writer", "writer"):
                            q = body_json.get("query", "")
                            ans = body_json.get("answer", "")
                            claims = body_json.get("claims", [])
                            v_text = "\n".join([f"- {c.get('claim_text', str(c)) if isinstance(c, dict) else str(c)}" for c in claims])
                            sys_inst = "You are an elite scientific academic paper author. Generate a comprehensive, publication-grade academic report formatted in Markdown."
                            prompt = (
                                f"Research Question: {q}\n\n"
                                f"Executive Summary:\n{ans}\n\n"
                                f"Verified Scientific Evidence:\n{v_text}\n\n"
                                "Write a structured, formal academic paper formatted in Markdown."
                            )
                        else:
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
                        resp = await _http_client.post(
                            f"{active_url}/v1/chat/completions",
                            json=vllm_payload,
                            timeout=120.0
                        )
                        data = resp.json()
                        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                        # For JSON endpoints, parse text into structured dict if valid
                        if path in ("api/router", "router", "api/summarize/generator", "summarize/generator", "api/summarize/critic", "summarize/critic"):
                            try:
                                m = re.search(r"\{.*\}", text, re.DOTALL)
                                if m:
                                    parsed = py_json.loads(m.group(0))
                                    if isinstance(parsed, dict):
                                        parsed.setdefault("model", "Qwen2.5-32B-Instruct-AWQ")
                                        parsed.setdefault("text", text)
                                        parsed.setdefault("output", text)
                                    return JSONResponse(content=parsed, status_code=resp.status_code)
                            except Exception:
                                pass

                        if path in ("api/writer", "writer"):
                            return JSONResponse(
                                content={"report": text, "metadata": {"model": "Qwen2.5-32B-Instruct-AWQ"}},
                                status_code=resp.status_code
                            )

                        return JSONResponse(
                            content={"text": text, "output": text, "response": text, "model": "Qwen2.5-32B-Instruct-AWQ"},
                            status_code=resp.status_code
                        )
                    except Exception as e:
                        return JSONResponse(status_code=500, content={"error": f"vLLM Proxy error: {e}"})

            # Forward request to active worker
            target_url = f"{active_url}/{path}"
            resp = await _http_client.request(
                method=method,
                url=target_url,
                content=body,
                headers=headers,
                params=request.query_params,
            )
            response_headers = {
                k: v for k, v in resp.headers.items()
                if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")
            }
            return Response(content=resp.content, status_code=resp.status_code, headers=response_headers)

    except WorkerSwitchingError as e:
        return JSONResponse(
            status_code=503,
            content={
                "error": str(e),
                "transition_state": worker_manager.transition_state,
                "transition_id": worker_manager.transition_id,
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "error": f"GPU Worker tạm thời không phản hồi ({e}). Vui lòng thử lại sau.",
                "transition_state": worker_manager.transition_state,
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
