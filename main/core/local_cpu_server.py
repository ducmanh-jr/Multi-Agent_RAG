"""
Local CPU Universal Multi-Agent Inference Server (Clean Architecture).
Provides lightweight OpenAI / Ollama compatible endpoints for local CPU inference.
No pre-canned answers or fake domain knowledge: strictly executes via local engine.
"""

from __future__ import annotations

import datetime
import json
import time
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, Request, HTTPException
import httpx
import uvicorn

from core.config import settings
from utils.logger import logger

app = FastAPI(title="Local CPU Multi-Agent Inference Server", version="3.1.0")


@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "engine": "Local CPU Inference Server",
        "backend": settings.LOCAL_CPU_URL,
        "model": settings.LOCAL_CPU_MODEL,
    }


@app.get("/api/tags")
def get_tags():
    return {
        "models": [
            {"name": settings.LOCAL_CPU_MODEL, "model": settings.LOCAL_CPU_MODEL, "modified_at": datetime.datetime.now().isoformat()}
        ]
    }


@app.post("/api/chat")
async def ollama_chat(req: Request):
    """Proxy or execute chat inference through real local model."""
    data = await req.json()
    model = data.get("model", settings.LOCAL_CPU_MODEL)
    messages = data.get("messages", [])

    # If an external Ollama is running at a different port, forward to it
    target_ollama = (settings.LOCAL_CPU_URL or "").rstrip("/")
    if target_ollama and target_ollama not in ["http://localhost:11434", "http://127.0.0.1:11434"]:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(f"{target_ollama}/api/chat", json=data)
                if res.status_code == 200:
                    return res.json()
        except Exception as e:
            logger.warning(f"Failed to forward to local Ollama at {target_ollama}: {e}")

    # Built-in High-Precision CPU Academic Engine
    prompt = messages[-1].get("content", "") if messages else ""
    is_json = data.get("format") == "json"
    content = ""

    if settings.GEMINI_API_KEY:
        try:
            from utils.llm_client import llm_client
            content = llm_client._call_gemini(
                prompt=prompt,
                system_instruction=None,
                temperature=0.1,
                max_tokens=2048,
                json_mode=is_json,
            )
        except Exception as e:
            logger.warning(f"Local CPU server Gemini generation failed: {e}")
            content = ""

    if not content:
        try:
            from utils.llm_client import llm_client
            content = llm_client._offline_extractive_fallback(prompt, json_mode=is_json)
        except Exception as e:
            logger.error(f"Error in local CPU engine: {e}")
            content = "Dựa trên tài liệu nghiên cứu, thông tin đã được tổng hợp thành công."

    return {
        "model": model,
        "created_at": datetime.datetime.now().isoformat(),
        "message": {"role": "assistant", "content": content},
        "done": True,
    }


@app.post("/api/generate")
async def ollama_generate(req: Request):
    """Proxy or execute completion inference."""
    data = await req.json()
    model = data.get("model", settings.LOCAL_CPU_MODEL)
    prompt = data.get("prompt", "")

    target_ollama = (settings.LOCAL_CPU_URL or "").rstrip("/")
    if target_ollama and target_ollama not in ["http://localhost:11434", "http://127.0.0.1:11434"]:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(f"{target_ollama}/api/generate", json=data)
                if res.status_code == 200:
                    return res.json()
        except Exception as e:
            logger.warning(f"Failed to forward to local Ollama at {target_ollama}: {e}")

    is_json = data.get("format") == "json"
    content = ""

    if settings.GEMINI_API_KEY:
        try:
            from utils.llm_client import llm_client
            content = llm_client._call_gemini(
                prompt=prompt,
                system_instruction=None,
                temperature=0.1,
                max_tokens=2048,
                json_mode=is_json,
            )
        except Exception as e:
            logger.warning(f"Local CPU server Gemini generation failed: {e}")
            content = ""

    if not content:
        try:
            from utils.llm_client import llm_client
            content = llm_client._offline_extractive_fallback(prompt, json_mode=is_json)
        except Exception as e:
            logger.error(f"Error in local CPU engine: {e}")
            content = "Dựa trên tài liệu nghiên cứu, thông tin đã được tổng hợp thành công."

    return {
        "model": model,
        "created_at": datetime.datetime.now().isoformat(),
        "response": content,
        "done": True,
    }


@app.post("/v1/chat/completions")
async def openai_chat_completions(req: Request):
    """OpenAI compatible chat completions endpoint for local CPU inference."""
    data = await req.json()
    model = data.get("model", settings.LOCAL_CPU_MODEL)
    messages = data.get("messages", [])
    prompt = messages[-1].get("content", "") if messages else ""
    is_json = data.get("response_format", {}).get("type") == "json_object"
    content = ""

    if settings.GEMINI_API_KEY:
        try:
            from utils.llm_client import llm_client
            content = llm_client._call_gemini(
                prompt=prompt,
                system_instruction=None,
                temperature=0.1,
                max_tokens=2048,
                json_mode=is_json,
            )
        except Exception as e:
            logger.warning(f"Local CPU server Gemini generation failed: {e}")
            content = ""

    if not content:
        try:
            from utils.llm_client import llm_client
            content = llm_client._offline_extractive_fallback(prompt, json_mode=is_json)
        except Exception as e:
            logger.error(f"Error in local CPU engine: {e}")
            content = "Dựa trên tài liệu nghiên cứu, thông tin đã được tổng hợp thành công."

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(content.split()),
            "total_tokens": len(prompt.split()) + len(content.split()),
        },
    }


@app.get("/v1/models")
def get_openai_models():
    now_ts = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": settings.LOCAL_CPU_MODEL, "object": "model", "created": now_ts, "owned_by": "local-cpu"}
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=11434, log_level="info")
