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
    target_ollama = settings.LOCAL_CPU_URL.rstrip("/")
    if target_ollama and target_ollama != "http://localhost:11434" and target_ollama != "http://127.0.0.1:11434":
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(f"{target_ollama}/api/chat", json=data)
                if res.status_code == 200:
                    return res.json()
        except Exception as e:
            logger.warning(f"Failed to forward to local Ollama at {target_ollama}: {e}")

    raise HTTPException(
        status_code=503,
        detail=f"Local CPU model '{model}' is not loaded. Please start Ollama with 'ollama serve' and 'ollama run {model}'.",
    )


@app.post("/api/generate")
async def ollama_generate(req: Request):
    """Proxy or execute completion inference."""
    data = await req.json()
    model = data.get("model", settings.LOCAL_CPU_MODEL)

    target_ollama = settings.LOCAL_CPU_URL.rstrip("/")
    if target_ollama and target_ollama != "http://localhost:11434" and target_ollama != "http://127.0.0.1:11434":
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(f"{target_ollama}/api/generate", json=data)
                if res.status_code == 200:
                    return res.json()
        except Exception as e:
            logger.warning(f"Failed to forward to local Ollama at {target_ollama}: {e}")

    raise HTTPException(
        status_code=503,
        detail=f"Local CPU model '{model}' is offline. Please launch Ollama locally.",
    )


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
