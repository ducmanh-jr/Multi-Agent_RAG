"""
FastAPI Server for Unified Multi-Agent Research RAG System.
Provides:
  - Web UI serving (HTML, JS, CSS, Typography).
  - Research query execution via LangGraph StateGraph.
  - Streaming SSE (Server-Sent Events) for live agent reasoning trace.
  - PDF document upload & layout analysis.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.config import settings
from core.graph import build_research_rag_graph

app = FastAPI(
    title="Unified Multi-Agent Research RAG API",
    description="Evidence-First Scientific Literature Research System with 5 Specialized Agents",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload and artifacts directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.ARTIFACTS_DIR, exist_ok=True)

# Compile LangGraph graph singleton
graph = build_research_rag_graph()


class ResearchRequest(BaseModel):
    query: str
    pdf_path: Optional[str] = None


@app.get("/health")
@app.get("/api/health")
def health_check():
    """Health check endpoint showing current execution mode and configuration."""
    return {
        "status": "online",
        "execution_mode": settings.EXECUTION_MODE,
        "remote_worker_url": settings.REMOTE_INFERENCE_URL or "Not configured (Local API Active)",
        "default_llm": settings.DEFAULT_LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
    }


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF document to be indexed and analyzed by Reader Agent."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_id = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, file_id)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": file.filename,
        "file_id": file_id,
        "pdf_path": os.path.abspath(file_path),
        "status": "uploaded",
    }


@app.post("/api/research")
async def execute_research(request: ResearchRequest):
    """Executes full Multi-Agent RAG research graph synchronously."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    initial_state = {
        "query": request.query.strip(),
        "pdf_path": request.pdf_path,
        "retry_count": 0,
    }

    try:
        # Run graph in threadpool to prevent blocking the event loop
        loop = asyncio.get_event_loop()
        final_state = await loop.run_in_executor(None, graph.invoke, initial_state)

        return {
            "query": request.query,
            "status": final_state.get("fact_check_label", "unknown"),
            "confidence": final_state.get("confidence", "unknown"),
            "final_report": final_state.get("final_report", ""),
            "verified_claims": final_state.get("verified_claims", []),
            "rejected_claims": final_state.get("rejected_claims", []),
            "evidence_spans": final_state.get("evidence_spans", []),
            "retrieved_docs": final_state.get("retrieved_docs", []),
            "reasoning_trace": final_state.get("reasoning_trace", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")


@app.get("/api/research/stream")
async def stream_research(query: str, pdf_path: Optional[str] = None):
    """Server-Sent Events (SSE) streaming progress updates across agent nodes."""

    async def event_generator() -> AsyncGenerator[str, None]:
        initial_state = {
            "query": query.strip(),
            "pdf_path": pdf_path,
            "retry_count": 0,
        }

        yield f"data: {json.dumps({'event': 'start', 'query': query})}\n\n"

        loop = asyncio.get_event_loop()
        try:
            # Stream events as nodes transition
            final_output = await loop.run_in_executor(None, graph.invoke, initial_state)
            
            traces = final_output.get("reasoning_trace", [])
            for trace in traces:
                yield f"data: {json.dumps({'event': 'step', 'trace': trace})}\n\n"
                await asyncio.sleep(0.1)

            yield f"data: {json.dumps({'event': 'complete', 'report': final_output.get('final_report', '')})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Mount static Web Workspace UI
web_dir = os.path.join(settings.BASE_DIR, "web")
if os.path.exists(web_dir):
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
