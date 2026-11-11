"""
Autonomous PDF Downloader & SQLite Ingestion for Agent 0 (Scout).
Downloads academic papers directly from arXiv into uploads/ and registers them in workspace.db.
Validates PDF magic bytes (%PDF-) to prevent corrupted HTML rate-limit downloads.
"""

from __future__ import annotations

import os
import re
import httpx
from typing import Any
from core.config import settings
from core.db import save_document
from utils.logger import logger


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def sanitize_filename(name: str) -> str:
    """Creates a clean, filesystem-safe filename."""
    clean = re.sub(r'[\\/*?:"<>|]', "", name)
    clean = re.sub(r"\s+", "_", clean).strip("_")
    return clean[:60]


def download_and_register_paper(paper: dict[str, Any], timeout: int = 25) -> dict[str, Any]:
    """
    Downloads the paper PDF from its direct link and registers it in workspace.db.
    Validates %PDF- header magic bytes.
    Returns the updated paper dictionary with local_pdf_path.
    """
    pdf_url = paper.get("pdf_url")
    if not pdf_url:
        logger.error(f"[Scout Downloader] No PDF URL available for paper: {paper.get('title')}")
        return paper

    title = paper.get("title", "academic_paper")
    arxiv_id = paper.get("arxiv_id", "unknown").replace("/", "_")
    safe_title = sanitize_filename(title)
    
    filename = f"{arxiv_id}_{safe_title}.pdf"
    target_dir = os.path.join(settings.BASE_DIR, "uploads")
    os.makedirs(target_dir, exist_ok=True)
    local_path = os.path.join(target_dir, filename)

    # Check if already downloaded locally and valid
    is_cached = False
    if os.path.exists(local_path) and os.path.getsize(local_path) > 10000:
        try:
            with open(local_path, "rb") as f:
                header = f.read(1024)
                if b"%PDF-" in header:
                    is_cached = True
        except Exception:
            is_cached = False

    if is_cached:
        logger.info(f"[Scout Downloader] Valid paper already cached locally: {local_path} ({os.path.getsize(local_path)} bytes)")
    else:
        logger.info(f"[Scout Downloader] Streaming PDF from {pdf_url} to {local_path}...")
        total_bytes = 0
        header_checked = False
        tmp_path = f"{local_path}.part_{os.getpid()}"
        try:
            with httpx.Client(follow_redirects=True, timeout=float(timeout), headers={"User-Agent": USER_AGENT}) as client:
                with client.stream("GET", pdf_url) as resp:
                    resp.raise_for_status()
                    with open(tmp_path, "wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                            if not header_checked and chunk:
                                if b"%PDF-" not in chunk[:1024]:
                                    raise ValueError("Downloaded file does not contain valid PDF magic bytes (%PDF-). Server may have returned an HTML captcha/error.")
                                header_checked = True
                            f.write(chunk)
                            total_bytes += len(chunk)

            if total_bytes < 5000:
                raise ValueError(f"Downloaded file too small ({total_bytes} bytes) - not a valid academic paper.")
            os.replace(tmp_path, local_path)
            logger.info(f"[Scout Downloader] Successfully streamed {total_bytes} bytes to {local_path}.")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    file_size = os.path.getsize(local_path)
    doc_id = paper.get("id") or f"doc_{arxiv_id}"

    # Register into SQLite workspace.db
    source_url = paper.get("source_url") or f"https://arxiv.org/abs/{arxiv_id}"
    internal_pdf_url = f"/api/documents/{doc_id}/file"
    metadata = {
        "title": title,
        "authors": paper.get("authors", []),
        "year": paper.get("year", 2024),
        "source": paper.get("source", "arXiv"),
        "source_url": source_url,
        "abstract": paper.get("abstract", "")[:500],
        "pdf_url": internal_pdf_url,
        "has_pdf": True,
        "auto_scouted": True,
        "doc_type": "pdf"
    }
    
    save_document(
        doc_id=doc_id,
        name=f"[{paper.get('year', 2024)}] {title}.pdf",
        path=local_path,
        size=file_size,
        doc_type="pdf",
        metadata=metadata
    )
    
    logger.info(f"[Scout Downloader] Document registered in SQLite DB: id='{doc_id}', path='{local_path}'")
    
    result = dict(paper)
    result["local_pdf_path"] = local_path
    result["doc_id"] = doc_id
    result["file_size"] = file_size
    result["has_pdf"] = True
    result["pdf_url"] = internal_pdf_url
    result["source_url"] = source_url
    result["doc_type"] = "pdf"
    return result
