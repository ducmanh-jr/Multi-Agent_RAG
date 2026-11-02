"""
Autonomous PDF Downloader & SQLite Ingestion for Agent 0 (Scout).
Downloads academic papers directly from arXiv into uploads/ and registers them in workspace.db.
Validates PDF magic bytes (%PDF-) to prevent corrupted HTML rate-limit downloads.
"""

from __future__ import annotations

import os
import re
import urllib.request
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
        logger.info(f"[Scout Downloader] Downloading PDF from {pdf_url} to {local_path}...")
        req = urllib.request.Request(pdf_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            
            # Guard against HTML captcha / rate limit blocks
            if len(data) < 5000:
                raise ValueError(f"Downloaded file too small ({len(data)} bytes) - not a valid academic paper.")
            if b"%PDF-" not in data[:1024]:
                raise ValueError("Downloaded file does not contain valid PDF magic bytes (%PDF-). Server may have returned an HTML error/captcha.")
                
            with open(local_path, "wb") as f:
                f.write(data)
        logger.info(f"[Scout Downloader] Successfully downloaded {len(data)} bytes.")

    file_size = os.path.getsize(local_path)
    doc_id = paper.get("id") or f"doc_{arxiv_id}"

    # Register into SQLite workspace.db
    metadata = {
        "title": title,
        "authors": paper.get("authors", []),
        "year": paper.get("year", 2024),
        "source": paper.get("source", "arXiv"),
        "abstract": paper.get("abstract", "")[:500],
        "pdf_url": pdf_url,
        "auto_scouted": True,
    }
    
    save_document(
        doc_id=doc_id,
        name=f"[{paper.get('year', 2024)}] {title}",
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
    return result
