"""
CrossRef REST API Client for Agent 0 (Scout).
Retrieves citation statistics and DOI metadata to enrich paper evaluation.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any
from utils.logger import logger

CROSSREF_API_BASE = "https://api.crossref.org/works"
USER_AGENT = "ResearchDesk-Agent0/1.0 (mailto:academic-rag@antigravity.ai)"


def get_citation_count_for_title(title: str, timeout: int = 8) -> int:
    """
    Looks up CrossRef citation count for a paper title.
    Returns the number of times the work is referenced, or 0 if not found.
    """
    clean_title = title.replace("\n", " ").strip()
    query_encoded = urllib.parse.quote(clean_title[:100])
    url = f"{CROSSREF_API_BASE}?query.bibliographic={query_encoded}&rows=1&mailto=academic-rag@antigravity.ai"
    
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            items = data.get("message", {}).get("items", [])
            if items:
                count = items[0].get("is-referenced-by-count", 0)
                return int(count) if count else 0
    except Exception as e:
        logger.debug(f"[CrossRef Scout] Could not fetch citations for '{clean_title[:30]}...': {e}")
        
    return 0
