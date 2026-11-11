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
    # Word-boundary truncation to avoid cutting off mid-word
    trunc_title = clean_title[:120].rsplit(" ", 1)[0] if len(clean_title) > 120 else clean_title
    query_encoded = urllib.parse.quote(trunc_title)
    url = f"{CROSSREF_API_BASE}?query.bibliographic={query_encoded}&rows=3&mailto=academic-rag@antigravity.ai"

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        import re
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            items = data.get("message", {}).get("items", [])
            t_words = set(re.findall(r"\w+", clean_title.lower()))
            for it in items:
                titles = it.get("title", [])
                item_title = titles[0] if titles else ""
                it_words = set(re.findall(r"\w+", item_title.lower()))
                # Require significant title token overlap (>= 40%) before trusting citation count
                if t_words and it_words:
                    overlap = len(t_words & it_words) / max(len(t_words | it_words), 1)
                    if overlap >= 0.4:
                        count = it.get("is-referenced-by-count", 0)
                        return int(count) if count else 0
    except Exception as e:
        logger.debug(f"[CrossRef Scout] Could not fetch citations for '{clean_title[:30]}...': {e}")
        
    return 0
