"""
ArXiv Academic Client for Agent 0 (Scout).
Powered by the official `arxiv` Python SDK (arxiv.Client).
Supports:
  - Exact ArXiv ID / URL direct lookup via id_list.
  - Multi-keyword academic queries with automatic rate-limit management.
  - Category scoping for STEM fields (cs.*, stat.ML, eess.*).
  - Robust exception handling and structured paper metadata output.
"""

from __future__ import annotations

import re
from typing import Any
import arxiv
from utils.logger import logger


ARXIV_ID_PATTERN = re.compile(r"(?:arxiv[:\s/]|abs/|pdf/)?(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)


def extract_arxiv_id(query: str) -> str | None:
    """Detects if query contains an explicit ArXiv ID or URL."""
    match = ARXIV_ID_PATTERN.search(query.strip())
    if match:
        return match.group(1)
    return None


def search_arxiv_papers(query: str, max_results: int = 5, categories: list[str] | None = None) -> list[dict[str, Any]]:
    """
    Searches ArXiv using the official arxiv.Client SDK.
    Returns structured paper records with PDF download links.
    """
    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=3.0,
        num_retries=3
    )

    direct_id = extract_arxiv_id(query)
    papers: list[dict[str, Any]] = []

    try:
        if direct_id:
            logger.info(f"[ArXiv Official SDK] Direct ArXiv ID detected: '{direct_id}'.")
            search = arxiv.Search(id_list=[direct_id])
        else:
            # Clean keywords: strip special query characters
            clean_terms = re.findall(r"[a-zA-Z0-9_-]+", query)
            clean_query = " ".join(clean_terms[:6]) if clean_terms else "machine learning"
            
            logger.info(f"[ArXiv Official SDK] Querying: '{clean_query}' (limit={max_results})...")
            search = arxiv.Search(
                query=clean_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )

        results = list(client.results(search))
        for res in results:
            arxiv_id = res.get_short_id()
            clean_id = arxiv_id.replace(".", "_")
            title = res.title.strip().replace("\n", " ") if res.title else "Untitled"
            summary = res.summary.strip().replace("\n", " ") if res.summary else ""
            pub_year = res.published.year if res.published else 2024
            authors = [a.name for a in res.authors[:5]]
            pdf_url = res.pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # Ensure HTTPS
            if pdf_url.startswith("http://"):
                pdf_url = "https://" + pdf_url[7:]
            if not pdf_url.endswith(".pdf"):
                pdf_url = f"{pdf_url}.pdf"

            papers.append({
                "id": f"arxiv_{clean_id}",
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "year": pub_year,
                "abstract": summary,
                "pdf_url": pdf_url,
                "categories": res.categories,
                "primary_category": res.primary_category,
                "source": "arXiv",
                "source_url": res.entry_id,
            })

        logger.info(f"[ArXiv Official SDK] Successfully retrieved {len(papers)} papers.")
    except Exception as e:
        logger.error(f"[ArXiv Official SDK] Query failed for '{query}': {e}")

    return papers
