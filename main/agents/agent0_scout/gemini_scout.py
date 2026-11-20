"""
Backward-compatibility alias module for legacy imports.

NOTE: This module does NOT call Gemini API. All scholarly discovery is performed
via OpenAlex Global Scholarly Index in `agents.agent0_scout.openalex_scout`.
New callers should import directly from `agents.agent0_scout.openalex_scout`.
"""

from __future__ import annotations

from agents.agent0_scout.openalex_scout import (
    COMMON_CONCEPT_TRANSLATIONS,
    USER_AGENT,
    VIETNAMESE_STOP_PATTERNS,
    clean_and_expand_query,
    download_open_access_pdf,
    reconstruct_abstract,
    search_openalex_academic_papers,
    search_openalex_works,
)

# Backward-compatible alias for legacy callers (e.g. main/api/app.py)
search_gemini_academic_papers = search_openalex_academic_papers

__all__ = [
    "USER_AGENT",
    "VIETNAMESE_STOP_PATTERNS",
    "COMMON_CONCEPT_TRANSLATIONS",
    "clean_and_expand_query",
    "reconstruct_abstract",
    "search_openalex_works",
    "download_open_access_pdf",
    "search_openalex_academic_papers",
    "search_gemini_academic_papers",
]
