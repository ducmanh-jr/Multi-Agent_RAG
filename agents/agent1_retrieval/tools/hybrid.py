"""
Hybrid Reciprocal Rank Fusion (RRF) and Deduplication Engine.
Combines BM25 and Dense/Semantic ranking lists into a balanced optimal ranking.
"""

from __future__ import annotations

from typing import Any
from core.state import compute_context_hash


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    RRF Algorithm: RRF_score(d) = sum(1 / (k + rank_i(d)))
    """
    doc_scores: dict[str, float] = {}
    doc_registry: dict[str, dict[str, Any]] = {}

    for r_list in ranked_lists:
        for rank, doc in enumerate(r_list):
            doc_id = doc.get("id") or compute_context_hash(
                doc.get("text", ""), doc.get("doc_id", ""), doc.get("page_no", 1)
            )
            doc_registry[doc_id] = doc
            score_increment = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score_increment

    sorted_doc_ids = sorted(doc_scores.keys(), key=lambda d_id: doc_scores[d_id], reverse=True)

    fused_results: list[dict[str, Any]] = []
    for d_id in sorted_doc_ids[:top_n]:
        item = dict(doc_registry[d_id])
        item["rrf_score"] = round(doc_scores[d_id], 5)
        if "context_hash" not in item:
            item["context_hash"] = compute_context_hash(
                item.get("text", ""), item.get("doc_id", ""), item.get("page_no", 1)
            )
        fused_results.append(item)

    return fused_results
