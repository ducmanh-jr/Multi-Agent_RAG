"""
Hybrid Reciprocal Rank Fusion (RRF) and Deduplication Engine.
Combines BM25 and Dense/Semantic ranking lists into a balanced optimal ranking.
"""

from __future__ import annotations

from typing import Any, Optional
from core.state import compute_context_hash


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
    top_n: int = 5,
    weights: Optional[list[float]] = None,
) -> list[dict[str, Any]]:
    """
    Weighted RRF Algorithm:
      RRF_score(d) = sum_i( w_i / (k + rank_i(d) + 1) )
    Supports dynamic alpha blending between Lexical/Sparse lists and Semantic/Dense lists.
    """
    doc_scores: dict[str, float] = {}
    doc_registry: dict[str, dict[str, Any]] = {}

    for list_idx, r_list in enumerate(ranked_lists):
        w = weights[list_idx] if (weights and list_idx < len(weights)) else 1.0
        for rank, doc in enumerate(r_list):
            raw_id = doc.get("id")
            parent_doc = doc.get("doc_id", "")
            doc_id = f"{parent_doc}:{raw_id}" if (raw_id and parent_doc) else (raw_id or compute_context_hash(
                doc.get("text", ""), parent_doc, doc.get("page_no", 1)
            ))
            doc_registry[doc_id] = doc
            score_increment = float(w) / (k + rank + 1)
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
