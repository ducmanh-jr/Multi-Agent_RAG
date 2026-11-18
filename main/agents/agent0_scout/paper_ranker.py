"""
Paper Ranker & Semantic Relevance Gatekeeper for Agent 0 (Scout).
Scores candidate papers using a hybrid of Semantic Cosine Similarity (SentenceTransformer)
and Exact Keyword Overlap to strictly eliminate spurious collisions (e.g. EcoTIM tractor).
"""

from __future__ import annotations

import os
import re
from typing import Any
from utils.logger import logger

_EMBED_MODEL = None


def get_embed_model():
    """Lazily loads local embedding model."""
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
        
    try:
        from sentence_transformers import SentenceTransformer
        model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "multilingual-e5-small")
        if os.path.exists(model_path):
            _EMBED_MODEL = SentenceTransformer(model_path)
            logger.info(f"[Scout Ranker] Loaded local SentenceTransformer from: {model_path}")
        else:
            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        logger.warning(f"[Scout Ranker] Could not load SentenceTransformer: {e}")
        _EMBED_MODEL = False
        
    return _EMBED_MODEL


def calculate_keyword_score(paper: dict[str, Any], query_keywords: list[str]) -> tuple[float, int]:
    """Calculates keyword match count in title and abstract."""
    title_text = paper.get("title", "").lower()
    abstract_text = paper.get("abstract", "").lower()
    
    score = 0.0
    matches = 0
    for kw in query_keywords:
        kw_clean = kw.lower().strip()
        if not kw_clean or len(kw_clean) < 2:
            continue
        if re.search(rf"\b{re.escape(kw_clean)}\b", title_text):
            score += 3.0
            matches += 1
        elif re.search(rf"\b{re.escape(kw_clean)}\b", abstract_text):
            score += 1.0
            matches += 1
            
    return score, matches


def rank_and_select_papers(
    papers: list[dict[str, Any]], 
    user_query: str, 
    academic_keywords: str = "",
    top_k: int = 3,
    min_threshold: float = 1.5
) -> list[dict[str, Any]]:
    """
    Ranks papers and strictly applies Bidirectional Relevance Gating.
    Eliminates candidate papers that fail semantic topical alignment.
    """
    if not papers:
        return []

    ref_text = academic_keywords.strip() if academic_keywords.strip() else user_query
    meta_stop = {"architecture", "benchmark", "performance", "method", "comparison", "evaluation", "recent", "2024", "2025"}
    keywords = [w.strip() for w in re.findall(r"[a-zA-Z0-9_-]+", ref_text.lower()) if len(w.strip()) > 1 and w.strip() not in meta_stop]
    if not keywords:
        keywords = [w.strip() for w in re.findall(r"[a-zA-Z0-9_-]+", ref_text.lower()) if len(w.strip()) > 1]

    embed_model = get_embed_model()
    q_emb = None
    if embed_model:
        try:
            q_emb = embed_model.encode(f"query: {ref_text}")
        except Exception as e:
            logger.warning(f"[Scout Ranker] Failed to encode query vector: {e}")

    scored_papers = []
    seen_ids = set()

    for paper in papers:
        pid = paper.get("arxiv_id") or paper.get("id")
        if not pid or pid in seen_ids:
            continue
        seen_ids.add(pid)

        kw_score, matches = calculate_keyword_score(paper, keywords)
        
        # Semantic similarity
        semantic_sim = 0.0
        if embed_model and q_emb is not None:
            try:
                from sentence_transformers import util
                passage_text = f"passage: {paper.get('title', '')}. {paper.get('abstract', '')[:400]}"
                p_emb = embed_model.encode(passage_text)
                semantic_sim = float(util.cos_sim(q_emb, p_emb)[0][0])
            except Exception as e:
                logger.warning(f"[Scout Ranker] Failed to compute semantic similarity: {e}")

        # Combined composite score
        composite_score = kw_score + (semantic_sim * 4.0)

        # RELEVANCE HARD GATE:
        # If paper has ZERO keyword matches, it must have extremely high semantic similarity (>0.82) to be accepted
        if matches == 0 and semantic_sim < 0.82:
            logger.info(f"[Scout Hard Gate] Filtered out irrelevant paper '{paper.get('title')}' (matches={matches}, sim={semantic_sim:.3f})")
            continue

        if composite_score < min_threshold:
            logger.info(f"[Scout Hard Gate] Score below threshold '{paper.get('title')}' (score={composite_score:.2f})")
            continue

        paper_copy = dict(paper)
        paper_copy["relevance_score"] = round(composite_score, 2)
        paper_copy["semantic_similarity"] = round(semantic_sim, 3)
        paper_copy["match_count"] = matches
        scored_papers.append(paper_copy)

    # Sort descending by composite score, then by recency
    scored_papers.sort(
        key=lambda p: (p.get("relevance_score", 0), p.get("year", 0)), 
        reverse=True
    )

    selected = scored_papers[:top_k]
    logger.info(f"[Scout Ranker] Selected {len(selected)}/{len(papers)} top papers (Threshold: {min_threshold})")
    return selected
