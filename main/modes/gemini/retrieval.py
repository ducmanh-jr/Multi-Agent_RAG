"""
Gemini Mode - Agentic Retrieval Agent (L3-L4, Quota-Optimized).
Features:
  - Strictly scopes to user-selected PDF (state["pdf_path"]) via load_or_parse_selected_pdf().
  - Combines Supervisor DAG sub_queries + STORM perspectives.
  - Hybrid FTS5 + BM25 + Entity/Numeric Reranking (0 unnecessary API calls during local indexing).
"""

from __future__ import annotations

import json
import os
from typing import Any

from agents.agent1_retrieval.tools.bm25 import SimpleBM25
from agents.agent1_retrieval.tools.fts5_search import SQLiteFTS5Searcher
from agents.agent1_retrieval.tools.hybrid import reciprocal_rank_fusion
from agents.agent1_retrieval.tools.semantic_pipeline import (
    boost_entities_and_tables,
    build_cross_lingual_subqueries,
    enhance_with_dense_and_rerank,
    extract_searchable_figure_chunks,
)
from core.base_agent import BaseAgent
from core.state import AgentState
from core.supervisor import supervisor_planner
from core.tools import load_or_parse_selected_pdf
from utils.llm_client import llm_client


class GeminiRetrievalAgent(BaseAgent):
    """Agentic Retrieval for Gemini Mode: Document-scoped multi-query RRF + Entity Rerank."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="GeminiRetrievalAgent", max_internal_retry=max_internal_retry)
        self.fts5_searcher = SQLiteFTS5Searcher()
        self.bm25_fallback = SimpleBM25()
        self.corpus_docs: list[dict[str, Any]] = []
        self._load_corpus()

    def _load_corpus(self):
        corpus_path = self.settings.CORPUS_PATH
        docs = []
        if os.path.exists(corpus_path):
            try:
                with open(corpus_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            docs.append(json.loads(line))
            except Exception as e:
                self.logger.warning(f"[Gemini Retrieval] Error loading corpus: {e}")
        self.corpus_docs = docs
        if docs:
            self.fts5_searcher.fit(docs)
            self.bm25_fallback.fit(docs)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        pdf_path = state.get("pdf_path")
        retry_count = state.get("retry_count", 0)
        seen_chunks = state.get("seen_chunks", []) or []
        sub_queries = state.get("sub_queries") or []
        required_entities = state.get("required_entities") or []
        top_k = int(state.get("top_k") or 7)

        if not sub_queries:
            plan = supervisor_planner.plan_query(query, state.get("evidence_gaps"), retry_count)
            sub_queries = plan.sub_queries
            required_entities = plan.required_entities
            top_k = plan.top_k

        self.logger.info(
            f"[Gemini Retrieval] Query='{query[:65]}' | SubQueries={len(sub_queries)} | "
            f"PDF={os.path.basename(pdf_path) if pdf_path else 'Corpus'}"
        )

        pdf_paths = state.get("pdf_paths") or ([pdf_path] if pdf_path else [])
        # 1. Scope strictly to user-selected PDF(s) if provided
        pdf_chunks: list[dict[str, Any]] = []
        for p in pdf_paths:
            if p and os.path.exists(p):
                chunks = load_or_parse_selected_pdf(p)
                if chunks:
                    pdf_chunks.extend(chunks)
        figure_chunks = extract_searchable_figure_chunks(pdf_paths)
        if figure_chunks:
            pdf_chunks.extend(figure_chunks)
        if pdf_chunks:
            active_fts = SQLiteFTS5Searcher()
            active_bm25 = SimpleBM25()
            active_fts.fit(pdf_chunks)
            active_bm25.fit(pdf_chunks)
            active_pool = pdf_chunks
        else:
            active_fts = self.fts5_searcher
            active_bm25 = self.bm25_fallback
            active_pool = self.corpus_docs

        # 2. Cross-lingual bilingual bridging + multi-query hybrid search over FTS5 + BM25
        effective_q = str(state.get("resolved_query") or query).strip() or query
        bridged = build_cross_lingual_subqueries(effective_q, active_pool)
        search_queries = list(dict.fromkeys([*sub_queries[:4], *bridged]))[:6]
        candidate_limit = 2 * top_k

        all_results = []
        for sq in search_queries:
            res_fts = active_fts.search(sq, top_k=candidate_limit)
            res_bm25 = active_bm25.search(sq, top_k=candidate_limit)
            if res_fts:
                all_results.append(res_fts)
            if res_bm25:
                all_results.append(res_bm25)

        sparse_fused = reciprocal_rank_fusion(all_results, top_n=candidate_limit) if all_results else []
        if not sparse_fused and active_pool:
            sparse_fused = active_pool[: min(candidate_limit, len(active_pool))]

        # 3. Optional Gemini Dense Embedding + Page Diversity
        fused, dense_trace, rerank_trace = enhance_with_dense_and_rerank(
            effective_q,
            sparse_fused,
            active_pool,
            top_k=max(top_k + 2, top_k),
            http_client=llm_client.http_client,
            enable_kaggle_dense=False,
            enable_gemini_dense=bool(self.settings.GEMINI_DENSE_ENABLED and self.settings.GEMINI_API_KEY),
            enable_rerank=False,
            page_diverse=True,
        )

        # 4. Entity & Intact-Table/Equation priority boost
        fused = boost_entities_and_tables(effective_q, fused, required_entities, top_k)

        seen_ids = {c.get("id") for c in seen_chunks if isinstance(c, dict)}
        fresh_docs = [d for d in fused if d.get("id") not in seen_ids]
        no_new_evidence = len(fresh_docs) == 0 and retry_count > 0
        if not fresh_docs and fused:
            fresh_docs = fused[:top_k]

        all_seen = list(seen_chunks) + [d for d in fresh_docs if d.get("id") not in seen_ids]

        fallback_chain = (
            [{"stage": "dense", "reason": str(dense_trace.get("reason") or "unknown")}]
            if dense_trace.get("status") == "fallback"
            else []
        )
        strategy_used = "hybrid" if dense_trace.get("status") == "used" else "sparse"

        trace = self.log_action(
            "gemini_retrieval_completed",
            {
                "scoped_to_selected_pdf": bool(pdf_chunks),
                "top_k": top_k,
                "strategy_used": strategy_used,
                "fallback_chain": fallback_chain,
                "sub_queries_count": len(search_queries),
                "cross_lingual_queries": bridged,
                "dense": dense_trace,
                "rerank": rerank_trace,
                "retrieved_count": len(fresh_docs),
            },
        )

        return {
            "sub_queries": search_queries,
            "required_entities": required_entities,
            "retrieved_docs": fresh_docs,
            "seen_chunks": all_seen,
            "no_new_evidence": no_new_evidence,
            "reasoning_trace": [trace],
        }
