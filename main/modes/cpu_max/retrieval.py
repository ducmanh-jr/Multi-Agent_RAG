"""
CPU Max Mode - Agentic Retrieval Agent (L3).
Maximal local performance on Lenovo ThinkBook 14 G8+ AHP (AMD Ryzen 7 H 255 + Radeon 780M):
  - Document-scoped indexing of user-selected PDF (state["pdf_path"]) + Global Corpus fallback.
  - Stanford STORM Multi-Perspective Query Decomposition + Supervisor DAG sub_queries.
  - Hybrid FTS5 + BM25 + Entity Coverage Reranking with Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from agents.agent1_retrieval.tools.bm25 import SimpleBM25
from agents.agent1_retrieval.tools.fts5_search import SQLiteFTS5Searcher
from agents.agent1_retrieval.tools.hybrid import reciprocal_rank_fusion
from agents.agent1_retrieval.tools.perspective import generate_research_perspectives
from core.base_agent import BaseAgent
from core.state import AgentState
from core.supervisor import supervisor_planner
from core.tools import load_or_parse_selected_pdf


class CPUMaxRetrievalAgent(BaseAgent):
    """Retrieval Agent for CPU Max: Full STORM expansion + Document-Scoped Hybrid RRF + Entity Rerank."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="CPUMaxRetrievalAgent", max_internal_retry=max_internal_retry)
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
                self.logger.warning(f"[CPU Max Retrieval] Corpus load failed: {e}")

        self.corpus_docs = docs
        if docs:
            self.fts5_searcher.fit(docs)
            self.bm25_fallback.fit(docs)
            self.logger.info(f"[CPU Max Retrieval] Indexed {len(docs)} documents into C-Speed FTS5.")

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        pdf_path = state.get("pdf_path")
        retry_count = state.get("retry_count", 0)
        evidence_gaps = state.get("evidence_gaps", [])
        seen_chunks = state.get("seen_chunks", []) or []
        plan_subs = state.get("sub_queries") or []
        required_entities = state.get("required_entities") or []
        top_k = int(state.get("top_k") or 7)

        if not plan_subs:
            plan = supervisor_planner.plan_query(query, evidence_gaps, retry_count)
            plan_subs = plan.sub_queries
            required_entities = plan.required_entities
            top_k = plan.top_k

        self.logger.info(
            f"[CPU Max Retrieval] Running STORM + Hybrid RRF for: '{query[:65]}' "
            f"(Retry={retry_count}, PDF={os.path.basename(pdf_path) if pdf_path else 'Corpus'})"
        )

        # 1. Scope strictly to user-selected PDF if provided
        pdf_chunks = load_or_parse_selected_pdf(pdf_path) if pdf_path else []
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

        # 2. Combine Supervisor sub_queries + STORM multi-perspective sub_queries
        perspectives, storm_subs = generate_research_perspectives(query)
        combined_queries: list[str] = []
        for q_cand in [query] + list(plan_subs) + list(storm_subs):
            if q_cand and q_cand not in combined_queries:
                combined_queries.append(q_cand)
        combined_queries = combined_queries[:5]

        # 3. Multi-query FTS5 + BM25 search
        all_candidate_lists = []
        for q in combined_queries:
            fts5_res = active_fts.search(q, top_k=top_k)
            bm25_res = active_bm25.search(q, top_k=top_k)
            if fts5_res:
                all_candidate_lists.append(fts5_res)
            if bm25_res:
                all_candidate_lists.append(bm25_res)

        fused = reciprocal_rank_fusion(all_candidate_lists, top_n=top_k + 3) if all_candidate_lists else []

        if not fused and active_pool:
            fused = active_pool[: min(top_k, len(active_pool))]

        # 4. Cross-Encoder style Entity & Numeric Reranking on fused candidates
        if fused and required_entities:
            def _rerank_score(idx_and_doc: tuple[int, dict[str, Any]]) -> float:
                rank_idx, d = idx_and_doc
                base_rrf = 1.0 / (rank_idx + 1)
                txt_l = str(d.get("text", "")).lower()
                ent_matches = sum(1 for e in required_entities if e.lower() in txt_l)
                return base_rrf + 0.35 * ent_matches

            indexed_fused = list(enumerate(fused))
            indexed_fused.sort(key=_rerank_score, reverse=True)
            fused = [d for _, d in indexed_fused[:top_k]]
        else:
            fused = fused[:top_k]

        seen_ids = {c.get("id") for c in seen_chunks if isinstance(c, dict)}
        fresh_docs = [d for d in fused if d.get("id") not in seen_ids]
        no_new_evidence = len(fresh_docs) == 0 and retry_count > 0
        if not fresh_docs and fused:
            fresh_docs = fused[:top_k]

        all_seen = list(seen_chunks) + [d for d in fresh_docs if d.get("id") not in seen_ids]

        trace = self.log_action(
            "cpu_max_retrieval_completed",
            {
                "engine": "STORM_Hybrid_RRF_EntityRerank",
                "scoped_to_selected_pdf": bool(pdf_chunks),
                "queries_executed": len(combined_queries),
                "retrieved_count": len(fresh_docs),
                "perspectives_count": len(perspectives),
            },
        )

        return {
            "perspectives": perspectives,
            "sub_queries": combined_queries,
            "required_entities": required_entities,
            "retrieved_docs": fresh_docs,
            "seen_chunks": all_seen,
            "no_new_evidence": no_new_evidence,
            "reasoning_trace": [trace],
        }
