"""
CPU Max Mode - Agentic Retrieval Agent (L3).
Maximal local performance on Lenovo ThinkBook 14 G8+ AHP (AMD Ryzen 7 H 255 + Radeon 780M):
  - Multi-document scoped indexing across all user-selected PDFs (state["pdf_paths"]).
  - Stanford STORM Multi-Perspective Query Decomposition + Supervisor DAG sub_queries.
  - Hybrid FTS5 + BM25 + Entity Coverage Reranking with Reciprocal Rank Fusion (RRF).
  - Distributed Page-Sampling for full-document & multi-document summary queries.
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
    """Retrieval Agent for CPU Max: Full STORM expansion + Multi-Doc Scoped Hybrid RRF + Entity Rerank."""

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
        clean_q = (query or "").strip()
        pdf_path = state.get("pdf_path")
        pdf_paths = state.get("pdf_paths") or ([pdf_path] if pdf_path else [])
        retry_count = state.get("retry_count", 0)
        evidence_gaps = state.get("evidence_gaps", [])
        seen_chunks = state.get("seen_chunks", []) or []
        plan_subs = state.get("sub_queries") or []
        required_entities = state.get("required_entities") or []
        top_k = int(state.get("top_k") or 8)

        if not plan_subs:
            plan = supervisor_planner.plan_query(clean_q, evidence_gaps, retry_count)
            plan_subs = plan.sub_queries
            required_entities = plan.required_entities
            top_k = plan.top_k

        # Detect summary / overview / comparison intent
        is_summary_query = any(
            w in clean_q.lower()
            for w in [
                "tóm tắt", "tổng quan", "nội dung chính", "chính", "summary", "overview",
                "abstract", "giới thiệu", "phương pháp", "kết quả", "đóng góp", "2 bài báo",
                "các bài báo", "nói về gì", "nói về cái gì", "/tom-tat", "tom-tat",
                "so sánh", "đối chiếu", "so sanh", "doi chieu", "khác nhau", "khác biệt", "so-sanh", "so_sanh", "compare", "/so-sanh"
            ]
        ) or bool(state.get("is_comparative_query") or state.get("intent") == "compare")

        self.logger.info(
            f"[CPU Max Retrieval] Running STORM + Hybrid RRF for: '{clean_q[:65]}' "
            f"(Retry={retry_count}, PDFs={len(pdf_paths)}, IsSummary={is_summary_query})"
        )

        # 1. Scope strictly to all user-selected PDFs if provided
        pdf_chunks: list[dict[str, Any]] = []
        for p in pdf_paths:
            if p and os.path.exists(str(p)):
                chunks = load_or_parse_selected_pdf(p)
                if chunks:
                    pdf_chunks.extend(chunks)

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

        # 2. Retrieval Strategy:
        # If this is a general summary/overview query: sample chunks distributed across ALL pages of ALL documents!
        perspectives = []
        if is_summary_query and active_pool:
            sampled_summary: list[dict[str, Any]] = []
            seen_page_keys: set[tuple[str, int]] = set()
            for doc in active_pool:
                d_id = str(doc.get("doc_id", "doc"))
                p_no = int(doc.get("page_no", 1))
                key = (d_id, p_no)
                if key not in seen_page_keys:
                    seen_page_keys.add(key)
                    sampled_summary.append(doc)
            fused = sampled_summary if sampled_summary else active_pool[: max(top_k, 10)]
            combined_queries = [clean_q]
        else:
            # Multi-perspective STORM expansion (bypassed if Ollama local server is offline)
            from utils.llm_client import probe_ollama_models
            ollama_stat = probe_ollama_models(timeout=0.2)
            if ollama_stat.get("online"):
                perspectives, storm_subs = generate_research_perspectives(clean_q)
            else:
                perspectives, storm_subs = [], []
            combined_queries = []
            for q_cand in [clean_q] + list(plan_subs) + list(storm_subs):
                if q_cand and q_cand not in combined_queries:
                    combined_queries.append(q_cand)
            combined_queries = combined_queries[:5]

            # Multi-query FTS5 + BM25 search
            all_candidate_lists = []
            for q in combined_queries:
                fts5_res = active_fts.search(q, top_k=top_k)
                bm25_res = active_bm25.search(q, top_k=top_k)
                if fts5_res:
                    all_candidate_lists.append(fts5_res)
                if bm25_res:
                    all_candidate_lists.append(bm25_res)

            fused = reciprocal_rank_fusion(all_candidate_lists, top_n=top_k + 3) if all_candidate_lists else []

            # Only fallback to first chunks if it's a summary query (never for specific questions, to avoid hallucinations!)
            if not fused and active_pool and is_summary_query:
                fused = active_pool[: min(top_k, len(active_pool))]

        # 3. Cross-Encoder style Entity & Numeric Reranking on fused candidates
        if fused and required_entities and not is_summary_query:
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
            fused = fused[: max(top_k, 8)]

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
                "scoped_to_selected_pdfs": len(pdf_paths),
                "is_summary_query": is_summary_query,
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
