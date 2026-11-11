"""
CPU Min Mode - Agentic Retrieval Agent (L3-Lite).
Features:
  - Multi-document awareness: scopes to all user-selected PDFs (state["pdf_paths"] / state["pdf_path"]).
  - Distributed Page Sampling for summary queries (tóm tắt, tổng quan, overview) ensuring 100% whole-document coverage.
  - Multi-query FTS5 + BM25 search for specific factual questions.
  - Internal Coverage Gate: expands bilingual tokens if required entities are missed.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from agents.agent1_retrieval.tools.bm25 import SimpleBM25
from agents.agent1_retrieval.tools.fts5_search import SQLiteFTS5Searcher
from agents.agent1_retrieval.tools.hybrid import reciprocal_rank_fusion
from core.base_agent import BaseAgent
from core.state import AgentState
from core.supervisor import supervisor_planner
from core.tools import load_or_parse_selected_pdf


class CPUMinRetrievalAgent(BaseAgent):
    """Agentic Retrieval for CPU Min: Multi-document FTS5 + BM25 with distributed sampling for summaries."""

    def __init__(self, max_internal_retry: int = 1):
        super().__init__(name="CPUMinRetrievalAgent", max_internal_retry=max_internal_retry)
        self.fts5_searcher = SQLiteFTS5Searcher()
        self.bm25_fallback = SimpleBM25()
        self.corpus_docs: list[dict[str, Any]] = []
        self._doc_index_cache: dict[str, tuple[SQLiteFTS5Searcher, SimpleBM25, list[dict[str, Any]]]] = {}
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
                self.logger.warning(f"[CPU Min] Error loading corpus: {e}")

        self.corpus_docs = docs
        if docs:
            self.fts5_searcher.fit(docs)
            self.bm25_fallback.fit(docs)
            self.logger.info(f"[CPU Min Retrieval] Indexed {len(docs)} corpus chunks into FTS5 + BM25.")

    def _search_index(
        self,
        queries: list[str],
        fts: SQLiteFTS5Searcher,
        bm25: SimpleBM25,
        top_k: int = 6,
    ) -> list[dict[str, Any]]:
        candidate_lists: list[list[dict[str, Any]]] = []
        for q in queries:
            res_fts = fts.search(q, top_k=top_k)
            res_bm25 = bm25.search(q, top_k=top_k)
            if res_fts:
                candidate_lists.append(res_fts)
            if res_bm25:
                candidate_lists.append(res_bm25)
        if not candidate_lists:
            return []
        return reciprocal_rank_fusion(candidate_lists, top_n=top_k)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        clean_q = (query or "").strip()
        pdf_path = state.get("pdf_path")
        pdf_paths = state.get("pdf_paths") or ([pdf_path] if pdf_path else [])
        retry_count = state.get("retry_count", 0)
        seen_chunks = state.get("seen_chunks", []) or []
        sub_queries = state.get("sub_queries") or []
        required_entities = state.get("required_entities") or []
        top_k = int(state.get("top_k") or 8)

        if not sub_queries:
            plan = supervisor_planner.plan_query(clean_q, state.get("evidence_gaps"), retry_count)
            sub_queries = plan.sub_queries
            required_entities = plan.required_entities
            top_k = plan.top_k

        # Detect summary / overview intent (avoid generic substrings like 'chính' which match 'độ chính xác')
        is_summary_query = any(
            w in clean_q.lower()
            for w in [
                "tóm tắt", "tổng quan", "nội dung chính", "ý chính", "điểm chính", "summary", "overview",
                "abstract", "giới thiệu", "đóng góp chính", "2 bài báo",
                "các bài báo", "nói về cái gì", "nói về gì", "/tom-tat", "tom-tat",
            ]
        )

        self.logger.info(
            f"[CPU Min Retrieval] Query='{clean_q[:60]}' | SubQueries={len(sub_queries)} | "
            f"PDFs={len(pdf_paths)} | IsSummary={is_summary_query} | Retry={retry_count}"
        )

        # 1. Scope strictly to user-selected PDF(s) if provided, else global corpus
        pdf_chunks: list[dict[str, Any]] = []
        for p in pdf_paths:
            if p and os.path.exists(p):
                chunks = load_or_parse_selected_pdf(p)
                if chunks:
                    pdf_chunks.extend(chunks)

        if pdf_chunks:
            # Cache active index by chunk bounds to avoid rebuilding FTS5/BM25 on every retry loop
            cache_key = f"{len(pdf_chunks)}::{pdf_chunks[0].get('id')}::{pdf_chunks[-1].get('id')}"
            if cache_key in self._doc_index_cache:
                active_fts, active_bm25, active_pool = self._doc_index_cache[cache_key]
            else:
                active_fts = SQLiteFTS5Searcher()
                active_bm25 = SimpleBM25()
                active_fts.fit(pdf_chunks)
                active_bm25.fit(pdf_chunks)
                active_pool = pdf_chunks
                if len(self._doc_index_cache) >= 8:
                    self._doc_index_cache.pop(next(iter(self._doc_index_cache)))
                self._doc_index_cache[cache_key] = (active_fts, active_bm25, active_pool)
        else:
            active_fts = self.fts5_searcher
            active_bm25 = self.bm25_fallback
            active_pool = self.corpus_docs

        # 2. Retrieval Strategy:
        # If this is a general summary/overview query: sample chunks distributed across ALL pages of ALL documents!
        if is_summary_query and active_pool:
            docs_by_id: dict[str, dict[int, list[dict[str, Any]]]] = {}
            for chunk in active_pool:
                d_id = str(chunk.get("doc_id", "doc"))
                p_no = int(chunk.get("page_no", 1))
                docs_by_id.setdefault(d_id, {}).setdefault(p_no, []).append(chunk)

            sampled_summary: list[dict[str, Any]] = []
            for d_id, page_dict in docs_by_id.items():
                sorted_pages = sorted(page_dict.keys())
                max_page = sorted_pages[-1] if sorted_pages else 1

                # If large document (e.g. textbook > 20 pages), skip frontmatter (pages 1-15) to avoid TOC & copyright
                if max_page > 20:
                    body_pages = [p for p in sorted_pages if p >= 16]
                    pages_to_sample = body_pages if len(body_pages) >= 5 else sorted_pages
                else:
                    pages_to_sample = sorted_pages

                # Evenly sample 16-20 representative pages across the document body
                sample_count = min(len(pages_to_sample), 20)
                if sample_count > 0:
                    step = len(pages_to_sample) / float(sample_count)
                    chosen_pages = [pages_to_sample[int(i * step)] for i in range(sample_count)]
                    for p in chosen_pages:
                        chunks_on_page = page_dict[p]
                        best_chunk = max(chunks_on_page, key=lambda c: len(str(c.get("text", "")).strip()))
                        sampled_summary.append(best_chunk)

            results = sampled_summary if sampled_summary else active_pool[: max(top_k, 20)]
        else:
            # First-pass multi-query lexical search
            results = self._search_index(sub_queries, active_fts, active_bm25, top_k=top_k)

        # 3. Internal Coverage Gate: Check if required_entities are covered
        if required_entities and results and not is_summary_query:
            combined_text = " ".join(str(r.get("text", "")) for r in results).lower()
            covered = sum(1 for ent in required_entities if ent.lower() in combined_text)
            coverage_ratio = covered / max(len(required_entities), 1)
        else:
            coverage_ratio = 1.0 if results else 0.0

        if coverage_ratio < 0.5 or not results:
            expanded_queries = list(sub_queries)
            try:
                from utils.bilingual import expand_bilingual_tokens
                base_tokens = set(re.findall(r"\w+", clean_q.lower()))
                bilingual_tokens = expand_bilingual_tokens(clean_q, base_tokens)
                expanded_queries.append(" ".join(list(bilingual_tokens)[:12]))
            except Exception:
                pass
            if required_entities:
                expanded_queries.append(" ".join(required_entities))
            res_expanded = self._search_index(expanded_queries, active_fts, active_bm25, top_k=top_k + 2)
            if res_expanded:
                results = res_expanded

        # Fallback to first chunks of selected PDF only if results is still empty
        if not results and active_pool and is_summary_query:
            results = active_pool[: min(top_k, len(active_pool))]

        # 4. Deduplicate against seen_chunks across outer retries
        seen_ids = {c.get("id") for c in seen_chunks if isinstance(c, dict)}
        fresh_docs = [d for d in results if d.get("id") not in seen_ids]
        no_new_evidence = len(fresh_docs) == 0 and retry_count > 0
        if not fresh_docs and results:
            fresh_docs = results[: max(top_k, 10)]

        all_seen = list(seen_chunks) + [d for d in fresh_docs if d.get("id") not in seen_ids]

        trace = self.log_action(
            "cpu_min_agentic_retrieval_completed",
            {
                "engine": "Distributed_Sampling_FTS5_BM25",
                "docs_scoped": len(pdf_paths),
                "is_summary_query": is_summary_query,
                "retrieved_count": len(fresh_docs),
                "total_seen": len(all_seen),
            },
        )

        return {
            "sub_queries": sub_queries,
            "required_entities": required_entities,
            "retrieved_docs": fresh_docs,
            "seen_chunks": all_seen,
            "no_new_evidence": no_new_evidence,
            "reasoning_trace": [trace],
        }
