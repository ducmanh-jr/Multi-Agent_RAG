"""
GPU Max Mode - Agentic Retrieval Agent (L3-L4 32B Reasoning on Dual T4).
Features:
  - Uses fast /no_think mode on Qwen3-32B-AWQ for sub-query planning (avoids CoT latency during routing).
  - Strictly scopes to user-selected PDF (state["pdf_path"]) via load_or_parse_selected_pdf().
  - Multi-query Hybrid FTS5 + BM25 + RRF with Entity & Numeric priority reranking.
"""

from __future__ import annotations

import json
import os
from typing import Any

from agents.agent1_retrieval.tools.bm25 import SimpleBM25
from agents.agent1_retrieval.tools.fts5_search import SQLiteFTS5Searcher
from agents.agent1_retrieval.tools.hybrid import reciprocal_rank_fusion
from agents.agent1_retrieval.tools.semantic_pipeline import (
    boost_entities_tables_and_equations,
    build_cross_lingual_subqueries,
    extract_searchable_figure_chunks,
)
from core.base_agent import BaseAgent
from core.schemas import RouterPlanOutput
from core.state import AgentState
from core.supervisor import supervisor_planner
from core.tools import load_or_parse_selected_pdf
from utils.llm_client import llm_client


class GPUMaxRetrievalAgent(BaseAgent):
    """Retrieval Agent for GPU Max: 32B /no_think planner + Document-Scoped Hybrid RRF."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="GPUMaxRetrievalAgent", max_internal_retry=max_internal_retry)
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
                self.logger.warning(f"[GPU Max Retrieval] Corpus load failed: {e}")
        self.corpus_docs = docs
        if docs:
            self.fts5_searcher.fit(docs)
            self.bm25_fallback.fit(docs)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query") or state.get("question", "")
        pdf_path = state.get("pdf_path")
        pdf_paths = state.get("pdf_paths") or ([pdf_path] if pdf_path else [])
        retry_count = state.get("retry_count", 0)
        seen_chunks = state.get("seen_chunks", []) or []

        local_plan = supervisor_planner.plan_query(query, state.get("evidence_gaps"), retry_count)
        sub_queries = list(state.get("sub_queries") or local_plan.sub_queries)
        required_entities = list(state.get("required_entities") or local_plan.required_entities)
        top_k = int(state.get("top_k") or 8)

        self.logger.info(f"[GPU Max Retrieval] 32B /no_think routing + Hybrid search for: '{query[:65]}'")

        # Fast /no_think 32B query decomposition on first pass if complex
        if self.settings.REMOTE_INFERENCE_URL and retry_count == 0 and local_plan.complexity == "HIGH":
            try:
                res_32b = llm_client.call_gpu_max_32b(
                    prompt=(
                        f"Phân tách câu hỏi nghiên cứu sau thành tối đa 3 câu truy vấn tìm kiếm con (sub_queries) "
                        f"ngắn gọn, chính xác: \"{query}\".\n"
                        f'Chỉ trả về JSON: {{"sub_queries": ["...", "..."], "top_k": 8}}'
                    ),
                    enable_thinking=False,  # Fast /no_think mode for tool routing!
                    temperature=0.1,
                    max_tokens=256,
                    json_mode=True,
                )
                parsed_plan = RouterPlanOutput.from_dict(res_32b.get("parsed", {}), default_query=query)
                if parsed_plan.sub_queries:
                    sub_queries = list(dict.fromkeys(parsed_plan.sub_queries + sub_queries))[:4]
            except Exception as exc:
                self.logger.info(f"[GPU Max Retrieval] Using local Supervisor sub_queries ({exc})")

        # 1. Scope strictly to user-selected PDF(s) if provided
        pdf_chunks: list[dict[str, Any]] = []
        for p in pdf_paths:
            if p and os.path.exists(p):
                chunks = load_or_parse_selected_pdf(p)
                if chunks:
                    pdf_chunks.extend(chunks)
        figure_chunks = extract_searchable_figure_chunks(pdf_paths)
        if figure_chunks:
            existing_cids = {c.get("id") for c in pdf_chunks if isinstance(c, dict)}
            pdf_chunks.extend(fc for fc in figure_chunks if fc.get("id") not in existing_cids)
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

        # 2. Multi-query hybrid search (with cross-lingual VI<->EN & structural bridging)
        effective_q = str(state.get("resolved_query") or query).strip() or query
        bridged = build_cross_lingual_subqueries(effective_q, active_pool)
        search_queries = list(dict.fromkeys([*sub_queries[:4], *bridged]))[:6]
        all_lists = []
        for q in search_queries:
            res_fts = active_fts.search(q, top_k=top_k)
            res_bm25 = active_bm25.search(q, top_k=top_k)
            if res_fts:
                all_lists.append(res_fts)
            if res_bm25:
                all_lists.append(res_bm25)

        is_summary_query = any(
            w in query.lower()
            for w in [
                "tóm tắt", "tổng quan", "nội dung chính", "chính", "summary", "overview",
                "abstract", "giới thiệu", "phương pháp", "kết quả", "đóng góp", "2 bài báo",
                "các bài báo", "nói về gì", "nói về cái gì", "/tom-tat", "tom-tat",
                "so sánh", "đối chiếu", "so sanh", "doi chieu", "khác nhau", "khác biệt", "so-sanh", "so_sanh", "compare", "/so-sanh"
            ]
        ) or bool(state.get("is_comparative_query") or state.get("intent") == "compare")

        fused = reciprocal_rank_fusion(all_lists, top_n=top_k + 4) if all_lists else []
        if not fused and active_pool and is_summary_query:
            fused = active_pool[: min(top_k, len(active_pool))]

        # 3. Structural Equation/Table & Entity priority boost
        if fused:
            fused = boost_entities_tables_and_equations(effective_q, fused, required_entities, top_k)

        seen_ids = {c.get("id") for c in seen_chunks if isinstance(c, dict)}
        fresh_docs = [d for d in fused if d.get("id") not in seen_ids]
        no_new_evidence = len(fresh_docs) == 0 and retry_count > 0
        if not fresh_docs and fused:
            fresh_docs = fused[:top_k]

        all_seen = list(seen_chunks) + [d for d in fresh_docs if d.get("id") not in seen_ids]

        trace = self.log_action(
            "gpu_max_retrieval_completed",
            {
                "gpu": "Dual_T4_32GB_TP2",
                "model": "Qwen3-32B-AWQ (/no_think router)",
                "strategy_used": "sparse",
                "top_k": top_k,
                "cross_lingual_queries": bridged,
                "dense": {"status": "disabled"},
                "rerank": {"status": "disabled"},
                "scoped_to_selected_pdf": bool(pdf_chunks),
                "sub_queries": search_queries,
                "retrieved_count": len(fresh_docs),
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


