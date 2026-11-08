"""
GPU FT Mode - Agentic Retrieval Agent (L3 Multi-LoRA Specialist).
Leverages Remote Kaggle GPU Worker with Fine-Tuned 'router' (agent1_router) QLoRA:
  - Calls /api/router (or /generate with adapter='router') to obtain intent, sub_queries, alpha, top_k.
  - Dynamically scopes to user-selected PDF (state["pdf_path"]) via load_or_parse_selected_pdf().
  - Executes multi-query Hybrid FTS5 + BM25 + RRF search with Entity Coverage boost.
"""

from __future__ import annotations

import json
import os
from typing import Any

from agents.agent1_retrieval.tools.bm25 import SimpleBM25
from agents.agent1_retrieval.tools.fts5_search import SQLiteFTS5Searcher
from agents.agent1_retrieval.tools.hybrid import reciprocal_rank_fusion
from core.base_agent import BaseAgent
from core.schemas import RouterPlanOutput
from core.state import AgentState
from core.supervisor import supervisor_planner
from core.tools import load_or_parse_selected_pdf
from utils.llm_client import llm_client


ROUTER_TRAINING_SYSTEM_PROMPT = (
    "You are an intelligent routing agent for an Academic RAG system. "
    "Analyze the user query and output a JSON object with: intent, sub_queries, alpha (0.0 to 1.0), top_k (1 to 15), and reasoning."
)


class GPUFTRetrievalAgent(BaseAgent):
    """Retrieval Agent powered by GPU Fine-Tuned Router QLoRA ('router' / 'agent1_router')."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="GPUFTRetrievalAgent", max_internal_retry=max_internal_retry)
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
                self.logger.warning(f"[GPU FT Retrieval] Corpus load failed: {e}")
        self.corpus_docs = docs
        if docs:
            self.fts5_searcher.fit(docs)
            self.bm25_fallback.fit(docs)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        pdf_path = state.get("pdf_path")
        retry_count = state.get("retry_count", 0)
        seen_chunks = state.get("seen_chunks", []) or []

        # Start with local Supervisor plan, then enrich with fine-tuned Router LoRA on Kaggle
        local_plan = supervisor_planner.plan_query(query, state.get("evidence_gaps"), retry_count)
        sub_queries = list(state.get("sub_queries") or local_plan.sub_queries)
        required_entities = list(state.get("required_entities") or local_plan.required_entities)
        alpha = float(state.get("alpha") or local_plan.alpha)
        top_k = int(state.get("top_k") or local_plan.top_k)
        intent = local_plan.intent

        self.logger.info(f"[GPU FT Retrieval] Calling fine-tuned 'router' adapter for: '{query[:65]}'")

        if self.settings.REMOTE_INFERENCE_URL and retry_count == 0:
            try:
                router_res = llm_client.call_kaggle_specialist(
                    endpoint="api/router",
                    payload={"query": query, "adapter": "router"},
                    fallback_prompt=query,
                    fallback_system=ROUTER_TRAINING_SYSTEM_PROMPT,
                    adapter="router",
                )
                lora_plan = RouterPlanOutput.from_dict(router_res, default_query=query)
                if lora_plan.sub_queries:
                    # Merge LoRA sub_queries with supervisor sub_queries
                    merged_subs = list(dict.fromkeys(lora_plan.sub_queries + sub_queries))[:4]
                    sub_queries = merged_subs
                alpha = lora_plan.alpha
                top_k = lora_plan.top_k
                intent = lora_plan.intent
            except Exception as e:
                self.logger.warning(f"[GPU FT Retrieval] Remote router adapter fallback to Supervisor ({e})")

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

        # 2. Multi-query hybrid search over FTS5 + BM25
        candidate_lists = []
        for sq in sub_queries[:4]:
            res_fts = active_fts.search(sq, top_k=top_k)
            res_bm25 = active_bm25.search(sq, top_k=top_k)
            if res_fts:
                candidate_lists.append(res_fts)
            if res_bm25:
                candidate_lists.append(res_bm25)

        is_summary_query = any(
            w in query.lower()
            for w in [
                "tóm tắt", "tổng quan", "nội dung chính", "chính", "summary", "overview",
                "abstract", "giới thiệu", "phương pháp", "kết quả", "đóng góp", "2 bài báo",
                "các bài báo", "nói về gì", "nói về cái gì", "/tom-tat", "tom-tat",
                "so sánh", "đối chiếu", "so sanh", "doi chieu", "khác nhau", "khác biệt", "so-sanh", "so_sanh", "compare", "/so-sanh"
            ]
        ) or bool(state.get("is_comparative_query") or state.get("intent") == "compare")

        fused = reciprocal_rank_fusion(candidate_lists, top_n=top_k + 2) if candidate_lists else []
        if not fused and active_pool and is_summary_query:
            fused = active_pool[: min(top_k, len(active_pool))]

        # 3. Entity priority boost
        if fused and required_entities:
            def _score_fn(idx_doc: tuple[int, dict[str, Any]]) -> float:
                r_idx, d = idx_doc
                txt_l = str(d.get("text", "")).lower()
                ent_hits = sum(1 for e in required_entities if e.lower() in txt_l)
                return (1.0 / (r_idx + 1)) + 0.35 * ent_hits

            indexed = list(enumerate(fused))
            indexed.sort(key=_score_fn, reverse=True)
            fused = [d for _, d in indexed[:top_k]]
        else:
            fused = fused[:top_k]

        seen_ids = {c.get("id") for c in seen_chunks if isinstance(c, dict)}
        fresh_docs = [d for d in fused if d.get("id") not in seen_ids]
        no_new_evidence = len(fresh_docs) == 0 and retry_count > 0
        if not fresh_docs and fused:
            fresh_docs = fused[:top_k]

        all_seen = list(seen_chunks) + [d for d in fresh_docs if d.get("id") not in seen_ids]

        trace = self.log_action(
            "gpu_ft_retrieval_completed",
            {
                "remote_adapter": "router",
                "intent": intent,
                "alpha": alpha,
                "top_k": top_k,
                "sub_queries": sub_queries,
                "scoped_to_selected_pdf": bool(pdf_chunks),
                "retrieved_count": len(fresh_docs),
            },
        )

        return {
            "intent": intent,
            "alpha": alpha,
            "top_k": top_k,
            "sub_queries": sub_queries,
            "required_entities": required_entities,
            "retrieved_docs": fresh_docs,
            "seen_chunks": all_seen,
            "no_new_evidence": no_new_evidence,
            "reasoning_trace": [trace],
        }


