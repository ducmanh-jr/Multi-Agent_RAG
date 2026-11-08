"""
Hybrid Mode - Adaptive Retrieval Agent (L4 SOTA Cascade).
Hierarchical 3-Tier Autonomous Resilience:
  - Tier 1 (Primary - Cloud API): Gemini 2.5 Flash for deep intent reasoning & STORM sub-queries.
  - Tier 2 (Self-Healing Fallback 1): GPU Finetune Router LoRA (01_agent1_router) on Kaggle GPU.
  - Tier 3 (Self-Healing Fallback 2 - 100% Offline): Local Supervisor Planner + C-Engine FTS5 + BM25 search.
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
from core.schemas import RouterPlanOutput
from core.state import AgentState
from core.supervisor import supervisor_planner
from core.tools import load_or_parse_selected_pdf
from utils.llm_client import llm_client
from utils.local_vulkan_setup import probe_all_capabilities


class HybridRetrievalAgent(BaseAgent):
    """Flagship Hybrid Retrieval: Gemini Router First -> GPU Finetune LoRA -> Local FTS5/BM25."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="HybridRetrievalAgent", max_internal_retry=max_internal_retry)
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
                self.logger.warning(f"[Hybrid Retrieval] Corpus load failed: {e}")
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

        # Default Tier 3 Plan: Local Supervisor
        plan = supervisor_planner.plan_query(query, state.get("evidence_gaps"), retry_count)
        sub_queries = list(state.get("sub_queries") or plan.sub_queries)
        required_entities = list(state.get("required_entities") or plan.required_entities)
        top_k = int(state.get("top_k") or plan.top_k)
        alpha = float(state.get("alpha") or getattr(plan, "alpha", 0.5) or 0.5)
        planning_backend = "local_supervisor"

        caps = probe_all_capabilities()

        # =====================================================================
        # Tier 1 (Primary - Gemini API): Deep Intent & STORM Planning
        # =====================================================================
        if caps.get("gemini", {}).get("online") and retry_count == 0:
            try:
                gemini_prompt = (
                    f"You are a master academic research routing agent. "
                    f"Analyze this query and output JSON with keys: intent, sub_queries (3 distinct search angles), "
                    f"required_entities (key terms), and alpha (0.0 to 1.0).\n\nQuery: {query}"
                )
                raw_gemini = llm_client.generate(gemini_prompt, json_mode=True, temperature=0.1)
                m = re.search(r"\{.*\}", raw_gemini, re.DOTALL)
                if m:
                    g_data = json.loads(m.group(0))
                    g_subs = g_data.get("sub_queries", [])
                    if isinstance(g_subs, list) and g_subs:
                        sub_queries = list(dict.fromkeys(g_subs + sub_queries))[:4]
                        planning_backend = "gemini_cloud_router"
                        alpha = float(g_data.get("alpha", alpha))
            except Exception as e:
                self.logger.warning(f"[Hybrid Retrieval] Tier 1 Gemini planning failed ({e}), cascading to Tier 2...")

        # =====================================================================
        # Tier 2 (Fallback 1 - GPU Finetune Router LoRA): If Gemini failed/offline
        # =====================================================================
        is_gpu_online = caps.get("gpu_finetune", {}).get("online") or caps.get("gpu_ft", {}).get("online")
        if planning_backend == "local_supervisor" and is_gpu_online and retry_count == 0:
            try:
                router_res = llm_client.call_kaggle_specialist(
                    endpoint="api/router",
                    payload={"query": query, "adapter": "router"},
                    adapter="router",
                )
                lora_plan = RouterPlanOutput.from_dict(router_res, default_query=query)
                for lsq in lora_plan.sub_queries[:2]:
                    if lsq not in sub_queries:
                        sub_queries.append(lsq)
                alpha = lora_plan.alpha
                planning_backend = "gpu_finetune_router_lora"
            except Exception:
                pass

        # Generate STORM research perspectives to augment queries
        perspectives, storm_subs = generate_research_perspectives(query)
        for sq in storm_subs[:2]:
            if sq not in sub_queries:
                sub_queries.append(sq)
        sub_queries = sub_queries[:5]

        # =====================================================================
        # Scope strictly to user-selected PDF(s) if provided
        # =====================================================================
        pdf_chunks = []
        for p in pdf_paths:
            if p:
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

        # Multi-query FTS5 + BM25 search with weighted RRF
        all_lists = []
        list_weights = []
        lexical_weight = max(0.1, 1.0 - alpha)
        expansion_weight = max(0.1, alpha)

        for q in sub_queries:
            res_fts = active_fts.search(q, top_k=top_k)
            res_bm25 = active_bm25.search(q, top_k=top_k)
            if res_fts:
                all_lists.append(res_fts)
                list_weights.append(lexical_weight)
            if res_bm25:
                all_lists.append(res_bm25)
                list_weights.append(lexical_weight)

        fused = reciprocal_rank_fusion(all_lists, top_n=top_k + 4, weights=list_weights) if all_lists else []
        if not fused and active_pool:
            fused = active_pool[: min(top_k, len(active_pool))]

        # Priority reranking via required entities
        if fused and required_entities:
            def _score_fn(idx_doc: tuple[int, dict[str, Any]]) -> float:
                r_idx, d = idx_doc
                txt_l = str(d.get("text", "")).lower()
                ent_hits = sum(1 for e in required_entities if e.lower() in txt_l)
                return (1.0 / (r_idx + 1)) + 0.40 * ent_hits

            indexed = list(enumerate(fused))
            indexed.sort(key=_score_fn, reverse=True)
            fused = [d for _, d in indexed[:top_k]]
        else:
            fused = fused[:top_k]

        seen_ids = {c.get("id") for c in seen_chunks if isinstance(c, dict)}
        fresh_docs = [d for d in fused if d.get("id") not in seen_ids]
        no_new_evidence = len(fresh_docs) == 0 and retry_count > 0

        merged_docs = list(fresh_docs) if fresh_docs else list(fused[:top_k])
        for sc in seen_chunks:
            if isinstance(sc, dict) and sc.get("id") not in {d.get("id") for d in merged_docs}:
                merged_docs.append(sc)

        all_seen = list(seen_chunks) + [d for d in fresh_docs if d.get("id") not in seen_ids]

        trace = self.log_action(
            f"Truy hồi FTS5 + BM25 ({planning_backend}) kết nối {len(merged_docs)} đoạn trích",
            {
                "planning_backend": planning_backend,
                "scoped_to_selected_pdf": bool(pdf_chunks),
                "sub_queries": sub_queries,
                "retrieved_count": len(merged_docs),
                "fresh_count": len(fresh_docs),
            },
        )

        return {
            "perspectives": perspectives,
            "sub_queries": sub_queries,
            "required_entities": required_entities,
            "retrieved_docs": merged_docs[: top_k + 4],
            "seen_chunks": all_seen,
            "no_new_evidence": no_new_evidence,
            "reasoning_trace": [trace],
        }
