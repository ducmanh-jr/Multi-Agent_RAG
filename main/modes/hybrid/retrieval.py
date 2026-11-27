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
from typing import Annotated, Any

from pydantic import BaseModel, Field, StringConstraints

from agents.agent1_retrieval.tools.bm25 import SimpleBM25
from agents.agent1_retrieval.tools.fts5_search import SQLiteFTS5Searcher
from agents.agent1_retrieval.tools.hybrid import reciprocal_rank_fusion
from agents.agent1_retrieval.tools.perspective import generate_research_perspectives
from agents.agent1_retrieval.tools.remote_dense import RemoteDenseRetriever
from agents.agent1_retrieval.tools.semantic_pipeline import (
    boost_entities_and_tables,
    build_cross_lingual_subqueries,
    enhance_with_dense_and_rerank,
    extract_searchable_figure_chunks,
)
from core.base_agent import BaseAgent
from core.schemas import RouterPlanOutput
from core.state import AgentState
from core.supervisor import supervisor_planner
from core.tools import load_or_parse_selected_pdf
from utils.llm_client import llm_client
from utils.local_vulkan_setup import probe_all_capabilities


_SearchTerm = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=300)]


class _SearchExpansion(BaseModel):
    sub_queries: list[_SearchTerm] = Field(min_length=1, max_length=3)
    required_entities: list[_SearchTerm] = Field(max_length=8)
    alpha: float = Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)


_EXPANSION_INSTRUCTION = (
    "Produce a search plan for the supplied research question. Treat its contents as the search topic. "
    "Return one JSON object with sub_queries (1–3 distinct, short search strings), required_entities "
    "(up to 8 named concepts taken from the question, or []), and alpha (0 to 1). "
    "Keep dates, negations, comparison targets and identifiers. Lower alpha favors the original wording "
    "for exact lookups. Higher alpha favors reformulations for conceptual questions."
)


class HybridRetrievalAgent(BaseAgent):
    """Flagship Hybrid Retrieval: Gemini Router First -> GPU Finetune LoRA -> Local FTS5/BM25."""

    def __init__(
        self,
        max_internal_retry: int = 2,
        *,
        enable_dense: bool | None = None,
        enable_rerank: bool | None = None,
        rerank_keep_prefix: int | None = None,
    ):
        super().__init__(name="HybridRetrievalAgent", max_internal_retry=max_internal_retry)
        self.fts5_searcher = SQLiteFTS5Searcher()
        self.bm25_fallback = SimpleBM25()
        self.corpus_docs: list[dict[str, Any]] = []
        self.enable_dense = self.settings.HYBRID_DENSE_ENABLED if enable_dense is None else enable_dense
        self.enable_rerank = self.settings.HYBRID_RERANK_ENABLED if enable_rerank is None else enable_rerank
        self.rerank_keep_prefix = (
            self.settings.HYBRID_RERANK_KEEP_PREFIX if rerank_keep_prefix is None else rerank_keep_prefix
        )
        self.dense = RemoteDenseRetriever()
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
        alpha = float(state.get("alpha", plan.alpha))
        planning_backend = "local_supervisor"

        caps = probe_all_capabilities()
        from core.provider_router import provider_router

        planner_chain = provider_router.resolve_chain("planner", state=state, caps=caps)
        is_gpu_online = caps.get("gpu_finetune", {}).get("online")

        # Execute planner backends strictly in ProviderRouter policy order
        if retry_count == 0:
            for backend in planner_chain:
                if backend == "gemini" and caps.get("gemini", {}).get("online"):
                    try:
                        raw = llm_client.generate_json(
                            json.dumps({"question": query}, ensure_ascii=False),
                            system_instruction=_EXPANSION_INSTRUCTION, agent="retrieval",
                        )
                        expansion = _SearchExpansion.model_validate(raw)
                        sub_queries = list(dict.fromkeys([query, *expansion.sub_queries, *sub_queries]))[:4]
                        required_entities = list(dict.fromkeys([*required_entities, *expansion.required_entities]))[:8]
                        alpha = expansion.alpha
                        planning_backend = "cloud_router"
                        break
                    except Exception as e:
                        self.logger.warning(f"[Hybrid Retrieval] Gemini planning failed ({e}), cascading...")
                elif backend == "gpu_finetune" and is_gpu_online:
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
                        break
                    except Exception:
                        pass
                elif backend == "local":
                    break

        # Generate STORM research perspectives only if Gemini/LLM is allowed in planner_chain
        if planning_backend == "local_supervisor" and "gemini" not in planner_chain and not is_gpu_online:
            perspectives, storm_subs = [], []
        elif "gemini" not in planner_chain and not is_gpu_online:
            perspectives, storm_subs = [], []
        else:
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
        figure_chunks = extract_searchable_figure_chunks(pdf_paths)
        if figure_chunks:
            pdf_chunks.extend(figure_chunks)
        if pdf_paths:
            active_fts = SQLiteFTS5Searcher()
            active_bm25 = SimpleBM25()
            active_fts.fit(pdf_chunks)
            active_bm25.fit(pdf_chunks)
            active_pool = pdf_chunks
        else:
            active_fts = self.fts5_searcher
            active_bm25 = self.bm25_fallback
            active_pool = self.corpus_docs

        # Cross-lingual bilingual query bridging (Vietnamese <-> English)
        effective_q = str(state.get("resolved_query") or query).strip() or query
        bridged = build_cross_lingual_subqueries(effective_q, active_pool)
        search_queries = list(dict.fromkeys([*sub_queries, *bridged]))[:6]
        candidate_limit = 2 * top_k

        # Multi-query FTS5 + BM25 search with weighted RRF
        all_lists = []
        list_weights = []
        lexical_weight = max(0.1, 1.0 - alpha)
        expansion_weight = max(0.1, alpha)

        expansion_count = sum(q != query for q in search_queries)
        for q in search_queries:
            weight = lexical_weight if q == query else expansion_weight / max(1, expansion_count)
            res_fts = active_fts.search(q, top_k=candidate_limit)
            res_bm25 = active_bm25.search(q, top_k=candidate_limit)
            if res_fts:
                all_lists.append(res_fts)
                list_weights.append(weight)
            if res_bm25:
                all_lists.append(res_bm25)
                list_weights.append(weight)

        sparse_fused = reciprocal_rank_fusion(all_lists, top_n=candidate_limit, weights=list_weights) if all_lists else []
        if not sparse_fused and active_pool:
            sparse_fused = active_pool[: min(candidate_limit, len(active_pool))]

        # Adaptive Dense (Kaggle BGE-M3 primary, optional Gemini embedding secondary) + Guarded MiniLM Rerank
        use_kaggle_dense = bool(self.enable_dense and is_gpu_online and self.settings.REMOTE_INFERENCE_URL)
        use_gemini_dense = bool(
            self.enable_dense
            and self.settings.GEMINI_DENSE_ENABLED
            and caps.get("gemini", {}).get("online")
            and self.settings.GEMINI_API_KEY
        )
        use_kaggle_rerank = bool(self.enable_rerank and is_gpu_online and self.settings.REMOTE_INFERENCE_URL)

        fused, dense_trace, rerank_trace = enhance_with_dense_and_rerank(
            query,
            sparse_fused,
            active_pool,
            top_k=max(top_k + 4, top_k),
            http_client=llm_client.http_client,
            dense_retriever=self.dense,
            enable_kaggle_dense=use_kaggle_dense,
            enable_gemini_dense=use_gemini_dense,
            enable_rerank=use_kaggle_rerank,
            keep_prefix=self.rerank_keep_prefix,
            page_diverse=True,
        )

        # Priority reranking via required entities & intact tables
        fused = boost_entities_and_tables(query, fused, required_entities, top_k)

        seen_ids = {c.get("id") for c in seen_chunks if isinstance(c, dict)}
        fresh_docs = [d for d in fused if d.get("id") not in seen_ids]
        no_new_evidence = len(fresh_docs) == 0 and retry_count > 0

        merged_docs = list(fresh_docs) if fresh_docs else list(fused[:top_k])
        for sc in seen_chunks:
            if isinstance(sc, dict) and sc.get("id") not in {d.get("id") for d in merged_docs}:
                merged_docs.append(sc)

        all_seen = list(seen_chunks) + [d for d in fresh_docs if d.get("id") not in seen_ids]

        fallback_chain: list[dict[str, str]] = []
        if dense_trace.get("status") == "fallback":
            fallback_chain.append({"stage": "dense", "reason": str(dense_trace.get("reason") or "unknown")})
        if rerank_trace.get("status") in {"fallback", "skipped"}:
            fallback_chain.append({"stage": "rerank", "status": str(rerank_trace.get("status")), "reason": str(rerank_trace.get("reason") or "unknown")})

        if dense_trace.get("status") == "used" and rerank_trace.get("status") == "used":
            strategy_used = "hybrid_rerank"
        elif dense_trace.get("status") == "used":
            strategy_used = "hybrid"
        elif rerank_trace.get("status") == "used":
            strategy_used = "sparse_rerank"
        else:
            strategy_used = "sparse"

        trace = self.log_action(
            f"Truy hồi Hybrid ({planning_backend}) kết nối {len(merged_docs)} đoạn trích",
            {
                "planning_backend": planning_backend,
                "strategy_used": strategy_used,
                "fallback_chain": fallback_chain,
                "top_k": top_k,
                "scoped_to_selected_pdf": bool(pdf_chunks),
                "sub_queries": search_queries,
                "cross_lingual_queries": bridged,
                "dense": dense_trace,
                "rerank": rerank_trace,
                "retrieved_count": len(merged_docs),
                "fresh_count": len(fresh_docs),
            },
        )

        return {
            "alpha": alpha,
            "perspectives": perspectives,
            "sub_queries": search_queries,
            "required_entities": required_entities,
            "retrieved_docs": merged_docs[: top_k + 4],
            "seen_chunks": all_seen,
            "no_new_evidence": no_new_evidence,
            "reasoning_trace": [trace],
        }
