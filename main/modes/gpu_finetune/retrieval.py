"""
GPU Finetune Mode - Agentic Retrieval Agent (7-Model SOTA Pipeline).
Leverages Remote GPU Worker powered by:
  1. Fine-tuned 'router' (01_agent1_router) QLoRA on Qwen2.5-7B:
     - Predicts query intent, multi-angle sub-queries, dynamic alpha (dense/sparse weight), top_k.
  2. A versioned BGE-M3 dense encoder with FTS5/BM25 fallback.
  3. A versioned BGE cross-encoder for optional reranking.
  4. Scopes strictly to user-selected PDF via load_or_parse_selected_pdf().
"""

from __future__ import annotations

import json
import os
from typing import Any

from agents.agent1_retrieval.tools.bm25 import SimpleBM25
from agents.agent1_retrieval.tools.fts5_search import SQLiteFTS5Searcher
from agents.agent1_retrieval.tools.hybrid import reciprocal_rank_fusion
from agents.agent1_retrieval.tools.remote_dense import (
    DEFAULT_DENSE_RRF_K,
    DEFAULT_DENSE_WEIGHT,
    DenseRetrievalError,
    RemoteDenseRetriever,
)
from agents.agent1_retrieval.tools.remote_rerank import MAX_CANDIDATES, RerankError, rerank_passages
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


ROUTER_TRAINING_SYSTEM_PROMPT = (
    "You are an intelligent routing agent for an Academic RAG system. "
    "Analyze the user query and output a JSON object with: intent, sub_queries, alpha (0.0 to 1.0), top_k (1 to 15), and reasoning."
)


def _prefer_distinct_pages(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first hit per document/page ahead of lower-ranked passages from that page."""
    seen_pages = set()
    first_hits = []
    remaining = []
    for chunk in chunks:
        document = chunk.get("doc_id")
        page = chunk.get("page_no") or chunk.get("page")
        if document and page:
            key = (document, page)
            if key in seen_pages:
                remaining.append(chunk)
                continue
            seen_pages.add(key)
        first_hits.append(chunk)
    return first_hits + remaining


class GPUFinetuneRetrievalAgent(BaseAgent):
    """Scoped retrieval with optional dense embeddings and cross-encoder reranking."""

    def __init__(
        self, max_internal_retry: int = 2, *, enable_dense: bool | None = None,
        enable_remote_router: bool = True, enable_rerank: bool | None = None, dense_weight: float = DEFAULT_DENSE_WEIGHT,
        dense_rrf_k: int = DEFAULT_DENSE_RRF_K, page_diverse: bool | None = None,
        rerank_keep_prefix: int | None = None,
    ):
        super().__init__(name="GPUFinetuneRetrievalAgent", max_internal_retry=max_internal_retry)
        self.fts5_searcher = SQLiteFTS5Searcher()
        self.bm25_fallback = SimpleBM25()
        self.corpus_docs: list[dict[str, Any]] = []
        if enable_dense is None:
            enable_dense = self.settings.GPU_FINETUNE_DENSE_ENABLED
        self.dense = RemoteDenseRetriever() if enable_dense else None
        self.enable_remote_router = enable_remote_router
        self.enable_rerank = self.settings.GPU_FINETUNE_RERANK_ENABLED if enable_rerank is None else enable_rerank
        self.rerank_keep_prefix = (
            self.settings.GPU_FINETUNE_RERANK_KEEP_PREFIX if rerank_keep_prefix is None else rerank_keep_prefix
        )
        if not 0 <= self.rerank_keep_prefix < MAX_CANDIDATES:
            raise ValueError("Reranker protected prefix is outside the supported candidate budget")
        self.dense_weight = dense_weight
        self.dense_rrf_k = dense_rrf_k
        self.page_diverse = self.settings.GPU_FINETUNE_PAGE_DIVERSITY_ENABLED if page_diverse is None else page_diverse
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
                self.logger.warning(f"[GPU Finetune Retrieval] Corpus load failed: {e}")
        self.corpus_docs = docs
        if docs:
            self.fts5_searcher.fit(docs)
            self.bm25_fallback.fit(docs)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        pdf_path = state.get("pdf_path")
        pdf_paths = state.get("pdf_paths") or ([pdf_path] if pdf_path else [])
        retry_count = state.get("retry_count", 0)
        seen_chunks = state.get("seen_chunks", []) or []

        # 1. Start with local Supervisor plan, enrich with fine-tuned Router QLoRA
        local_plan = supervisor_planner.plan_query(query, state.get("evidence_gaps"), retry_count)
        sub_queries = list(state.get("sub_queries") or local_plan.sub_queries)
        required_entities = list(state.get("required_entities") or local_plan.required_entities)
        alpha = float(state.get("alpha") or local_plan.alpha)
        top_k = int(state.get("top_k") or local_plan.top_k)
        intent = local_plan.intent

        if self.enable_remote_router and self.settings.REMOTE_INFERENCE_URL and retry_count == 0 and not sub_queries:
            try:
                router_res = llm_client.call_kaggle_specialist(
                    endpoint="api/router",
                    payload={"query": query, "adapter": "router"},
                    fallback_prompt=query,
                    fallback_system=ROUTER_TRAINING_SYSTEM_PROMPT,
                    adapter="router",
                    timeout=20.0,
                )
                lora_plan = RouterPlanOutput.from_dict(router_res, default_query=query)
                if lora_plan.sub_queries:
                    merged_subs = list(dict.fromkeys(lora_plan.sub_queries + sub_queries))[:4]
                    sub_queries = merged_subs
                alpha = lora_plan.alpha
                top_k = lora_plan.top_k
                intent = lora_plan.intent
                self.logger.info(f"[GPU Finetune Retrieval] Router fine-tuned plan: intent={intent}, sub_queries={len(sub_queries)}, alpha={alpha:.2f}")
            except Exception as e:
                self.logger.warning(f"[GPU Finetune Retrieval] Remote router fallback to Supervisor ({e})")

        # 2. Scope strictly to user-selected PDF if provided
        pdf_chunks = []
        for path in dict.fromkeys(pdf_paths):
            pdf_chunks.extend(load_or_parse_selected_pdf(path))
        figure_chunks = extract_searchable_figure_chunks(pdf_paths)
        if figure_chunks:
            existing_cids = {c.get("id") for c in pdf_chunks if isinstance(c, dict)}
            pdf_chunks.extend(fc for fc in figure_chunks if fc.get("id") not in existing_cids)
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

        # 3. Multi-query hybrid search over FTS5 + BM25 (with VI<->EN & structural bridging)
        effective_query = str(state.get("resolved_query") or query).strip() or query
        bridged_queries = build_cross_lingual_subqueries(effective_query, active_pool)
        search_queries = list(dict.fromkeys([*sub_queries[:4], *bridged_queries]))[:6]
        candidate_limit = 2 * top_k if self.page_diverse else top_k
        candidate_lists = []
        for sq in search_queries:
            res_fts = active_fts.search(sq, top_k=candidate_limit)
            res_bm25 = active_bm25.search(sq, top_k=candidate_limit)
            if res_fts:
                candidate_lists.append(res_fts)
            if res_bm25:
                candidate_lists.append(res_bm25)

        # 4. Fuse the lexical rankings, then combine them with scoped dense results.
        fused_docs = reciprocal_rank_fusion(candidate_lists, k=60, top_n=candidate_limit) if candidate_lists else []
        dense_trace = {"status": "disabled"}
        sparse_hashes = None
        if self.dense is not None:
            try:
                if not self.settings.REMOTE_INFERENCE_URL:
                    raise DenseRetrievalError("embedding_worker_not_configured")
                dense_docs, revision = self.dense.rank(
                    query, active_pool, top_k=candidate_limit, client=llm_client.http_client,
                    url=self.settings.REMOTE_INFERENCE_URL, model=self.settings.EMBEDDING_MODEL,
                    timeout=float(self.settings.REMOTE_TIMEOUT_SECONDS),
                )
            except DenseRetrievalError as exc:
                dense_trace = {"status": "fallback", "reason": str(exc)}
                self.logger.warning("[GPU Finetune Retrieval] Dense fallback to Sparse: %s", exc)
            else:
                sparse_hashes = {doc["context_hash"] for doc in fused_docs}
                fusion_limit = 2 * candidate_limit if self.page_diverse else top_k
                fused_docs = reciprocal_rank_fusion(
                    [fused_docs, dense_docs], k=self.dense_rrf_k, top_n=fusion_limit, weights=[1.0, self.dense_weight],
                )
                dense_trace = {"status": "used", "model": self.settings.EMBEDDING_MODEL, "revision": revision}
        if not fused_docs and active_pool:
            fused_docs = active_pool[:candidate_limit]
        # Before page-diversity deduplication, ensure structured table/equation/figure chunks
        # win their page's primary slot over unstructured plain-text page dumps!
        if any(
            str(d.get("unit_type") or "") in {"equation", "table", "figure", "ocr_scan"}
            for d in fused_docs
        ):
            fused_docs = boost_entities_tables_and_equations(
                effective_query, fused_docs, required_entities, len(fused_docs)
            )
        if self.page_diverse:
            unseen_docs = [doc for doc in fused_docs if doc.get("id") not in seen_chunks]
            fused_docs = _prefer_distinct_pages(unseen_docs or fused_docs)[:top_k]

        # Deduplicate seen chunks
        final_docs = [d for d in fused_docs if d.get("id") not in seen_chunks][:top_k]
        if not final_docs and fused_docs:
            final_docs = fused_docs[:top_k]
        if sparse_hashes is not None:
            dense_trace["admitted_passages"] = sum(doc["context_hash"] not in sparse_hashes for doc in final_docs)

        rerank_trace = {"status": "disabled"}
        if self.enable_rerank:
            if dense_trace["status"] == "fallback":
                rerank_trace = {"status": "skipped", "reason": "dense_fallback"}
            elif min(len(final_docs), MAX_CANDIDATES) - self.rerank_keep_prefix < 2:
                rerank_trace = {"status": "skipped", "reason": "insufficient_candidates"}
            else:
                try:
                    if not self.settings.REMOTE_INFERENCE_URL:
                        raise RerankError("reranker_not_configured")
                    final_docs, revision, changed = rerank_passages(
                        query, final_docs, client=llm_client.http_client,
                        url=self.settings.REMOTE_INFERENCE_URL, model=self.settings.RERANKER_MODEL,
                        timeout=float(self.settings.REMOTE_TIMEOUT_SECONDS),
                        keep_prefix=self.rerank_keep_prefix,
                    )
                except RerankError as exc:
                    rerank_trace = {"status": "fallback", "reason": str(exc)}
                    self.logger.warning("[GPU Finetune Retrieval] Keeping original ranking: %s", exc)
                else:
                    rerank_trace = {
                        "status": "used", "model": self.settings.RERANKER_MODEL,
                        "revision": revision, "changed": changed,
                        "kept_prefix": self.rerank_keep_prefix,
                    }

        if any(
            str(d.get("unit_type") or "") in {"equation", "table", "figure", "ocr_scan"}
            for d in final_docs
        ) and dense_trace.get("status") != "used" and rerank_trace.get("status") != "used":
            final_docs = boost_entities_tables_and_equations(effective_query, final_docs, required_entities, top_k)

        new_seen = list(set(seen_chunks + [d.get("id") for d in final_docs if d.get("id")]))

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
            "gpu_finetune_retrieval_completed",
            {
                "query": query,
                "retrieved_count": len(final_docs),
                "intent": intent,
                "sub_queries": sub_queries,
                "cross_lingual_queries": bridged_queries,
                "strategy_used": strategy_used,
                "fallback_chain": fallback_chain,
                "alpha": alpha,
                "top_k": top_k,
                "dense": dense_trace,
                "rerank": rerank_trace,
            },
        )

        return {
            "retrieved_docs": final_docs,
            "sub_queries": sub_queries,
            "required_entities": required_entities,
            "alpha": alpha,
            "top_k": top_k,
            "seen_chunks": new_seen,
            "reasoning_trace": [trace],
        }
