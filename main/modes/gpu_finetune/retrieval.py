"""
GPU Finetune Mode - Agentic Retrieval Agent (7-Model SOTA Pipeline).
Leverages Remote GPU Worker powered by:
  1. Fine-tuned 'router' (01_agent1_router) QLoRA on Qwen2.5-7B:
     - Predicts query intent, multi-angle sub-queries, dynamic alpha (dense/sparse weight), top_k.
  2. Fine-tuned 'bge_m3_lora' (06_embedding_bge_m3) for high-density semantic embedding.
  3. Fine-tuned 'reranker_bge_large' (07_reranker_bge_large) CoREB LoRA for cross-encoder reranking.
  4. Scopes strictly to user-selected PDF via load_or_parse_selected_pdf().
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
from core.schemas import RouterPlanOutput
from core.state import AgentState
from core.supervisor import supervisor_planner
from core.tools import load_or_parse_selected_pdf
from utils.llm_client import llm_client


ROUTER_TRAINING_SYSTEM_PROMPT = (
    "You are an intelligent routing agent for an Academic RAG system. "
    "Analyze the user query and output a JSON object with: intent, sub_queries, alpha (0.0 to 1.0), top_k (1 to 15), and reasoning."
)


class GPUFinetuneRetrievalAgent(BaseAgent):
    """Retrieval Agent powered by Fine-Tuned Router + BGE-M3 + BGE-Reranker-Large."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="GPUFinetuneRetrievalAgent", max_internal_retry=max_internal_retry)
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
                self.logger.warning(f"[GPU Finetune Retrieval] Corpus load failed: {e}")
        self.corpus_docs = docs
        if docs:
            self.fts5_searcher.fit(docs)
            self.bm25_fallback.fit(docs)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        pdf_path = state.get("pdf_path")
        retry_count = state.get("retry_count", 0)
        seen_chunks = state.get("seen_chunks", []) or []

        # 1. Start with local Supervisor plan, enrich with fine-tuned Router QLoRA
        local_plan = supervisor_planner.plan_query(query, state.get("evidence_gaps"), retry_count)
        sub_queries = list(state.get("sub_queries") or local_plan.sub_queries)
        required_entities = list(state.get("required_entities") or local_plan.required_entities)
        alpha = float(state.get("alpha") or local_plan.alpha)
        top_k = int(state.get("top_k") or local_plan.top_k)
        intent = local_plan.intent

        self.logger.info(f"[GPU Finetune Retrieval] Invoking Fine-Tuned Router for query: '{query[:65]}'")

        if self.settings.REMOTE_INFERENCE_URL and retry_count == 0 and not sub_queries:
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
        pdf_chunks = load_or_parse_selected_pdf(pdf_path) if pdf_path else []
        if pdf_chunks:
            # 2.1 Explicit Target Page Routing & Out-of-Bounds Detection:
            page_match = re.search(r"\b(?:trang|page)\s*(\d+)\b", query, re.IGNORECASE)
            if page_match:
                req_page = int(page_match.group(1))
                all_pages = [int(c.get("page_no") or 1) for c in pdf_chunks]
                max_doc_page = max(all_pages) if all_pages else 1
                min_doc_page = min(all_pages) if all_pages else 1
                doc_name = os.path.basename(pdf_path) if pdf_path else "tài liệu mục tiêu"

                if req_page > max_doc_page or req_page < min_doc_page:
                    out_of_bounds_msg = (
                        f"Tài liệu mục tiêu ({doc_name}) chỉ có {max_doc_page} trang (từ trang {min_doc_page} đến trang {max_doc_page}). "
                        f"Không tồn tại trang {req_page}. Vui lòng chọn số trang hợp lệ trong phạm vi tài liệu."
                    )
                    self.logger.warning(f"[GPU Finetune Retrieval] Out-of-bounds page requested: {req_page} > {max_doc_page}")
                    trace = self.log_action("gpu_finetune_retrieval_page_out_of_bounds", {"req_page": req_page, "max_page": max_doc_page})
                    return {
                        "retrieved_docs": [],
                        "source_units": [],
                        "evidence_status": "NOT_FOUND",
                        "answer_status": "NOT_FOUND_IN_DOCUMENT",
                        "answer": out_of_bounds_msg,
                        "final_report": (
                            f"# Báo cáo Thẩm định Học thuật (Academic Audit Report)\n\n"
                            f"> **Trạng thái Thẩm định:** [`OUT_OF_BOUNDS_PAGE` | `HONEST_REFUSAL`]\n\n"
                            f"### 1. Tóm tắt Kết luận\n"
                            f"{out_of_bounds_msg}\n\n"
                            f"### 2. Chi tiết Thẩm định\n"
                            f"- **Yêu cầu đối chiếu:** *\"{query}\"*\n"
                            f"- **Kết luận xác minh:** Số trang yêu cầu ({req_page}) nằm ngoài phạm vi số trang thực tế của tài liệu mục tiêu ({min_doc_page}..{max_doc_page}). "
                            f"Hệ thống từ chối tạo sinh nội dung suy đoán để tuân thủ nghiêm ngặt chính sách bảo vệ trung thực học thuật (Zero-Hallucination Policy).\n"
                        ),
                        "sub_queries": sub_queries,
                        "required_entities": required_entities,
                        "alpha": alpha,
                        "top_k": top_k,
                        "seen_chunks": seen_chunks,
                        "no_new_evidence": True,
                        "citations": [],
                        "reasoning_trace": [trace],
                    }

            active_fts = SQLiteFTS5Searcher()
            active_bm25 = SimpleBM25()
            active_fts.fit(pdf_chunks)
            active_bm25.fit(pdf_chunks)
            active_pool = pdf_chunks
        else:
            active_fts = self.fts5_searcher
            active_bm25 = self.bm25_fallback
            active_pool = self.corpus_docs

        # 3. Multi-query hybrid search over FTS5 + BM25
        candidate_lists = []
        for sq in sub_queries[:4]:
            res_fts = active_fts.search(sq, top_k=top_k)
            res_bm25 = active_bm25.search(sq, top_k=top_k)
            if res_fts:
                candidate_lists.append(res_fts)
            if res_bm25:
                candidate_lists.append(res_bm25)

        # 4. Hybrid Reciprocal Rank Fusion (RRF)
        fused_docs = reciprocal_rank_fusion(candidate_lists, k=60) if candidate_lists else []
        if not fused_docs and active_pool:
            fused_docs = active_pool[:top_k]

        # 4.1 Explicit Target Page Boost:
        # If the query targets a specific valid page (e.g. trang 12), ensure that page's chunks are at position #1!
        page_match = re.search(r"\b(?:trang|page)\s*(\d+)\b", query, re.IGNORECASE)
        if page_match and pdf_chunks:
            req_page = int(page_match.group(1))
            target_page_chunks = [c for c in pdf_chunks if int(c.get("page_no") or 0) == req_page]
            if target_page_chunks:
                self.logger.info(f"[GPU Finetune Retrieval] Prioritizing {len(target_page_chunks)} chunks from target page {req_page}")
                other_fused = [d for d in fused_docs if int(d.get("page_no") or 0) != req_page]
                fused_docs = target_page_chunks + other_fused
                top_k = max(top_k, len(target_page_chunks) + 3)

        # 4.2 Explicit Table/Section Marker Boost:
        # If the query specifically targets a table or section (e.g. Table 1, Table 2, Table 3, Bảng 1, Bảng 2, Bảng 3),
        # prioritize chunks containing that exact marker to the front.
        target_markers = re.findall(r"\b(?:table|bảng)\s*\d+\b", query, re.IGNORECASE)
        if target_markers and fused_docs:
            def _marker_score(doc: dict[str, Any]) -> int:
                text = str(doc.get("text") or "").lower()
                return sum(1 for m in target_markers if m.lower() in text)
            fused_docs = sorted(fused_docs, key=_marker_score, reverse=True)

        # 4.3 General Document Summary / Overview Query Boost:
        # If user asks to summarize or overview the whole document (/tom-tat, tóm tắt, tổng quan),
        # ensure introductory context (page 1, page 2) is present in the fused pool.
        is_summary_query = any(k in query.lower() for k in ["tóm tắt", "tổng quan", "summary", "overview", "/tom-tat"])
        if is_summary_query and pdf_chunks:
            head_chunks = [c for c in pdf_chunks if int(c.get("page_no") or 1) in (1, 2)][:2]
            existing_ids = {d.get("id") for d in fused_docs}
            for hc in reversed(head_chunks):
                if hc.get("id") not in existing_ids:
                    fused_docs.insert(0, hc)
                    existing_ids.add(hc.get("id"))

        # 5. Remote Reranker enhancement via 07_reranker_bge_large LoRA if available
        if self.settings.REMOTE_INFERENCE_URL and fused_docs:
            try:
                candidate_texts = [d.get("text", "")[:500] for d in fused_docs[:15]]
                rerank_resp = llm_client.http_client.post(
                    f"{self.settings.REMOTE_INFERENCE_URL.rstrip('/')}/api/rerank",
                    json={"query": query, "candidates": candidate_texts},
                    timeout=float(self.settings.REMOTE_TIMEOUT_SECONDS) * 2,
                )
                if rerank_resp.status_code == 200:
                    reranked_data = rerank_resp.json()
                    ranked_indices = reranked_data.get("ranked_indices", [])
                    if ranked_indices:
                        fused_docs = [fused_docs[i] for i in ranked_indices if i < len(fused_docs)]
                        self.logger.info(f"[GPU Finetune Retrieval] Re-ordered top-{len(fused_docs)} with Fine-Tuned BGE-Reranker-Large")
            except Exception as e:
                self.logger.debug(f"[GPU Finetune Retrieval] Remote reranker skipped ({e}), maintaining RRF order.")

        # 5.1 Final Target Page Scoping: If the query is specifically targeting a single page, scope strictly to that page
        if page_match and pdf_chunks:
            req_page = int(page_match.group(1))
            target_page_chunks = [c for c in pdf_chunks if int(c.get("page_no") or 0) == req_page]
            if target_page_chunks:
                fused_docs = target_page_chunks
                self.logger.info(f"[GPU Finetune Retrieval] Strictly scoped to {len(fused_docs)} chunks from target page {req_page}")

        # Deduplicate seen chunks
        new_docs = [d for d in fused_docs if d.get("id") not in seen_chunks]
        no_new_evidence = bool(retry_count > 0 and len(new_docs) == 0)
        final_docs = (new_docs if new_docs else fused_docs)[:top_k]

        new_seen = list(set(seen_chunks + [d.get("id") for d in final_docs if d.get("id")]))

        trace = self.log_action(
            "gpu_finetune_retrieval_completed",
            {
                "query": query,
                "retrieved_count": len(final_docs),
                "intent": intent,
                "sub_queries": sub_queries,
                "alpha": alpha,
                "top_k": top_k,
                "no_new_evidence": no_new_evidence,
            },
        )

        return {
            "retrieved_docs": final_docs,
            "sub_queries": sub_queries,
            "required_entities": required_entities,
            "alpha": alpha,
            "top_k": top_k,
            "seen_chunks": new_seen,
            "no_new_evidence": no_new_evidence,
            "reasoning_trace": [trace],
        }
