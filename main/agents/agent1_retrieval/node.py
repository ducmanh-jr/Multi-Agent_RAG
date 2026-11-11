"""
Agent 1: Retrieval Agent (Multi-Perspective Hybrid RAG).
Combines:
  1. Stanford STORM Multi-Perspective Query Decomposition.
  2. Hybrid BM25 + Dense RRF Fusion.
  3. Iterative Evidence Gap Re-retrieval.
  4. Cumulative seen_chunks preservation.
"""

from __future__ import annotations

import json
import os
from typing import Any

from core.base_agent import BaseAgent
from core.state import AgentState
from agents.agent1_retrieval.tools.bm25 import SimpleBM25
from agents.agent1_retrieval.tools.fts5_search import SQLiteFTS5Searcher
from agents.agent1_retrieval.tools.hybrid import reciprocal_rank_fusion
from agents.agent1_retrieval.tools.perspective import generate_research_perspectives
from agents.agent1_retrieval.tools.router import make_decision


class RetrievalAgent(BaseAgent):
    """Retrieval Agent handling query routing, multi-perspective expansion, and hybrid search."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="RetrievalAgent", max_internal_retry=max_internal_retry)
        self.fts5_searcher = SQLiteFTS5Searcher()
        self.bm25_fallback = SimpleBM25()
        self._corpus_loaded = False
        self._load_corpus()

    def _load_corpus(self):
        """Loads corpus from data/corpus.jsonl if present."""
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
                self.logger.warning(f"Could not load corpus from {corpus_path}: {e}")

        if docs:
            self.fts5_searcher.fit(docs)
            self.bm25_fallback.fit(docs)
            self._corpus_loaded = True
            self.logger.info(f"RetrievalAgent: Indexed {len(docs)} documents into C-Speed FTS5 engine.")
        else:
            self.logger.info("RetrievalAgent: Kho dữ liệu trống. Đang chờ tài liệu từ người dùng hoặc Scout Agent.")

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        retry_count = state.get("retry_count", 0)
        evidence_gaps = state.get("evidence_gaps", [])
        seen_chunks = state.get("seen_chunks", []) or []

        self.logger.info(f"Running Retrieval for query: '{query}' (retry: {retry_count})")

        # 1. Routing decision (integrating Agent 1 fine-tuned router policy)
        router_decision = make_decision(query, history_trials=[{"is_relevant": False}] if retry_count > 0 else [])
        chosen_tool = router_decision.get("tool_selected", "dense")
        self.logger.info(f"Router Policy selected: '{chosen_tool}' - {router_decision.get('thought')}")

        # 2. Query expansion based on chosen strategy
        perspectives = []
        sub_queries = []

        if chosen_tool == "decompose":
            perspectives, sub_queries = generate_research_perspectives(query)
        elif chosen_tool == "bm25":
            perspectives = ["Tra cứu từ khóa chính xác"]
            sub_queries = [query]
        elif chosen_tool == "hyde":
            perspectives = ["Giải thích cơ chế chuyên sâu (HyDE)"]
            sub_queries = [query, f"cơ chế {query}", f"nguyên lý {query}"]
        elif chosen_tool == "rewrite":
            perspectives = ["Mở rộng truy vấn làm rõ ngữ nghĩa"]
            sub_queries = [query, f"tổng quan {query}"]
        else:
            # Default / dense
            perspectives, sub_queries = generate_research_perspectives(query)

        # Incorporate queries from evidence gaps if retrying
        for gap in evidence_gaps:
            suggested = gap.get("suggested_query")
            if suggested and suggested not in sub_queries:
                sub_queries.append(suggested)

        # 3. Execute retrieval per sub-query using native C FTS5 with BM25 fallback
        ranked_runs = []
        for q in sub_queries:
            matches = self.fts5_searcher.search(q, top_k=self.settings.MAX_RETRIEVED_CHUNKS)
            if not matches:
                matches = self.bm25_fallback.search(q, top_k=self.settings.MAX_RETRIEVED_CHUNKS)
            if matches:
                ranked_runs.append(matches)

        # 4. Fallback to ArXiv Internet Search if local corpus has no matches
        if not ranked_runs:
            try:
                from agents.agent1_retrieval.tools.arxiv_search import search_arxiv
                arxiv_docs = search_arxiv(query, max_results=self.settings.TOP_K_RERANK)
                if arxiv_docs:
                    ranked_runs.append(arxiv_docs)
                    self.fts5_searcher.add_documents(arxiv_docs)
            except Exception as e:
                self.logger.warning(f"ArXiv live search error: {e}")

        # 5. Hybrid Reciprocal Rank Fusion
        if not ranked_runs:
            fused_docs = []
            no_evidence = True
            self.logger.warning("[Retrieval] Không tìm thấy bằng chứng trong kho tài liệu hoặc arXiv. Báo cáo trung thực no_evidence=True.")
        else:
            fused_docs = reciprocal_rank_fusion(
                ranked_runs,
                top_n=self.settings.TOP_K_RERANK,
            )
            no_evidence = False

        # 6. Update cumulative seen_chunks to prevent losing contradiction sources
        updated_seen_chunks = list(seen_chunks)
        seen_hashes = {c.get("context_hash") for c in updated_seen_chunks if "context_hash" in c}
        new_chunks_added = 0
        for doc in fused_docs:
            h = doc.get("context_hash")
            if h and h not in seen_hashes:
                updated_seen_chunks.append(doc)
                seen_hashes.add(h)
                new_chunks_added += 1

        no_new_evidence = (retry_count >= 1 and new_chunks_added == 0)
        if no_new_evidence:
            self.logger.warning("[Retrieval] Early Stopping triggered: 0 new chunks discovered in retry round.")

        trace = self.log_action(
            "hybrid_retrieval_completed",
            {
                "tool_selected": chosen_tool,
                "router_thought": router_decision.get("thought"),
                "sub_queries": sub_queries,
                "fused_count": len(fused_docs),
                "total_seen": len(updated_seen_chunks),
                "new_chunks_added": new_chunks_added,
                "no_new_evidence": no_new_evidence,
            },
        )

        return {
            "sub_queries": sub_queries,
            "perspectives": perspectives,
            "retrieved_docs": fused_docs,
            "seen_chunks": updated_seen_chunks,
            "retrieval_strategy": f"{chosen_tool}_hybrid_rrf",
            "router_decision": router_decision,
            "no_new_evidence": no_new_evidence,
            "reasoning_trace": [trace],
        }
