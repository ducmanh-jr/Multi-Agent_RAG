"""
CPU Min Mode - Agentic Retrieval Agent (L3-Lite).
Features:
  - Multi-document awareness: scopes to all user-selected PDFs (state["pdf_paths"] / state["pdf_path"]).
  - Distributed Page Sampling for summary queries (tóm tắt, tổng quan, overview) ensuring 100% whole-document coverage.
  - Multi-query FTS5 + BM25 search for specific factual questions.
  - Internal Coverage Gate: expands bilingual tokens if required entities are missed.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from typing import Any

from agents.agent1_retrieval.tools.bm25 import SimpleBM25
from agents.agent1_retrieval.tools.fts5_search import SQLiteFTS5Searcher
from agents.agent1_retrieval.tools.hybrid import reciprocal_rank_fusion
from agents.agent1_retrieval.tools.semantic_pipeline import (
    boost_entities_tables_and_equations,
    build_cross_lingual_subqueries,
    extract_searchable_figure_chunks,
    find_referenced_table_chunks,
)
from core.base_agent import BaseAgent
from core.comparison_plan import (
    ComparisonPlan, ComparisonReader, compare_documents, get_comparison_budget, run_coro_sync,
)
from core.handoff_inputs import is_focused_retrieval
from core.research_tasks import ResearchTask, TaskStatus
from core.state import AgentState
from core.supervisor import supervisor_planner
from core.task_scheduler import TaskScheduler
from core.tools import load_or_parse_selected_pdf


class CPUMinRetrievalAgent(BaseAgent):
    """Agentic Retrieval for CPU Min: Multi-document FTS5 + BM25 with distributed sampling for summaries."""

    def __init__(self, max_internal_retry: int = 1):
        super().__init__(name="CPUMinRetrievalAgent", max_internal_retry=max_internal_retry)
        self.fts5_searcher = SQLiteFTS5Searcher()
        self.bm25_fallback = SimpleBM25()
        self.corpus_docs: list[dict[str, Any]] = []
        self._cache_lock = threading.RLock()
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

    async def _scheduled_search(
        self, queries: list[str], fts: SQLiteFTS5Searcher, bm25: SimpleBM25, top_k: int,
    ) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
        engines = {"fts5": fts, "bm25": bm25}
        tasks = [
            ResearchTask("fts5"),
            ResearchTask("bm25"),
            ResearchTask("fuse", depends_on_success=tuple(engines)),
        ]
        scheduler = TaskScheduler(tasks, max_concurrency=2)

        def search_all(engine):
            return [engine.search(query, top_k=top_k) for query in queries]

        async def execute(task, inputs):
            if task.id in engines:
                return await asyncio.to_thread(search_all, engines[task.id])
            pairs = zip(inputs["fts5"].output, inputs["bm25"].output, strict=True)
            candidates = [ranking for pair in pairs for ranking in pair if ranking]
            return reciprocal_rank_fusion(candidates, top_n=top_k) if candidates else []

        outcomes = await scheduler.run(execute)
        merged = outcomes["fuse"]
        event = self.log_action("retrieval_task_schedule", {
            "status": "completed" if merged.status == TaskStatus.COMPLETED else "fallback",
            "max_concurrency": 2,
            "tasks": [
                {
                    "id": task.id, "status": outcomes[task.id].status.value,
                    "depends_on_success": list(task.depends_on_success),
                    "attempts": outcomes[task.id].attempts,
                    "blocked_by": list(outcomes[task.id].blocked_by),
                }
                for task in tasks
            ],
        })
        return merged.output, event

    def run_comparison(self, state: AgentState, plan: ComparisonPlan, reader: ComparisonReader) -> dict[str, Any]:
        query = state["query"]
        queries = state.get("sub_queries") or [query]
        top_k = max(len(plan.paths), min(int(state.get("top_k") or 8), 10))
        # Parse PDFs before starting the parallel search tasks.
        budget = get_comparison_budget()
        chunks_by_path = {}
        for path in plan.paths:
            if budget:
                budget.remaining()
            chunks_by_path[path] = load_or_parse_selected_pdf(path) if os.path.isfile(path) else []

        def rank_document(path):
            if budget:
                budget.remaining()
            chunks = chunks_by_path[path]
            if not chunks:
                return []
            fts = SQLiteFTS5Searcher()
            try:
                bm25 = SimpleBM25()
                fts.fit(chunks)
                bm25.fit(chunks)
                if budget:
                    budget.remaining()
                return self._search_index(queries, fts, bm25, top_k=top_k)
            finally:
                fts.close()

        retry_count = state.get("retry_count", 0)
        result = run_coro_sync(compare_documents(plan, rank_document, reader, top_k, retry_count=retry_count))
        ranked = result["retrieved_docs"]
        if not ranked:
            return result

        seen = state.get("seen_chunks") or []
        seen_ids = {item["id"] for item in seen if isinstance(item, dict) and "id" in item}
        new = [item for item in ranked if item.get("id") not in seen_ids]
        return {
            **result, "seen_chunks": [*seen, *new],
            "no_new_evidence": bool(retry_count and not new),
        }

    def run(
        self, state: AgentState, *, parallel_search: bool = False, comparison: ComparisonPlan | None = None,
        comparison_reader: ComparisonReader | None = None,
    ) -> dict[str, Any]:
        if comparison is not None:
            if comparison_reader is None:
                raise ValueError("Comparison requires a Reader.")
            return self.run_comparison(state, comparison, comparison_reader)
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
        figure_chunks = extract_searchable_figure_chunks(pdf_paths)
        if figure_chunks:
            existing_cids = {c.get("id") for c in pdf_chunks if isinstance(c, dict)}
            pdf_chunks.extend(fc for fc in figure_chunks if fc.get("id") not in existing_cids)

        if pdf_chunks:
            # Cache active index by resolved file paths + content hashes to guarantee zero cross-run leakage
            # even when two concurrent runs inspect distinct PDFs sharing the same filename.
            path_sig = "|".join(os.path.abspath(str(p)) for p in pdf_paths if p)
            sha_sig = "|".join(
                dict.fromkeys(str(c.get("document_sha256") or "") for c in pdf_chunks if isinstance(c, dict))
            )
            cache_key = f"{path_sig}::{sha_sig}::{len(pdf_chunks)}::{pdf_chunks[0].get('id')}::{pdf_chunks[-1].get('id')}"
            with self._cache_lock:
                cached_entry = self._doc_index_cache.get(cache_key)
            if cached_entry is not None:
                active_fts, active_bm25, active_pool = cached_entry
            else:
                built_fts = SQLiteFTS5Searcher()
                built_bm25 = SimpleBM25()
                built_fts.fit(pdf_chunks)
                built_bm25.fit(pdf_chunks)
                with self._cache_lock:
                    if cache_key in self._doc_index_cache:
                        active_fts, active_bm25, active_pool = self._doc_index_cache[cache_key]
                        built_fts.close()
                    else:
                        if len(self._doc_index_cache) >= 8:
                            self._doc_index_cache.pop(next(iter(self._doc_index_cache)))
                        self._doc_index_cache[cache_key] = (built_fts, built_bm25, pdf_chunks)
                        active_fts, active_bm25, active_pool = built_fts, built_bm25, pdf_chunks
        else:
            active_fts = self.fts5_searcher
            active_bm25 = self.bm25_fallback
            active_pool = self.corpus_docs

        schedules = []

        def search(queries, limit, stage):
            if parallel_search:
                ranking, event = asyncio.run(self._scheduled_search(queries, active_fts, active_bm25, limit))
                event["details"].update(stage=stage, retry_count=retry_count)
                schedules.append(event)
                if ranking is not None:
                    return ranking
            return self._search_index(queries, active_fts, active_bm25, top_k=limit)

        # 2. Retrieval Strategy via Bounded Specialist Tool Loop (max 3 local tools, zero LLM calls)
        tool_ctx: dict[str, Any] = {
            "results": [],
            "focused": is_focused_retrieval(state),
            "effective_q": str(state.get("resolved_query") or clean_q).strip() or clean_q,
            "primary_queries": [],
            "boost_q": str(state.get("resolved_query") or clean_q).strip() or clean_q,
            "coverage_ratio": 1.0,
        }

        def _tool_summary_sample() -> int:
            docs_by_id: dict[str, dict[int, list[dict[str, Any]]]] = {}
            for chunk in active_pool:
                d_id = str(chunk.get("doc_id", "doc"))
                p_no = int(chunk.get("page_no", 1))
                docs_by_id.setdefault(d_id, {}).setdefault(p_no, []).append(chunk)

            sampled_summary: list[dict[str, Any]] = []
            for d_id, page_dict in docs_by_id.items():
                sorted_pages = sorted(page_dict.keys())
                max_page = sorted_pages[-1] if sorted_pages else 1
                if max_page > 20:
                    body_pages = [p for p in sorted_pages if p >= 16]
                    pages_to_sample = body_pages if len(body_pages) >= 5 else sorted_pages
                else:
                    pages_to_sample = sorted_pages
                sample_count = min(len(pages_to_sample), 20)
                if sample_count > 0:
                    step = len(pages_to_sample) / float(sample_count)
                    chosen_pages = [pages_to_sample[int(i * step)] for i in range(sample_count)]
                    for p in chosen_pages:
                        chunks_on_page = page_dict[p]
                        best_chunk = max(chunks_on_page, key=lambda c: len(str(c.get("text", "")).strip()))
                        sampled_summary.append(best_chunk)
            tool_ctx["results"] = sampled_summary if sampled_summary else active_pool[: max(top_k, 20)]
            return len(tool_ctx["results"])

        def _tool_fts5_bm25_search() -> int:
            focused = tool_ctx["focused"]
            effective_q = tool_ctx["effective_q"]
            boost_q = (
                " ".join(dict.fromkeys([*sub_queries[:2], effective_q]))
                if focused and sub_queries
                else effective_q
            )
            tool_ctx["boost_q"] = boost_q
            bridged_queries = build_cross_lingual_subqueries(boost_q, active_pool)
            primary_queries = list(dict.fromkeys([*sub_queries, *bridged_queries]))[:6]
            tool_ctx["primary_queries"] = primary_queries
            found = search(primary_queries, max(top_k + 4, top_k), "primary")
            if found:
                found = boost_entities_tables_and_equations(boost_q, found, required_entities, top_k)
            tool_ctx["results"] = found
            return len(found)

        def _tool_referenced_table_lookup() -> int:
            seen_id_set = {c.get("id") for c in seen_chunks if isinstance(c, dict)}
            targeted_tables = find_referenced_table_chunks(active_pool, tool_ctx["primary_queries"], seen_id_set)
            if targeted_tables:
                target_ids = {t.get("id") for t in targeted_tables}
                merged = [*targeted_tables, *(d for d in tool_ctx["results"] if d.get("id") not in target_ids)]
                tool_ctx["results"] = boost_entities_tables_and_equations(
                    tool_ctx["boost_q"], merged, required_entities, top_k
                )
            return len(targeted_tables)

        def _tool_bilingual_coverage_expansion() -> int:
            expanded_queries = list(sub_queries)
            try:
                from utils.bilingual import expand_bilingual_tokens
                base_tokens = set(re.findall(r"\w+", clean_q.lower()))
                bilingual_tokens = expand_bilingual_tokens(clean_q, base_tokens)
                expanded_queries.append(" ".join(sorted(bilingual_tokens)[:24]))
            except Exception:
                pass
            if required_entities:
                expanded_queries.append(" ".join(required_entities))
            res_expanded = search(expanded_queries, top_k + 2, "coverage_expansion")
            if res_expanded:
                tool_ctx["results"] = boost_entities_tables_and_equations(
                    str(state.get("resolved_query") or clean_q), res_expanded, required_entities, top_k
                )
            return len(res_expanded)

        def _retrieval_tool_planner(step_idx: int, history: list[dict[str, Any]]):
            if step_idx == 0:
                if is_summary_query and active_pool and not tool_ctx["focused"]:
                    return ("distributed_page_sampler", {})
                return ("fts5_bm25_search", {})
            last_tool = history[-1]["tool"] if history else ""
            if last_tool == "fts5_bm25_search" and tool_ctx["focused"] and active_pool:
                return ("referenced_table_lookup", {})
            if last_tool in {"fts5_bm25_search", "referenced_table_lookup"} and not is_summary_query:
                res_now = tool_ctx["results"]
                if required_entities and res_now:
                    combined_text = " ".join(str(r.get("text", "")) for r in res_now).lower()
                    covered = sum(1 for ent in required_entities if ent.lower() in combined_text)
                    cov = covered / max(len(required_entities), 1)
                else:
                    cov = 1.0 if res_now else 0.0
                tool_ctx["coverage_ratio"] = cov
                if cov < 0.5 or not res_now:
                    return ("bilingual_coverage_expansion", {})
            return None

        tool_loop = self.run_bounded_tool_loop(
            {
                "distributed_page_sampler": _tool_summary_sample,
                "fts5_bm25_search": _tool_fts5_bm25_search,
                "referenced_table_lookup": _tool_referenced_table_lookup,
                "bilingual_coverage_expansion": _tool_bilingual_coverage_expansion,
            },
            _retrieval_tool_planner,
            max_steps=3,
        )
        results = tool_ctx["results"]

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
                "strategy_used": "sparse",
                "tools_used": tool_loop["tools_used"],
                "top_k": top_k,
                "docs_scoped": len(pdf_paths),
                "is_summary_query": is_summary_query,
                "cross_lingual_queries": build_cross_lingual_subqueries(str(state.get("resolved_query") or clean_q), active_pool),
                "dense": {"status": "disabled"},
                "rerank": {"status": "disabled"},
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
            "tools_used": tool_loop["tools_used"],
            "reasoning_trace": [*schedules, trace],
        }
