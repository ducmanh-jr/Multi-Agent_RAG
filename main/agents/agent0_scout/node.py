"""
Agent 0: Autonomous Literature Scout Node.
Orchestrates autonomous paper discovery, intent analysis, query expansion, multi-source search,
enforces bidirectional relevance filtering, and downloads primary PDF for zero-click research workflows.
"""

from __future__ import annotations

import os
from typing import Any

from core.base_agent import BaseAgent
from core.state import AgentState
from agents.agent0_scout.query_expander import analyze_intent
from agents.agent0_scout.arxiv_client import search_arxiv_papers
from agents.agent0_scout.paper_ranker import rank_and_select_papers
from agents.agent0_scout.downloader import download_and_register_paper
from agents.agent0_scout.gemini_scout import search_gemini_academic_papers


class ScoutAgent(BaseAgent):
    """
    Agent 0 (Literature Scout):
    Autonomous paper discovery and ingestion agent.
    If no PDF is attached, Scout identifies relevant academic literature,
    downloads the primary paper, and registers it in the system workspace.
    """

    def __init__(self, *, auto_scout: bool = True):
        super().__init__(name="LiteratureScout")
        self.auto_scout = auto_scout

    def _terminal(self, action, decision, status, message, **details):
        return {
            "scout_decision": decision,
            "scout_message": message,
            "answer_status": status,
            "answer": message,
            "final_report": message,
            "reasoning_trace": [self.log_action(action, {**details, "message": message})],
        }

    def _gemini_result(self, papers, perspectives, provider):
        for paper in papers:
            path = paper.get("local_pdf_path") or paper.get("path") or paper.get("pdf_path")
            if path and path.lower().endswith(".pdf") and os.path.isfile(path):
                trace = self.log_action("paper_discovered_and_ingested", {
                    "provider": provider, "title": paper.get("title"), "path": path,
                    "candidates_count": len(papers),
                })
                return {
                    "scout_decision": "success",
                    "pdf_path": path,
                    "pdf_paths": [path],
                    "retrieved_docs": [{
                        "id": paper.get("id"), "name": paper.get("title"), "path": path,
                        "title": paper.get("title"), "authors": paper.get("authors", []),
                        "year": paper.get("year", 2024), "abstract": paper.get("abstract", ""),
                        "text": f"# {paper.get('title')}\n\n{paper.get('abstract')}",
                        "auto_scouted": True,
                    }],
                    "perspectives": [p["angle"] for p in perspectives],
                    "sub_queries": [p["query"] for p in perspectives],
                    "reasoning_trace": [trace],
                }
        return self._terminal(
            "download_failed", "download_failed", "DOWNLOAD_FAILED",
            "Đã tìm thấy tài liệu liên quan nhưng chưa tải được bản toàn văn để kiểm chứng.",
        )

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Executes the autonomous scouting process with intent intelligence and relevance gating.
        """
        user_query = state.get("query", "").strip()
        selected = state.get("pdf_paths") or ([state["pdf_path"]] if state.get("pdf_path") else [])

        # 1. If user already specified a valid PDF, bypass discovery
        if selected and all(os.path.isfile(path) for path in selected):
            existing_pdf = selected[0]
            self.logger.info(f"[LiteratureScout] PDF already specified ({existing_pdf}). Passing through.")
            trace = self.log_action("bypass_discovery", {"pdf_path": existing_pdf, "status": "retained"})
            return {
                "pdf_path": existing_pdf,
                "pdf_paths": selected,
                "reasoning_trace": [trace],
            }
        if not self.auto_scout:
            return self._terminal(
                "waiting_for_document", "waiting_for_document", "WAITING_FOR_DOCUMENT",
                "Tài liệu đã chọn không còn truy cập được. Hãy chọn tài liệu trong kho hoặc bật Auto-Scout.",
            )

        self.logger.info(f"[LiteratureScout] No PDF attached. Activating Autonomous Paper Discovery for: '{user_query}'...")

        # 2. Analyze Intent (3-Way: SEARCH / CLARIFY / REJECT)
        intent = analyze_intent(user_query)
        decision = str(intent.get("decision", "SEARCH")).upper()
        domain = intent.get("domain", "STEM")
        
        if decision == "REJECT":
            message = intent.get("message") or "Vui lòng nêu chủ đề nghiên cứu cụ thể hoặc tải lên tài liệu PDF."
            self.logger.warning(f"[LiteratureScout] Query rejected: {intent.get('reason')} - {intent.get('message')}")
            trace = self.log_action("discovery_rejected", {
                "query": user_query,
                "reason": intent.get("reason"),
                "message": intent.get("message")
            })
            return {
                "scout_decision": "reject",
                "scout_message": message,
                "answer_status": "OUT_OF_SCOPE",
                "answer": message,
                "final_report": message,
                "reasoning_trace": [trace]
            }
            
        if decision == "CLARIFY":
            message = intent.get("message") or "Bạn muốn tìm tài liệu về khía cạnh nào của chủ đề này?"
            self.logger.info(f"[LiteratureScout] Clarification required: {intent.get('message')}")
            trace = self.log_action("discovery_clarify", {
                "query": user_query,
                "reason": intent.get("reason"),
                "message": intent.get("message")
            })
            return {
                "scout_decision": "clarify",
                "scout_message": message,
                "answer_status": "CLARIFICATION_NEEDED",
                "answer": message,
                "final_report": message,
                "reasoning_trace": [trace]
            }

        # Query expansion into academic keywords and multi-perspectives
        perspectives = intent["perspectives"]
        academic_query = intent["academic_query"]

        # 3. NON-STEM ROUTE: Route directly to Gemini Academic Explorer
        if domain == "NON_STEM":
            self.logger.info(f"[LiteratureScout] Non-STEM humanities/history topic: '{user_query}'. Routing to Gemini Academic Explorer...")
            gemini_papers = search_gemini_academic_papers(user_query, max_results=3)
            if gemini_papers:
                return self._gemini_result(gemini_papers, perspectives, "Gemini Academic Explorer")

        # 4. STEM ROUTE: Multi-angle search on ArXiv
        all_candidates: list[dict[str, Any]] = []
        for p in perspectives:
            p_papers = search_arxiv_papers(p["query"], max_results=3, recent=p["angle"] == "Recent")
            all_candidates.extend(p_papers)

        if not all_candidates:
            self.logger.warning("[LiteratureScout] Multi-perspective search yielded 0 results, running fallback query...")
            all_candidates = search_arxiv_papers(academic_query, max_results=5)

        # 5. Rank candidates and enforce Bidirectional Relevance Gate
        top_papers = []
        if all_candidates:
            top_papers = rank_and_select_papers(
                all_candidates, 
                user_query, 
                academic_keywords=academic_query, 
                top_k=3,
                min_threshold=2.0
            )

        # If STEM ArXiv search yielded 0 relevant papers, fallback to Gemini Academic Explorer
        if not top_papers:
            self.logger.info(f"[LiteratureScout] ArXiv yielded 0 papers for '{user_query}'. Activating Gemini Academic Explorer fallback...")
            gemini_papers = search_gemini_academic_papers(user_query, max_results=3)
            if gemini_papers:
                return self._gemini_result(gemini_papers, perspectives, "Gemini Academic Explorer (Fallback)")

            return self._terminal(
                "discovery_failed", "no_relevant_papers", "NO_RELEVANT_PAPERS",
                f"Không tìm thấy bài báo đủ liên quan đến chủ đề '{user_query}'. Bạn có thể tải tài liệu vào kho.",
                query=user_query, reason="candidates_below_relevance_threshold",
            )

        primary_paper = top_papers[0]
        self.logger.info(f"[LiteratureScout] Golden Paper Selected: '{primary_paper.get('title')}' ({primary_paper.get('year')})")

        # 6. Download primary paper and register in SQLite
        try:
            downloaded = download_and_register_paper(primary_paper)
            local_pdf_path = downloaded.get("local_pdf_path")
            if not local_pdf_path or not os.path.isfile(local_pdf_path):
                raise OSError("Downloaded PDF is unavailable")
        except Exception as exc:
            self.logger.error("[LiteratureScout] Failed to download paper: %s", exc)
            return self._terminal(
                "download_failed", "download_failed", "DOWNLOAD_FAILED",
                "Không thể tải bản toàn văn của bài báo đã chọn.",
            )

        # 7. Build reasoning trace and state updates
        trace = self.log_action(
            "paper_discovered_and_ingested",
            {
                "title": downloaded.get("title"),
                "year": downloaded.get("year"),
                "arxiv_id": downloaded.get("arxiv_id"),
                "pdf_path": local_pdf_path,
                "candidates_count": len(all_candidates),
                "top_candidates": [
                    {"title": p.get("title"), "year": p.get("year"), "id": p.get("id"), "relevance": p.get("relevance_score")}
                    for p in top_papers
                ]
            }
        )

        discovered_doc_record = {
            "id": downloaded.get("doc_id"),
            "name": f"[{downloaded.get('year', 2024)}] {downloaded.get('title')}",
            "path": local_pdf_path,
            "title": downloaded.get("title"),
            "authors": downloaded.get("authors", []),
            "year": downloaded.get("year", 2024),
            "abstract": downloaded.get("abstract", ""),
            "auto_scouted": True,
        }

        return {
            "scout_decision": "success",
            "pdf_path": local_pdf_path,
            "pdf_paths": [local_pdf_path],
            "retrieved_docs": [discovered_doc_record],
            "perspectives": [p["angle"] for p in perspectives],
            "sub_queries": [p["query"] for p in perspectives],
            "reasoning_trace": [trace],
        }
