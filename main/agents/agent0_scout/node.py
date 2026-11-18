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
from agents.agent0_scout.query_expander import (
    analyze_intent,
    generate_search_perspectives,
    translate_to_academic_english
)
from agents.agent0_scout.arxiv_client import search_arxiv_papers
from agents.agent0_scout.paper_ranker import rank_and_select_papers
from agents.agent0_scout.downloader import download_and_register_paper
from agents.agent0_scout.gemini_scout import search_gemini_academic_papers
from utils.logger import logger


class ScoutAgent(BaseAgent):
    """
    Agent 0 (Literature Scout):
    Autonomous paper discovery and ingestion agent.
    If no PDF is attached, Scout identifies relevant academic literature,
    downloads the primary paper, and registers it in the system workspace.
    """

    def __init__(self):
        super().__init__(name="LiteratureScout")

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Executes the autonomous scouting process with intent intelligence and relevance gating.
        """
        user_query = state.get("query", "").strip()
        existing_pdf = state.get("pdf_path")

        # 1. If user already specified a valid PDF, bypass discovery
        if existing_pdf and os.path.exists(existing_pdf) and os.path.getsize(existing_pdf) > 1000:
            self.logger.info(f"[LiteratureScout] PDF already specified ({existing_pdf}). Passing through.")
            trace = self.log_action("bypass_discovery", {"pdf_path": existing_pdf, "status": "retained"})
            return {
                "reasoning_trace": [trace]
            }

        self.logger.info(f"[LiteratureScout] No PDF attached. Activating Autonomous Paper Discovery for: '{user_query}'...")

        # 2. Analyze Intent (3-Way: SEARCH / CLARIFY / REJECT)
        intent = analyze_intent(user_query)
        decision = intent.get("decision", "search")
        domain = intent.get("domain", "STEM")
        
        if decision == "reject":
            self.logger.warning(f"[LiteratureScout] Query rejected: {intent.get('reason')} - {intent.get('message')}")
            trace = self.log_action("discovery_rejected", {
                "query": user_query,
                "reason": intent.get("reason"),
                "message": intent.get("message")
            })
            return {
                "scout_decision": "reject",
                "scout_message": intent.get("message"),
                "reasoning_trace": [trace]
            }
            
        if decision == "clarify":
            self.logger.info(f"[LiteratureScout] Clarification required: {intent.get('message')}")
            trace = self.log_action("discovery_clarify", {
                "query": user_query,
                "reason": intent.get("reason"),
                "message": intent.get("message")
            })
            return {
                "scout_decision": "clarify",
                "scout_message": intent.get("message"),
                "reasoning_trace": [trace]
            }

        # Query expansion into academic keywords and multi-perspectives
        perspectives = generate_search_perspectives(user_query)
        academic_query = translate_to_academic_english(user_query)

        # 3. NON-STEM ROUTE: Route directly to Gemini Academic Explorer
        if domain == "NON_STEM":
            self.logger.info(f"[LiteratureScout] Non-STEM humanities/history topic: '{user_query}'. Routing to Gemini Academic Explorer...")
            gemini_papers = search_gemini_academic_papers(user_query, max_results=3)
            if gemini_papers:
                primary = gemini_papers[0]
                local_path = primary.get("local_pdf_path") or primary.get("path") or primary.get("pdf_path")
                doc_record = {
                    "id": primary.get("id"),
                    "name": primary.get("title"),
                    "path": local_path,
                    "title": primary.get("title"),
                    "authors": primary.get("authors", []),
                    "year": primary.get("year", 2024),
                    "abstract": primary.get("abstract", ""),
                    "text": f"# {primary.get('title')}\n\n{primary.get('abstract')}",
                    "auto_scouted": True,
                }
                trace = self.log_action(
                    "paper_discovered_and_ingested",
                    {
                        "provider": "Gemini Academic Explorer",
                        "title": primary.get("title"),
                        "path": local_path,
                        "candidates_count": len(gemini_papers),
                    }
                )
                return {
                    "scout_decision": "success",
                    "pdf_path": local_path,
                    "retrieved_docs": [doc_record],
                    "perspectives": [p["angle"] for p in perspectives],
                    "sub_queries": [p["query"] for p in perspectives],
                    "reasoning_trace": [trace],
                }

        # 4. STEM ROUTE: Multi-angle search on ArXiv
        all_candidates: list[dict[str, Any]] = []
        for p in perspectives:
            p_papers = search_arxiv_papers(p["query"], max_results=3)
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
                primary = gemini_papers[0]
                local_path = primary.get("local_pdf_path") or primary.get("path") or primary.get("pdf_path")
                doc_record = {
                    "id": primary.get("id"),
                    "name": primary.get("title"),
                    "path": local_path,
                    "title": primary.get("title"),
                    "authors": primary.get("authors", []),
                    "year": primary.get("year", 2024),
                    "abstract": primary.get("abstract", ""),
                    "text": f"# {primary.get('title')}\n\n{primary.get('abstract')}",
                    "auto_scouted": True,
                }
                trace = self.log_action(
                    "paper_discovered_and_ingested",
                    {
                        "provider": "Gemini Academic Explorer (Fallback)",
                        "title": primary.get("title"),
                        "path": local_path,
                        "candidates_count": len(gemini_papers),
                    }
                )
                return {
                    "scout_decision": "success",
                    "pdf_path": local_path,
                    "retrieved_docs": [doc_record],
                    "perspectives": [p["angle"] for p in perspectives],
                    "sub_queries": [p["query"] for p in perspectives],
                    "reasoning_trace": [trace],
                }

            self.logger.warning(f"[LiteratureScout] All candidate papers failed relevance threshold for query: '{user_query}'")
            trace = self.log_action("discovery_failed", {
                "query": user_query, 
                "reason": "candidates_below_relevance_threshold"
            })
            return {
                "scout_decision": "no_relevant_papers",
                "scout_message": f"Các bài báo tìm được không đạt ngưỡng liên quan học thuật đến chủ đề '{user_query}'. Bạn có thể tải tài liệu PDF trực tiếp vào Kho Tài Liệu.",
                "reasoning_trace": [trace]
            }

        primary_paper = top_papers[0]
        self.logger.info(f"[LiteratureScout] Golden Paper Selected: '{primary_paper.get('title')}' ({primary_paper.get('year')})")

        # 6. Download primary paper and register in SQLite
        try:
            downloaded = download_and_register_paper(primary_paper)
            local_pdf_path = downloaded.get("local_pdf_path")
        except Exception as e:
            self.logger.error(f"[LiteratureScout] Failed to download paper: {e}")
            trace = self.log_action("download_failed", {"error": str(e)})
            return {
                "scout_decision": "download_failed",
                "scout_message": f"Không thể tải tệp PDF bài báo: {e}",
                "reasoning_trace": [trace]
            }

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
            "retrieved_docs": [discovered_doc_record],
            "perspectives": [p["angle"] for p in perspectives],
            "sub_queries": [p["query"] for p in perspectives],
            "reasoning_trace": [trace],
        }
