"""
CPU Min Mode - Writer Agent.
Fast academic report generation:
  - Formats verified claims with markdown citations.
  - Generates clear structured summary, bullet points, and source bibliography.
"""

from __future__ import annotations

from typing import Any
from core.base_agent import BaseAgent
from core.state import AgentState


class CPUMinWriterAgent(BaseAgent):
    """Writer Agent for CPU Min: Fast template-based academic report generator."""

    def __init__(self, max_internal_retry: int = 1):
        super().__init__(name="CPUMinWriterAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        answer = state.get("answer", "")
        verified_claims = state.get("verified_claims", []) or []
        retrieved_docs = state.get("retrieved_docs", []) or []

        self.logger.info(f"[CPU Min Writer] Compiling report for query: '{query}'")

        # Compile bibliography
        biblio_lines = []
        for idx, doc in enumerate(retrieved_docs[:5], 1):
            title = doc.get("title") or doc.get("doc_id") or f"Tài liệu {idx}"
            biblio_lines.append(f"[{idx}] {title}. (Trích đoạn: {doc.get('id', '')})")
        biblio_str = "\n".join(biblio_lines) if biblio_lines else "*(Không có tài liệu tham khảo)*"

        # Build Verified findings section
        findings_lines = []
        for idx, vc in enumerate(verified_claims, 1):
            findings_lines.append(f"{idx}. {vc.get('claim_text') or vc.get('claim', '')} `[Kiểm chứng: Hợp lệ]`")
        findings_str = "\n".join(findings_lines) if findings_lines else "- Chưa có luận điểm nào được xác thực độc lập."

        final_report = (
            f"# BÁO CÁO NGHIÊN CỨU KHOA HỌC (CPU MIN MODE)\n\n"
            f"> **Chủ đề**: {query}\n\n"
            f"## 1. Tóm Tắt Tổng Quan (Executive Summary)\n"
            f"{answer}\n\n"
            f"## 2. Các Luận Điểm Đã Được Đối Soát (Verified Findings)\n"
            f"{findings_str}\n\n"
            f"## 3. Tài Liệu Tham Khảo (References)\n"
            f"{biblio_str}\n"
        )

        trace = self.log_action("cpu_min_report_written", {"length": len(final_report)})

        return {
            "final_report": final_report,
            "written_report_data": {"title": query, "sections": 3},
            "reasoning_trace": [trace],
        }
