"""
Unit tests for Agent 5 (Writer):
  - Report synthesis from verified claim pool
  - Disclaimer creation for rejected claims
  - In-text citation anchors
"""

from agents.agent5_writer.components.report_builder import build_academic_report
from agents.agent5_writer.node import WriterAgent


def test_build_academic_report():
    verified_claims = [
        {"claim": "Transformer uses multi-head attention.", "evidence": "Vaswani 2017"}
    ]
    rejected_claims = [
        {"claim": "Transformer was proposed in 1990.", "reasoning": "Contradicts 2017 date."}
    ]
    retrieved_docs = [
        {"title": "Attention Is All You Need", "doc_id": "vaswani_2017", "page_no": 1, "text": "Paper excerpt..."}
    ]

    report, report_data = build_academic_report("Transformer query", verified_claims, rejected_claims, retrieved_docs)
    assert "# Báo cáo Nghiên cứu: Transformer query" in report
    assert "Nhận định 1" in report
    assert "Các điểm chưa đủ căn cứ hoặc bị bác bỏ" in report
    assert "[1]" in report


def test_writer_agent_run():
    agent = WriterAgent()
    state = {
        "query": "Overview of Attention",
        "verified_claims": [{"claim": "Attention is effective.", "evidence": "doc_1"}],
        "rejected_claims": [],
        "retrieved_docs": [{"doc_id": "doc_1", "title": "Paper 1", "page_no": 1, "text": "Text"}],
    }
    output = agent.run(state)
    assert "final_report" in output
    assert "written_report_data" in output
    assert output["confidence"] == "high"


def test_writer_clarification_status():
    report, report_data = build_academic_report(
        query="kết quả thực nghiệm",
        verified_claims=[],
        rejected_claims=[],
        retrieved_docs=[{"doc_id": "doc1", "title": "Paper 1", "page_no": 1}],
        answer="Dựa trên tài liệu, có nhiều kết quả khác nhau. Vui lòng cung cấp thêm chi tiết để tôi hỗ trợ chính xác hơn dựa trên tài liệu.",
        answer_status="CLARIFICATION_NEEDED",
    )
    assert report_data["status"] == "CLARIFICATION_NEEDED"
    assert "CLARIFICATION_NEEDED" in report

