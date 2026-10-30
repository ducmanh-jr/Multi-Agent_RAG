"""
Integration Tests cho Unified LangGraph Pipeline.
Đã gỡ bỏ @pytest.mark.skip vì toàn bộ 5 Agent đã được hiện thực hoàn chỉnh.
"""

import pytest
from core.graph import build_research_rag_graph


def test_graph_build_success():
    """Kiểm tra graph có thể build thành công và biên dịch state graph."""
    app = build_research_rag_graph()
    assert app is not None


def test_graph_standard_flow():
    """Kiểm tra luồng chuẩn không có PDF (START -> Retrieval -> Reader -> Summarizer -> FactCheck -> Writer -> END)"""
    app = build_research_rag_graph()
    initial_state = {
        "query": "What is LoRA?",
        "pdf_path": None,
        "retry_count": 0,
        "reasoning_trace": [],
    }
    result = app.invoke(initial_state)
    assert result["fact_check_label"] in ["support", "contradict", "not_enough_info"]
    assert "Báo cáo Nghiên cứu" in result.get("final_report", "") or "CẢNH BÁO" in result.get("final_report", "")
    assert len(result["reasoning_trace"]) >= 4


def test_graph_escalation_flow():
    """Kiểm tra khi retry vượt quá giới hạn 3 lần -> chuyển sang EscalationNode"""
    app = build_research_rag_graph()
    initial_state = {
        "query": "[test_retry] Câu hỏi không thể trả lời",
        "pdf_path": None,
        "retry_count": 3,
        "fact_check_label": "not_enough_info",
        "reasoning_trace": [],
    }
    result = app.invoke(initial_state)
    assert result.get("confidence") in ["high", "low"]
    assert "reasoning_trace" in result
