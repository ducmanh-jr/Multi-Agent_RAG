"""
Unit tests for Agent 3 (Summarizer):
  - Generator extraction
  - Critic filtering and warnings
"""

from agents.agent3_summarizer.components.critic import critique_summary_and_claims
from agents.agent3_summarizer.node import SummarizerAgent


def test_critic_filtering():
    claims = [
        {"id": "c1", "text": "Claim 1 supported by evidence"},
        {"id": "c2", "text": "Claim 2 ungrounded assertion"},
    ]
    evidence_spans = [
        {"quote": "Claim 1 is verified in the text.", "unit_id": "u1"}
    ]
    ans, filtered, warnings = critique_summary_and_claims("Draft answer", claims, evidence_spans)
    assert len(filtered) > 0


def test_summarizer_agent_run():
    agent = SummarizerAgent()
    state = {
        "query": "What is QLoRA?",
        "source_units": [
            {
                "id": "u1",
                "text": "QLoRA reduces memory usage to finetune a 65B model on a single 48GB GPU.",
                "doc_id": "qlora_2023",
                "page_no": 1,
            }
        ],
    }
    output = agent.run(state)
    assert "answer" in output
    assert "claims" in output
    assert "evidence_spans" in output
    assert len(output["claims"]) > 0


def test_out_of_domain_response_principles():
    from utils.domain_guard import build_out_of_domain_response
    res = build_out_of_domain_response("Lịch sử phát triển tàu vũ trụ Apollo")
    assert "hoàn toàn không được đề cập" in res["answer"]
    assert "Vui lòng hỏi các câu hỏi liên quan đến nội dung tài liệu được cung cấp" in res["answer"]
    assert res["is_out_of_domain"] is True
    assert res.get("answer_status") == "NOT_FOUND_IN_DOCUMENT"


def test_ambiguous_clarification_response_principles():
    from utils.domain_guard import build_ambiguous_clarification_response
    branches = [
        {"topic": "Mô hình Base", "ref": "Bảng 2, Trang 7", "preview": "BLEU đạt 27.3"},
        {"topic": "Mô hình Big", "ref": "Bảng 2, Trang 7", "preview": "BLEU đạt 28.4"},
    ]
    res = build_ambiguous_clarification_response("kết quả thực nghiệm", branches)
    assert "Vui lòng cung cấp thêm chi tiết để tôi hỗ trợ chính xác hơn dựa trên tài liệu." in res["answer"]
    assert "Mô hình Base" in res["answer"]
    assert "Mô hình Big" in res["answer"]
    assert res["is_ambiguous"] is True
    assert res["answer_status"] == "CLARIFICATION_NEEDED"

