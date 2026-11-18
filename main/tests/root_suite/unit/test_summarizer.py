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
