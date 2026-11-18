"""
Unit tests for Agent 2 (Reader):
  - SourceUnit creation and offset mapping
  - Context hash calculation
  - Document schema generation
"""

from core.state import SourceUnit, compute_context_hash
from agents.agent2_reader.node import ReaderAgent


def test_source_unit_and_context_hash():
    unit = SourceUnit(
        id="unit_1",
        text="Sample scientific finding about neural scaling laws.",
        doc_id="paper_2024",
        page_no=2,
    )
    assert unit.context_hash != ""
    assert len(unit.context_hash) == 16
    assert compute_context_hash("Sample scientific finding about neural scaling laws.", "paper_2024", 2) == unit.context_hash


def test_reader_agent_retrieved_docs_mapping():
    agent = ReaderAgent()
    state = {
        "pdf_path": None,
        "retrieved_docs": [
            {"id": "doc_a", "text": "Section on Transformer attention matrices.", "page_no": 1},
            {"id": "doc_b", "text": "Empirical evaluation on GLUE benchmark.", "page_no": 3},
        ],
    }
    output = agent.run(state)
    assert "source_units" in output
    assert len(output["source_units"]) == 2
    assert "document_schema" in output
    assert output["source_units"][0]["doc_id"] == "doc_a"
