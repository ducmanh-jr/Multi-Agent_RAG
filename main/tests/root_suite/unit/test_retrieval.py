"""
Unit tests for Agent 1 (Retrieval):
  - BM25 indexing and searching
  - Reciprocal Rank Fusion (RRF)
  - Perspective decomposition
"""

from agents.agent1_retrieval.tools.bm25 import SimpleBM25
from agents.agent1_retrieval.tools.hybrid import reciprocal_rank_fusion
from agents.agent1_retrieval.node import RetrievalAgent


def test_simple_bm25_search():
    bm25 = SimpleBM25()
    corpus = [
        {"id": "doc_1", "text": "Deep learning architectures including Convolutional Neural Networks and Transformers."},
        {"id": "doc_2", "text": "Graph Convolutional Networks applied to molecular property prediction."},
        {"id": "doc_3", "text": "Attention mechanisms in sequence to sequence models."},
    ]
    bm25.fit(corpus)
    results = bm25.search("Transformer attention", top_k=2)
    assert len(results) > 0
    assert any("Transformer" in r["text"] or "Attention" in r["text"] for r in results)


def test_reciprocal_rank_fusion():
    list1 = [{"id": "d1", "text": "doc one"}, {"id": "d2", "text": "doc two"}]
    list2 = [{"id": "d2", "text": "doc two"}, {"id": "d3", "text": "doc three"}]
    fused = reciprocal_rank_fusion([list1, list2], k=60, top_n=2)
    assert len(fused) == 2
    # d2 appears in both lists, so its RRF score should be the highest
    assert fused[0]["id"] == "d2"
    assert "rrf_score" in fused[0]


def test_retrieval_agent_run():
    agent = RetrievalAgent()
    state = {
        "query": "Attention mechanism in Transformers",
        "retry_count": 0,
        "evidence_gaps": [],
        "seen_chunks": [],
    }
    output = agent.run(state)
    assert "retrieved_docs" in output
    assert "sub_queries" in output
    assert "perspectives" in output
    assert len(output["retrieved_docs"]) > 0
