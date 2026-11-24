import asyncio
from threading import Barrier

from modes.cpu_min.retrieval import CPUMinRetrievalAgent


def test_scheduler_runs_both_engines_at_the_same_time_and_preserves_ranking():
    retriever = CPUMinRetrievalAgent()
    barrier = Barrier(2)

    class Engine:
        def __init__(self, prefix):
            self.prefix = prefix

        def search(self, query, top_k):
            if query == "first" and not self.prefix.startswith("reference"):
                barrier.wait(timeout=5)
            key = self.prefix.removeprefix("reference-") + query
            return [{"id": key, "doc_id": key, "text": key, "page_no": 1}]

    queries = ["first", "second"]
    reference = retriever._search_index(queries, Engine("reference-fts"), Engine("reference-bm"), top_k=4)
    ranked, event = asyncio.run(retriever._scheduled_search(queries, Engine("fts"), Engine("bm"), top_k=4))
    assert event["details"]["status"] == "completed"
    assert ranked == reference


def test_parallel_branch_failure_records_fallback_to_the_normal_ranker(monkeypatch):
    retriever = CPUMinRetrievalAgent()
    original = retriever.fts5_searcher.search
    calls = 0

    def fail_once(query, top_k):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("Search unavailable")
        return original(query, top_k)

    docs = [{"id": "proof", "doc_id": "paper", "title": "Observations", "text": "Trial yield was 21."}]
    retriever.fts5_searcher.fit(docs)
    retriever.bm25_fallback.fit(docs)
    retriever.corpus_docs = docs
    monkeypatch.setattr(retriever.fts5_searcher, "search", fail_once)
    result = retriever.run({"query": "trial yield", "sub_queries": ["trial yield"]}, parallel_search=True)
    event = result["reasoning_trace"][0]
    assert event["details"]["status"] == "fallback"
    assert event["details"]["tasks"][-1]["blocked_by"] == ["fts5"]
    assert result["retrieved_docs"][0]["id"] == "proof"
