import json

import httpx
import pytest

from agents.agent1_retrieval.tools.remote_dense import RemoteDenseRetriever
from core.config import RunConfig, use_run_config
from modes.gpu_finetune import retrieval
from utils.llm_client import llm_client


MODEL = "BAAI/bge-m3"
REVISION = "a" * 40
SOURCES = {
    "selected.pdf": [
        {"id": "chosen-1", "doc_id": "selected", "page_no": 1, "text": "Thermal curing in polymers."},
        {"id": "chosen-2", "doc_id": "selected", "page_no": 2, "text": "Deferring answers when evidence is insufficient."},
    ],
}


def _vector(first: float, second: float) -> list[float]:
    return [first, second] + [0.0] * 1022


def _embedding_response(texts, revision=REVISION):
    embeddings = [_vector(1, 0) if "Thermal" in text else _vector(0, 2) for text in texts]
    return httpx.Response(200, json={"model": MODEL, "revision": revision, "embeddings": embeddings})


def test_dense_adds_a_semantic_hit_and_only_embeds_selected_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(retrieval, "load_or_parse_selected_pdf", SOURCES.__getitem__)
    calls = []

    def endpoint(request):
        assert request.url.path == "/api/embed"
        texts = json.loads(request.content)["texts"]
        calls.append(texts)
        assert "unselected" not in " ".join(texts)
        return _embedding_response(texts)

    config = RunConfig.capture(remote_inference_url="https://worker.invalid", corpus_path=str(tmp_path / "none.jsonl"))
    with httpx.Client(transport=httpx.MockTransport(endpoint)) as client, use_run_config(config):
        monkeypatch.setattr(llm_client, "http_client", client)
        state = {"query": "Khi nào hệ thống từ chối trả lời?", "pdf_paths": ["selected.pdf"], "top_k": 1}
        agent = retrieval.GPUFinetuneRetrievalAgent(enable_dense=True, enable_rerank=False, enable_remote_router=False)
        agent.corpus_docs = [{"id": "unselected", "text": "unselected"}]
        first = agent.run(state)
        second = agent.run(state)

    assert first["retrieved_docs"][0]["id"] == "chosen-2"
    assert second["retrieved_docs"] == first["retrieved_docs"]
    assert first["reasoning_trace"][0]["details"]["dense"] == {
        "status": "used", "model": MODEL, "revision": REVISION, "admitted_passages": 1,
    }
    assert [len(texts) for texts in calls] == [1, 2, 1]


def test_hybrid_can_admit_dense_only_passages_when_sparse_fills_top_ten(monkeypatch, tmp_path):
    lexical = [
        {"id": f"sparse-{i}", "doc_id": "selected", "page_no": i, "text": f"Calibration experiment {i}"}
        for i in range(1, 11)
    ]
    dense_only = [
        {"id": f"dense-{i}", "doc_id": "selected", "page_no": i + 10, "text": f"Instrument adjustment {i}"}
        for i in range(1, 11)
    ]
    monkeypatch.setattr(retrieval, "load_or_parse_selected_pdf", lambda _: lexical + dense_only)

    def endpoint(request):
        texts = json.loads(request.content)["texts"]
        vectors = [_vector(0, 1) if text.startswith("Calibration ") else _vector(1, 0) for text in texts]
        return httpx.Response(200, json={"model": MODEL, "revision": REVISION, "embeddings": vectors})

    config = RunConfig.capture(remote_inference_url="https://worker.invalid", corpus_path=str(tmp_path / "none.jsonl"))
    state = {"query": "Calibration", "pdf_paths": ["selected.pdf"], "top_k": 10}
    with httpx.Client(transport=httpx.MockTransport(endpoint)) as client, use_run_config(config):
        monkeypatch.setattr(llm_client, "http_client", client)
        sparse = retrieval.GPUFinetuneRetrievalAgent(enable_dense=False, enable_rerank=False).run(state)
        hybrid = retrieval.GPUFinetuneRetrievalAgent(enable_dense=True, enable_rerank=False).run(state)

    sparse_ids = {passage["id"] for passage in sparse["retrieved_docs"]}
    hybrid_ids = {passage["id"] for passage in hybrid["retrieved_docs"]}
    assert sparse_ids == {passage["id"] for passage in lexical}
    assert len(hybrid_ids) == 10
    assert hybrid_ids - sparse_ids
    assert hybrid["reasoning_trace"][0]["details"]["dense"]["admitted_passages"] == len(hybrid_ids - sparse_ids)


def test_page_diversity_fills_from_deeper_candidates_and_preserves_sparse_fallback(monkeypatch, tmp_path):
    def passage(i, doc, page):
        return {"id": f"{doc}-{i}", "doc_id": doc, "page_no": page, "text": f"calibration report {i:02d}"}

    first = [passage(i, "first", 1) for i in range(10)]
    second = [passage(i, "second", i + 1) for i in range(10)]
    sources = {"first.pdf": first, "second.pdf": second}
    monkeypatch.setattr(retrieval, "load_or_parse_selected_pdf", sources.__getitem__)
    endpoint = httpx.MockTransport(lambda request: httpx.Response(503))
    config = RunConfig.capture(remote_inference_url="https://worker.invalid", corpus_path=str(tmp_path / "none.jsonl"))
    state = {"query": "calibration", "pdf_paths": list(sources), "top_k": 10}
    with httpx.Client(transport=endpoint) as client, use_run_config(config):
        monkeypatch.setattr(llm_client, "http_client", client)
        control = retrieval.GPUFinetuneRetrievalAgent(enable_dense=False, enable_rerank=False).run(state)
        sparse = retrieval.GPUFinetuneRetrievalAgent(
            enable_dense=False, enable_rerank=False, page_diverse=True,
        ).run(state)
        hybrid = retrieval.GPUFinetuneRetrievalAgent(enable_dense=True, enable_rerank=True, page_diverse=True).run(state)
        retry = retrieval.GPUFinetuneRetrievalAgent(
            enable_dense=False, enable_rerank=False, page_diverse=True,
        ).run({**state, "seen_chunks": [second[0]["id"]]})

    assert len({(doc["doc_id"], doc["page_no"]) for doc in control["retrieved_docs"]}) < 10
    selected = {(doc["doc_id"], doc["page_no"]) for doc in sparse["retrieved_docs"]}
    assert len(selected) == 10
    assert {("first", 1), ("second", 1)} <= selected
    assert hybrid["retrieved_docs"] == sparse["retrieved_docs"]
    assert hybrid["reasoning_trace"][0]["details"]["dense"]["reason"] == "embedding_http_503"
    assert hybrid["reasoning_trace"][0]["details"]["rerank"] == {"status": "skipped", "reason": "dense_fallback"}
    assert len(retry["retrieved_docs"]) == 10
    assert second[0]["id"] not in {doc["id"] for doc in retry["retrieved_docs"]}


@pytest.mark.parametrize(("problem", "reason"), [
    ("unavailable", "embedding_http_503"),
    ("timeout", "embedding_timeout"),
    ("invalid_vector", "invalid_embedding_response"),
    ("new_revision", "embedding_revision_changed"),
    ("wrong_model", "embedding_model_mismatch"),
])
def test_dense_failures_return_the_exact_sparse_ranking(monkeypatch, tmp_path, problem, reason):
    monkeypatch.setattr(retrieval, "load_or_parse_selected_pdf", SOURCES.__getitem__)

    def endpoint(request):
        assert request.url.path == "/api/embed"
        if problem == "timeout":
            raise httpx.ReadTimeout("test timeout")
        if problem == "unavailable":
            return httpx.Response(503, json={"detail": "Embedding unavailable"})
        if problem == "wrong_model":
            return httpx.Response(200, json={"model": "different"})
        if problem == "invalid_vector":
            return httpx.Response(200, json={"model": MODEL, "revision": REVISION, "embeddings": [[0] * 1024]})
        texts = json.loads(request.content)["texts"]
        return _embedding_response(texts, revision="b" * 40 if len(texts) > 1 else REVISION)

    state = {"query": "Thermal curing", "pdf_paths": ["selected.pdf"], "top_k": 2}
    config = RunConfig.capture(remote_inference_url="https://worker.invalid", corpus_path=str(tmp_path / "none.jsonl"))
    with httpx.Client(transport=httpx.MockTransport(endpoint)) as client, use_run_config(config):
        monkeypatch.setattr(llm_client, "http_client", client)
        sparse = retrieval.GPUFinetuneRetrievalAgent(
            enable_dense=False, enable_rerank=False, enable_remote_router=False,
        ).run(state)
        hybrid = retrieval.GPUFinetuneRetrievalAgent(enable_dense=True, enable_remote_router=False).run(state)
    assert hybrid["retrieved_docs"] == sparse["retrieved_docs"]
    assert hybrid["reasoning_trace"][0]["details"]["dense"] == {"status": "fallback", "reason": reason}


def test_checkpoint_change_between_queries_invalidates_cached_passages():
    revision = REVISION
    calls = []

    def endpoint(request):
        texts = json.loads(request.content)["texts"]
        calls.append(texts)
        return _embedding_response(texts, revision=revision)

    retriever = RemoteDenseRetriever()
    with httpx.Client(transport=httpx.MockTransport(endpoint)) as client:
        settings = {"top_k": 1, "client": client, "url": "https://worker.invalid", "model": MODEL, "timeout": 5}
        retriever.rank("question", SOURCES["selected.pdf"], **settings)
        revision = "b" * 40
        _, used_revision = retriever.rank("another question", SOURCES["selected.pdf"], **settings)

    assert used_revision == revision
    assert [len(batch) for batch in calls] == [1, 2, 1, 2]
