import json

import httpx
import pytest

from agents.agent1_retrieval.tools.remote_rerank import RerankError, rerank_passages
from core.config import RunConfig, use_run_config
from modes.gpu_finetune import retrieval
from utils.llm_client import llm_client


MODEL = "BAAI/bge-reranker-v2-m3"
REVISION = "d" * 40
PASSAGES = [
    {"id": "p1", "doc_id": "paper", "page_no": 1, "text": "Calibration study of irrelevant conditions."},
    {"id": "p2", "doc_id": "paper", "page_no": 2, "text": "Calibration study of the required conditions."},
]


@pytest.mark.parametrize(("scores", "indices", "changed"), [
    ([1.0, 7.0], [1, 0], True),
    ([7.0, 1.0], [0, 1], False),
    ([1.0, 1.0], [0, 1], False),
])
def test_scores_define_stable_permutation(scores, indices, changed):
    def worker(request):
        assert request.url.path == "/api/rerank"
        body = json.loads(request.content)
        assert body["candidates"] == [row["text"] for row in PASSAGES]
        return httpx.Response(200, json={"model": MODEL, "revision": REVISION, "scores": scores, "ranked_indices": indices})

    with httpx.Client(transport=httpx.MockTransport(worker)) as client:
        result = rerank_passages("Calibration?", PASSAGES, client=client, url="https://worker.invalid", model=MODEL, timeout=2)
    assert result == ([PASSAGES[i] for i in indices], REVISION, changed)


def test_guarded_rerank_keeps_the_first_three_and_resorts_only_remaining_candidates():
    candidates = [{"id": str(i), "text": f"Passage {i}"} for i in range(5)]
    response = {
        "model": MODEL, "revision": REVISION, "scores": [1, 2, 3, 4, 5],
        "ranked_indices": [4, 3, 2, 1, 0],
    }
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))) as client:
        ordered, revision, changed = rerank_passages(
            "Calibration?", candidates, client=client, url="https://worker.invalid", model=MODEL,
            timeout=2, keep_prefix=3,
        )
    assert ordered == candidates[:3] + candidates[3:][::-1]
    assert revision == REVISION and changed


@pytest.mark.parametrize("payload", [
    {"ranked_indices": [0, 1]},
    {"model": MODEL, "revision": REVISION, "scores": [1.0, 2.0], "ranked_indices": [0, 0]},
    {"model": MODEL, "revision": REVISION, "scores": [1.0, 2.0], "ranked_indices": [-1, 0]},
    {"model": MODEL, "revision": REVISION, "scores": [1.0, 2.0], "ranked_indices": [True, 0]},
    {"model": MODEL, "revision": REVISION, "scores": [1.0, 2.0], "ranked_indices": [1]},
    {"model": MODEL, "revision": REVISION, "scores": [1.0, 2.0], "ranked_indices": [0, 1]},
    {"model": MODEL, "revision": REVISION, "scores": [1.0, "2.0"], "ranked_indices": [1, 0]},
])
def test_malformed_or_placeholder_response_is_rejected(payload):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as client:
        with pytest.raises(RerankError, match="invalid_reranker_response"):
            rerank_passages("Calibration?", PASSAGES, client=client, url="https://worker.invalid", model=MODEL, timeout=2)


@pytest.mark.parametrize(("failure", "reason"), [
    ("timeout", "reranker_timeout"), ("unavailable", "reranker_http_503"),
    ("legacy", "invalid_reranker_response"),
])
def test_agent_keeps_original_ranking_on_rerank_failure(monkeypatch, tmp_path, failure, reason):
    monkeypatch.setattr(retrieval, "load_or_parse_selected_pdf", lambda _: PASSAGES)

    def worker(request):
        assert request.url.path == "/api/rerank"
        if failure == "timeout":
            raise httpx.ReadTimeout("worker timed out")
        if failure == "legacy":
            return httpx.Response(200, json={"ranked_indices": [0, 1]})
        return httpx.Response(503)

    config = RunConfig.capture(remote_inference_url="https://worker.invalid", corpus_path=str(tmp_path / "none.jsonl"))
    state = {"query": "Calibration", "pdf_paths": ["paper.pdf"], "top_k": 2}
    with httpx.Client(transport=httpx.MockTransport(worker)) as client, use_run_config(config):
        monkeypatch.setattr(llm_client, "http_client", client)
        base = retrieval.GPUFinetuneRetrievalAgent(enable_dense=False, enable_rerank=False).run(state)
        reranked = retrieval.GPUFinetuneRetrievalAgent(enable_dense=False, enable_rerank=True).run(state)
    assert reranked["retrieved_docs"] == base["retrieved_docs"]
    assert reranked["reasoning_trace"][0]["details"]["rerank"] == {"status": "fallback", "reason": reason}
