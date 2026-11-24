import importlib.util
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from core.config import settings
from core.schemas import SummarizerCriticOutput
from modes.gpu_finetune.fact_checker import GPUFinetuneFactCheckerAgent
from utils.llm_client import llm_client


@pytest.fixture
def worker(monkeypatch):
    torch = ModuleType("torch")
    torch.no_grad = nullcontext
    torch.softmax = lambda values, dim: SimpleNamespace(
        squeeze=lambda: SimpleNamespace(tolist=lambda: values),
    )
    torch.argmax = lambda values, dim: SimpleNamespace(item=lambda: values.index(max(values)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    path = Path(__file__).resolve().parents[2] / "kaggle" / "kaggle_finetune_server.py"
    spec = importlib.util.spec_from_file_location("finetune_worker_under_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("raw", [
    "{broken",
    '{"action":"accept","issues_found":[]}',
    '{"is_supported":"true","action":"accept","issues_found":[]}',
    '{"is_supported":true,"action":"approve","issues_found":["Missing quote"]}',
])
def test_worker_critic_rejects_invalid_output(worker, monkeypatch, raw):
    monkeypatch.setattr(worker, "generate_with_qwen", lambda *a, **kw: raw)
    review = worker.summarize_critic(worker.CriticRequest(query="Q", context="C", answer="A", claims=[]))
    assert review["action"] == "rewrite"
    assert review["is_supported"] is False


def test_worker_generator_surfaces_invalid_schema(worker, monkeypatch):
    monkeypatch.setattr(worker, "generate_with_qwen", lambda *a, **kw: '{"answer":"Incomplete"}')
    with pytest.raises(HTTPException) as error:
        worker.summarize_generator(worker.SummarizeRequest(query="Q", context="C"))
    assert error.value.status_code == 502


def test_embedding_worker_returns_version_and_reports_model_unavailability(worker, monkeypatch):
    revision = "a" * 40
    monkeypatch.setattr(worker, "load_embedding_model", lambda: (object(), object(), revision))
    vectors = [[1.0] + [0.0] * 1023, [0.0, 1.0] + [0.0] * 1022]
    monkeypatch.setattr(worker, "encode_dense_embeddings", lambda model, tokenizer, texts: (vectors, 12))
    response = worker.Response()
    assert worker.embed_endpoint(worker.EmbedRequest(texts=["question", "passage"]), response) == {
        "model": "BAAI/bge-m3", "revision": revision, "embeddings": vectors,
    }
    assert response.headers["X-Research-Input-Tokens"] == "12"
    assert response.headers["X-Research-Output-Tokens"] == "0"

    def unavailable():
        raise OSError("model files unavailable")

    monkeypatch.setattr(worker, "load_embedding_model", unavailable)
    with pytest.raises(HTTPException) as error:
        worker.embed_endpoint(worker.EmbedRequest(texts=["question"]))
    assert error.value.status_code == 503
    with pytest.raises(ValidationError):
        worker.EmbedRequest(texts=["   "])
    with pytest.raises(ValidationError):
        worker.EmbedRequest(texts=["input"] * 17)


def test_rerank_worker_orders_by_model_scores_and_returns_checkpoint(worker, monkeypatch):
    revision = "d" * 40
    monkeypatch.setattr(worker, "load_reranker_model", lambda: (object(), object(), revision))
    request = worker.RerankRequest(query="Calibration method?", candidates=["irrelevant", "partially relevant", "exact"])

    def score(model, tokenizer, query, candidates):
        assert query == request.query and candidates == request.candidates
        return [-3.0, 1.5, 4.0], 20

    monkeypatch.setattr(worker, "score_candidate_pairs", score)
    response = worker.Response()
    assert worker.rerank_endpoint(request, response) == {
        "model": "BAAI/bge-reranker-v2-m3", "revision": revision,
        "scores": [-3.0, 1.5, 4.0], "ranked_indices": [2, 1, 0],
    }
    assert response.headers["X-Research-Input-Tokens"] == "20"
    assert response.headers["X-Research-Output-Tokens"] == "0"
    with pytest.raises(ValidationError):
        worker.RerankRequest(query="Calibration?", candidates=[""] * 16)


@pytest.mark.parametrize("failure", ["load_failed", "invalid_scores"])
def test_rerank_worker_returns_503_instead_of_identity_on_failure(worker, monkeypatch, failure):
    def load():
        if failure == "load_failed":
            raise OSError("checkpoint missing")
        return object(), object(), "d" * 40

    monkeypatch.setattr(worker, "load_reranker_model", load)
    monkeypatch.setattr(worker, "score_candidate_pairs", lambda *_: ([1.0, float("nan")], 20))
    with pytest.raises(HTTPException) as error:
        worker.rerank_endpoint(worker.RerankRequest(query="Q", candidates=["A", "B"]))
    assert error.value.status_code == 503


def test_worker_does_not_use_previous_adapter_when_critic_is_missing(worker, monkeypatch):
    monkeypatch.setattr(worker, "_loaded_adapters", ["generator"])
    monkeypatch.setattr(worker, "load_qwen_lora_cluster", lambda: (object(), object()))
    with pytest.raises(HTTPException) as error:
        worker.generate_with_qwen("Review the draft", adapter="critic")
    assert error.value.status_code == 503


def test_qwen_response_reports_the_pinned_base_and_active_adapter(worker, monkeypatch, tmp_path):
    adapters = {}
    for name, role in (("generator_qlora", "generator"), ("critic_qlora", "critic")):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
        (directory / "adapter_model.safetensors").write_bytes(role.encode())
        adapters[name] = directory
    monkeypatch.setattr(worker, "find_weight_path", lambda path: adapters.get(Path(path).name))
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2, memory_allocated=lambda _: 0)
    monkeypatch.setattr(worker.torch, "cuda", cuda, raising=False)
    monkeypatch.setattr(worker.torch, "float16", object(), raising=False)
    monkeypatch.setenv("BASE_QWEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    revision = "a" * 40
    hub = ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda *args, **kwargs: str(tmp_path / "snapshots" / revision / "config.json")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    loaded = []

    class Model:
        device = "cpu"
        current_adapter = None
        generated_with = None

        def load_adapter(self, path, adapter_name):
            assert path == adapters[f"{adapter_name}_qlora"]

        def set_adapter(self, adapter):
            self.current_adapter = adapter

        @contextmanager
        def disable_adapter(self):
            former, self.current_adapter = self.current_adapter, None
            try:
                yield
            finally:
                self.current_adapter = former

        def generate(self, **inputs):
            self.generated_with = self.current_adapter
            return [[1, 2]]

    model = Model()

    class Inputs(dict):
        input_ids = [[1]]

        def to(self, device):
            return self

    class Tokenizer:
        pad_token = eos_token = "pad"
        pad_token_id = eos_token_id = 0

        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, *args, **kwargs):
            return Inputs()

        def decode(self, *args, **kwargs):
            if model.generated_with == "generator":
                return '{"answer":"A","claims":[],"evidence_spans":[]}'
            return model.generated_with or "base"

    def pretrained(asset, value):
        def load(model_id, **options):
            loaded.append((asset, model_id, options["revision"]))
            return value
        return SimpleNamespace(from_pretrained=load)

    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = pretrained("model", model)
    transformers.AutoTokenizer = pretrained("tokenizer", Tokenizer())
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    peft = ModuleType("peft")
    peft.PeftModel = SimpleNamespace(from_pretrained=lambda model, path, adapter_name: model)
    monkeypatch.setitem(sys.modules, "peft", peft)

    with TestClient(worker.app) as client:
        generator = client.post("/api/summarize/generator", json={"query": "Q", "context": "C"})
        critic = client.post("/generate", json={"prompt": "Review this", "adapter": "critic"})
        base = client.post("/generate", json={"prompt": "Explain this", "adapter": "none"})
        missing = client.post("/generate", json={"prompt": "Rank this", "adapter": "writer"})
    assert loaded == [("tokenizer", "Qwen/Qwen2.5-7B-Instruct", revision), ("model", "Qwen/Qwen2.5-7B-Instruct", revision)]
    assert generator.json() == {"answer": "A", "claims": [], "evidence_spans": []}
    assert critic.json() == {"text": "critic", "adapter": "critic"}
    assert base.json() == {"text": "base", "adapter": "none"}
    for response, adapter in ((generator, "generator"), (critic, "critic"), (base, "none")):
        assert response.status_code == 200
        assert response.headers["X-Research-Model"] == "Qwen/Qwen2.5-7B-Instruct"
        assert response.headers["X-Research-Revision"] == revision
        assert response.headers["X-Research-Adapter"] == adapter
        assert response.headers["X-Research-Input-Tokens"] == "1"
        assert response.headers["X-Research-Output-Tokens"] == "1"
    assert generator.headers["X-Research-Adapter-SHA256"] != critic.headers["X-Research-Adapter-SHA256"]
    assert len(generator.headers["X-Research-Adapter-SHA256"]) == 64
    assert "X-Research-Adapter-SHA256" not in base.headers
    assert missing.status_code == 503 and "X-Research-Revision" not in missing.headers


def test_nli_reports_loaded_bundle_digest_and_distinguishes_unavailable(worker, monkeypatch, tmp_path):
    base = tmp_path / "checkpoint"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    (base / "tokenizer.json").write_text("{}", encoding="utf-8")
    weights = base / "model.safetensors"
    weights.write_bytes(b"original weights")
    monkeypatch.setattr(worker, "find_weight_path", lambda _: None)
    monkeypatch.setattr(worker.torch, "cuda", SimpleNamespace(is_available=lambda: False, device_count=lambda: 0), raising=False)
    monkeypatch.setattr(worker.torch, "float16", object(), raising=False)

    class Classifier:
        device = "cpu"
        config = SimpleNamespace(id2label={0: "NEUTRAL", 1: "ENTAILMENT", 2: "CONTRADICTION"})

        def eval(self):
            return self

        def __call__(self, **inputs):
            return SimpleNamespace(logits=[0.1, 0.8, 0.1])

    tokenizer = lambda *args, **kwargs: SimpleNamespace(to=lambda device: {"input_ids": [[1, 2, 3]]})
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)
    transformers.AutoModelForSequenceClassification = SimpleNamespace(from_pretrained=lambda *args, **kwargs: Classifier())
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    request = {"claims": ["The pilot lasted 7 days."], "context": "The pilot lasted 7 days."}
    with TestClient(worker.app) as client:
        unavailable = client.post("/api/factcheck", json=request)
        assert unavailable.json()["evaluations"][0]["verdict"] == "NOT_ENOUGH_INFO"
        assert unavailable.headers["X-Research-Model-Status"] == "unavailable"
        assert unavailable.headers["X-Research-Input-Tokens"] == "0"
        assert "X-Research-Revision" not in unavailable.headers
        monkeypatch.setattr(worker, "find_weight_path", lambda _: str(base))
        classified = client.post("/api/factcheck", json=request)
        assert classified.json()["evaluations"][0]["verdict"] == "SUPPORT"
        assert classified.headers["X-Research-Model"] == "research-finetuned-nli"
        assert classified.headers["X-Research-Input-Tokens"] == "3"
        assert classified.headers["X-Research-Output-Tokens"] == "0"
        digest = classified.headers["X-Research-Revision"]
        assert len(digest) == 64 and digest == worker._checkpoint_sha256(base)
        weights.write_bytes(b"changed weights")
        assert worker._checkpoint_sha256(base) != digest
        assert client.post("/api/factcheck", json=request).headers["X-Research-Revision"] == digest


def test_app_critic_rejects_a_malformed_issues_field():
    review = SummarizerCriticOutput.from_dict({
        "is_supported": True, "action": "accept", "issues_found": "missing evidence",
    })
    assert review.action == "lower_confidence"
    assert review.is_supported is False


@pytest.mark.parametrize(("label", "expected"), [
    ("ENTAILMENT", "SUPPORT"),
    ("CONTRADICTION", "CONTRADICT"),
    ("NEUTRAL", "NOT_ENOUGH_INFO"),
    ("LABEL_1", "NOT_ENOUGH_INFO"),
    (None, "NOT_ENOUGH_INFO"),
])
def test_worker_uses_checkpoint_label_or_abstains(worker, monkeypatch, label, expected):
    class Classifier:
        device = "cpu"
        config = SimpleNamespace(id2label={0: "NEUTRAL", 1: label, 2: "CONTRADICTION"})

        def __call__(self, **inputs):
            return SimpleNamespace(logits=[0.1, 0.8, 0.1])

    tokenizer = lambda *args, **kwargs: SimpleNamespace(to=lambda device: {})
    models = (Classifier(), tokenizer) if label is not None else (None, None)
    monkeypatch.setattr(worker, "load_factcheck_model", lambda: models)
    finding, = worker.factcheck_endpoint(worker.FactCheckRequest(claims=["claim"], context="source"))["evaluations"]
    assert finding["verdict"] == expected
    if label in {None, "LABEL_1"}:
        assert finding["confidence"] == 0.0


@pytest.mark.parametrize(("status", "payload"), [
    (503, {"error": "unavailable"}),
    (200, {"results": [{"claim_id": "claim", "status": "SUPPORT", "quote": "The pilot lasted 7 days."}]}),
    (200, {"evaluations": [{"claim": "The pilot lasted 7 days.", "verdict": "SUPPORT"}]}),
])
def test_unavailable_or_malformed_nli_has_no_generator_fallback(monkeypatch, status, payload):
    paths = []

    def worker_response(request):
        paths.append(request.url.path)
        return httpx.Response(status, json=payload)

    monkeypatch.setattr(settings, "REMOTE_INFERENCE_URL", "https://nli.invalid")
    with httpx.Client(transport=httpx.MockTransport(worker_response)) as client:
        monkeypatch.setattr(llm_client, "http_client", client)
        statement = "The pilot lasted 7 days."
        result = GPUFinetuneFactCheckerAgent().run({
            "query": "How long was the pilot?",
            "answer": statement,
            "claims": [{"id": "claim", "text": statement}],
            "source_units": [{"id": "source", "text": statement, "doc_id": "paper", "page_no": 1}],
        })
    assert paths == ["/api/factcheck"]
    assert result["fact_check_label"] == "not_enough_info"
    assert result["verified_claims"] == []
