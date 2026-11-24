import asyncio
import json
from threading import Event

import fitz
import httpx
import pytest
from fastapi.testclient import TestClient

from api import app as api_module, auth
from api import research_runs as run_service
from api.run_manifest import get_run_manifest
from core import graph
from core.config import RunConfig, get_run_settings, settings, use_run_config
from modes import registry
from modes.gpu_finetune.reader import GPUFinetuneReaderAgent
from utils import local_vulkan_setup, llm_client as llm_module


@pytest.mark.integration
def test_concurrent_api_runs_keep_worker_mode_model_and_cache_identity(monkeypatch, tmp_path, research_db):
    pdf = tmp_path / "paper.pdf"
    with fitz.open() as doc:
        doc.new_page().insert_text((72, 72), "The pilot lasted seven days.")
        doc.save(pdf)

    monkeypatch.setattr(auth, "AUTH_SECRET_PASSWORD", "runtime-test")
    monkeypatch.setattr(auth, "AUTH_DEV_MODE", False)
    monkeypatch.setattr(auth, "_sessions", {})
    monkeypatch.setattr(settings, "EXECUTION_MODE", "gpu_finetune")
    monkeypatch.setattr(settings, "REMOTE_INFERENCE_URL", "https://worker-a.invalid")
    monkeypatch.setattr(settings, "LOCAL_CPU_MODEL", "configured-model-a")
    monkeypatch.setattr(api_module, "_legacy_mode_override", None)
    monkeypatch.setattr(api_module, "_write_env_settings", lambda values: None)
    monkeypatch.setattr(api_module, "get_config", lambda: {"mode": settings.EXECUTION_MODE})

    research_db.save_document("paper", pdf.name, str(pdf), pdf.stat().st_size)
    monkeypatch.setattr(graph.ScoutAgent, "run", lambda self, state: {})
    monkeypatch.setattr(graph, "add_figure_evidence", lambda inputs, update, mode: update)

    a_started = Event()
    b_started = Event()
    calls = []
    cache = {}
    monkeypatch.setattr(llm_module, "get_cached_llm", cache.get)
    monkeypatch.setattr(llm_module, "set_cached_llm", lambda key, prompt, answer: cache.update({key: answer}))

    def inference(request):
        host, path = request.url.host, request.url.path
        if path == "/generate":
            payload = json.loads(request.content)
            calls.append((host, path, payload["adapter"]))
            if host == "worker-a.invalid":
                a_started.set()
                assert b_started.wait(5), "Second run did not reach its worker"
            else:
                b_started.set()
            headers = {
                "X-Research-Model": "Qwen/Qwen2.5-7B-Instruct",
                "X-Research-Revision": ("a" if host == "worker-a.invalid" else "b") * 40,
                "X-Research-Adapter": payload["adapter"],
            }
            if payload["adapter"] == "router":
                headers["X-Research-Adapter-SHA256"] = "f" * 64
            return httpx.Response(200, json={"text": host}, headers=headers)
        if path == "/api/reader/layout":
            calls.append((host, path, get_run_settings().LOCAL_CPU_MODEL))
            return httpx.Response(200, json={"extracted_tables": [{"page_no": 1, "text": "| Duration | seven days |"}]})
        raise AssertionError(path)

    class TestAgent:
        def __init__(self, role, mode):
            self.role = role
            self.mode = mode

        def run(self, state):
            if self.role == "retrieval":
                text = llm_module.llm_client.generate("A shared prompt", agent="retrieval")
                return {"retrieved_docs": [{"id": "p1", "doc_id": "paper", "page_no": 1, "text": text}]}
            if self.role == "reader":
                return GPUFinetuneReaderAgent().run(state)
            if self.role == "factchecker":
                return {"fact_check_label": "support"}
            if self.role == "writer":
                text = llm_module.llm_client.generate("A shared prompt", agent="retrieval")
                return {"final_report": f"{self.mode} {get_run_settings().LOCAL_CPU_MODEL} {text}"}
            return {}

    monkeypatch.setattr(registry, "get_agent", TestAgent)
    real_async_client = httpx.AsyncClient

    async def run_requests():
        transport = httpx.ASGITransport(app=api_module.app)
        with httpx.Client(transport=httpx.MockTransport(inference)) as worker:
            monkeypatch.setattr(llm_module.llm_client, "http_client", worker)
            async with real_async_client(transport=transport, base_url="https://workspace.test") as client:
                monkeypatch.setattr(httpx, "AsyncClient", lambda **options: real_async_client(
                    transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"status": "ok"})),
                    **options,
                ))
                login = await client.post("/api/auth/login", json={"password": "runtime-test"})
                assert login.status_code == 200
                try:
                    first = await client.post("/api/runs", json={"question": "Inspect the pilot", "document_id": "paper"})
                    assert first.status_code == 200
                    assert await asyncio.to_thread(a_started.wait, 5)
                    assert (await client.post("/api/config/remote-url", json={"remote_url": "https://worker-b.invalid"})).status_code == 200
                    assert (await client.post("/api/config/mode", json={"mode": "gpu_max"})).status_code == 200
                    api_module._set_run_defaults({"LOCAL_CPU_MODEL": "configured-model-b"})
                    second = await client.post("/api/runs", json={"question": "Inspect the pilot", "document_id": "paper"})
                    assert second.status_code == 200
                    await asyncio.wait_for(asyncio.gather(*tuple(api_module.research_runs.tasks.values())), 10)
                    return first.json()["run_manifest"], second.json()["run_manifest"]
                finally:
                    b_started.set()

    first_admission, second_admission = asyncio.run(run_requests())
    first, second = (research_db.get_run(item["run_id"]) for item in (first_admission, second_admission))
    assert first["status"] == second["status"] == "completed"
    assert "gpu_finetune configured-model-a worker-a.invalid" in first["final_report"]
    assert "gpu_max configured-model-b worker-b.invalid" in second["final_report"]
    assert ("worker-a.invalid", "/generate", "router") in calls
    assert ("worker-b.invalid", "/generate", "none") in calls
    assert ("worker-a.invalid", "/api/reader/layout", "configured-model-a") in calls
    assert ("worker-b.invalid", "/api/reader/layout", "configured-model-b") in calls
    assert len(cache) == 2  # Writer reuses its own run's cached retrieval output.
    for admission, record, mode, model in [
        (first_admission, first, "gpu_finetune", "configured-model-a"),
        (second_admission, second, "gpu_max", "configured-model-b"),
    ]:
        manifest = get_run_manifest(record)
        configuration = manifest["configuration"]
        assert configuration == admission["configuration"]
        assert set(configuration["agent_modes"].values()) == {mode}
        assert configuration["models"]["local_cpu"] == model
        assert admission["observed"]["worker_inference"] == []
        observed, cached = manifest["observed"]["worker_inference"]
        adapter, revision = ("router", "a") if mode == "gpu_finetune" else ("none", "b")
        assert observed["status"] == "reported" and observed["adapter"] == adapter
        assert observed["revision"] == revision * 40
        assert cached == {"component": adapter, "endpoint": "/generate", "status": "cache_hit"}


def test_failure_and_capability_probe_are_partitioned_by_worker(monkeypatch):
    monkeypatch.setattr(settings, "REMOTE_INFERENCE_URL", "https://worker-a.invalid")
    first = RunConfig.capture()
    monkeypatch.setattr(settings, "REMOTE_INFERENCE_URL", "https://worker-b.invalid")
    second = RunConfig.capture()
    assert first.REMOTE_INFERENCE_URL == "https://worker-a.invalid"
    with pytest.raises(AttributeError):
        first.REMOTE_INFERENCE_URL = "https://changed.invalid"

    model_client = llm_module.UnifiedLLMClient()
    model_client.http_client.close()

    def inference(request):
        if request.url.host == "worker-a.invalid":
            raise httpx.ConnectError("worker unavailable", request=request)
        return httpx.Response(200, json={"text": "second worker"})

    monkeypatch.setattr(local_vulkan_setup, "_PROBE_CACHE", None)
    monkeypatch.setattr(local_vulkan_setup, "probe_ollama_models", lambda: {
        "online": False, "cpu_max_model": "local", "models": [],
    })
    monkeypatch.setattr(local_vulkan_setup, "probe_kaggle_worker", lambda: {
        "online": True, "url": get_run_settings().REMOTE_INFERENCE_URL, "adapters": [],
    })
    with httpx.Client(transport=httpx.MockTransport(inference)) as provider:
        model_client.http_client = provider
        with use_run_config(first):
            with pytest.raises(RuntimeError, match="worker-a"):
                model_client._call_remote_kaggle("generate", {"prompt": "first"})
            assert local_vulkan_setup.probe_all_capabilities()["kaggle_max"]["url"] == first.REMOTE_INFERENCE_URL
        with use_run_config(second):
            assert model_client._call_remote_kaggle("generate", {"prompt": "second"}) == "second worker"
            assert local_vulkan_setup.probe_all_capabilities()["kaggle_max"]["url"] == second.REMOTE_INFERENCE_URL


def test_gemini_request_uses_the_model_from_its_snapshot(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-provider-key")
    monkeypatch.setattr(settings, "DEFAULT_LLM_MODEL", "model-a")
    first = RunConfig.capture()
    monkeypatch.setattr(settings, "DEFAULT_LLM_MODEL", "model-b")
    second = RunConfig.capture()
    paths = []

    def inference(request):
        paths.append(request.url.path)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "answer"}]}}]})

    monkeypatch.setattr(llm_module, "log_api_usage", lambda *args: None)
    with httpx.Client(transport=httpx.MockTransport(inference)) as provider:
        monkeypatch.setattr(llm_module.llm_client, "http_client", provider)
        for snapshot in (first, second):
            with use_run_config(snapshot):
                assert llm_module.llm_client._call_gemini("Prompt") == "answer"
    assert paths == ["/v1beta/models/model-a:generateContent", "/v1beta/models/model-b:generateContent"]


@pytest.mark.integration
def test_direct_and_streaming_routes_capture_fresh_defaults(monkeypatch, research_db):
    monkeypatch.setattr(auth, "AUTH_SECRET_PASSWORD", "runtime-test")
    monkeypatch.setattr(auth, "_sessions", {})
    monkeypatch.setattr(api_module, "_legacy_mode_override", "cpu_min")
    monkeypatch.setattr(settings, "REMOTE_INFERENCE_URL", "https://worker-a.invalid")
    seen = []

    class TestGraph:
        async def astream(self, state, stream_mode):
            yield {"writer_node": {"final_report": "Ready"}}

    def build(mode, *, run_config, auto_scout, parallel_retrieval, comparison_plan):
        assert not parallel_retrieval and not comparison_plan
        seen.append((mode, run_config.REMOTE_INFERENCE_URL, auto_scout))
        return TestGraph()

    monkeypatch.setattr(run_service, "build_research_rag_graph", build)
    with TestClient(api_module.app, base_url="https://workspace.test") as client:
        assert client.post("/api/auth/login", json={"password": "runtime-test"}).status_code == 200
        direct = client.post("/api/research", json={"query": "Question"}).json()
        assert direct["final_report"] == "Ready"
        assert research_db.get_run(direct["run_id"])["status"] == "completed"
        monkeypatch.setattr(settings, "REMOTE_INFERENCE_URL", "https://worker-b.invalid")
        monkeypatch.setattr(api_module, "_legacy_mode_override", "gpu_finetune")
        response = client.post("/api/research/stream", json={"query": "Question"})
        assert response.status_code == 200
        assert '"event": "finished"' in response.text
    assert seen == [("cpu_min", "https://worker-a.invalid", True), ("gpu_finetune", "https://worker-b.invalid", True)]
