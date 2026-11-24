import importlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from core.config import agent_inference_mode, settings


@pytest.fixture
def offline_llm(monkeypatch, tmp_path):
    if "core.db" not in sys.modules:
        workspace_db = str(Path(settings.BASE_DIR) / "data" / "workspace.db")
        with monkeypatch.context() as disk:
            disk.setattr(settings, "BASE_DIR", str(tmp_path))
            database = importlib.import_module("core.db")
        database.DB_PATH = workspace_db

    module = importlib.import_module("utils.llm_client")
    cache = {}
    calls = []
    monkeypatch.setattr(module, "get_cached_llm", cache.get)
    monkeypatch.setattr(module, "set_cached_llm", lambda key, prompt, answer: cache.__setitem__(key, answer))
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test")

    def local(**options):
        calls.append(("ollama", options["system_instruction"]))
        return f"local answer {len(calls)}"

    def gemini(prompt, system_instruction, temperature, max_tokens, json_mode):
        calls.append(("gemini", system_instruction))
        return "Gemini answer"

    client = module.UnifiedLLMClient()
    monkeypatch.setattr(client, "_call_local_cpu", local)
    monkeypatch.setattr(client, "_call_gemini", gemini)
    monkeypatch.setattr(client.http_client, "send", lambda *a, **kw: pytest.fail("Unexpected network call"))
    try:
        yield client, calls
    finally:
        client.http_client.close()


def test_inference_mode_restored_after_nested_failure(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_WRITER_MODE", "gemini")
    with agent_inference_mode("cpu_max"):
        with pytest.raises(RuntimeError, match="failed agent"):
            with agent_inference_mode("gpu_finetune"):
                assert settings.get_inference_mode("writer") == "gpu_finetune"
                raise RuntimeError("failed agent")
        assert settings.get_inference_mode("writer") == "cpu_max"
    assert settings.get_inference_mode("writer") == "gemini"


def test_model_cache_partitions_modes_and_generation_options(offline_llm):
    client, calls = offline_llm
    with agent_inference_mode("cpu_max"):
        assert client.generate("same prompt", "one", agent="writer") == "local answer 1"
        assert client.generate("same prompt", "two", agent="writer") == "local answer 2"
        assert client.generate("same prompt", "one", max_tokens=8, agent="writer") == "local answer 3"
    with agent_inference_mode("gemini"):
        assert client.generate("same prompt", "one", agent="writer") == "Gemini answer"
        assert client.generate("same prompt", "one", agent="writer") == "Gemini answer"
    with agent_inference_mode("cpu_max"):
        assert client.generate("same prompt", "one", agent="writer") == "local answer 1"
    assert calls == [("ollama", "one"), ("ollama", "two"), ("ollama", "one"), ("gemini", "one")]


@pytest.mark.integration
def test_overlapping_graphs_keep_their_own_agents_and_provider_modes(monkeypatch, offline_llm):
    pytest.importorskip("langgraph")
    graph = importlib.import_module("core.graph")
    registry = importlib.import_module("modes.registry")
    client, calls = offline_llm
    overlap = Barrier(2, timeout=10)

    class TestAgent:
        def __init__(self, role, mode):
            self.role, self.mode = role, mode

        def run(self, state):
            if self.role == "retrieval":
                overlap.wait()
            assert settings.get_inference_mode(self.role) == self.mode
            if self.role == "factchecker":
                return {"fact_check_label": "support"}
            if self.role == "writer":
                return {"final_report": client.generate("shared prompt", agent="writer")}
            return {}

    monkeypatch.setattr(settings, "EXECUTION_MODE", "gpu_finetune")
    initial_agent_modes = settings.get_all_agent_modes()
    monkeypatch.setattr(registry, "get_agent", TestAgent)
    monkeypatch.setattr(graph.ScoutAgent, "run", lambda self, state: {})
    local = graph.build_research_rag_graph("cpu_max")
    cloud = graph.build_research_rag_graph("gemini")

    with ThreadPoolExecutor(max_workers=2) as pool:
        local_run = pool.submit(local.invoke, {"query": "same question"})
        cloud_run = pool.submit(cloud.invoke, {"query": "same question"})
        assert local_run.result(timeout=15)["final_report"].startswith("local answer ")
        assert cloud_run.result(timeout=15)["final_report"] == "Gemini answer"

    assert sorted(calls) == [("gemini", None), ("ollama", None)]
    assert settings.EXECUTION_MODE == "gpu_finetune"
    assert settings.get_all_agent_modes() == initial_agent_modes


def test_hybrid_summarizer_fallback_uses_local_provider_and_keeps_review_feedback(monkeypatch, offline_llm):
    module = importlib.import_module("modes.hybrid.summarizer")
    client, calls = offline_llm
    summarizer = module.HybridSummarizerAgent(max_internal_retry=3)
    monkeypatch.setattr(settings, "HYBRID_PROVIDER_PROFILE", "auto")
    monkeypatch.setattr(module, "probe_all_capabilities", lambda: {
        "gpu_finetune": {"online": True}, "gemini": {"online": False},
    })
    state = {"query": "Revise the answer", "retry_count": 2, "critic_feedback": {"issues_found": ["Wrong page"]}}
    attempted_modes = []

    def gpu_unavailable(inputs):
        attempted_modes.append(settings.get_inference_mode("summarizer"))
        assert inputs is state
        raise RuntimeError("Test GPU unavailable")

    def run_local(inputs):
        attempted_modes.append(settings.get_inference_mode("summarizer"))
        assert inputs is state
        return {"answer": client.generate(inputs["query"], agent="summarizer"), "critic_action": "approve"}

    monkeypatch.setattr(summarizer._gpu_finetune, "run", gpu_unavailable)
    monkeypatch.setattr(summarizer._cpu_max, "run", run_local)
    with agent_inference_mode("hybrid"):
        result = summarizer.run(state)
        assert settings.get_inference_mode("summarizer") == "hybrid"

    assert summarizer._cpu_max.max_internal_retry == 3
    assert attempted_modes == ["gpu_finetune", "cpu_max"]
    assert result["answer"] == "local answer 1"
    assert calls == [("ollama", None)]
