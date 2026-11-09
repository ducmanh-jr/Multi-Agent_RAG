"""
Tests for Modular Sub-packages Architecture (Phương án A).
Verifies that each of the 5 modes (gemini, cpu_min, cpu_max, kaggle_fn, kaggle_max)
independently provides the complete suite of 5 specialized agents.
"""

import pytest
from core.base_agent import BaseAgent
from core.graph import build_research_rag_graph
from modes.registry import get_mode_agents, get_agent, normalize_mode, MODE_MAP


MODES = ["gemini", "cpu_min", "cpu_max", "gpu_ft", "gpu_finetune", "gpu_max", "hybrid"]
EXPECTED_AGENTS = ["retrieval", "reader", "summarizer", "fact_checker", "writer"]


@pytest.mark.parametrize("mode", MODES)
def test_mode_package_contains_all_5_agents(mode):
    """Ensure that every mode provides all 5 agents and they inherit from BaseAgent."""
    agents = get_mode_agents(mode)
    assert len(agents) == 5, f"Mode '{mode}' must provide exactly 5 agents."
    for name in EXPECTED_AGENTS:
        assert name in agents, f"Agent '{name}' missing from mode '{mode}'"
        agent_instance = agents[name]
        assert isinstance(agent_instance, BaseAgent), f"Agent '{name}' in mode '{mode}' is not a BaseAgent"
        assert callable(getattr(agent_instance, "run", None)), f"Agent '{name}' in mode '{mode}' lacks run() method"


@pytest.mark.parametrize("mode", MODES)
def test_graph_compiles_for_all_modes(mode):
    """Ensure that LangGraph successfully compiles a runnable workflow for each mode."""
    graph = build_research_rag_graph(mode=mode)
    assert graph is not None


def test_normalize_mode_aliases():
    assert normalize_mode("API") == "gemini"
    assert normalize_mode("local_api") == "gemini"
    assert normalize_mode("cpu") == "cpu_max"
    assert normalize_mode("kaggle") == "gpu_max"
    assert normalize_mode("kaggle_max") == "gpu_max"
    assert normalize_mode("kaggle_fn") == "gpu_ft"
    assert normalize_mode("gpu_ft") == "gpu_ft"
    assert normalize_mode("gpu_finetune") == "gpu_finetune"
    assert normalize_mode("cpu_min") == "cpu_min"
