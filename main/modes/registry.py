"""
Mode Registry & Dynamic Agent Factory for Multi-Agent Research RAG.
Maps the 6 distinct execution modes (CPU Min, CPU Max, Gemini API, GPU FT, GPU Max, Hybrid)
to their independent 5-agent implementations.
"""

from __future__ import annotations

from typing import Optional
from core.base_agent import BaseAgent
from core.config import settings
from utils.logger import logger

MODE_MAP = {
    "gemini": "modes.gemini",
    "cpu_min": "modes.cpu_min",
    "cpu_max": "modes.cpu_max",
    "gpu_ft": "modes.gpu_ft",
    "gpu_finetune": "modes.gpu_finetune",
    "gpu_max": "modes.gpu_max",
    "hybrid": "modes.hybrid",
    # Backward-compatible aliases
    "kaggle_fn": "modes.gpu_ft",
    "kaggle_max": "modes.gpu_max",
}

MODE_ALIASES = {
    "api": "gemini",
    "local_api": "gemini",
    "cpu": "cpu_max",
    "model": "gpu_max",
    "kaggle": "gpu_max",
    "kaggle_gpu": "gpu_max",
    "kaggle_max": "gpu_max",
    "gpumax": "gpu_max",
    "gpu_max": "gpu_max",
    "kaggle_fn": "gpu_ft",
    "fn": "gpu_ft",
    "gpuft": "gpu_ft",
    "gpu_ft": "gpu_ft",
    "ft": "gpu_ft",
    "gpu_finetune": "gpu_finetune",
    "finetune": "gpu_finetune",
    "gpufinetune": "gpu_finetune",
    "gpu_ft_new": "gpu_finetune",
    "auto": "hybrid",
    "optimal": "hybrid",
}


def normalize_mode(mode: Optional[str]) -> str:
    """Normalize user or config mode string to one of the canonical modes."""
    if not mode:
        return "cpu_min"
    clean = mode.lower().strip()
    return MODE_ALIASES.get(clean, clean)


def get_agent_class(agent_name: str, mode: str):
    """Dynamically imports and returns the Agent class for the given agent and mode."""
    canon_mode = normalize_mode(mode)
    clean_agent = agent_name.lower().replace("agent", "").replace("_", "").strip()

    if canon_mode == "cpu_min":
        import modes.cpu_min as m
        mapping = {
            "retrieval": m.CPUMinRetrievalAgent,
            "reader": m.CPUMinReaderAgent,
            "summarizer": m.CPUMinSummarizerAgent,
            "factchecker": m.CPUMinFactCheckerAgent,
            "writer": m.CPUMinWriterAgent,
        }
    elif canon_mode == "gemini":
        import modes.gemini as m
        mapping = {
            "retrieval": m.GeminiRetrievalAgent,
            "reader": m.GeminiReaderAgent,
            "summarizer": m.GeminiSummarizerAgent,
            "factchecker": m.GeminiFactCheckerAgent,
            "writer": m.GeminiWriterAgent,
        }
    elif canon_mode in ("gpu_ft", "kaggle_fn"):
        import modes.gpu_ft as m
        mapping = {
            "retrieval": m.GPUFTRetrievalAgent,
            "reader": m.GPUFTReaderAgent,
            "summarizer": m.GPUFTSummarizerAgent,
            "factchecker": m.GPUFTFactCheckerAgent,
            "writer": m.GPUFTWriterAgent,
        }
    elif canon_mode == "gpu_finetune":
        import modes.gpu_finetune as m
        mapping = {
            "retrieval": m.GPUFinetuneRetrievalAgent,
            "reader": m.GPUFinetuneReaderAgent,
            "summarizer": m.GPUFinetuneSummarizerAgent,
            "factchecker": m.GPUFinetuneFactCheckerAgent,
            "writer": m.GPUFinetuneWriterAgent,
        }
    elif canon_mode in ("gpu_max", "kaggle_max"):
        import modes.gpu_max as m
        mapping = {
            "retrieval": m.GPUMaxRetrievalAgent,
            "reader": m.GPUMaxReaderAgent,
            "summarizer": m.GPUMaxSummarizerAgent,
            "factchecker": m.GPUMaxFactCheckerAgent,
            "writer": m.GPUMaxWriterAgent,
        }
    elif canon_mode == "hybrid":
        import modes.hybrid as m
        mapping = {
            "retrieval": m.HybridRetrievalAgent,
            "reader": m.HybridReaderAgent,
            "summarizer": m.HybridSummarizerAgent,
            "factchecker": m.HybridFactCheckerAgent,
            "writer": m.HybridWriterAgent,
        }
    else:  # default cpu_max
        import modes.cpu_max as m
        mapping = {
            "retrieval": m.CPUMaxRetrievalAgent,
            "reader": m.CPUMaxReaderAgent,
            "summarizer": m.CPUMaxSummarizerAgent,
            "factchecker": m.CPUMaxFactCheckerAgent,
            "writer": m.CPUMaxWriterAgent,
        }

    cls = mapping.get(clean_agent)
    if not cls:
        raise ValueError(f"Unknown agent '{clean_agent}' for mode '{canon_mode}'")
    return cls


def get_agent(agent_name: str, mode: Optional[str] = None) -> BaseAgent:
    """Instantiate a single agent for a specific mode."""
    target_mode = mode or settings.get_agent_mode(agent_name) or settings.EXECUTION_MODE
    cls = get_agent_class(agent_name, target_mode)
    return cls()


def get_mode_agents(mode: str) -> dict[str, BaseAgent]:
    """Instantiate all 5 agents strictly within the same mode."""
    canon_mode = normalize_mode(mode)
    logger.info(f"[Mode Registry] Initializing all 5 agents under mode: '{canon_mode}'")
    return {
        "retrieval": get_agent_class("retrieval", canon_mode)(),
        "reader": get_agent_class("reader", canon_mode)(),
        "summarizer": get_agent_class("summarizer", canon_mode)(),
        "fact_checker": get_agent_class("factchecker", canon_mode)(),
        "writer": get_agent_class("writer", canon_mode)(),
    }


def get_active_agents() -> dict[str, BaseAgent]:
    """
    Instantiate agents according to current settings:
    Allows per-agent custom configuration or unified configuration under settings.EXECUTION_MODE.
    """
    global_mode = normalize_mode(settings.EXECUTION_MODE)
    agents = {}
    for agent_key in ["retrieval", "reader", "summarizer", "fact_checker", "writer"]:
        clean_key = agent_key.replace("_", "")
        agent_mode = settings.get_agent_mode(agent_key)
        mode_to_use = normalize_mode(agent_mode) if agent_mode else global_mode
        agents[agent_key] = get_agent_class(clean_key, mode_to_use)()
    return agents
