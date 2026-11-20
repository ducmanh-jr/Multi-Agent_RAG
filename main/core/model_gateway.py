"""
Centralized Model Gateway & Provider Policy Registry.

Enforces strict provider isolation across execution modes per AGENT.md (Section 16):
  - Official execution modes: cpu_min, cpu_max, gpu_finetune, gpu_max, hybrid, gemini.
  - Gemini API is permitted ONLY when execution mode is strictly "gemini".
  - Modes cpu_min, cpu_max, gpu_finetune, gpu_max, and hybrid are
    strictly forbidden from instantiating Gemini clients/agents, calling Gemini API,
    or falling back to Gemini on error.
  - Hybrid mode backend selection is centralized here:
      GPU Finetune (gpu_finetune) -> GPU Max (gpu_max) -> Local CPU (zero Gemini).
"""

from __future__ import annotations

from typing import Any, Optional

from utils.logger import logger

OFFICIAL_EXECUTION_MODES: frozenset[str] = frozenset(
    {"cpu_min", "cpu_max", "gpu_finetune", "gpu_max", "hybrid", "gemini"}
)

GEMINI_ALLOWED_MODES: frozenset[str] = frozenset({"gemini"})

NON_GEMINI_MODES: frozenset[str] = frozenset(
    {
        "cpu_min",
        "cpu_max",
        "cpu",
        "local_cpu",
        "gpu_finetune",
        "gpu_max",
        "kaggle_fn",
        "kaggle_max",
        "kaggle_gpu",
        "kaggle",
        "model",
        "hybrid",
    }
)

HYBRID_ALLOWED_BACKENDS: tuple[str, ...] = ("gpu_finetune", "gpu_max", "cpu_max")


class ProviderPolicyError(RuntimeError):
    """Raised when a caller attempts to use a provider forbidden in the active execution mode."""


class ProviderConfigurationError(RuntimeError):
    """Raised when an allowed provider is missing required configuration (e.g. API key)."""


def _clean_mode(mode: Optional[str]) -> str:
    return (mode or "").strip().lower()


def validate_execution_mode(mode: Optional[str]) -> str:
    """Validates that `mode` is one of the 6 official execution modes."""
    clean = _clean_mode(mode)
    if clean not in OFFICIAL_EXECUTION_MODES:
        raise ValueError(
            f"Unsupported execution mode '{clean or 'unset'}'. "
            f"Official modes: {sorted(OFFICIAL_EXECUTION_MODES)}."
        )
    return clean


def is_gemini_allowed(mode: Optional[str]) -> bool:
    """Returns True ONLY if execution mode is strictly 'gemini'."""
    return _clean_mode(mode) in GEMINI_ALLOWED_MODES


def assert_provider_allowed(
    provider: str,
    mode: Optional[str],
    caller: str = "",
) -> None:
    """
    Validates that `provider` is permitted under `mode`.
    Raises ProviderPolicyError on any violation.
    """
    clean_provider = (provider or "").strip().lower()
    clean_mode = _clean_mode(mode)

    if clean_provider == "gemini" and clean_mode not in GEMINI_ALLOWED_MODES:
        ctx = f" (caller={caller})" if caller else ""
        msg = (
            f"Provider policy violation: 'gemini' is strictly forbidden in execution mode "
            f"'{clean_mode or 'unset'}'{ctx}. Allowed modes: {sorted(GEMINI_ALLOWED_MODES)}."
        )
        logger.error(f"[ModelGateway] {msg}")
        raise ProviderPolicyError(msg)


def assert_gemini_ready(
    mode: Optional[str],
    api_key: Optional[str],
    caller: str = "",
) -> None:
    """
    Enforces both policy and configuration checks before any Gemini API call:
      1. Mode must be strictly 'gemini'.
      2. GEMINI_API_KEY must be present and non-empty (no silent provider switch).
    """
    assert_provider_allowed("gemini", mode, caller=caller)
    if not (api_key or "").strip():
        ctx = f" (caller={caller})" if caller else ""
        msg = (
            f"Gemini configuration error: execution mode is 'gemini' but GEMINI_API_KEY is missing{ctx}. "
            f"Set GEMINI_API_KEY in environment or settings."
        )
        logger.error(f"[ModelGateway] {msg}")
        raise ProviderConfigurationError(msg)


def guard_agent_instantiation(
    agent_name: str,
    provider: str,
    mode: Optional[str],
) -> None:
    """Blocks direct instantiation of provider-bound agents outside their permitted modes."""
    assert_provider_allowed(provider, mode, caller=agent_name)


def _is_gpu_finetune_online(caps: dict[str, Any]) -> bool:
    return bool(caps.get("gpu_finetune", {}).get("online"))


def _is_gpu_max_online(caps: dict[str, Any]) -> bool:
    kmax = caps.get("kaggle_max", {}) or caps.get("gpu_max", {}) or {}
    return bool(kmax.get("online") and kmax.get("has_vllm_32b", True))


def get_hybrid_backend_chain(
    caps: Optional[dict[str, Any]],
    stage: str = "general",
) -> list[str]:
    """
    Returns ordered backend candidates for Hybrid mode:
      GPU Finetune -> GPU Max (where supported) -> Local CPU.
    Gemini is NEVER included in any Hybrid chain.
    """
    safe_caps = caps or {}
    clean_stage = (stage or "general").strip().lower()
    chain: list[str] = []

    if _is_gpu_finetune_online(safe_caps):
        chain.append("gpu_finetune")

    if clean_stage in ("summarizer", "fact_checker", "factchecker", "general"):
        if _is_gpu_max_online(safe_caps):
            chain.append("gpu_max")

    # Local CPU is always available as deterministic/local fallback in Hybrid mode.
    chain.append("cpu_max")
    return chain


def select_hybrid_backend(
    caps: Optional[dict[str, Any]],
    stage: str = "general",
) -> str:
    """Returns the highest-priority permitted backend for Hybrid mode."""
    return get_hybrid_backend_chain(caps, stage=stage)[0]


class ModelGateway:
    """Facade object for centralized provider policy and Hybrid backend selection."""

    OFFICIAL_EXECUTION_MODES = OFFICIAL_EXECUTION_MODES
    GEMINI_ALLOWED_MODES = GEMINI_ALLOWED_MODES
    HYBRID_ALLOWED_BACKENDS = HYBRID_ALLOWED_BACKENDS

    def validate_execution_mode(self, mode: Optional[str]) -> str:
        return validate_execution_mode(mode)

    def is_gemini_allowed(self, mode: Optional[str]) -> bool:
        return is_gemini_allowed(mode)

    def assert_provider_allowed(
        self,
        provider: str,
        mode: Optional[str],
        caller: str = "",
    ) -> None:
        assert_provider_allowed(provider, mode, caller=caller)

    def assert_gemini_ready(
        self,
        mode: Optional[str],
        api_key: Optional[str],
        caller: str = "",
    ) -> None:
        assert_gemini_ready(mode, api_key, caller=caller)

    def guard_agent_instantiation(
        self,
        agent_name: str,
        provider: str,
        mode: Optional[str],
    ) -> None:
        guard_agent_instantiation(agent_name, provider, mode)

    def get_hybrid_backend_chain(
        self,
        caps: Optional[dict[str, Any]],
        stage: str = "general",
    ) -> list[str]:
        return get_hybrid_backend_chain(caps, stage=stage)

    def select_hybrid_backend(
        self,
        caps: Optional[dict[str, Any]],
        stage: str = "general",
    ) -> str:
        return select_hybrid_backend(caps, stage=stage)


model_gateway = ModelGateway()
