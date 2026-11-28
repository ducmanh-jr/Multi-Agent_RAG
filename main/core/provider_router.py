"""
Unified Hybrid Provider Router & Policy Engine (PR-03).
Cleanly separates:
  - ExecutionMode: cpu_min | cpu_max | gpu_finetune | gpu_max | hybrid | gemini
  - ModelProvider: gemini | kaggle | local
  - AgentRole: planner | retrieval | embedding | reranker | reader | generator | critic | fact_checker | writer | citation_audit

Guarantees:
  1. Single-mode isolation: cpu_min, cpu_max, gpu_finetune, gpu_max NEVER silently call Gemini.
  2. Hybrid routing: supports Gemini Flash (configured via DEFAULT_LLM_MODEL, e.g. gemini-3.5-flash)
     as primary LLM reasoning provider for complex/multi-doc tasks (or when HYBRID_PROVIDER_PROFILE=gemini_first),
     with Kaggle Qwen as fast-path/fallback.
  3. Grounding invariants: embedding (BGE-M3), reranker (Guarded MiniLM), fact_checker (Local/Kaggle BGE NLI),
     and citation_audit remain strictly local/Kaggle primary (never replaced by Gemini, and verifier must be
     cross-family from the generator).
"""

from __future__ import annotations

import json
from typing import Any, Literal, Mapping, Optional

from core.config import get_run_settings
from utils.logger import logger


ExecutionMode = Literal["cpu_min", "cpu_max", "gpu_finetune", "gpu_max", "hybrid", "gemini"]
ModelProvider = Literal["gemini", "kaggle", "local"]
AgentRole = Literal[
    "planner", "retrieval", "embedding", "reranker", "reader",
    "generator", "critic", "fact_checker", "writer", "citation_audit",
]

DEFAULT_HYBRID_PROVIDER_POLICY: dict[str, str] = {
    "planner": "gemini",
    "reader": "local",
    "retrieval": "kaggle",
    "embedding": "kaggle",
    "reranker": "kaggle",
    "generator": "gemini",
    "critic": "gemini",
    "fact_checker": "kaggle",
    "writer": "gemini",
    "citation_audit": "local",
}

GEMINI_ALLOWED_EXECUTION_MODES: frozenset[str] = frozenset({"gemini", "api", "local_api", "hybrid", "auto", "optimal"})
STRICT_LOCAL_ROLES: frozenset[str] = frozenset({"embedding", "reranker", "fact_checker", "citation_audit"})
VALID_PROVIDERS: frozenset[str] = frozenset({"gemini", "kaggle", "local"})


class ProviderPolicyError(RuntimeError):
    """Raised when a caller attempts to invoke a provider forbidden in the active execution mode."""


def _clean(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def is_provider_allowed(provider: str, execution_mode: Optional[str]) -> bool:
    """Checks whether `provider` is permitted under `execution_mode` without breaking mode isolation."""
    prov = _clean(provider)
    mode = _clean(execution_mode) or "cpu_min"
    if prov == "gemini":
        return mode in GEMINI_ALLOWED_EXECUTION_MODES
    if prov == "kaggle":
        return mode in {"gpu_finetune", "gpu_max", "kaggle_max", "kaggle", "hybrid", "auto", "optimal"}
    if prov == "local":
        return True
    return False


def assert_provider_allowed(provider: str, execution_mode: Optional[str], *, role: str = "") -> None:
    """Raises ProviderPolicyError if `provider` violates mode isolation."""
    if not is_provider_allowed(provider, execution_mode):
        raise ProviderPolicyError(
            f"Provider '{provider}' is not permitted under execution mode '{execution_mode}' (role='{role}')."
        )


def normalize_model_family(family: Optional[str]) -> str:
    """Normalize a model/backend identifier into a coarse family token for cross-family verification checks."""
    low = _clean(family)
    if not low or low == "unknown":
        return "unknown"
    if "gemini" in low or "google" in low:
        return "gemini"
    if "qwen" in low or "gpu_max" in low or "kaggle_max" in low:
        return "qwen"
    if "bge" in low or "nli" in low or "minicheck" in low or "deberta" in low:
        return "nli"
    if "deepseek" in low:
        return "deepseek"
    if "gpt" in low or "openai" in low:
        return "openai"
    if "cpu" in low or "local" in low or "extractive" in low or "ollama" in low:
        return "local"
    return low


def is_cross_family_verifier(generator_family: Optional[str], verifier_backend: str) -> bool:
    """
    Ensures the fact-checker does not use the same LLM family that generated the draft.
    Dedicated NLI / deterministic local verifiers (`gpu_finetune` BGE-reranker NLI, `cpu_max` MiniCheck/deterministic)
    are always cross-family relative to generative LLMs (Gemini, Qwen, DeepSeek, OpenAI).
    """
    gen_fam = normalize_model_family(generator_family)
    ver_key = _clean(verifier_backend)
    if ver_key in {"gpu_finetune", "cpu_max", "local"}:
        return True
    ver_fam = normalize_model_family(ver_key)
    if gen_fam == "unknown" or ver_fam == "unknown":
        return ver_fam != "gemini"
    return gen_fam != ver_fam


def is_gemini_permitted(execution_mode: Optional[str], *, role: str = "") -> bool:
    """
    Single centralized policy check governing whether any Gemini API call is allowed.
    Enforces:
      1. Execution-mode isolation (single local/GPU modes never call Gemini).
      2. HYBRID_PROVIDER_PROFILE=local_only zero-Gemini mode.
      3. STRICT_LOCAL_ROLES invariant in hybrid mode (embedding, reranker, fact_checker, citation_audit
         never route to Gemini).
    """
    mode = _clean(execution_mode) or "cpu_min"
    role_key = _clean(role)
    if not is_provider_allowed("gemini", mode):
        return False
    if mode in {"hybrid", "auto", "optimal"}:
        if role_key in STRICT_LOCAL_ROLES:
            return False
        profile = _clean(getattr(get_run_settings(), "HYBRID_PROVIDER_PROFILE", "auto")) or "auto"
        if profile == "local_only":
            return False
    return True


def load_hybrid_provider_policy(overrides: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """
    Loads the effective Hybrid role -> provider policy from defaults + settings + optional overrides.
    Always protects STRICT_LOCAL_ROLES from routing to 'gemini' and keeps citation_audit strictly 'local'.
    """
    policy = dict(DEFAULT_HYBRID_PROVIDER_POLICY)
    settings = get_run_settings()
    raw_json = getattr(settings, "HYBRID_PROVIDER_POLICY", "") or ""
    if isinstance(raw_json, str) and raw_json.strip():
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                for role, prov in parsed.items():
                    r_key = _clean(role)
                    p_val = _clean(prov)
                    if r_key in policy and p_val in VALID_PROVIDERS:
                        policy[r_key] = p_val
        except (ValueError, TypeError) as exc:
            logger.warning("[ProviderRouter] Ignoring malformed HYBRID_PROVIDER_POLICY JSON: %s", exc)
    if overrides:
        for role, prov in overrides.items():
            r_key = _clean(role)
            p_val = _clean(prov)
            if r_key in policy and p_val in VALID_PROVIDERS:
                policy[r_key] = p_val
    # Grounding invariants: STRICT_LOCAL_ROLES never route to Gemini; citation_audit stays strictly local.
    for strict_role in STRICT_LOCAL_ROLES:
        if policy.get(strict_role) == "gemini":
            policy[strict_role] = DEFAULT_HYBRID_PROVIDER_POLICY.get(strict_role, "local")
    policy["citation_audit"] = "local"
    return policy


def _is_complex_reasoning_task(state: Mapping[str, Any]) -> bool:
    """Detects multi-document, comparative, or HIGH-complexity synthesis tasks suited for Gemini primary."""
    complexity = str(state.get("complexity") or "").upper()
    if complexity == "HIGH":
        return True
    if state.get("is_comparative_query"):
        return True
    pdf_paths = state.get("pdf_paths") or []
    if len(set(p for p in pdf_paths if p)) >= 2:
        return True
    units = state.get("source_units") or []
    if len({u.get("doc_id") for u in units if isinstance(u, dict) and u.get("doc_id")}) >= 2:
        return True
    return False


class HybridProviderRouter:
    """
    Resolves ordered backend execution chains for each AgentRole in Hybrid mode,
    separating ExecutionMode from ModelProvider and preserving local verification invariants.
    """

    def resolve_chain(
        self,
        role: str,
        *,
        state: Optional[Mapping[str, Any]] = None,
        caps: Optional[Mapping[str, Any]] = None,
        policy_overrides: Optional[Mapping[str, str]] = None,
    ) -> list[str]:
        """
        Returns an ordered list of concrete backend keys to attempt for `role`:
          - Generator/Summarizer/Writer keys: 'gemini', 'gpu_finetune', 'gpu_max', 'cpu_max'
          - Fact-Checker keys: 'gpu_finetune', 'cpu_max', 'gpu_max', 'gemini'
          - Planner keys: 'gemini', 'gpu_finetune', 'local'
        Only includes online/available remote backends plus guaranteed local fallback.
        """
        role_key = _clean(role)
        state = state or {}
        caps = caps or {}
        settings = get_run_settings()
        profile = _clean(getattr(settings, "HYBRID_PROVIDER_PROFILE", "auto")) or "auto"
        policy = load_hybrid_provider_policy(policy_overrides)

        gemini_online = bool(caps.get("gemini", {}).get("online"))
        gpu_ft_online = bool(caps.get("gpu_finetune", {}).get("online"))
        gpu_max_online = bool(caps.get("kaggle_max", {}).get("online") and caps.get("kaggle_max", {}).get("has_vllm_32b"))

        if role_key == "fact_checker":
            # Verification invariants:
            # 1. STRICT_LOCAL_ROLES: Gemini is NEVER placed in the Hybrid fact_checker chain.
            # 2. Cross-family rule: verifier must not share the generator's model family.
            gen_family = str(state.get("generator_model_family") or "unknown")
            chain: list[str] = []
            if gpu_ft_online:
                chain.append("gpu_finetune")
            chain.append("cpu_max")
            if gpu_max_online and is_cross_family_verifier(gen_family, "gpu_max"):
                chain.append("gpu_max")
            return chain

        if role_key in {"embedding", "reranker"}:
            return ["gpu_finetune"] if gpu_ft_online else ["local"]

        if role_key == "citation_audit":
            return ["local"]

        if role_key == "planner":
            preferred = policy.get("planner", "gemini")
            chain = []
            if profile == "local_only":
                if gpu_ft_online:
                    chain.append("gpu_finetune")
            elif preferred == "gemini":
                if gemini_online:
                    chain.append("gemini")
                if gpu_ft_online:
                    chain.append("gpu_finetune")
            else:
                if gpu_ft_online:
                    chain.append("gpu_finetune")
                if gemini_online:
                    chain.append("gemini")
            chain.append("local")
            return chain

        if role_key in {"generator", "summarizer", "critic", "writer"}:
            policy_role = "generator" if role_key == "summarizer" else role_key
            preferred = policy.get(policy_role, "gemini")
            use_gemini_primary = (
                gemini_online
                and preferred == "gemini"
                and (
                    profile == "gemini_first"
                    or (profile == "auto" and _is_complex_reasoning_task(state))
                )
            )
            chain = []
            if use_gemini_primary:
                chain.append("gemini")
                if gpu_ft_online:
                    chain.append("gpu_finetune")
                if gpu_max_online:
                    chain.append("gpu_max")
                chain.append("cpu_max")
            elif preferred == "local" or profile == "local_only":
                chain.append("cpu_max")
                if gpu_ft_online:
                    chain.append("gpu_finetune")
                if gemini_online and profile != "local_only":
                    chain.append("gemini")
            else:
                # Auto fast-path for simple/single-doc tasks: Kaggle Qwen 7B -> GPU Max -> Local CPU -> Gemini
                if gpu_ft_online:
                    chain.append("gpu_finetune")
                if gpu_max_online:
                    chain.append("gpu_max")
                chain.append("cpu_max")
                if gemini_online:
                    chain.append("gemini")
            return list(dict.fromkeys(chain))

        return ["cpu_max"]


provider_router = HybridProviderRouter()
