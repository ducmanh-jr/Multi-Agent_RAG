"""Record research settings at admission and read observations from the same run."""

from typing import Any

from core.config import RunConfig, use_run_config
from core.comparison_plan import COMPARISON_SCHEDULE
from core.coordination import reflection_limit
from core.token_usage import TOKEN_BUDGET_EXHAUSTED, TOKEN_USAGE_ACTION
from core.worker_observations import PROVIDER_INFERENCE_ACTION, WORKER_INFERENCE_ACTION
from modes.registry import normalize_mode, resolve_agent_modes


RUN_MANIFEST_ACTION = "run_manifest"


def manifest_event(
    config: RunConfig, *, mode: str | None, auto_scout: bool, scout_enabled: bool,
    reflection_loops: int | None, parallel_retrieval: bool = False, comparison_plan: bool = False,
    token_budget: int | None = None,
) -> dict[str, Any]:
    with use_run_config(config):
        limit = reflection_limit({} if reflection_loops is None else {"max_reflection_loops": reflection_loops})

    agent_modes = resolve_agent_modes(config, mode)
    # Allowlist replay metadata. RunConfig also contains provider credentials and private paths.
    return {
        "action": RUN_MANIFEST_ACTION,
        "schema_version": 1,
        "configuration": {
            "execution_mode": config.EXECUTION_MODE,
            "mode_override": normalize_mode(mode) if mode else None,
            "agent_modes": agent_modes,
            "auto_scout_requested": auto_scout,
            "auto_scout_enabled": scout_enabled,
            "comparison_plan_requested": comparison_plan,
            "comparison_plan_enabled": bool(comparison_plan),
            "token_budget": token_budget,
            "remote_inference_configured": bool(config.REMOTE_INFERENCE_URL),
            "local_cpu_configured": bool(config.LOCAL_CPU_URL),
            "models": {
                "default_llm": config.DEFAULT_LLM_MODEL,
                "fallback_llm": config.FALLBACK_LLM_MODEL,
                "local_cpu": config.LOCAL_CPU_MODEL,
                "embedding": config.EMBEDDING_MODEL,
                "fallback_embedding": config.FALLBACK_EMBEDDING_MODEL,
                "reranker": config.RERANKER_MODEL,
            },
            "retrieval": {
                "parallel_requested": parallel_retrieval,
                "parallel_enabled": parallel_retrieval and agent_modes["retrieval"] == "cpu_min",
                "dense_enabled": config.GPU_FINETUNE_DENSE_ENABLED,
                "page_diversity_enabled": config.GPU_FINETUNE_PAGE_DIVERSITY_ENABLED,
                "rerank_enabled": config.GPU_FINETUNE_RERANK_ENABLED,
                "rerank_keep_prefix": config.GPU_FINETUNE_RERANK_KEEP_PREFIX,
            },
            "figures": {
                "vision_provider": config.FIGURE_VISION_PROVIDER,
                "vision_model": config.FIGURE_VISION_MODEL,
                "ocr_language": config.FIGURE_OCR_LANGUAGE,
            },
            "limits": {
                "reflection_loops": limit,
                "requested_reflection_loops": reflection_loops,
                "default_max_internal_retry": config.DEFAULT_MAX_INTERNAL_RETRY,
                "max_factcheck_calls": config.MAX_FACTCHECK_CALLS,
                "max_retrieved_chunks": config.MAX_RETRIEVED_CHUNKS,
                "top_k_rerank": config.TOP_K_RERANK,
                "remote_timeout_seconds": config.REMOTE_TIMEOUT_SECONDS,
            },
        },
    }


def _component(details: Any, fields: set[str]) -> dict[str, Any]:
    return {key: value for key, value in details.items() if key in fields} if isinstance(details, dict) else {}


_RETRIEVAL_TRACE_ACTIONS = {
    "gpu_finetune_retrieval_completed",
    "hybrid_retrieval_completed",
    "gemini_retrieval_completed",
    "cpu_min_agentic_retrieval_completed",
    "cpu_max_retrieval_completed",
    "gpu_max_retrieval_completed",
}


def _retrieval_attempts(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attempts = []
    for event in trace:
        action = str(event.get("action") or "")
        if action not in _RETRIEVAL_TRACE_ACTIONS and not action.startswith("Truy hồi Hybrid"):
            continue
        details = event.get("details")
        if not isinstance(details, dict):
            continue
        attempt: dict[str, Any] = {
            "top_k": details.get("top_k"),
            "retrieved_count": details.get("retrieved_count"),
            "dense": _component(details.get("dense"), {"status", "backend", "reason", "model", "revision", "admitted_passages"}),
            "rerank": _component(details.get("rerank"), {"status", "reason", "model", "revision", "changed", "kept_prefix"}),
        }
        if action != "gpu_finetune_retrieval_completed":
            attempt["action"] = action
        if details.get("strategy_used"):
            attempt["strategy_used"] = details["strategy_used"]
        if details.get("cross_lingual_queries"):
            attempt["cross_lingual_queries"] = list(details["cross_lingual_queries"])
        if details.get("fallback_chain"):
            attempt["fallback_chain"] = list(details["fallback_chain"])
        attempts.append(attempt)
    return attempts


def _reported_revision(inferences: list[dict[str, Any]], component: str) -> str | None:
    calls = [
        call for call in inferences
        if call["component"] == component and call["status"] in {"reported", "unreported", "cache_hit"}
    ]
    if not calls or any(call["status"] != "reported" for call in calls):
        return None
    identities = {(call["model"], call["revision"], call.get("adapter_sha256")) for call in calls}
    if len(identities) == 1:
        _, revision, _ = identities.pop()
        return revision
    return None


def get_run_manifest(record: dict[str, Any]) -> dict[str, Any] | None:
    """Assemble the manifest from persisted admission, sources and run-local observations."""
    trace = [event for event in record.get("reasoning_trace") or [] if isinstance(event, dict)]
    admission = next((event for event in trace if event.get("action") == RUN_MANIFEST_ACTION), None)
    if admission is None:
        return None
    inferences = [
        {key: value for key, value in event.items() if key != "action"}
        for event in trace if event.get("action") == WORKER_INFERENCE_ACTION
    ]
    provider_inferences = [
        {key: value for key, value in event.items() if key != "action"}
        for event in trace if event.get("action") == PROVIDER_INFERENCE_ACTION
    ]
    usage = None
    if "token_budget" in admission["configuration"]:
        calls = [
            {key: value for key, value in event.items() if key != "action"}
            for event in trace if event.get("action") == TOKEN_USAGE_ACTION
        ]
        reported = [call for call in calls if call["status"] == "reported"]
        exhausted = any(event.get("answer_status") == TOKEN_BUDGET_EXHAUSTED for event in trace)
        usage = {
            "total_tokens": sum(call["total_tokens"] for call in reported),
            "reported_calls": len(reported),
            "unreported_calls": len(calls) - len(reported),
            "complete": len(reported) == len(calls) and (record["status"] == "completed" or exhausted),
            "calls": calls,
        }
    return {
        "schema_version": admission["schema_version"],
        "run_id": record["id"],
        "status": record["status"],
        "created_at": record["created_at"],
        "configuration": admission["configuration"],
        "documents": [
            {key: document[key] for key in ("id", "name", "sha256")} for document in record.get("documents") or []
        ],
        "observed": {
            "token_usage": usage,
            "retrieval_attempts": _retrieval_attempts(trace),
            "retrieval_schedules": [
                event["details"] for event in trace if event.get("action") == "retrieval_task_schedule"
            ],
            "comparison_schedules": [event["details"] for event in trace if event.get("action") == COMPARISON_SCHEDULE],
            "worker_inference": inferences,
            "provider_inference": provider_inferences,
            "generator_revision": _reported_revision(inferences, "generator"),
            "verifier_revision": _reported_revision(inferences, "verifier"),
        },
    }
