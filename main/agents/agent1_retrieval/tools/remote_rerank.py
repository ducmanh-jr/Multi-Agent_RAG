"""Cross-encoder client that preserves the candidate set and validates every score."""

import math
import re
from typing import Any

import httpx

from core.config import get_run_settings
from core.token_usage import metered_post


MAX_CANDIDATES = 15
MAX_TEXT_LENGTH = 4096


ALLOWED_QWEN3_RERANKER_MODELS = frozenset({
    "Qwen/Qwen3-Reranker-8B",
    "Qwen/Qwen3-Reranker-4B",
    "Qwen/Qwen3-Reranker-0.6B",
})


class RerankError(RuntimeError):
    """The caller can keep the original retrieval ranking."""


def rerank_passages(
    query: str, candidates: list[dict[str, Any]], *, client: httpx.Client, url: str, model: str, timeout: float,
    keep_prefix: int = 0, max_text_length: int = MAX_TEXT_LENGTH,
) -> tuple[list[dict[str, Any]], str, bool]:
    effective_max = max(256, min(int(max_text_length or MAX_TEXT_LENGTH), MAX_TEXT_LENGTH))
    texts = [str(doc.get("text") or "").strip()[:effective_max] for doc in candidates[:MAX_CANDIDATES]]
    clean_query = query.strip()[:MAX_TEXT_LENGTH]
    if not clean_query or not all(texts):
        raise RerankError("missing_rerank_input")
    token = (getattr(get_run_settings(), "REMOTE_INFERENCE_TOKEN", "") or "").strip()
    req_kwargs: dict[str, Any] = {"json": {"query": clean_query, "candidates": texts}, "timeout": timeout}
    if token:
        req_kwargs["headers"] = {"Authorization": f"Bearer {token}"}
    try:
        response = metered_post(
            client.post, f"{url.rstrip('/')}/api/rerank", provider="kaggle", operation="api/rerank", protocol="worker",
            **req_kwargs,
        )
    except httpx.TimeoutException as exc:
        raise RerankError("reranker_timeout") from exc
    except httpx.RequestError as exc:
        raise RerankError("reranker_unreachable") from exc
    if response.status_code != 200:
        raise RerankError(f"reranker_http_{response.status_code}")
    try:
        data = response.json()
        if data["model"] != model and data["model"] not in ALLOWED_QWEN3_RERANKER_MODELS:
            raise RerankError("reranker_model_mismatch")
        revision, scores, indices = data["revision"], data["scores"], data["ranked_indices"]
        if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise ValueError("Invalid reranker revision")
        if (not isinstance(scores, list) or len(scores) != len(texts)
                or any(type(score) not in (float, int) or not math.isfinite(score) for score in scores)):
            raise ValueError("Expected one finite score for each passage")
        original_order = list(range(len(texts)))
        expected_order = sorted(original_order, key=scores.__getitem__, reverse=True)
        if not isinstance(indices, list) or any(type(index) is not int for index in indices) or indices != expected_order:
            raise ValueError("Invalid reranker indices")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RerankError("invalid_reranker_response") from exc
    # Protect top-ranked evidence after validating the full model order.
    order = original_order[:keep_prefix] + [index for index in indices if index >= keep_prefix]
    ordered = [candidates[index] for index in order]
    return ordered + candidates[len(texts):], revision, order != original_order
