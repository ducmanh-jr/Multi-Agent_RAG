"""Collect worker checkpoint reports in the scope of one research run."""

from contextlib import contextmanager
from contextvars import ContextVar
import re
from threading import Lock
from typing import Any

import httpx


WORKER_INFERENCE_ACTION = "worker_inference"
PROVIDER_INFERENCE_ACTION = "provider_inference"
_ENDPOINTS = {
    "api/router", "api/summarize/generator", "api/summarize/critic",
    "api/writer", "api/factcheck", "generate", "v1/chat/completions",
}
_ADAPTERS = {"generator", "critic", "router", "writer", "none"}
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}(?:/[A-Za-z0-9][A-Za-z0-9_.-]{0,95})?")
_SHA256 = re.compile(r"[a-f0-9]{64}")
_MODEL_REVISION = re.compile(r"(?:[a-f0-9]{40}|[a-f0-9]{64})")


class _Observations:
    def __init__(self):
        self.lock = Lock()
        self.events: list[dict[str, Any]] = []


_observations: ContextVar[_Observations | None] = ContextVar("worker_observations", default=None)


@contextmanager
def capture_worker_observations():
    token = _observations.set(_Observations())
    try:
        yield
    finally:
        _observations.reset(token)


def drain_worker_observations() -> list[dict[str, Any]]:
    pending = _observations.get()
    if pending is None:
        return []
    with pending.lock:
        events, pending.events = pending.events, []
    return events


def record_worker_observation(
    endpoint: str, adapter: str | None, response: httpx.Response | None = None, *, status: str = "transport_error",
) -> None:
    pending = _observations.get()
    if pending is None or endpoint not in _ENDPOINTS:
        return
    component = "verifier" if endpoint == "api/factcheck" else adapter
    if component not in _ADAPTERS and component != "verifier":
        return
    event: dict[str, Any] = {
        "action": WORKER_INFERENCE_ACTION, "endpoint": "/" + endpoint, "component": component, "status": status,
    }
    if response is not None:
        if response.status_code != 200:
            event.update(status="http_error", http_status=response.status_code)
        else:
            _read_worker_headers(event, response.headers)
    with pending.lock:
        pending.events.append(event)


def _read_worker_headers(event: dict[str, Any], headers: httpx.Headers) -> None:
    event["status"] = "unreported"
    model = headers.get("X-Research-Model", "")
    revision = headers.get("X-Research-Revision", "")
    adapter = headers.get("X-Research-Adapter")
    adapter_sha256 = headers.get("X-Research-Adapter-SHA256", "")
    component = event["component"]
    if component == "verifier" and headers.get("X-Research-Model-Status") == "unavailable":
        event["status"] = "unavailable"
        return
    if not _MODEL_NAME.fullmatch(model):
        return
    if component == "verifier":
        if adapter is not None:
            return
    elif adapter not in {component, "none"}:
        return
    event["model"] = model
    if _MODEL_REVISION.fullmatch(revision):
        event["revision"] = revision
    if adapter is not None:
        event["adapter"] = adapter
        if adapter != "none" and _SHA256.fullmatch(adapter_sha256):
            event["adapter_sha256"] = adapter_sha256
    if "revision" in event and (adapter in {None, "none"} or "adapter_sha256" in event):
        event["status"] = "reported"


def record_provider_observation(
    *,
    provider: str,
    operation: str,
    status: str,
    requested_model: str | None = None,
    attempted_models: list[str] | tuple[str, ...] = (),
    served_model: str | None = None,
    model_version: str | None = None,
    fallback_reason: str | None = None,
    agent: str | None = None,
) -> None:
    """Record cloud/local/fallback LLM provider execution metadata within an active observation scope."""
    pending = _observations.get()
    if pending is None:
        return
    event: dict[str, Any] = {
        "action": PROVIDER_INFERENCE_ACTION,
        "provider": str(provider or "unknown"),
        "operation": str(operation or "generate"),
        "status": str(status or "unreported"),
    }
    if agent:
        event["agent"] = str(agent)
    if requested_model:
        event["requested_model"] = str(requested_model)
    if attempted_models:
        event["attempted_models"] = [str(m) for m in attempted_models if m]
    if served_model:
        event["served_model"] = str(served_model)
    if model_version:
        event["model_version"] = str(model_version)
    if fallback_reason:
        event["fallback_reason"] = str(fallback_reason)[:300]
    with pending.lock:
        pending.events.append(event)
