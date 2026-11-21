"""Collect worker checkpoint reports in the scope of one research run."""

from contextlib import contextmanager
from contextvars import ContextVar
import re
from threading import Lock
from typing import Any

import httpx


WORKER_INFERENCE_ACTION = "worker_inference"
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
    expected_adapter = None if component == "verifier" else component
    if adapter != expected_adapter or not _MODEL_NAME.fullmatch(model):
        return
    event["model"] = model
    revision_pattern = _SHA256 if component == "verifier" else _MODEL_REVISION
    if revision_pattern.fullmatch(revision):
        event["revision"] = revision
    if adapter is not None:
        event["adapter"] = adapter
        if adapter != "none" and _SHA256.fullmatch(adapter_sha256):
            event["adapter_sha256"] = adapter_sha256
    if "revision" in event and (adapter in {None, "none"} or "adapter_sha256" in event):
        event["status"] = "reported"
