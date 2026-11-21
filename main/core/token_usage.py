"""Account for provider-reported tokens and stop budgeted research runs."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from threading import Lock

import httpx


TOKEN_USAGE_ACTION = "provider_token_usage"
TOKEN_BUDGET_EXHAUSTED = "TOKEN_BUDGET_EXHAUSTED"
TOKEN_USAGE_UNAVAILABLE = "TOKEN_USAGE_UNAVAILABLE"
_INPUT_HEADER = "X-Research-Input-Tokens"
_OUTPUT_HEADER = "X-Research-Output-Tokens"


class TokenBudgetStop(RuntimeError):
    def __init__(self, status: str):
        self.status = status
        super().__init__(status)


class RunTokenLedger:
    def __init__(self, limit: int | None):
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("token_budget must be a positive integer.")
        self.limit = limit
        self._total = 0
        self._failure = None
        self._closed = False
        self._events: list[dict] = []
        self._lock = Lock()
        self._provider_gate = Lock()

    @property
    def stop_status(self) -> str | None:
        with self._lock:
            return self._failure

    def check(self) -> None:
        if failure := self.stop_status:
            raise TokenBudgetStop(failure)

    def drain(self) -> list[dict]:
        with self._lock:
            events, self._events = self._events, []
            return events

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def _before_call(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("Research run has ended.")
            if self._failure is None and self.limit is not None and self._total >= self.limit:
                self._failure = TOKEN_BUDGET_EXHAUSTED
            if self._failure is not None:
                raise TokenBudgetStop(self._failure)

    def _observe(self, provider, operation, response, protocol):
        usage = _reported_usage(response, protocol) if response is not None else None
        with self._lock:
            if self._closed:
                raise RuntimeError("Research run has ended.")
            event = {
                "action": TOKEN_USAGE_ACTION, "provider": provider, "operation": operation,
                "status": "reported" if usage else "unreported",
                "http_status": response.status_code if response is not None else None,
            }
            if usage:
                event.update(usage)
                self._total += usage["total_tokens"]
            self._events.append(event)
            if self.limit is not None:
                if not usage:
                    self._failure = TOKEN_USAGE_UNAVAILABLE
                elif self._total > self.limit:
                    self._failure = TOKEN_BUDGET_EXHAUSTED
                if self._failure is not None:
                    raise TokenBudgetStop(self._failure)

    def post(self, send, url, *, provider, operation, protocol, **kwargs):
        # Only budgeted runs serialize calls. Two concurrent calls cannot both spend the same balance.
        with self._provider_gate if self.limit is not None else nullcontext():
            self._before_call()
            try:
                response = send(url, **kwargs)
            except httpx.RequestError:
                self._observe(provider, operation, None, protocol)
                raise
            self._observe(provider, operation, response, protocol)
            return response


_active_usage: ContextVar[RunTokenLedger | None] = ContextVar("run_token_usage", default=None)


@contextmanager
def capture_token_usage(limit: int | None):
    ledger = RunTokenLedger(limit)
    token = _active_usage.set(ledger)
    try:
        yield ledger
    finally:
        _active_usage.reset(token)


def metered_post(send, url, *, provider, operation, protocol, **kwargs):
    ledger = _active_usage.get()
    if ledger is None:
        return send(url, **kwargs)
    return ledger.post(send, url, provider=provider, operation=operation, protocol=protocol, **kwargs)


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _header_count(value: str | None) -> int | None:
    if value and len(value) <= 18 and value.isascii() and value.isdecimal():
        return int(value)
    return None


def _reported_usage(response: httpx.Response, protocol: str) -> dict | None:
    if protocol == "worker":
        input_tokens = _header_count(response.headers.get(_INPUT_HEADER))
        output_tokens = _header_count(response.headers.get(_OUTPUT_HEADER))
        if input_tokens is None or output_tokens is None:
            return None
        return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}

    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    if protocol == "ollama":
        input_tokens = _count(body.get("prompt_eval_count"))
        output_tokens = _count(body.get("eval_count"))
        if input_tokens is None or output_tokens is None:
            return None
        total = input_tokens + output_tokens
    elif protocol == "gemini":
        usage = body.get("usageMetadata")
        if not isinstance(usage, dict):
            return None
        input_tokens = _count(usage.get("promptTokenCount"))
        output_tokens = _count(usage.get("candidatesTokenCount"))
        total = _count(usage.get("totalTokenCount"))
    else:  # OpenAI Chat Completions-compatible usage.
        usage = body.get("usage")
        if not isinstance(usage, dict):
            return None
        input_tokens = _count(usage.get("prompt_tokens"))
        output_tokens = _count(usage.get("completion_tokens"))
        total = _count(usage.get("total_tokens"))
    if total is None or total < (input_tokens or 0) + (output_tokens or 0):
        return None
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total}
