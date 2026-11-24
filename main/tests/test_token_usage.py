from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import httpx
import pytest

from core.token_usage import (
    TOKEN_BUDGET_EXHAUSTED, TOKEN_USAGE_UNAVAILABLE, TokenBudgetStop, capture_token_usage, metered_post,
)
from utils import llm_client as llm_module


@pytest.mark.parametrize(("protocol", "body", "headers", "expected"), [
    ("gemini", {"usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2, "totalTokenCount": 13}}, {},
     (4, 2, 13)),
    ("ollama", {"prompt_eval_count": 8, "eval_count": 2}, {}, (8, 2, 10)),
    ("openai", {"usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}, {}, (3, 2, 5)),
    ("worker", {"usage": {"total_tokens": 999}},
     {"X-Research-Input-Tokens": "6", "X-Research-Output-Tokens": "1"}, (6, 1, 7)),
    ("worker", {"usage": {"total_tokens": 1}}, {}, None),
    ("openai", {"usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 1}}, {}, None),
    ("gemini", {"usageMetadata": {"totalTokenCount": True}}, {}, None),
])
def test_meter_reads_only_valid_provider_reported_usage(protocol, body, headers, expected):
    response = httpx.Response(200, json=body, headers=headers)
    with capture_token_usage(None) as ledger:
        assert metered_post(
            lambda url: response, "https://worker.invalid", provider="fixture", operation="inference", protocol=protocol,
        ) is response
        event, = ledger.drain()
        assert "worker.invalid" not in str(event)
        if expected is None:
            assert event["status"] == "unreported" and "total_tokens" not in event
        else:
            assert event["status"] == "reported"
            assert tuple(event[key] for key in ("input_tokens", "output_tokens", "total_tokens")) == expected


def test_parallel_budgeted_calls_cannot_reuse_an_already_spent_balance():
    gate = Barrier(2)
    requests = []

    def send(url):
        requests.append(url)
        return httpx.Response(200, json={"usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}})

    with capture_token_usage(6) as ledger:
        def attempt():
            gate.wait(timeout=5)
            try:
                ledger.post(send, "test", provider="fixture", operation="inference", protocol="openai")
            except TokenBudgetStop as exc:
                return exc.status
            return "sent"

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(attempt) for _ in range(2)]
            assert sorted(future.result(timeout=5) for future in futures) == [TOKEN_BUDGET_EXHAUSTED, "sent"]
        assert len(requests) == 1
        assert len(ledger.drain()) == 1


def test_llm_cache_hit_does_not_consume_or_invent_provider_tokens(monkeypatch):
    monkeypatch.setattr(llm_module, "get_cached_llm", lambda _: "cached")
    with capture_token_usage(1) as ledger:
        assert llm_module.llm_client.generate("Cached prompt") == "cached"
        assert ledger.drain() == []


def test_transport_error_is_unknown_and_blocks_later_provider_calls():
    def unreachable(url):
        raise httpx.ConnectError("Connection dropped")

    def unexpected(url):
        pytest.fail("A provider was contacted after token accounting became unavailable")

    with capture_token_usage(5) as ledger:
        for send in (unreachable, unexpected):
            with pytest.raises(TokenBudgetStop, match=TOKEN_USAGE_UNAVAILABLE):
                metered_post(send, "fixture", provider="fixture", operation="inference", protocol="ollama")
        event, = ledger.drain()
        assert event["status"] == "unreported" and event["http_status"] is None


def test_failed_generation_still_consumes_reported_worker_tokens():
    rejected = httpx.Response(502, headers={"X-Research-Input-Tokens": "4", "X-Research-Output-Tokens": "2"})
    with capture_token_usage(6) as ledger:
        assert metered_post(
            lambda url: rejected, "fixture", provider="kaggle", operation="api/summarize/generator", protocol="worker",
        ).status_code == 502
        with pytest.raises(TokenBudgetStop, match=TOKEN_BUDGET_EXHAUSTED):
            metered_post(
                lambda url: pytest.fail("Fallback executed after consuming all tokens"),
                "fixture", provider="kaggle", operation="generate", protocol="worker",
            )
        event, = ledger.drain()
        assert event["http_status"] == 502 and event["total_tokens"] == 6
