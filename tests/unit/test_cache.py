"""
Unit tests for Persistent SQLite LLM Semantic Cache.
"""

import hashlib
from core.db import get_cached_llm, set_cached_llm
from utils.llm_client import llm_client


def test_sqlite_llm_cache_set_and_get():
    prompt = "Test prompt for caching"
    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    expected_response = '{"answer": "Cached answer content"}'

    set_cached_llm(key, prompt, expected_response)
    cached = get_cached_llm(key)
    assert cached == expected_response


def test_llm_client_uses_cache():
    prompt = "Cơ chế Self-Attention trong Transformer hoạt động như thế nào?"
    # First invocation should generate and populate cache
    resp1 = llm_client.generate(prompt)
    assert len(resp1) > 0

    # Second invocation should retrieve from SQLite cache
    key = hashlib.sha256(f"{prompt}______False".encode("utf-8")).hexdigest()
    cached = get_cached_llm(key)
    assert cached == resp1
