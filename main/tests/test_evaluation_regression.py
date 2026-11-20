"""Comprehensive Regression Suite for Đợt 7 Defect Fixes & Pre-Commit Audit.

Covers:
- DEF-01: Multi-dimensional cache key partitioning (mode, agent, temp, max_tokens, json_mode).
- DEF-02: Robust JSON decoding with LaTeX escaping (\\text, \\frac, \\sqrt, \\n, \\t, \\uXXXX, code blocks).
- DEF-03: Disambiguation of translation prompts, PDF queries, and general questions referencing 'tài liệu'.
- DEF-04: Task execution disambiguation (phần mềm là gì vs giải thích lệnh vs chạy lệnh).
- DEF-05: API /api/runs document handling, session isolation, and backward compatibility.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from unittest.mock import MagicMock, patch

import pytest
from core.supervisor import AgenticSupervisorPlanner
from utils.json_repair import extract_and_parse_json
from utils.llm_client import UnifiedLLMClient


# =========================================================================
# DEF-01: Cache Key Isolation Tests
# =========================================================================

def test_def01_cache_key_partitioning_by_all_dimensions():
    """Verify cache keys strictly isolate mode, agent, temperature, max_tokens, and prompt."""
    client = UnifiedLLMClient()
    prompt = "Explain self-attention"
    sys_inst = "You are a research assistant"

    keys = {}
    with patch("utils.llm_client.get_cached_llm") as mock_cache_get:
        mock_cache_get.return_value = "mock_response"

        # 1. Mode variation
        with patch.object(client.settings, "EXECUTION_MODE", "gemini"):
            client.generate(prompt=prompt, system_instruction=sys_inst, temperature=0.2, max_tokens=1024, json_mode=False)
            keys["gemini"] = mock_cache_get.call_args[0][0]

        with patch.object(client.settings, "EXECUTION_MODE", "cpu_max"):
            client.generate(prompt=prompt, system_instruction=sys_inst, temperature=0.2, max_tokens=1024, json_mode=False)
            keys["cpu_max"] = mock_cache_get.call_args[0][0]

        with patch.object(client.settings, "EXECUTION_MODE", "cpu_min"):
            client.generate(prompt=prompt, system_instruction=sys_inst, temperature=0.2, max_tokens=1024, json_mode=False)
            keys["cpu_min"] = mock_cache_get.call_args[0][0]

        # 2. Temperature variation
        with patch.object(client.settings, "EXECUTION_MODE", "gemini"):
            client.generate(prompt=prompt, system_instruction=sys_inst, temperature=0.7, max_tokens=1024, json_mode=False)
            keys["gemini_temp07"] = mock_cache_get.call_args[0][0]

        # 3. Max tokens variation
        with patch.object(client.settings, "EXECUTION_MODE", "gemini"):
            client.generate(prompt=prompt, system_instruction=sys_inst, temperature=0.2, max_tokens=2048, json_mode=False)
            keys["gemini_max2048"] = mock_cache_get.call_args[0][0]

        # 4. Agent variation
        with patch.object(client.settings, "EXECUTION_MODE", "gemini"):
            client.generate(prompt=prompt, system_instruction=sys_inst, temperature=0.2, max_tokens=1024, json_mode=False, agent="summarizer")
            keys["gemini_summarizer"] = mock_cache_get.call_args[0][0]

        # 5. JSON mode variation
        with patch.object(client.settings, "EXECUTION_MODE", "gemini"):
            client.generate(prompt=prompt, system_instruction=sys_inst, temperature=0.2, max_tokens=1024, json_mode=True)
            keys["gemini_json"] = mock_cache_get.call_args[0][0]

    # Every dimension MUST produce a unique cache key
    assert len(keys) == 7
    unique_hashes = set(keys.values())
    assert len(unique_hashes) == 7, "Cache collision detected across different generation parameters!"


# =========================================================================
# DEF-02: LaTeX JSON Escaping & Repair Tests
# =========================================================================

def test_def02_latex_escaping_and_valid_json_preservation():
    """Verify extract_and_parse_json preserves valid JSON escapes, code, unicode, and repairs LaTeX."""

    # 1. Complex formula with \\text, \\frac, \\sqrt, \\left, \\right
    raw_latex = r'''```json
    {
        "formula": "$$\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$",
        "description": "Standard scaled dot-product attention",
        "nested": {
            "matrix": "\\alpha + \\beta = \\gamma"
        }
    }
    ```'''
    res1 = extract_and_parse_json(raw_latex)
    assert r"\text{softmax}" in res1["formula"]
    assert r"\frac{QK^T}{\sqrt{d_k}}" in res1["formula"]
    assert r"\alpha + \beta = \gamma" in res1["nested"]["matrix"]

    # 2. Valid standard JSON escapes (\n, \t, \", \\, \uXXXX) MUST NOT be corrupted
    raw_standard = r'''{
        "lines": "Line 1\nLine 2\tIndented",
        "unicode": "\u00e9tudiant \u03b1",
        "escaped_slash": "C:\\Windows\\System32",
        "quote": "He said \"hello\""
    }'''
    res2 = extract_and_parse_json(raw_standard)
    assert "\n" in res2["lines"]
    assert "\t" in res2["lines"]
    assert res2["unicode"] == "étudiant α"
    assert res2["escaped_slash"] == "C:\\Windows\\System32"
    assert res2["quote"] == 'He said "hello"'

    # 3. Code block inside JSON containing backslashes
    raw_code = r'''{
        "code_snippet": "def regex_test():\n    return re.sub(r'\\d+', '', text)"
    }'''
    res3 = extract_and_parse_json(raw_code)
    assert "re.sub" in res3["code_snippet"]

    # 4. Malformed unrepairable JSON raises ValueError/JSONDecodeError
    with pytest.raises(Exception):
        extract_and_parse_json("{invalid json syntax, not an object")


# =========================================================================
# DEF-03: Translation vs Document QA Routing Tests
# =========================================================================

def test_def03_translation_and_document_routing_disambiguation():
    """Verify positive and negative intent classifications for translation and document references."""
    sup = AgenticSupervisorPlanner()

    # Case A: Translation of arbitrary text containing the word 'tài liệu' (No PDF needed)
    p1 = sup.plan_query("Dịch đoạn tài liệu kỹ thuật sau từ tiếng Anh sang tiếng Việt: 'Attention is all you need.'", pdf_path=None)
    assert p1.primary_intent == "TASK_ASSISTED_RESPONSE"
    assert p1.sub_intent == "Translation"
    assert p1.input_status == "OK"
    assert p1.needs_retrieval is False

    # Case B: Document question with NO PDF attached -> DOCUMENT_QA with NO_DOCUMENT_SELECTED
    p2 = sup.plan_query("Tài liệu PDF này nói gì?", pdf_path=None)
    assert p2.primary_intent == "DOCUMENT_QA"
    assert p2.input_status == "NO_DOCUMENT_SELECTED"
    assert p2.needs_retrieval is False

    # Case C: Document question WITH PDF attached -> DOCUMENT_QA with OK
    p3 = sup.plan_query("Tài liệu PDF này nói gì?", pdf_path="dummy.pdf")
    assert p3.primary_intent == "DOCUMENT_QA"
    assert p3.input_status == "OK"
    assert p3.needs_retrieval is True

    # Case D: Translation of specific PDF page -> DOCUMENT_QA (requires PDF)
    p4_no_pdf = sup.plan_query("Dịch nội dung trang 5 của PDF", pdf_path=None)
    assert p4_no_pdf.primary_intent == "DOCUMENT_QA"
    assert p4_no_pdf.input_status == "NO_DOCUMENT_SELECTED"

    p4_with_pdf = sup.plan_query("Dịch nội dung trang 5 của PDF", pdf_path="dummy.pdf")
    assert p4_with_pdf.primary_intent == "DOCUMENT_QA"
    assert p4_with_pdf.input_status == "OK"
    assert p4_with_pdf.needs_retrieval is True

    # Case E: General question containing the word 'tài liệu' -> GENERAL_QA
    p5 = sup.plan_query("Tài liệu là gì?", pdf_path=None)
    assert p5.primary_intent == "GENERAL_QA"
    assert p5.input_status == "OK"
    assert p5.needs_retrieval is False

    p6 = sup.plan_query("Cách viết tài liệu kỹ thuật tốt?", pdf_path=None)
    assert p6.primary_intent in ("GENERAL_QA", "TASK_ASSISTED_RESPONSE")
    assert p6.input_status == "OK"
    assert p6.needs_retrieval is False


# =========================================================================
# DEF-04: Task Execution vs General QA / Assistance Disambiguation
# =========================================================================

def test_def04_execution_keywords_do_not_leak_to_task_execution():
    """Verify keywords 'phần mềm', 'chương trình', 'lệnh' do not falsely trigger TASK_EXECUTION."""
    sup = AgenticSupervisorPlanner()

    # Case A: "Phần mềm là gì?" -> GENERAL_QA (NOT TASK_EXECUTION)
    p1 = sup.plan_query("Phần mềm là gì?", pdf_path=None)
    assert p1.primary_intent == "GENERAL_QA"
    assert p1.needs_retrieval is False

    # Case B: "Giải thích lệnh ls" -> TASK_ASSISTED_RESPONSE (NOT TASK_EXECUTION)
    p2 = sup.plan_query("Giải thích lệnh ls", pdf_path=None)
    assert p2.primary_intent == "TASK_ASSISTED_RESPONSE"
    assert p2.sub_intent == "Coding_Or_Guidance"
    assert p2.needs_retrieval is False

    # Case C: "Viết chương trình tính tổng hai số" -> TASK_ASSISTED_RESPONSE (NOT TASK_EXECUTION)
    p3 = sup.plan_query("Viết chương trình Python tính tổng hai số", pdf_path=None)
    assert p3.primary_intent == "TASK_ASSISTED_RESPONSE"
    assert p3.sub_intent == "Coding_Or_Guidance"
    assert p3.needs_retrieval is False

    # Case D: "Chạy lệnh ls trên máy tính của tôi" -> TASK_EXECUTION (Requires permission/tool)
    p4 = sup.plan_query("Chạy lệnh ls trên máy tính của tôi", pdf_path=None)
    assert p4.primary_intent == "TASK_EXECUTION"
    assert p4.needs_retrieval is False

    # Case E: "Chạy chương trình này trên máy tính của bạn và đọc file hệ thống." -> TASK_EXECUTION
    p5 = sup.plan_query("Chạy chương trình này trên máy tính của bạn và đọc file hệ thống.", pdf_path=None)
    assert p5.primary_intent == "TASK_EXECUTION"


# =========================================================================
# DEF-05: API Document Handling & Backward Compatibility
# =========================================================================

def test_def05_api_document_handling_and_session_isolation():
    """Verify API POST /api/runs handles no-document queries, preserves sessions, and maintains compatibility."""
    from api.app import app
    from httpx import AsyncClient, ASGITransport

    async def _test_api_scenarios():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:

            # Scenario 1: General query without PDF -> successfully accepted
            resp1 = await client.post("/api/runs", json={
                "query": "Thủ đô của Việt Nam là gì?",
                "execution_mode": "cpu_min",
                "document_ids": [],
                "session_id": "test_session_001",
            })
            assert resp1.status_code == 200
            d1 = resp1.json()
            assert "id" in d1
            assert d1["session_id"] == "test_session_001"
            assert d1["status"] in ("processing", "completed", "running")

            # Scenario 2: Legacy field name compatibility ('question' instead of 'query', 'document_id' instead of 'document_ids')
            resp2 = await client.post("/api/runs", json={
                "question": "Chào bạn, bạn có khỏe không?",
                "execution_mode": "cpu_min",
                "document_id": None,
                "session_id": "test_session_002",
            })
            assert resp2.status_code == 200
            d2 = resp2.json()
            assert "id" in d2
            assert d2["session_id"] == "test_session_002"

            # Scenario 3: Document question with NO PDF -> run created, status processing
            resp3 = await client.post("/api/runs", json={
                "query": "Tài liệu này nói gì về kiến trúc?",
                "execution_mode": "cpu_min",
                "document_ids": [],
            })
            assert resp3.status_code == 200
            d3 = resp3.json()
            assert "id" in d3

    asyncio.run(_test_api_scenarios())
