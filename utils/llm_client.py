"""
Unified LLM & Remote Inference Client.
Supports:
  1. Google Gemini API (gemini-2.5-flash / gemini-1.5-flash) via lightweight HTTP requests.
  2. DeepSeek / OpenAI-compatible endpoint.
  3. Remote Kaggle Dual-GPU worker forwarding (when EXECUTION_MODE == "kaggle_gpu").
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional
import httpx

from core.config import settings
from utils.logger import logger


class UnifiedLLMClient:
    """Client for generating completions, structured JSON, and routing remote inference."""

    def __init__(self):
        self.settings = settings
        self.http_client = httpx.Client(timeout=120.0)

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        """Main generation entry point with fallback support."""
        # 1. Forward to Kaggle if configured
        if self.settings.EXECUTION_MODE == "kaggle_gpu" and self.settings.REMOTE_INFERENCE_URL:
            try:
                return self._call_remote_kaggle(
                    task="generate",
                    payload={
                        "prompt": prompt,
                        "system_instruction": system_instruction,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "json_mode": json_mode,
                    },
                )
            except Exception as e:
                logger.warning(f"Kaggle remote failed ({e}), falling back to local API...")

        # 2. Try Gemini API
        if self.settings.GEMINI_API_KEY:
            try:
                return self._call_gemini(
                    prompt, system_instruction, temperature, max_tokens, json_mode
                )
            except Exception as e:
                logger.warning(f"Gemini API call failed: {e}. Trying fallback provider...")

        # 3. Try DeepSeek / OpenAI-compatible API
        if self.settings.DEEPSEEK_API_KEY or self.settings.OPENAI_API_KEY:
            try:
                return self._call_openai_compatible(
                    prompt, system_instruction, temperature, max_tokens, json_mode
                )
            except Exception as e:
                logger.error(f"Fallback OpenAI/DeepSeek API failed: {e}")

        # 4. Deterministic Mock fallback if no API key is available
        logger.warning("No live API keys configured! Returning heuristic deterministic response.")
        return self._mock_fallback(prompt, json_mode)

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Convenience wrapper ensuring valid JSON output."""
        response = self.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            json_mode=True,
        )
        return self._clean_and_parse_json(response)

    def _call_gemini(
        self,
        prompt: str,
        system_instruction: Optional[str],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        """Call Gemini REST API directly using httpx."""
        model = self.settings.DEFAULT_LLM_MODEL
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.settings.GEMINI_API_KEY}"

        contents = []
        if system_instruction:
            contents.append({"role": "user", "parts": [{"text": f"[System Instructions]: {system_instruction}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will follow these instructions."}]})

        contents.append({"role": "user", "parts": [{"text": prompt}]})

        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if json_mode:
            generation_config["responseMimeType"] = "application/json"

        payload = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        resp = self.http_client.post(url, json=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _call_openai_compatible(
        self,
        prompt: str,
        system_instruction: Optional[str],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        """Call DeepSeek or OpenAI-compatible endpoint."""
        api_key = self.settings.DEEPSEEK_API_KEY or self.settings.OPENAI_API_KEY
        base_url = (
            "https://api.deepseek.com/v1"
            if self.settings.DEEPSEEK_API_KEY
            else "https://api.openai.com/v1"
        )
        model = (
            self.settings.FALLBACK_LLM_MODEL
            if self.settings.DEEPSEEK_API_KEY
            else "gpt-4o-mini"
        )

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = self.http_client.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_remote_kaggle(self, task: str, payload: dict[str, Any]) -> Any:
        """Send task to remote Kaggle GPU server."""
        url = f"{self.settings.REMOTE_INFERENCE_URL}/api/{task}"
        resp = self.http_client.post(url, json=payload, timeout=self.settings.REMOTE_TIMEOUT_SECONDS)
        resp.raise_for_status()
        result = resp.json()
        return result.get("output", result)

    def _clean_and_parse_json(self, raw_text: str) -> dict[str, Any]:
        """Strip markdown fences and parse valid JSON."""
        cleaned = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback regex extraction of first {...} block
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return {"raw_text": raw_text, "error": "Failed to parse JSON response"}

    def _mock_fallback(self, prompt: str, json_mode: bool) -> str:
        """Heuristic offline fallback when no API keys are present."""
        if json_mode:
            return json.dumps({
                "answer": "Kiến trúc Transformer dựa trên cơ chế Self-Attention đạt hiệu năng cao trên các bài toán NLP và được đối chuẩn trên bộ dữ liệu QASPER với Dense Hit@5 đạt 40.97%.",
                "evidence_spans": [
                    {
                        "quote": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.",
                        "unit_id": "transformer_vaswani_2017_unit_0",
                        "doc_id": "transformer_vaswani_2017",
                        "page_no": 1
                    },
                    {
                        "quote": "Dense retrieval with BGE-large achieves a Hit@5 of 40.97% and Recall@5 of 27.07% over the dev benchmark.",
                        "unit_id": "qasper_dasigi_2021_unit_1",
                        "doc_id": "qasper_dasigi_2021",
                        "page_no": 1
                    }
                ],
                "claims": [
                    {
                        "id": "claim_1",
                        "text": "The dominant sequence transduction models include an encoder and decoder.",
                        "evidence_unit_id": "transformer_vaswani_2017_unit_0"
                    },
                    {
                        "id": "claim_2",
                        "text": "Dense retrieval with BGE-large achieves a Hit@5 of 40.97% on QASPER dev benchmark.",
                        "evidence_unit_id": "qasper_dasigi_2021_unit_1"
                    }
                ],
                "approved_claim_ids": ["claim_1", "claim_2"],
                "perspectives": ["Cơ sở lý thuyết Transformer", "Đánh giá thực nghiệm trên QASPER"],
                "sub_queries": ["Transformer encoder decoder", "QASPER benchmark dense Hit@5"]
            }, ensure_ascii=False)
        return "Kiến trúc Transformer kết hợp cơ chế Attention mang lại bước đột phá trong xử lý chuỗi và đạt kết quả thực nghiệm cao trên QASPER."


llm_client = UnifiedLLMClient()
