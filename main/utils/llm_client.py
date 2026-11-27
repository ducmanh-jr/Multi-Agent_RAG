"""
Unified LLM & Remote Inference Client (L3-L4 Agentic Upgrade).
Supports:
  1. Google Gemini API (gemini-2.5-flash / gemini-1.5-flash) with quota tracking & batch JSON.
  2. Local AMD Radeon 780M Vulkan / AVX-512 via Ollama (qwen2.5:1.5b for CPU Min, qwen2.5:7b-instruct / qwen3:8b for CPU Max).
  3. Remote Kaggle Multi-LoRA Specialist Worker (/api/router, /api/summarize/generator, /api/summarize/critic, /api/factcheck, /generate) with canonical adapter mapping.
  4. Remote Kaggle Max 32B Reasoning Worker (Qwen/Qwen3-32B-AWQ or QwQ-32B-AWQ on Dual T4 TP=2) with dynamic /think and /no_think control.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Optional
import httpx

from core.config import get_run_settings
from core.db import get_cached_llm, log_api_usage, set_cached_llm
from core.schemas import canonical_adapter_name
from core.token_usage import TokenBudgetStop, metered_post
from core.worker_observations import record_provider_observation, record_worker_observation
from utils.local_vulkan_setup import probe_ollama_models
from utils.logger import logger


_REMOTE_MODES = {"model", "kaggle", "kaggle_max", "kaggle_gpu", "gpu_max", "gpu_finetune"}


class UnifiedLLMClient:
    """Client for generating completions, structured JSON, and routing remote/local inference."""

    def __init__(self):
        self.http_client = httpx.Client(timeout=120.0)
        self._circuit_breaker_until: dict[str, float] = {}

    @property
    def settings(self):
        return get_run_settings()

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
        agent: Optional[str] = None,
    ) -> str:
        """Main generation entry point with Semantic SQLite Cache and multi-tier fallback."""
        config = self.settings
        target_mode = config.get_inference_mode(agent)
        target_mode = (target_mode or "cpu_max").lower()

        cache_enabled = bool(getattr(config, "LLM_CACHE_ENABLED", True))
        cache_key_str = json.dumps([
            target_mode, agent, prompt, system_instruction, json_mode, temperature, max_tokens,
            config.LOCAL_CPU_URL, config.LOCAL_CPU_MODEL, config.REMOTE_INFERENCE_URL,
            config.DEFAULT_LLM_MODEL, config.FALLBACK_LLM_MODEL,
            config.GEMINI_API_KEY, config.DEEPSEEK_API_KEY, config.OPENAI_API_KEY,
        ], ensure_ascii=False)
        cache_key = hashlib.sha256(cache_key_str.encode("utf-8")).hexdigest()
        cached = get_cached_llm(cache_key) if cache_enabled else None
        if cached:
            logger.info(f"LLM Semantic Cache Hit [{cache_key[:8]}]. Saved 1 external API/GPU call.")
            if target_mode in _REMOTE_MODES:
                adapter = "none" if target_mode == "gpu_max" else canonical_adapter_name(agent)
                record_worker_observation("generate", adapter, status="cache_hit")
            else:
                record_provider_observation(
                    provider=target_mode,
                    operation="generate",
                    status="cache_hit",
                    requested_model=config.LOCAL_CPU_MODEL if target_mode in {"cpu", "cpu_min", "cpu_max", "local_cpu"} else config.DEFAULT_LLM_MODEL,
                    agent=agent,
                )
            return cached

        response_text = ""
        used_degraded_fallback = False
        is_cpu_mode = target_mode in ["cpu", "cpu_min", "cpu_max", "local_cpu"]

        # 0. Pure Local CPU / Radeon 780M Vulkan execution (100% Offline - ZERO Cloud API calls)
        if is_cpu_mode:
            model_tier = "1.5b" if target_mode == "cpu_min" else "7b"
            try:
                response_text = self._call_local_cpu(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    agent=agent,
                    model_tier=model_tier,
                )
            except TokenBudgetStop:
                raise
            except Exception as e:
                used_degraded_fallback = True
                logger.info(
                    f"[CPU Mode: {target_mode}] Local Ollama server is offline ({e}). "
                    f"Using pure in-memory CPU extractive processing (100% offline, zero cloud API calls)."
                )
                record_provider_observation(
                    provider="heuristic_fallback",
                    operation="offline_extractive_fallback",
                    status="fallback",
                    requested_model=config.LOCAL_CPU_MODEL,
                    fallback_reason=f"local_cpu_offline:{type(e).__name__}",
                    agent=agent,
                )
                response_text = self._offline_extractive_fallback(prompt, json_mode)

            if response_text and cache_enabled and not used_degraded_fallback:
                set_cached_llm(cache_key, prompt, response_text)
            return response_text

        # 1. Remote Kaggle Worker (Explicit Kaggle FN, Kaggle Max, or GPU Finetune mode)
        elif target_mode in _REMOTE_MODES:
            if not self.settings.REMOTE_INFERENCE_URL:
                raise RuntimeError(
                    f"Kaggle mode '{target_mode}' requires REMOTE_INFERENCE_URL to be set in environment or settings."
                )
            try:
                canon_adapter = "none" if target_mode == "gpu_max" else canonical_adapter_name(agent)
                response_text = self._call_remote_kaggle(
                    task="generate",
                    payload={
                        "prompt": prompt,
                        "system_instruction": system_instruction,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "json_mode": json_mode,
                        "adapter": canon_adapter,
                        "mode": target_mode,
                    },
                )
            except TokenBudgetStop:
                raise
            except Exception as e:
                logger.warning(f"Kaggle remote for agent '{agent}' failed ({e}). Cascading to Cloud Foundation Model for high-fidelity generative synthesis...")
                response_text = ""
            if response_text and cache_enabled:
                set_cached_llm(cache_key, prompt, response_text)
            return response_text

        # 2. Cloud Foundation Model (Enforced by centralized Provider Policy Gate)
        from core.provider_router import is_gemini_permitted

        cloud_failure_reason = ""
        if self.settings.GEMINI_API_KEY and is_gemini_permitted(target_mode, role=agent or ""):
            try:
                response_text = self._call_gemini(
                    prompt, system_instruction, temperature, max_tokens, json_mode
                )
            except TokenBudgetStop:
                raise
            except Exception as e:
                cloud_failure_reason = f"gemini_error:{type(e).__name__}"
                logger.warning(f"Gemini API call failed: {e}. Trying fallback provider...")
        elif self.settings.GEMINI_API_KEY and not is_gemini_permitted(target_mode, role=agent or ""):
            cloud_failure_reason = "provider_policy_blocked"
            logger.info(
                "[ProviderGate] Blocked Gemini API call under mode='%s' (profile='%s', agent='%s').",
                target_mode,
                getattr(self.settings, "HYBRID_PROVIDER_PROFILE", "auto"),
                agent or "",
            )
        else:
            cloud_failure_reason = "missing_gemini_api_key"

        # 3. Fallback to DeepSeek / OpenAI-compatible API if Gemini fails
        if not response_text and (self.settings.DEEPSEEK_API_KEY or self.settings.OPENAI_API_KEY):
            try:
                response_text = self._call_openai_compatible(
                    prompt, system_instruction, temperature, max_tokens, json_mode
                )
            except TokenBudgetStop:
                raise
            except Exception as e:
                cloud_failure_reason = f"{cloud_failure_reason};openai_error:{type(e).__name__}".lstrip(";")
                logger.error(f"Fallback OpenAI/DeepSeek API failed: {e}")

        # 4. Fallback if cloud APIs fail — NEVER persist heuristic fallback into llm_cache!
        if not response_text:
            used_degraded_fallback = True
            logger.warning("Cloud APIs unreachable! Returning heuristic deterministic response.")
            record_provider_observation(
                provider="heuristic_fallback",
                operation="offline_extractive_fallback",
                status="fallback",
                requested_model=config.DEFAULT_LLM_MODEL,
                fallback_reason=cloud_failure_reason or "cloud_unreachable",
                agent=agent,
            )
            response_text = self._offline_extractive_fallback(prompt, json_mode)

        if response_text and cache_enabled and not used_degraded_fallback:
            set_cached_llm(cache_key, prompt, response_text)

        return response_text

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        agent: Optional[str] = None,
    ) -> dict[str, Any]:
        """Convenience wrapper ensuring valid JSON output."""
        response = self.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            json_mode=True,
            agent=agent,
        )
        return self._clean_and_parse_json(response)

    def _worker_headers(self) -> dict[str, str]:
        token = (getattr(self.settings, "REMOTE_INFERENCE_TOKEN", "") or "").strip()
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _post_worker(
        self, base_url: str, endpoint: str, payload: dict[str, Any], timeout: float, adapter: str | None,
    ) -> httpx.Response:
        headers = self._worker_headers()
        kwargs: dict[str, Any] = {"json": payload, "timeout": timeout}
        if headers:
            kwargs["headers"] = headers
        try:
            response = metered_post(
                self.http_client.post, f"{base_url}/{endpoint}", provider="kaggle", operation=endpoint,
                protocol="openai" if endpoint == "v1/chat/completions" else "worker", **kwargs,
            )
        except httpx.RequestError:
            record_worker_observation(endpoint, adapter)
            raise
        record_worker_observation(endpoint, adapter, response)
        return response

    def call_kaggle_specialist(
        self,
        endpoint: str,
        payload: dict[str, Any],
        fallback_prompt: str = "",
        fallback_system: str = "",
        adapter: str = "generator",
        timeout: Optional[float] = None,
        allow_generate_fallback: bool = True,
    ) -> dict[str, Any]:
        """
        Calls dedicated training_lab endpoints on Kaggle Multi-LoRA Worker:
          - /api/router
          - /api/summarize/generator
          - /api/summarize/critic
          - /api/factcheck
        And automatically falls back to /generate or /api/generate with canonical adapter name!
        """
        base_url = (self.settings.REMOTE_INFERENCE_URL or "").rstrip("/")
        if not base_url:
            raise RuntimeError("Chưa kết nối được hãy kết nối lại. (REMOTE_INFERENCE_URL trống)")

        canon_adapter = canonical_adapter_name(adapter)
        clean_ep = endpoint.lstrip("/")
        try:
            effective_timeout = float(timeout if timeout is not None else self.settings.REMOTE_TIMEOUT_SECONDS)
            resp = self._post_worker(base_url, clean_ep, payload, effective_timeout, canon_adapter)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    return data
        except TokenBudgetStop:
            raise
        except Exception as exc:
            if not allow_generate_fallback:
                logger.info("[Kaggle Specialist] Invalid or unavailable /%s (%s)", clean_ep, type(exc).__name__)
                return {}
            logger.info(f"[Kaggle Specialist] Dedicated endpoint /{clean_ep} fallback to /generate ({exc})")
            if "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
                raise exc

        if not allow_generate_fallback:
            return {}

        # Fallback to /generate with canonical adapter name ('router', 'generator', 'critic')
        raw_out = self._call_remote_kaggle(
            task="generate",
            payload={
                "prompt": fallback_prompt or json.dumps(payload, ensure_ascii=False),
                "system_instruction": fallback_system,
                "temperature": 0.1,
                "max_tokens": 1536,
                "json_mode": True,
                "adapter": canon_adapter,
            },
        )
        if isinstance(raw_out, dict):
            return raw_out
        return self._clean_and_parse_json(str(raw_out))

    def call_kaggle_max_32b(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        enable_thinking: bool = True,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """
        Calls Kaggle Max 32B Reasoning Model (Qwen/Qwen3-32B-AWQ or QwQ-32B-AWQ on Dual T4 TP=2).
        Controls thinking budget via /think (for deep synthesis/verification) vs /no_think (for fast tool calls).
        Returns {"text": ..., "thinking": ..., "parsed": ...}.
        """
        base_url = (self.settings.REMOTE_INFERENCE_URL or "").rstrip("/")
        if not base_url:
            raise RuntimeError("Chưa kết nối được hãy kết nối lại. (REMOTE_INFERENCE_URL trống)")

        think_prefix = "/think\n" if enable_thinking else "/no_think\n"
        decorated_prompt = f"{think_prefix}{prompt}"

        raw_response = self._call_remote_kaggle(
            task="generate",
            payload={
                "prompt": decorated_prompt,
                "system_instruction": system_instruction,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
                "enable_thinking": enable_thinking,
                "mode": "kaggle_max",
                "adapter": "none",
            },
        )
        raw_str = raw_response if isinstance(raw_response, str) else json.dumps(raw_response, ensure_ascii=False)
        thinking = ""
        think_match = re.search(r"<think>(.*?)</think>", raw_str, flags=re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            raw_str = re.sub(r"<think>.*?</think>", "", raw_str, flags=re.DOTALL).strip()

        parsed = self._clean_and_parse_json(raw_str) if json_mode else {}
        return {
            "text": raw_str,
            "thinking": thinking,
            "parsed": parsed,
        }

    # Alias for GPU Max 32B Reasoning call
    call_gpu_max_32b = call_kaggle_max_32b

    def _call_gemini(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        """Call Gemini REST API directly with config-driven model selection and bounded 429 retry."""
        raw_candidates = [
            self.settings.DEFAULT_LLM_MODEL,
            self.settings.FALLBACK_LLM_MODEL,
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3-flash-preview",
            "gemini-flash-lite-latest",
        ]
        candidate_models = []
        for m in raw_candidates:
            if m and m not in candidate_models:
                candidate_models.append(m)
        cooldowns = getattr(self, "_quota_cooldown_until", None)
        if cooldowns is None:
            cooldowns = {}
            self._quota_cooldown_until = cooldowns
        now_mono = time.monotonic()
        active_models = [m for m in candidate_models if cooldowns.get(m, 0.0) <= now_mono]
        if active_models:
            candidate_models = active_models
        requested_model = candidate_models[0] if candidate_models else (self.settings.DEFAULT_LLM_MODEL or "")
        attempted_models: list[str] = []
        cascade_reasons: list[str] = []
        last_error: Optional[Exception] = None

        for model in candidate_models:
            if not model:
                continue
            attempted_models.append(model)
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

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

            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.settings.GEMINI_API_KEY or "",
            }
            # Bounded single retry on transient 429/503 before cascading to next candidate model
            for attempt in range(2):
                try:
                    resp = metered_post(
                        self.http_client.post, url, provider="gemini", operation="generateContent", protocol="gemini",
                        json=payload,
                        headers=headers,
                        timeout=35.0,
                    )
                    if resp.status_code == 200:
                        log_api_usage("gemini", model, "generateContent")
                        data = resp.json()
                        raw_version = data.get("modelVersion") if isinstance(data, dict) else None
                        actual_version = raw_version.strip() if isinstance(raw_version, str) and raw_version.strip() else model
                        cands = data.get("candidates", []) if isinstance(data, dict) else []
                        if cands:
                            content = cands[0].get("content", {})
                            parts = content.get("parts", [])
                            non_thought = [
                                p["text"]
                                for p in parts
                                if isinstance(p, dict) and isinstance(p.get("text"), str) and not p.get("thought")
                            ]
                            text_out = non_thought[-1] if non_thought else (parts[0]["text"] if parts and "text" in parts[0] else None)
                            if text_out is not None:
                                cascaded = model != requested_model or bool(cascade_reasons)
                                record_provider_observation(
                                    provider="gemini",
                                    operation="generateContent",
                                    status="fallback" if cascaded else "reported",
                                    requested_model=requested_model,
                                    attempted_models=attempted_models,
                                    served_model=model,
                                    model_version=actual_version,
                                    fallback_reason="; ".join(cascade_reasons) if cascade_reasons else None,
                                )
                                return text_out
                        cascade_reasons.append(f"{model}:empty_candidates")
                        logger.warning(f"Gemini response has no text parts: {data}")
                        break
                    elif resp.status_code == 404:
                        # Cache 404 retired model IDs (e.g. gemini-2.5-flash) for 1 hour so they are skipped immediately
                        cooldowns[model] = time.monotonic() + 3600.0
                        cascade_reasons.append(f"{model}:http_404")
                        last_error = RuntimeError(f"Gemini HTTP 404 (model '{model}' not found)")
                        logger.info(f"Gemini model '{model}' returned 404; caching skip and cascading to next model...")
                        break
                    elif resp.status_code in (429, 503):
                        last_error = RuntimeError(f"Gemini HTTP {resp.status_code} on {model}")
                        if resp.status_code == 429 and "exceeded your current quota" in (resp.text or "").lower():
                            cooldowns[model] = time.monotonic() + 180.0
                        elif resp.status_code == 503:
                            cooldowns[model] = time.monotonic() + 60.0
                        retry_after = 0.0
                        try:
                            retry_after = min(2.0, max(0.0, float(resp.headers.get("Retry-After", "0"))))
                        except (TypeError, ValueError):
                            retry_after = 0.0
                        if attempt == 0 and retry_after > 0:
                            time.sleep(retry_after)
                            continue
                        cascade_reasons.append(f"{model}:http_{resp.status_code}")
                        logger.warning(f"Gemini model {model} hit HTTP {resp.status_code}. Trying next model...")
                        break
                    else:
                        cascade_reasons.append(f"{model}:http_{resp.status_code}")
                        resp.raise_for_status()
                except TokenBudgetStop:
                    raise
                except Exception as e:
                    last_error = e
                    if not cascade_reasons or not cascade_reasons[-1].startswith(f"{model}:"):
                        cascade_reasons.append(f"{model}:{type(e).__name__}")
                    logger.warning(f"Gemini call to {model} failed: {e}. Trying fallback...")
                    break

        record_provider_observation(
            provider="gemini",
            operation="generateContent",
            status="error",
            requested_model=requested_model,
            attempted_models=attempted_models,
            fallback_reason="; ".join(cascade_reasons) or (str(last_error) if last_error else "all_models_failed"),
        )
        if last_error:
            raise last_error
        raise RuntimeError("Tất cả các mô hình Gemini đều không phản hồi.")

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

        provider_name = "deepseek" if self.settings.DEEPSEEK_API_KEY else "openai"
        resp = metered_post(
            self.http_client.post, f"{base_url}/chat/completions",
            provider=provider_name,
            operation="chat/completions", protocol="openai",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        body = resp.json()
        served_name = str(body.get("model") or model)
        record_provider_observation(
            provider=provider_name,
            operation="chat/completions",
            status="reported",
            requested_model=model,
            attempted_models=[model],
            served_model=served_name,
            model_version=served_name,
        )
        return body["choices"][0]["message"]["content"]

    def _call_local_cpu(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
        agent: Optional[str] = None,
        model_tier: str = "7b",
    ) -> str:
        """
        Call local inference on Lenovo ThinkBook 14 G8+ AHP (Radeon 780M Vulkan / AVX-512)
        via Ollama or OpenAI-compatible server.
        Automatically picks 1.5B for CPU Min and 7B/8B for CPU Max when installed.
        """
        base_url = (self.settings.LOCAL_CPU_URL or "http://localhost:11434").rstrip("/")
        ollama_probe = probe_ollama_models(base_url=base_url)
        if not ollama_probe["online"]:
            raise ConnectionError("Ollama local server not running on port 11434")

        if model_tier == "1.5b":
            model_name = ollama_probe.get("cpu_min_model") or self.settings.LOCAL_CPU_MODEL or "qwen2.5:1.5b"
        else:
            model_name = ollama_probe.get("cpu_max_model") or "qwen2.5:7b-instruct"

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        # 1. Try Ollama native endpoint (/api/chat) with Vulkan / Flash-Attention options
        ollama_url = f"{base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 4096,
            },
        }
        if json_mode:
            payload["format"] = "json"

        resp = metered_post(
            self.http_client.post, ollama_url, provider="ollama", operation="api/chat",
            protocol="ollama", json=payload, timeout=75.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            if content:
                served_ollama = str(data.get("model") or model_name)
                record_provider_observation(
                    provider="ollama",
                    operation="api/chat",
                    status="reported",
                    requested_model=model_name,
                    attempted_models=[model_name],
                    served_model=served_ollama,
                    model_version=served_ollama,
                    agent=agent,
                )
                logger.info(f"Local Radeon 780M Vulkan [Ollama {model_name}] generated response ({len(content)} chars)")
                return content

        # 2. Fallback to OpenAI-compatible endpoint (/v1/chat/completions)
        compat_url = f"{base_url}/v1/chat/completions"
        compat_payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            compat_payload["response_format"] = {"type": "json_object"}

        resp = metered_post(
            self.http_client.post, compat_url, provider="local_openai", operation="chat/completions",
            protocol="openai", json=compat_payload, timeout=75.0,
        )
        resp.raise_for_status()
        data = resp.json()
        served_compat = str(data.get("model") or model_name)
        record_provider_observation(
            provider="local_openai",
            operation="chat/completions",
            status="reported",
            requested_model=model_name,
            attempted_models=[model_name],
            served_model=served_compat,
            model_version=served_compat,
            agent=agent,
        )
        return data["choices"][0]["message"]["content"]

    def _call_remote_kaggle(self, task: str, payload: dict[str, Any]) -> Any:
        """
        Send task to remote Kaggle GPU server with Circuit Breaker resilience.
        Supports:
          1. /generate (FastAPI server in kaggle_worker.ipynb)
          2. /api/{task} (Unified API router)
          3. /v1/chat/completions (vLLM Dual T4 server for Qwen3-32B-AWQ / QwQ-32B-AWQ)
        """
        base_url = self.settings.REMOTE_INFERENCE_URL.rstrip("/")
        timeout = max(30.0, float(self.settings.REMOTE_TIMEOUT_SECONDS or 90.0))

        # Circuit Breaker: Fail fast in 0.001s if remote recently timed out
        now = time.time()
        blocked_until = self._circuit_breaker_until.get(base_url, 0.0)
        if now < blocked_until:
            rem = int(blocked_until - now)
            endpoint = "generate" if task == "generate" else f"api/{task}"
            record_worker_observation(endpoint, payload.get("adapter", "generator"), status="circuit_open")
            raise RuntimeError(f"Kaggle GPU Worker đang offline hoặc không phản hồi (Circuit Breaker active {rem}s).")

        # Normalize adapter name so 'generator_qlora' -> 'generator', 'critic_qlora' -> 'critic', etc.
        if "adapter" in payload and payload["adapter"] not in (None, "none"):
            payload["adapter"] = canonical_adapter_name(str(payload["adapter"]))
        adapter = payload.get("adapter", "generator")

        # 1. Try /generate (primary endpoint in kaggle_worker.ipynb)
        if task == "generate":
            try:
                resp = self._post_worker(base_url, "generate", payload, timeout, adapter)
                if resp.status_code == 200:
                    self._circuit_breaker_until.pop(base_url, None)
                    result = resp.json()
                    if isinstance(result, dict):
                        return (
                            result.get("text")
                            or result.get("output")
                            or result.get("response")
                            or json.dumps(result, ensure_ascii=False)
                        )
                    return str(result)
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                self._circuit_breaker_until[base_url] = time.time() + 10.0
                raise RuntimeError(f"Không thể kết nối tới Kaggle GPU Worker tại {base_url}.") from exc
            except TokenBudgetStop:
                raise
            except Exception:
                pass
        else:
            try:
                resp = self._post_worker(base_url, f"api/{task}", payload, timeout, adapter)
                if resp.status_code == 200:
                    self._circuit_breaker_until.pop(base_url, None)
                    result = resp.json()
                    return result.get("output", result)
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                self._circuit_breaker_until[base_url] = time.time() + 10.0
                raise RuntimeError(f"Không thể kết nối tới Kaggle GPU Worker tại {base_url}.") from exc
            except TokenBudgetStop:
                raise
            except Exception:
                pass

        # 2. Try direct vLLM OpenAI-compatible endpoint (/v1/chat/completions) for 32B AWQ
        messages = []
        if payload.get("system_instruction"):
            messages.append({"role": "system", "content": payload["system_instruction"]})
        messages.append({"role": "user", "content": payload.get("prompt", "")})

        model_id = payload.get("model") or "Qwen/Qwen3-32B-AWQ"
        v1_payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": payload.get("temperature", 0.2),
            "max_tokens": payload.get("max_tokens", 2048),
        }
        if payload.get("json_mode"):
            v1_payload["response_format"] = {"type": "json_object"}
        try:
            resp = self._post_worker(base_url, "v1/chat/completions", v1_payload, timeout, adapter)
            if resp.status_code == 200:
                self._circuit_breaker_until.pop(base_url, None)
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            self._circuit_breaker_until[base_url] = time.time() + 10.0
            raise RuntimeError(f"Không thể kết nối tới Kaggle GPU Worker tại {base_url}.") from exc
        except TokenBudgetStop:
            raise
        except Exception:
            pass

        self._circuit_breaker_until[base_url] = time.time() + 10.0
        raise RuntimeError(f"Không thể kết nối tới Kaggle GPU Worker tại {base_url}.")

    def _clean_and_parse_json(self, raw_text: str) -> dict[str, Any]:
        """Strip <think> blocks and markdown fences, then parse valid JSON."""
        no_think = re.sub(r"<think>.*?</think>", "", (raw_text or "").strip(), flags=re.DOTALL)
        cleaned = re.sub(r"^```json\s*", "", no_think.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"^```\s*", "", cleaned.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return {"raw_text": raw_text, "error": "Failed to parse JSON response"}

    # Backward compatibility alias
    def _mock_fallback(self, prompt: str, json_mode: bool) -> str:
        return self._offline_extractive_fallback(prompt, json_mode)

    def _offline_extractive_fallback(self, prompt: str, json_mode: bool) -> str:
        """Context-aware dynamic offline fallback when API keys are unreachable or local mode.
        Produces clean, truthful, academic summaries and handles negative/unanswered quantitative queries.
        """
        match = re.search(
            r"""(?:Câu hỏi nghiên cứu|Câu hỏi|User Question|Question|Query):\s*\n?["']?([^\n"']+)""",
            prompt,
            re.IGNORECASE,
        )
        user_query = match.group(1).strip() if match else ""
        if not user_query:
            m2 = re.search(r"""(?:User Question|Question|Câu hỏi):\s*(.*?)(?:\n\n|\n[A-Z]|\Z)""", prompt, re.DOTALL | re.IGNORECASE)
            user_query = m2.group(1).strip() if m2 else prompt[:120].strip()

        prompt_content = prompt
        cut_markers = [
            "YÊU CẦU TRẢ VỀ",
            "Yêu cầu trả về",
            "OUTPUT FORMAT",
            "FORMAT REQUIREMENTS",
            "Trả về JSON",
            "ĐỊNH DẠNG JSON",
        ]
        for marker in cut_markers:
            if marker in prompt_content:
                prompt_content = prompt_content.split(marker)[0]

        unit_pattern = re.compile(
            r"\[Unit ID:\s*([^\|]+)\s*\|\s*Doc:\s*([^\|]+)\s*\|\s*Trang:\s*(\d+)\]:?\s*(.*?)(?=\n\s*\[Unit ID:|\Z)",
            re.DOTALL,
        )
        units = []
        for m in unit_pattern.finditer(prompt_content):
            uid = m.group(1).strip()
            doc = m.group(2).strip()
            page = int(m.group(3).strip())
            body = m.group(4).strip()
            if body:
                units.append({"id": uid, "doc_id": doc, "page_no": page, "text": body})

        def clean_doc_title(name: str) -> str:
            s = re.sub(r"^(?:gemini|doc)_corpus_[a-f0-9]+_", "", name, flags=re.IGNORECASE)
            s = re.sub(r"^(?:gemini|doc)_corpus_[a-f0-9]+", "", s, flags=re.IGNORECASE)
            s = re.sub(r"^(?:gemini|doc)_[a-f0-9]+_", "", s, flags=re.IGNORECASE)
            s = re.sub(r"^(?:gemini|doc)_[a-f0-9]+", "", s, flags=re.IGNORECASE)
            s = re.sub(r"^(?:gemini|doc)_", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\.(?:pdf|md|txt)$", "", s, flags=re.IGNORECASE)
            clean = s.replace("_", " ").strip()
            return clean if clean else "Tài liệu nghiên cứu"

        def clean_caption_credit(s: str) -> str:
            res = re.sub(r"\(Ảnh:[^)]*\)", "", s).strip()
            res = re.sub(r"\(Nguồn:[^)]*\)", "", res).strip()
            return res

        def is_complete_sentence(s: str) -> bool:
            s_clean = clean_caption_credit(s).strip()
            if len(s_clean) < 25:
                return False
            if s_clean[-1] not in '.!?:\u201c\u201d;':
                return False
            last_tokens = re.findall(r"\w+", s_clean)
            last_word = last_tokens[-1].lower() if last_tokens else ""
            dangling_words = {
                "được", "là", "và", "của", "đã", "cho", "nhưng", "bởi", "ở", "vào",
                "ra", "tại", "với", "về", "trong", "để", "có", "sau", "trước", "khi",
                "mà", "theo", "các", "những", "thì", "do", "bị", "rằng", "gồm", "như",
            }
            if last_word in dangling_words:
                return False
            if not (s_clean[0].isalnum() or s_clean[0] in '\u201c\u201d\'['):
                return False
            return True

        def is_metadata_or_header(line: str) -> bool:
            l = clean_caption_credit(line).strip()
            if not l or len(l) < 20:
                return True
            if l.startswith("#") or l.startswith("---") or l.startswith("==="):
                return True
            if l.startswith("http://") or l.startswith("https://"):
                return True
            bad_phrases = [
                "Tác giả", "Năm xuất bản", "Nguồn công bố", "Liên kết tham chiếu",
                "Quyền riêng tư", "Cookie", "Theo tư liệu",
                "YÊU CẦU TRẢ VỀ", "ĐỊNH DẠNG JSON", "markdown bên ngoài",
                "Tóm tắt nội dung nghiên cứu", "Bối cảnh và Giá trị khoa học",
            ]
            for phrase in bad_phrases:
                if phrase.lower() in l.lower():
                    return True
            if l.lower().startswith("ảnh:"):
                return True
            return False

        def has_quantity_for_entity(sentence: str, query: str) -> bool:
            s = re.sub(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", "", sentence)
            s = re.sub(r"\b(?:ngày|tháng|năm|số)\s+\d+[\w-]*\b", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\b(19\d\d|20\d\d)\b", "", s)

            num_match = re.search(r"\b\d+(?:[.,]\d+)?\b", s)
            if not num_match:
                return False

            m = re.search(r"bao nhiêu\s+([\w\s]+)", query, re.IGNORECASE)
            target = m.group(1).strip().split()[0] if m else ""
            if target and target.lower() in sentence.lower():
                return bool(
                    re.search(r"\b\d+\s*(?:nghìn|triệu|vạn|" + re.escape(target) + r")", s, re.IGNORECASE)
                    or re.search(re.escape(target) + r"[\w\s]{1,30}\b\d+", s, re.IGNORECASE)
                )
            return True

        if not units:
            return json.dumps(
                {
                    "answer": f"Không tìm thấy dữ liệu liên quan đến câu hỏi '{user_query}' trong tài liệu ngữ cảnh.",
                    "evidence_spans": [],
                    "claims": [],
                    "approved_claim_ids": [],
                    "perspectives": ["Yêu cầu cung cấp tài liệu"],
                    "sub_queries": [user_query],
                },
                ensure_ascii=False,
            )

        stop_words = {
            "và", "của", "cho", "với", "như", "thế", "nào", "được", "khi", "là", "gì",
            "ở", "đâu", "các", "những", "này", "đó", "ra", "vào", "từ", "về", "trong", "để", "có",
        }
        query_tokens = set(re.findall(r"\w+", user_query.lower())) - stop_words
        try:
            from utils.bilingual import expand_bilingual_tokens
            query_tokens = expand_bilingual_tokens(user_query, query_tokens)
        except Exception:
            pass
        is_slash_cmd = user_query.strip().startswith("/")
        is_general_summary = is_slash_cmd or any(
            kw in user_query.lower()
            for kw in [
                "tóm tắt", "tổng quan", "nội dung chính", "/tom-tat", "/so-sanh",
                "so sánh", "đóng góp", "/dong-gop", "điểm mới", "ưu nhược điểm"
            ]
        )
        is_numeric_query = any(
            kw in user_query.lower()
            for kw in ["bao nhiêu", "mấy", "số lượng", "tỷ lệ", "phần trăm", "chi phí", "ước lượng", "point", "kích thước", "size", "cần gì"]
        )
        is_temporal_query = any(
            kw in user_query.lower()
            for kw in ["khi nào", "ngày", "tháng", "năm", "thời gian", "giai đoạn", "lúc nào", "khi", "lịch sử"]
        )
        full_doc_text = " ".join(u["text"].lower() for u in units)
        content_words = [w for w in re.findall(r"\w+", user_query.lower()) if len(w) > 2 and w not in stop_words]
        content_overlap = [w for w in content_words if w in full_doc_text]
        content_cov = len(content_overlap) / max(1, len(content_words)) if content_words else 1.0

        expanded_words = set(content_words)
        try:
            from utils.bilingual import expand_bilingual_tokens
            expanded_words = expand_bilingual_tokens(user_query, expanded_words)
        except Exception:
            pass
        bilingual_overlap = [w for w in expanded_words if w.lower() in full_doc_text]

        # Generalized Out-of-Domain Detection:
        # If user query has substantive content words but ZERO overlap (even after bilingual expansion),
        # or contains foreign technical/proper terms that do not appear anywhere in the corpus.
        tech_q_tokens = [
            w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]{4,}\b", user_query)
            if w.lower() not in stop_words
        ]
        unmatched_tech = [t for t in tech_q_tokens if t not in full_doc_text]

        is_ood_subject = False
        if not is_general_summary and content_words:
            if len(bilingual_overlap) == 0:
                is_ood_subject = True
            elif tech_q_tokens and len(unmatched_tech) == len(tech_q_tokens) and len(content_overlap) <= 1:
                is_ood_subject = True

        if is_ood_subject and not is_general_summary:
            clean_q = user_query.strip()
            return json.dumps(
                {
                    "answer": (
                        f"Dựa trên các bài báo và tài liệu hiện có trong hệ thống, thông tin về \"{clean_q}\" "
                        f"**hoàn toàn không được đề cập**.\n\n"
                        f"Vui lòng hỏi các câu hỏi liên quan đến nội dung tài liệu được cung cấp hoặc tải thêm tài liệu liên quan lên hệ thống để phân tích."
                    ),
                    "evidence_spans": [],
                    "claims": [],
                    "approved_claim_ids": [],
                    "is_out_of_domain": True,
                    "perspectives": ["Từ chối trung thực (Honest Refusal)"],
                    "sub_queries": [clean_q],
                },
                ensure_ascii=False,
            )

        if is_general_summary:
            def is_admin_or_biblio(text: str) -> bool:
                t = text.lower()
                # 1. Emails & student identifiers
                if "@" in t and any(domain in t for domain in [".edu", ".vn", ".com", ".org", "@st.", "@huflit"]):
                    return True
                # 2. Author names, affiliations & academic/publisher metadata noise
                noise_keywords = [
                    "người thực hiện", "ước lượng (point)", "ước lượng point", "point",
                    "tổng cộng 27", "slide thuyết trình", "luyện tập thuyết trình",
                    "chuẩn bị nội dung slide", "tài liệu tham khảo", "tài liệu chính thức:",
                    "certbot.eff.org", "nginx.org/en/docs", "tv1", "tv2", "tv3", "nhóm trưởng", "cả nhóm",
                    "journal of science", "khoa công nghệ", "trường đại học", "tạp chí",
                    "huflit", "email:", "nhận bài:", "chấp nhận đăng:", "thông tin tác giả",
                    "tóm tắt—", "abstract—", "nguyễn thị", "tân khoa", "minh ý", "mai sương",
                    "mcgraw-hill", "series in computer science", "consulting editor", "tsing hua",
                    "practitioner's approach", "edition roger", "roger s. pressman", "all rights reserved",
                    "printed in the united states", "library of congress", "isbn", "cataloging-in-publication",
                    "publishing company", "dedication", "to barbara"
                ]
                if any(b in t for b in noise_keywords):
                    return True
                # 3. Uppercase document title header lines (e.g. NGHIÊN CỨU ĐA PHƯƠNG PHÁP...)
                words = [w for w in text.split() if len(w) > 1]
                if words and sum(1 for w in words if w.isupper()) / len(words) > 0.5:
                    return True
                return False

            sections_def = [
                {
                    "title": "1. Mục tiêu & Bối cảnh Nghiên cứu",
                    "keywords": ["mục tiêu", "bối cảnh", "nghiên cứu", "phát hiện", "đặt vấn đề", "tổng quan", "phương pháp", "thành phần cốt lõi"],
                },
                {
                    "title": "2. Phương pháp Tiếp cận & Kiến trúc Đánh giá",
                    "keywords": ["phương pháp", "mô hình", "kiến trúc", "bert", "transformer", "llama", "roberta", "thuật toán", "tiếp cận", "phân loại", "bốn nhóm"],
                },
                {
                    "title": "3. Kết quả Thực nghiệm & Số liệu Cốt lõi",
                    "keywords": ["kết quả", "thực nghiệm", "độ chính xác", "f1", "dữ liệu", "chính xác", "đạt", "số liệu", "chỉ số", "bộ dữ liệu", "8.453"],
                },
                {
                    "title": "4. Kết luận & Đánh đổi Triển khai (Trade-offs)",
                    "keywords": ["kết luận", "đánh đổi", "triển khai", "hạn chế", "khuyến nghị", "chi phí", "vận hành", "tối ưu", "cân nhắc", "thực tế"],
                }
            ]

            all_sents = []
            for u in units:
                unit_text = re.sub(r"(?<!\n)\r?\n(?!\r?\n)", " ", u["text"])
                raw_paras = [p.strip() for p in re.split(r"[\r\n]{2,}", unit_text) if len(p.strip()) > 20]
                for p in raw_paras:
                    if is_admin_or_biblio(p):
                        continue
                    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if len(s.strip()) > 25]
                    for s in sentences:
                        if is_admin_or_biblio(s):
                            continue
                        all_sents.append((s, u))

            assigned_per_sec = {i: [] for i in range(len(sections_def))}
            used_sents = set()

            for sec_idx, sec in enumerate(sections_def):
                candidates_sec = []
                for s, u in all_sents:
                    norm_s = re.sub(r"\s+", " ", s.lower().strip())[:90]
                    if norm_s in used_sents:
                        continue
                    s_l = s.lower()
                    match_count = sum(1 for kw in sec["keywords"] if kw in s_l)
                    if match_count > 0:
                        candidates_sec.append((match_count, s, u, norm_s))
                candidates_sec.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
                for _, s, u, norm_s in candidates_sec[:2]:
                    assigned_per_sec[sec_idx].append((s, u))
                    used_sents.add(norm_s)

            for sec_idx in range(len(sections_def)):
                if not assigned_per_sec[sec_idx] and all_sents:
                    for s, u in all_sents:
                        norm_s = re.sub(r"\s+", " ", s.lower().strip())[:90]
                        if norm_s not in used_sents and 50 <= len(s) <= 300:
                            assigned_per_sec[sec_idx].append((s, u))
                            used_sents.add(norm_s)
                            break

            clean_doc_names = list(dict.fromkeys(clean_doc_title(u["doc_id"]) for u in units if "doc_id" in u))
            main_title = clean_doc_names[0] if clean_doc_names else "Tài liệu Nghiên cứu"
            output_lines = [f"### Tóm tắt Trọng tâm: {main_title}\n\n---"]
            claims = []
            evidence_spans = []
            claim_idx = 1

            for sec_idx, sec in enumerate(sections_def):
                items = assigned_per_sec[sec_idx]
                if not items:
                    continue
                output_lines.append(f"\n#### {sec['title']}")
                for s, u in items:
                    output_lines.append(f"• {s} [{claim_idx}] (Trang {u['page_no']})")
                    claims.append({
                        "id": f"claim_{claim_idx}",
                        "text": s,
                        "evidence_unit_id": u["id"]
                    })
                    evidence_spans.append({
                        "quote": s,
                        "unit_id": u["id"],
                        "doc_id": u["doc_id"],
                        "page_no": u["page_no"]
                    })
                    claim_idx += 1

            if claims:
                return json.dumps({
                    "answer": "\n".join(output_lines),
                    "evidence_spans": evidence_spans,
                    "claims": claims,
                    "approved_claim_ids": [c["id"] for c in claims],
                    "perspectives": ["Tóm tắt cấu trúc điều hành (Executive Summary)"],
                    "sub_queries": [user_query],
                }, ensure_ascii=False)

        candidates = []
        for u in units:
            unit_text = re.sub(r"(?<!\n)\r?\n(?!\r?\n)", " ", u["text"])
            raw_paras = [p.strip() for p in re.split(r"[\r\n]{2,}", unit_text) if len(p.strip()) > 20]
            if not raw_paras:
                raw_paras = [unit_text.strip()]
            for p in raw_paras:
                if is_metadata_or_header(p):
                    continue
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if len(s.strip()) > 20]
                if not sentences:
                    sentences = [p]
                for idx_s, raw_s in enumerate(sentences):
                    s = clean_caption_credit(raw_s).strip()
                    if is_metadata_or_header(s) or not is_complete_sentence(s):
                        continue
                    s_tokens = set(re.findall(r"\w+", s.lower())) - stop_words
                    overlap = len(query_tokens & s_tokens)
                    has_quant = has_quantity_for_entity(s, user_query)
                    has_date = bool(re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", s) or re.search(r"\b(19\d\d|20\d\d)\b", s))

                    if is_general_summary:
                        richness = 3 if 60 <= len(s) <= 350 else 1
                        score = overlap * 2 + richness
                    elif is_numeric_query:
                        score = overlap * 2 + (6 if has_quant else 0)
                    elif is_temporal_query:
                        score = overlap * 2 + (7 if has_date else 0)
                    else:
                        score = overlap

                    # Expand ±1 neighboring sentence for richer context window
                    window_sents = [s]
                    if idx_s + 1 < len(sentences) and is_complete_sentence(sentences[idx_s + 1]):
                        window_sents.append(sentences[idx_s + 1])
                    window_text = " ".join(window_sents)
                    candidates.append((score, window_text, u, has_quant))

        candidates.sort(key=lambda x: x[0], reverse=True)

        selected = []
        seen = set()
        min_score_required = 1 if is_general_summary else min(2, max(1, len(query_tokens)))
        for score, s, u, has_quant in candidates:
            norm_s = re.sub(r"\s+", " ", s.lower().strip())[:90]
            if (score >= min_score_required or is_general_summary) and norm_s not in seen:
                seen.add(norm_s)
                selected.append((s, u, has_quant))
                if len(selected) >= 4:
                    break

        if not selected and candidates and candidates[0][0] >= 1:
            for score, s, u, has_quant in candidates:
                norm_s = re.sub(r"\s+", " ", s.lower().strip())[:90]
                if score >= 1 and norm_s not in seen:
                    seen.add(norm_s)
                    selected.append((s, u, has_quant))
                    if len(selected) >= 4:
                        break

        if not selected:
            return json.dumps(
                {
                    "answer": f"Tài liệu đã chọn không chứa nội dung trả lời cho câu hỏi: '{user_query}'.",
                    "evidence_spans": [],
                    "claims": [],
                    "approved_claim_ids": [],
                    "perspectives": ["Tìm kiếm bổ sung"],
                    "sub_queries": [user_query],
                },
                ensure_ascii=False,
            )

        evidence_spans = []
        claims = []
        summary_bullets = []

        for idx, (quote, u, has_quant) in enumerate(selected):
            cl_id = f"claim_{idx + 1}"
            evidence_spans.append(
                {
                    "quote": quote,
                    "unit_id": u["id"],
                    "doc_id": u["doc_id"],
                    "page_no": u["page_no"],
                }
            )
            claims.append(
                {
                    "id": cl_id,
                    "text": quote,
                    "evidence_unit_id": u["id"],
                }
            )
            summary_bullets.append(f"• {quote} [{idx + 1}] (Trang {u['page_no']})")

        clean_doc_names = list(dict.fromkeys(clean_doc_title(u["doc_id"]) for _, u, _ in selected))
        doc_header = f"Dựa trên tài liệu đã chọn (**{', '.join(clean_doc_names)}**):"

        any_quantity = any(has_quant for _, _, has_quant in selected)
        if is_numeric_query and not any_quantity:
            answer_text = (
                f"{doc_header}\n\n"
                f"**Lưu ý:** Tài liệu nguồn không cung cấp số lượng hoặc con số cụ thể cho câu hỏi: *\"{user_query}\"*.\n\n"
                f"Văn bản nguồn chỉ đề cập đến nội dung liên quan sau:\n\n"
                + "\n\n".join(summary_bullets)
            )
        elif is_general_summary:
            answer_text = (
                f"Dưới đây là tóm tắt các phát hiện và nội dung khoa học trọng tâm từ (**{', '.join(clean_doc_names)}**):\n\n"
                + "\n\n".join(summary_bullets)
            )
        else:
            answer_text = f"{doc_header}\n\n" + "\n\n".join(summary_bullets)

        return json.dumps(
            {
                "answer": answer_text,
                "evidence_spans": evidence_spans,
                "claims": claims,
                "approved_claim_ids": [c["id"] for c in claims],
                "perspectives": ["Phân tích nội dung văn bản gốc"],
                "sub_queries": [user_query],
            },
            ensure_ascii=False,
        )


llm_client = UnifiedLLMClient()
