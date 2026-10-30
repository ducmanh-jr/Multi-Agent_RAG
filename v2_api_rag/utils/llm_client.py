import os
import json
from typing import Type, TypeVar, Optional, Any
from pydantic import BaseModel
from google import genai
from core.config import settings
from utils.logger import logger

T = TypeVar("T", bound=BaseModel)

_client: Optional[genai.Client] = None

def get_genai_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        _client = genai.Client(api_key=api_key)
    return _client

def call_structured_llm(
    prompt: str,
    schema: Type[T],
    system_instruction: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2
) -> T:
    """
    Gọi Gemini API với Structured Output trực tiếp bằng Pydantic Schema.
    Đảm bảo 100% trả về đúng định dạng mà không cần parse regex hay fine-tune.
    """
    client = get_genai_client()
    model_name = model or settings.DEFAULT_FAST_LLM

    config_params: dict[str, Any] = {
        "response_mime_type": "application/json",
        "response_schema": schema,
        "temperature": temperature,
    }
    if system_instruction:
        config_params["system_instruction"] = system_instruction

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config_params,
        )
        return response.parsed
    except Exception as e:
        logger.error(f"[call_structured_llm Error] Model {model_name} thất bại: {e}")
        raise e

def call_text_llm(
    prompt: str,
    system_instruction: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.3
) -> str:
    """
    Gọi Gemini API trả về text tự nhiên (dành cho viết báo cáo nghiên cứu hoàn chỉnh).
    """
    client = get_genai_client()
    model_name = model or settings.DEFAULT_REASONING_LLM

    config_params: dict[str, Any] = {
        "temperature": temperature,
    }
    if system_instruction:
        config_params["system_instruction"] = system_instruction

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config_params,
        )
        return response.text or ""
    except Exception as e:
        logger.error(f"[call_text_llm Error] Model {model_name} thất bại: {e}")
        raise e
