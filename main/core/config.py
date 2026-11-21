"""
Unified Configuration for Multi-Agent Research RAG System.
Supports:
  1. local_api: Lightweight execution on Laptop (ThinkBook G8+) using Gemini / DeepSeek API + CPU search.
  2. kaggle_gpu: Remote high-throughput execution forwarding heavy inference to Kaggle Dual Tesla T4 worker.
  3. local_gpu: On-premise execution with local CUDA GPUs.
"""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
from threading import RLock
from types import MappingProxyType
from typing import Mapping
from dotenv import load_dotenv

_core_dir = os.path.dirname(os.path.abspath(__file__))
_main_dir = os.path.dirname(_core_dir)
_root_dir = os.path.dirname(_main_dir)

for _candidate in [os.path.join(_root_dir, ".env"), os.path.join(_main_dir, ".env")]:
    if os.path.exists(_candidate):
        load_dotenv(_candidate)
        break
else:
    load_dotenv()


_inference_mode: ContextVar[str | None] = ContextVar("agent_inference_mode", default=None)
_run_config: ContextVar["RunConfig | None"] = ContextVar("run_config", default=None)
run_defaults_lock = RLock()
_AGENT_MODE_FIELDS = {
    "retrieval": "AGENT_RETRIEVAL_MODE",
    "reader": "AGENT_READER_MODE",
    "summarizer": "AGENT_SUMMARIZER_MODE",
    "factchecker": "AGENT_FACTCHECKER_MODE",
    "writer": "AGENT_WRITER_MODE",
}


@contextmanager
def agent_inference_mode(mode: str):
    """Pin shared LLM calls to the mode of the graph node currently executing."""
    token = _inference_mode.set(mode)
    try:
        yield
    finally:
        _inference_mode.reset(token)


class _AgentModeSettings:
    def get_agent_mode(self, agent_name: str) -> str:
        key = agent_name.lower().replace("agent", "").replace("_", "").strip()
        return getattr(self, _AGENT_MODE_FIELDS.get(key, f"AGENT_{key.upper()}_MODE"), "api")

    def get_inference_mode(self, agent_name: str | None = None) -> str:
        mode = _inference_mode.get()
        if mode is not None:
            return mode
        return self.get_agent_mode(agent_name) if agent_name else self.EXECUTION_MODE

    def get_all_agent_modes(self) -> dict[str, str]:
        return {role: getattr(self, name) for role, name in _AGENT_MODE_FIELDS.items()}


@dataclass
class Settings(_AgentModeSettings):
    # Execution Mode: "local_api" | "kaggle_gpu" | "local_gpu"
    EXECUTION_MODE: str = field(
        default_factory=lambda: os.getenv("EXECUTION_MODE", "local_api").lower()
    )

    # Remote Kaggle Worker Endpoint (Cloudflare Tunnel or ngrok URL)
    REMOTE_INFERENCE_URL: str = field(
        default_factory=lambda: os.getenv("REMOTE_INFERENCE_URL", "").rstrip("/")
    )
    REMOTE_INFERENCE_TOKEN: str = field(
        default_factory=lambda: os.getenv("REMOTE_INFERENCE_TOKEN", "").strip()
    )
    REMOTE_TIMEOUT_SECONDS: int = int(os.getenv("REMOTE_TIMEOUT_SECONDS", "90"))

    # Local CPU Inference Engine (Ollama / Llama.cpp / GGUF on user's PC)
    LOCAL_CPU_URL: str = field(
        default_factory=lambda: os.getenv("LOCAL_CPU_URL", "http://localhost:11434").rstrip("/")
    )
    LOCAL_CPU_MODEL: str = field(
        default_factory=lambda: os.getenv("LOCAL_CPU_MODEL", "qwen2.5:1.5b")
    )

    # Granular Per-Agent Execution Modes: "api" | "model" | "cpu"
    AGENT_RETRIEVAL_MODE: str = field(
        default_factory=lambda: os.getenv("AGENT_RETRIEVAL_MODE", "api").lower()
    )
    AGENT_READER_MODE: str = field(
        default_factory=lambda: os.getenv("AGENT_READER_MODE", "api").lower()
    )
    AGENT_SUMMARIZER_MODE: str = field(
        default_factory=lambda: os.getenv("AGENT_SUMMARIZER_MODE", "api").lower()
    )
    AGENT_FACTCHECKER_MODE: str = field(
        default_factory=lambda: os.getenv("AGENT_FACTCHECKER_MODE", "api").lower()
    )
    AGENT_WRITER_MODE: str = field(
        default_factory=lambda: os.getenv("AGENT_WRITER_MODE", "api").lower()
    )

    def set_agent_mode(self, agent_name: str, mode: str):
        """Sets 'api' or 'model' for the specified agent."""
        clean_name = agent_name.lower().replace("agent", "").replace("_", "").strip()
        attr = _AGENT_MODE_FIELDS.get(clean_name, f"AGENT_{clean_name.upper()}_MODE")
        if hasattr(self, attr):
            setattr(self, attr, mode.lower())

    # API Keys for Cloud Foundation Models
    GEMINI_API_KEY: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    DEEPSEEK_API_KEY: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    COHERE_API_KEY: str = field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""))

    # Foundation LLM Models
    DEFAULT_LLM_MODEL: str = field(
        default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "gemini-3.1-flash-lite")
    )
    FALLBACK_LLM_MODEL: str = field(
        default_factory=lambda: os.getenv("FALLBACK_LLM_MODEL", "gemini-flash-lite-latest")
    )

    # Embedding & Retrieval Models
    EMBEDDING_MODEL: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    )
    GPU_FINETUNE_DENSE_ENABLED: bool = field(
        default_factory=lambda: os.getenv("GPU_FINETUNE_DENSE_ENABLED", "false").strip().lower() == "true"
    )
    GPU_FINETUNE_PAGE_DIVERSITY_ENABLED: bool = field(
        default_factory=lambda: os.getenv("GPU_FINETUNE_PAGE_DIVERSITY_ENABLED", "false").strip().lower() == "true"
    )
    GPU_FINETUNE_RERANK_ENABLED: bool = field(
        default_factory=lambda: os.getenv("GPU_FINETUNE_RERANK_ENABLED", "false").strip().lower() == "true"
    )
    GPU_FINETUNE_RERANK_KEEP_PREFIX: int = field(
        default_factory=lambda: int(os.getenv("GPU_FINETUNE_RERANK_KEEP_PREFIX", "0"))
    )
    HYBRID_DENSE_ENABLED: bool = field(
        default_factory=lambda: os.getenv("HYBRID_DENSE_ENABLED", "true").strip().lower() == "true"
    )
    HYBRID_RERANK_ENABLED: bool = field(
        default_factory=lambda: os.getenv("HYBRID_RERANK_ENABLED", "true").strip().lower() == "true"
    )
    HYBRID_RERANK_KEEP_PREFIX: int = field(
        default_factory=lambda: int(os.getenv("HYBRID_RERANK_KEEP_PREFIX", "3"))
    )
    GEMINI_DENSE_ENABLED: bool = field(
        default_factory=lambda: os.getenv("GEMINI_DENSE_ENABLED", "false").strip().lower() == "true"
    )
    GEMINI_EMBEDDING_MODEL: str = field(
        default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
    )
    FALLBACK_EMBEDDING_MODEL: str = field(
        default_factory=lambda: os.getenv("FALLBACK_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    )
    RERANKER_MODEL: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    )

    # Multi-Agent StateGraph Limits & Budget Guards
    MAX_ORCHESTRATOR_RETRY: int = int(os.getenv("MAX_ORCHESTRATOR_RETRY", "3"))
    DEFAULT_MAX_INTERNAL_RETRY: int = int(os.getenv("DEFAULT_MAX_INTERNAL_RETRY", "2"))
    MAX_FACTCHECK_CALLS: int = int(os.getenv("MAX_FACTCHECK_CALLS", "120"))
    MAX_RETRIEVED_CHUNKS: int = int(os.getenv("MAX_RETRIEVED_CHUNKS", "10"))
    TOP_K_RERANK: int = int(os.getenv("TOP_K_RERANK", "5"))

    # Data & Storage Paths
    BASE_DIR: str = field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    CORPUS_PATH: str = field(
        default_factory=lambda: os.getenv(
            "CORPUS_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "corpus.jsonl")
        )
    )
    INDEX_DIR: str = field(
        default_factory=lambda: os.getenv(
            "INDEX_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "index")
        )
    )
    ARTIFACTS_DIR: str = field(
        default_factory=lambda: os.getenv(
            "ARTIFACTS_DIR", os.path.join(os.path.dirname(__file__), "..", "artifacts")
        )
    )
    UPLOAD_DIR: str = field(
        default_factory=lambda: os.getenv(
            "UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "..", "uploads")
        )
    )

    @property
    def FIGURE_VISION_PROVIDER(self) -> str:
        return os.getenv("FIGURE_VISION_PROVIDER", "").strip().lower()

    @property
    def FIGURE_VISION_MODEL(self) -> str:
        return os.getenv("FIGURE_VISION_MODEL", "").strip()

    @property
    def FIGURE_OCR_LANGUAGE(self) -> str:
        return os.getenv("FIGURE_OCR_LANGUAGE", "eng")


settings = Settings()


@dataclass(frozen=True, eq=False)
class RunConfig(_AgentModeSettings):
    """Read-only copy of the settings consumed by an admitted research run."""

    _values: Mapping[str, str | int] = field(repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_values", MappingProxyType(dict(self._values)))

    def __getattr__(self, name: str):
        try:
            return object.__getattribute__(self, "_values")[name]
        except KeyError:
            raise AttributeError(name) from None

    @classmethod
    def capture(
        cls, *, remote_inference_url: str | None = None, corpus_path: str | None = None,
    ) -> "RunConfig":
        with run_defaults_lock:
            values = {entry.name: getattr(settings, entry.name) for entry in fields(settings)}
            for name in ("FIGURE_VISION_PROVIDER", "FIGURE_VISION_MODEL", "FIGURE_OCR_LANGUAGE"):
                values[name] = getattr(settings, name)
            if remote_inference_url is not None:
                values["REMOTE_INFERENCE_URL"] = remote_inference_url
            if corpus_path is not None:
                values["CORPUS_PATH"] = corpus_path
            return cls(values)


def get_run_settings() -> RunConfig | Settings:
    return _run_config.get() or settings


@contextmanager
def use_run_config(config: RunConfig):
    token = _run_config.set(config)
    try:
        yield
    finally:
        _run_config.reset(token)
