"""
Unified Configuration for Multi-Agent Research RAG System.
Supports:
  1. local_api: Lightweight execution on Laptop (ThinkBook G8+) using Gemini / DeepSeek API + CPU search.
  2. kaggle_gpu: Remote high-throughput execution forwarding heavy inference to Kaggle Dual Tesla T4 worker.
  3. local_gpu: On-premise execution with local CUDA GPUs.
"""

import os
from dataclasses import dataclass, field
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


@dataclass
class Settings:
    # Execution Mode: "local_api" | "kaggle_gpu" | "local_gpu"
    EXECUTION_MODE: str = field(
        default_factory=lambda: os.getenv("EXECUTION_MODE", "local_api").lower()
    )

    # Remote Kaggle Worker Endpoint (Cloudflare Tunnel or ngrok URL)
    REMOTE_INFERENCE_URL: str = field(
        default_factory=lambda: os.getenv("REMOTE_INFERENCE_URL", "").rstrip("/")
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

    def get_agent_mode(self, agent_name: str) -> str:
        """Returns 'api' or 'model' for the specified agent."""
        clean_name = agent_name.lower().replace("agent", "").replace("_", "").strip()
        name_map = {
            "retrieval": "AGENT_RETRIEVAL_MODE",
            "reader": "AGENT_READER_MODE",
            "summarizer": "AGENT_SUMMARIZER_MODE",
            "factchecker": "AGENT_FACTCHECKER_MODE",
            "writer": "AGENT_WRITER_MODE",
        }
        attr = name_map.get(clean_name, f"AGENT_{clean_name.upper()}_MODE")
        return getattr(self, attr, "api")

    def set_agent_mode(self, agent_name: str, mode: str):
        """Sets 'api' or 'model' for the specified agent."""
        clean_name = agent_name.lower().replace("agent", "").replace("_", "").strip()
        name_map = {
            "retrieval": "AGENT_RETRIEVAL_MODE",
            "reader": "AGENT_READER_MODE",
            "summarizer": "AGENT_SUMMARIZER_MODE",
            "factchecker": "AGENT_FACTCHECKER_MODE",
            "writer": "AGENT_WRITER_MODE",
        }
        attr = name_map.get(clean_name, f"AGENT_{clean_name.upper()}_MODE")
        if hasattr(self, attr):
            setattr(self, attr, mode.lower())

    def get_all_agent_modes(self) -> dict[str, str]:
        return {
            "retrieval": self.AGENT_RETRIEVAL_MODE,
            "reader": self.AGENT_READER_MODE,
            "summarizer": self.AGENT_SUMMARIZER_MODE,
            "factchecker": self.AGENT_FACTCHECKER_MODE,
            "writer": self.AGENT_WRITER_MODE,
        }

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


settings = Settings()
