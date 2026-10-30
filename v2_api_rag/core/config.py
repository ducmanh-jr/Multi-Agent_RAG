import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Tìm và load file .env trong v2_api_rag hoặc root
ENV_PATHS = [
    Path(__file__).resolve().parent.parent / ".env",
    Path(__file__).resolve().parent.parent.parent / ".env",
]
for p in ENV_PATHS:
    if p.exists():
        load_dotenv(p)
        break

@dataclass
class Settings:
    # API Keys
    GEMINI_API_KEY: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    GROQ_API_KEY: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))

    # Model Settings (Sử dụng model frontier sẵn có, tối ưu độ trễ & chi phí)
    DEFAULT_FAST_LLM: str = field(default_factory=lambda: os.getenv("DEFAULT_FAST_LLM", "gemini-2.5-flash"))
    DEFAULT_REASONING_LLM: str = field(default_factory=lambda: os.getenv("DEFAULT_REASONING_LLM", "gemini-2.5-flash"))
    DEFAULT_EMBEDDING_MODEL: str = field(default_factory=lambda: os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-004"))

    # Graph Retries
    MAX_ORCHESTRATOR_RETRY: int = int(os.getenv("MAX_ORCHESTRATOR_RETRY", "3"))
    DEFAULT_MAX_INTERNAL_RETRY: int = int(os.getenv("DEFAULT_MAX_INTERNAL_RETRY", "2"))

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    CORPUS_PATH: str = field(default_factory=lambda: str(Path(__file__).resolve().parent.parent / "data" / "corpus.jsonl"))

settings = Settings()
