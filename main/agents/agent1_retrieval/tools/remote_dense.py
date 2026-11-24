"""Scoped dense retrieval using the versioned worker embedding contract."""

from collections import OrderedDict
import re
import threading
import time
from typing import Any

import httpx
import numpy as np

from core.config import get_run_settings
from core.token_usage import metered_post


DEFAULT_DENSE_WEIGHT = 0.67
# The second pass combines two short rankings. A smaller offset lets dense-only hits compete.
DEFAULT_DENSE_RRF_K = 2


class DenseRetrievalError(RuntimeError):
    """A dense failure that allows the caller to keep its sparse ranking."""


class RemoteDenseRetriever:
    BATCH_SIZE = 16
    MAX_TEXT_LENGTH = 4096
    DIMENSION = 1024
    CACHE_SIZE = 4096

    def __init__(self) -> None:
        self._identity: tuple[str, str, str] | None = None
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.RLock()

    def _embed(
        self, client: httpx.Client, url: str, model: str, texts: list[str], deadline: float,
    ) -> tuple[str, np.ndarray]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DenseRetrievalError("embedding_timeout")
        token = (getattr(get_run_settings(), "REMOTE_INFERENCE_TOKEN", "") or "").strip()
        req_kwargs: dict[str, Any] = {"json": {"texts": texts}, "timeout": remaining}
        if token:
            req_kwargs["headers"] = {"Authorization": f"Bearer {token}"}
        try:
            response = metered_post(
                client.post, f"{url}/api/embed", provider="kaggle", operation="api/embed",
                protocol="worker", **req_kwargs,
            )
        except httpx.TimeoutException as exc:
            raise DenseRetrievalError("embedding_timeout") from exc
        except httpx.RequestError as exc:
            raise DenseRetrievalError("embedding_worker_unreachable") from exc
        if time.monotonic() >= deadline:
            raise DenseRetrievalError("embedding_timeout")
        if response.status_code != 200:
            raise DenseRetrievalError(f"embedding_http_{response.status_code}")

        try:
            data = response.json()
            if data["model"] != model:
                raise DenseRetrievalError("embedding_model_mismatch")
            revision = data["revision"]
            if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
                raise ValueError("Invalid checkpoint revision")
            rows = data["embeddings"]
            if (not isinstance(rows, list) or len(rows) != len(texts)
                    or any(not isinstance(row, list) or len(row) != self.DIMENSION
                           or any(type(value) not in (float, int) for value in row) for row in rows)):
                raise ValueError("Expected one numeric vector for each input")
            vectors = np.asarray(rows, dtype=np.float32)
            if vectors.shape != (len(texts), self.DIMENSION) or not np.isfinite(vectors).all():
                raise ValueError("Invalid embedding shape or non-finite values")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or (norms <= 0).any():
                raise ValueError("Invalid embedding norm")
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
            raise DenseRetrievalError("invalid_embedding_response") from exc
        return revision, vectors / norms

    def rank(
        self, query: str, passages: list[dict[str, Any]], *, top_k: int, client: httpx.Client,
        url: str, model: str, timeout: float,
    ) -> tuple[list[dict[str, Any]], str]:
        texts = [str(passage.get("text") or "").strip()[:self.MAX_TEXT_LENGTH] for passage in passages]
        candidates = [(index, text) for index, text in enumerate(texts) if text]
        if not candidates or not query.strip():
            raise DenseRetrievalError("missing_embedding_input")

        deadline = time.monotonic() + timeout
        url = url.rstrip("/")
        revision, query_vectors = self._embed(client, url, model, [query.strip()[:self.MAX_TEXT_LENGTH]], deadline)
        identity = (url, model, revision)

        unique_texts = dict.fromkeys(text for _, text in candidates)
        # Serialize cache lookup + missing-passage embedding across parallel DAG branches so the same PDF
        # is never embedded 3x concurrently over the remote Kaggle tunnel.
        with self._lock:
            if self._identity != identity:
                self._cache.clear()
                self._identity = identity
            vectors = {text: self._cache[text] for text in unique_texts if text in self._cache}
            missing_texts = [text for text in unique_texts if text not in vectors]
            for start in range(0, len(missing_texts), self.BATCH_SIZE):
                batch = missing_texts[start:start + self.BATCH_SIZE]
                batch_revision, batch_vectors = self._embed(client, url, model, batch, deadline)
                if batch_revision != revision:
                    raise DenseRetrievalError("embedding_revision_changed")
                vectors.update(zip(batch, batch_vectors, strict=True))
            for text, vector in vectors.items():
                self._cache[text] = vector
                self._cache.move_to_end(text)
            while len(self._cache) > self.CACHE_SIZE:
                self._cache.popitem(last=False)

        matrix = np.stack([vectors[text] for _, text in candidates])
        scores = matrix @ query_vectors[0]
        if time.monotonic() >= deadline:
            raise DenseRetrievalError("embedding_timeout")
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [passages[candidates[index][0]] for index in order], revision
