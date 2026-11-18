"""
In-memory BM25 Keyword Search Engine.
Provides rapid lexical retrieval with zero heavy external dependencies.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


class SimpleBM25:
    """Okapi BM25 implementation for lexical document retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[dict[str, Any]] = []
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: list[Counter] = []
        self.idf: dict[str, float] = {}

    def tokenize(self, text: str) -> list[str]:
        """Case-insensitive word tokenizer supporting alphanumeric & unicode."""
        return re.findall(r"\w+", text.lower())

    def fit(self, docs: list[dict[str, Any]]):
        """Index a list of documents where each doc has at least 'text' and 'id'."""
        self.docs = docs
        self.corpus_size = len(docs)
        if self.corpus_size == 0:
            return

        total_length = 0
        df: Counter = Counter()
        self.doc_freqs = []

        for doc in docs:
            tokens = self.tokenize(doc.get("text", ""))
            total_length += len(tokens)
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            for token in freqs.keys():
                df[token] += 1

        self.avgdl = total_length / self.corpus_size

        # Compute IDF with smoothing
        self.idf = {}
        for token, freq in df.items():
            self.idf[token] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve top_k documents matching query with cross-lingual expansion."""
        if self.corpus_size == 0:
            return []

        q_tokens = self.tokenize(query)
        try:
            from utils.bilingual import expand_bilingual_tokens
            q_tokens = list(expand_bilingual_tokens(query, set(q_tokens)))
        except Exception:
            pass
        scores: list[float] = [0.0] * self.corpus_size

        for i, freqs in enumerate(self.doc_freqs):
            doc_len = sum(freqs.values())
            for q in q_tokens:
                if q not in freqs:
                    continue
                tf = freqs[q]
                idf = self.idf.get(q, 0.1)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * (doc_len / (self.avgdl or 1)))
                scores[i] += idf * (num / den)

        ranked = sorted(
            range(self.corpus_size), key=lambda idx: scores[idx], reverse=True
        )

        results = []
        for idx in ranked[:top_k]:
            if scores[idx] > 0:
                doc = dict(self.docs[idx])
                doc["bm25_score"] = round(scores[idx], 4)
                results.append(doc)

        return results
