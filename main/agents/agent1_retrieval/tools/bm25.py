"""
In-memory BM25 Keyword Search Engine.
Provides rapid lexical retrieval with zero heavy external dependencies.
"""

from __future__ import annotations

import math
import unicodedata
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
        """Case-insensitive word tokenizer supporting alphanumeric & unicode with NFC normalization."""
        if not text:
            return []
        norm_text = unicodedata.normalize("NFC", str(text).lower())
        return re.findall(r"\w+", norm_text)

    def fit(self, docs: list[dict[str, Any]]):
        """Index a list of documents where each doc has at least 'text' and 'id'."""
        self.docs = docs
        self.corpus_size = len(docs)
        if self.corpus_size == 0:
            return

        total_length = 0
        df: Counter = Counter()
        self.doc_freqs = []
        self.doc_lens = []

        for doc in docs:
            tokens = self.tokenize(doc.get("text", ""))
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_length += doc_len
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
        stop_words = {
            "là", "gì", "và", "của", "cho", "với", "như", "thế", "nào", "được", "khi",
            "ở", "đâu", "các", "những", "này", "đó", "ra", "vào", "từ", "về", "trong",
            "để", "có", "hãy", "biết", "ai", "mấy"
        }
        content_q_tokens = [t for t in q_tokens if len(t) > 1 and t.lower() not in stop_words]
        if content_q_tokens:
            q_tokens = content_q_tokens
        try:
            from utils.bilingual import expand_bilingual_tokens
            q_tokens = list(expand_bilingual_tokens(query, set(q_tokens)))
        except Exception as e:
            logger.debug(f"[BM25] Bilingual token expansion bypassed: {e}")
        scores: list[float] = [0.0] * self.corpus_size

        for i, freqs in enumerate(self.doc_freqs):
            doc_len = self.doc_lens[i] if i < len(self.doc_lens) else sum(freqs.values())
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
