"""
Bilingual Academic Concept Mapping (Vietnamese <-> English).
Bridges the lexical vocabulary gap in pure CPU RAG when queries are in Vietnamese
and literature/papers are in English (or vice versa).
"""

from __future__ import annotations

import re
from typing import Set

BILINGUAL_PHRASE_MAP: dict[str, list[str]] = {
    # Deep Learning & NLP Concepts
    "cơ chế chú ý": ["attention", "mechanism"],
    "chú ý": ["attention"],
    "tự chú ý": ["self", "attention"],
    "đa đầu": ["multi", "head"],
    "tinh chỉnh": ["fine", "tuning", "finetuning", "adaptation"],
    "tham số": ["parameters", "params", "weights"],
    "trọng số": ["weights"],
    "nén": ["compression", "quantization", "pruning"],
    "lượng tử hóa": ["quantization", "quantized", "bits"],
    "thứ hạng thấp": ["low", "rank", "lora"],
    "hạ bậc": ["low", "rank"],
    "bộ nhớ": ["memory", "vram", "sram", "footprint", "bandwidth"],
    "băng thông": ["bandwidth", "throughput", "traffic"],
    "độ trễ": ["latency", "delay", "speed"],
    "tốc độ": ["speed", "throughput", "fps", "tokens"],
    "độ chính xác": ["accuracy", "precision", "recall", "f1", "score"],
    "mô hình ngôn ngữ": ["language", "model", "models", "llm", "llms"],
    "mô hình": ["model", "architecture", "framework"],
    "học máy": ["machine", "learning"],
    "trí tuệ nhân tạo": ["artificial", "intelligence", "ai"],
    "tóm tắt": ["summary", "abstract", "overview", "introduction"],
    "đóng góp": ["contribution", "contributions", "propose", "proposed"],
    "phương pháp": ["method", "methodology", "approach", "technique"],
    "kết quả": ["results", "evaluation", "experiments", "performance"],
    "thực nghiệm": ["empirical", "experiment", "experimental"],
    "huấn luyện": ["training", "train", "optimization"],
    "suy luận": ["inference", "decoding", "generation"],
    "tiền huấn luyện": ["pretraining", "pretrained"],
    "tập dữ liệu": ["dataset", "benchmark", "corpus"],
    "phần cứng": ["hardware", "gpu", "cpu", "tpu"],
    "hạn chế": ["limitation", "limitations", "drawback"],
    "ưu điểm": ["advantage", "advantages", "benefit"],
    "kiến trúc": ["architecture", "transformer", "network"],
    "hiện vật": ["artifacts", "specimens", "relics", "exhibits"],
    "bảo tàng": ["museum"],
    "thành lập": ["founded", "established", "established"],
    "khai trương": ["inaugurated", "opened"],
}

# English -> Vietnamese reverse map
ENGLISH_TO_VIETNAMESE: dict[str, list[str]] = {
    "attention": ["chú", "ý"],
    "transformer": ["kiến", "trúc", "transformer"],
    "lora": ["tinh", "chỉnh", "thứ", "hạng", "thấp"],
    "qlora": ["lượng", "tử", "hóa", "tinh", "chỉnh"],
    "memory": ["bộ", "nhớ", "vram"],
    "parameters": ["tham", "số"],
    "weights": ["trọng", "số"],
    "latency": ["độ", "trễ"],
    "throughput": ["băng", "thông", "tốc", "độ"],
    "fine-tuning": ["tinh", "chỉnh"],
    "accuracy": ["độ", "chính", "xác"],
    "training": ["huấn", "luyện"],
    "inference": ["suy", "luận"],
    "quantization": ["lượng", "tử", "hóa"],
    "abstract": ["tóm", "tắt", "nội", "dung"],
    "summary": ["tóm", "tắt"],
}


def expand_bilingual_tokens(query: str, current_tokens: Set[str]) -> Set[str]:
    """
    Expands token set with bilingual synonyms (Vietnamese <-> English).
    Returns an expanded set of keyword tokens.
    """
    expanded = set(current_tokens)
    q_low = query.lower()

    # 1. Match Vietnamese phrases -> inject English tokens
    for vn_phrase, en_tokens in BILINGUAL_PHRASE_MAP.items():
        if vn_phrase in q_low:
            expanded.update(en_tokens)

    # 2. Match English words -> inject Vietnamese tokens
    for en_word, vn_tokens in ENGLISH_TO_VIETNAMESE.items():
        if re.search(r'\b' + re.escape(en_word) + r'\b', q_low):
            expanded.update(vn_tokens)

    return expanded
