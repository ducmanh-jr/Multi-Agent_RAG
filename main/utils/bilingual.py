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
    "đa đầu": ["multi", "head", "heads"],
    "đầu chú ý": ["attention", "head", "heads"],
    "tầng con": ["sub-layers", "sub-layer", "sublayer", "sublayers"],
    "hệ số tỉ lệ": ["scale", "scaled", "scaling", "factor"],
    "tỉ lệ": ["scale", "scaled", "ratio"],
    "phép nhân vô hướng": ["dot", "product", "dot-product"],
    "tích vô hướng": ["dot", "product", "dot-product"],
    "mạng nơ-ron truyền thẳng": ["feed-forward", "feedforward", "ffn"],
    "truyền thẳng": ["feed-forward", "feedforward"],
    "mã hóa vị trí": ["positional", "encoding", "encodings"],
    "bước khởi động": ["warmup", "steps"],
    "điều chuẩn": ["regularization", "regularizer", "label", "smoothing"],
    "làm mịn nhãn": ["label", "smoothing"],
    "mặt nạ": ["mask", "masked", "masking"],
    "kết nối tàn dư": ["residual", "connection", "connections"],
    "tàn dư": ["residual"],
    "chuẩn hóa lớp": ["layer", "normalization", "layernorm", "norm"],
    "hàm kích hoạt": ["activation", "function", "relu"],
    "độ phức tạp": ["computational", "complexity", "path", "length", "sequential"],
    "tự hồi quy": ["autoregressive", "leftward", "subsequent"],
    "chiều ẩn": ["dimension", "dimensions", "d_model", "d_ff", "d_k", "d_v"],
    "tối ưu hóa": ["optimizer", "adam", "learning", "rate"],
    "bộ tối ưu": ["optimizer", "adam"],
    "điểm bleu": ["bleu", "score"],
    "dịch máy": ["machine", "translation", "wmt", "english", "german", "french"],
    "tinh chỉnh": ["fine", "tuning", "finetuning", "adaptation"],
    "tham số": ["parameters", "params", "weights"],
    "trọng số": ["weights", "weight"],
    "đóng băng": ["freeze", "frozen", "freezes", "freezing"],
    "phân rã": ["decomposition", "decompose", "factorization"],
    "hạng thấp": ["low", "rank", "lora"],
    "hạng": ["rank", "ranks"],
    "ma trận": ["matrix", "matrices"],
    "mục đích": ["purpose", "objective", "goal"],
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
    "attention": ["chú ý", "attention"],
    "transformer": ["kiến trúc", "mô hình transformer", "transformer"],
    "lora": ["tinh chỉnh", "thứ hạng thấp", "lora"],
    "qlora": ["lượng tử hóa", "tinh chỉnh qlora", "qlora"],
    "memory": ["bộ nhớ", "vram"],
    "parameters": ["tham số"],
    "weights": ["trọng số"],
    "latency": ["độ trễ"],
    "throughput": ["băng thông", "tốc độ xử lý"],
    "fine-tuning": ["tinh chỉnh", "fine-tuning"],
    "accuracy": ["độ chính xác", "accuracy"],
    "training": ["huấn luyện", "đào tạo"],
    "inference": ["suy luận"],
    "quantization": ["lượng tử hóa"],
    "abstract": ["tóm tắt", "nội dung chính"],
    "summary": ["tóm tắt", "tổng quan"],
}

# Pre-compiled regex patterns for high-frequency lookup avoidance
_COMPILED_EN_PATTERNS: dict[str, re.Pattern] = {
    en_word: re.compile(r'\b' + re.escape(en_word) + r'\b')
    for en_word in ENGLISH_TO_VIETNAMESE
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

    # 2. Match English words via pre-compiled patterns -> inject Vietnamese tokens
    for en_word, pattern in _COMPILED_EN_PATTERNS.items():
        if pattern.search(q_low):
            expanded.update(ENGLISH_TO_VIETNAMESE[en_word])

    return expanded
