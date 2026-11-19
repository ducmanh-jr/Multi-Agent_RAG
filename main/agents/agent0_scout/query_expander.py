"""Classify discovery requests and build academic search queries locally."""

from __future__ import annotations

import re
from typing import Any


OUT_OF_DOMAIN_PATTERNS = [
    (r"\b(sau\s+giải\s+phóng|sau\s+giai\s+phong|giải\s+phóng\s+miền\s+nam|cách\s+mạng\s+tháng\s+tám|chiến\s+tranh\s+thế\s+giới|lịch\s+sử|triều\s+đại|nhà\s+trần|nhà\s+nguyễn)\b",
     "Lịch sử / Chính trị địa phương"),
    (r"\b(thơ\s+nguyễn\s+du|truyện\s+kiều|văn\s+học|tiểu\s+thuyết|thơ\s+ca|nhạc\s+trịnh)\b",
     "Văn học / Nghệ thuật"),
    (r"\b(luật\s+đất\s+đai|luật\s+giao\s+thông|nghị\s+định|thông\s+tư|bộ\s+luật\s+hình\s+sự|văn\s+bản\s+pháp\s+luật)\b",
     "Văn bản Pháp luật địa phương"),
    (r"\b(cách\s+nấu|nấu\s+ăn|món\s+ăn|phở\s+bò|bánh\s+chưng|công\s+thức\s+nấu)\b",
     "Ẩm thực / Đời sống"),

]


NON_STEM_PATTERNS = [
    (r"\b(ô\s*tô|xe\s*hơi|người\s*phát\s*minh|phát\s*minh\s*ra|karl\s*benz|daimler|đại\s*học\s*xây\s*dựng|tô\s*lâm|chính\s*trị|triết\s*học)\b",
     "Lịch sử / Khoa học Xã hội / Nhân văn"),
]

CONVERSATIONAL_BOILERPLATE = [
    r"\btìm\s+bài\s+báo\s+về\b", r"\btim\s+bai\s+bao\s+ve\b",
    r"\btìm\s+bài\s+báo\b", r"\btim\s+bai\s+bao\b",
    r"\btìm\s+giúp\s+tôi\b", r"\btim\s+giup\s+toi\b",
    r"\bcho\s+mình\s+xin\b", r"\bcho\s+minh\s+xin\b",
    r"\bliệt\s+kê\s+các\s+công\s+trình\s+gần\s+đây\s+về\b",
    r"\bliệt\s+kê\s+các\s+công\s+trình\s+về\b",
    r"\bliệt\s+kê\b", r"\bgần\s+đây\b",
    r"\bvài\s+bài\s+báo\s+về\b", r"\bbài\s+báo\s+về\b", r"\bbai\s+bao\s+ve\b",
    r"\btài\s+liệu\s+nghiên\s+cứu\s+về\b", r"\btài\s+liệu\s+về\b", r"\btai\s+lieu\s+ve\b",
    r"\bnghiên\s+cứu\s+về\b", r"\bnghien\s+cuu\s+ve\b",
    r"\bcó\s+bài\s+nào\s+nói\s+về\b", r"\bnói\s+về\b", r"\bkhông\s*\??$",
    r"\bsearch\s+papers\s+về\b", r"\bsearch\s+papers\b", r"\bsearch\b", r"\bpapers\b",
    r"\btìm\s+bài\s+về\b", r"\btim\s+bai\s+ve\b", r"\btìm\s+bài\b", r"\btim\s+bai\b",
    r"\btìm\s+tài\s+liệu\s+về\b", r"\btim\s+tai\s+lieu\s+ve\b",
    r"\bcho\s+mình\b", r"\bcho\s+tôi\b",
]

STEM_TRANSLATIONS = [
    (r"\bmô hình ngôn ngữ lớn\b", "large language models"),
    (r"\bmo hinh ngon ngu lon\b", "large language models"),
    (r"\bhọc tăng cường\b", "reinforcement learning"),
    (r"\bhoc tang cuong\b", "reinforcement learning"),
    (r"\bmạng nơ-ron đồ thị\b", "graph neural networks"),
    (r"\bmang no-ron do thi\b", "graph neural networks"),
    (r"\bthị giác máy tính\b", "computer vision"),
    (r"\bthi giac may tinh\b", "computer vision"),
    (r"\bxử lý ngôn ngữ tự nhiên\b", "natural language processing"),
    (r"\bxu ly ngon ngu tu nhien\b", "natural language processing"),
    (r"\btiếng việt\b", "vietnamese"),
    (r"\btieng viet\b", "vietnamese"),
    (r"\bhọc sâu\b", "deep learning"),
    (r"\bhoc sau\b", "deep learning"),
    (r"\bhọc máy\b", "machine learning"),
    (r"\bhoc may\b", "machine learning"),
    (r"\btrí tuệ nhân tạo\b", "artificial intelligence"),
    (r"\btri tue nhan tao\b", "artificial intelligence"),
    (r"\by tế\b", "medical healthcare"),
    (r"\by te\b", "medical healthcare"),
    (r"\btrong y học\b", "in medicine"),
    (r"\bbảo tồn di sản\b", "cultural heritage preservation"),
    (r"\bdự đoán ô nhiễm không khí\b", "air pollution prediction"),
    (r"\bso sánh cnn và transformer cho thị giác máy tính\b", "cnn transformer computer vision comparison"),
]

def clean_academic_keywords(text: str) -> str:
    """Strips conversational noise and translates key Vietnamese STEM terms into academic keywords."""
    s = text.strip()
    s = re.sub(r"[!?,.]+$", "", s).strip()
    for pat, rep in STEM_TRANSLATIONS:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    for bp in CONVERSATIONAL_BOILERPLATE:
        s = re.sub(bp, " ", s, flags=re.IGNORECASE)
    words = re.findall(r"[\w-]+", s, re.UNICODE)
    stop = {"về", "ve", "và", "va", "cho", "trong", "của", "cua", "là", "la", "cac", "các", "nhung", "những"}
    filtered = [w for w in words if w.lower() not in stop]
    return " ".join(filtered)

def _perspectives(keywords: str) -> list[dict[str, str]]:
    return [
        {"angle": "Architecture", "query": f"{keywords} architecture"},
        {"angle": "Benchmark", "query": f"{keywords} benchmark"},
        {"angle": "Recent", "query": keywords},
    ]


def analyze_intent(user_query: str) -> dict[str, Any]:
    """Fast local rule engine."""
    q_lower = user_query.strip().lower()
    
    if not q_lower or not re.search(r"\w", q_lower):
        return {
            "decision": "REJECT",
            "domain": "NON_STEM",
            "reason": "empty_query",
            "message": "Vui lòng nhập chủ đề hoặc từ khóa tìm kiếm.",
            "academic_query": "",
            "perspectives": []
        }

    # 1. Broad / Ambiguous Clarification
    if q_lower in {"ai", "trí tuệ nhân tạo", "tri tue nhan tao", "ml"}:
        return {
            "decision": "CLARIFY",
            "domain": "STEM",
            "reason": "too_broad",
            "message": f"Từ khóa '{user_query.strip()}' quá rộng. Bạn muốn tìm về mảng nào: LLM, Computer Vision, hay Reinforcement Learning?",
            "academic_query": "",
            "perspectives": []
        }

    if re.search(r"\bcat\b", q_lower) and "computer" not in q_lower and "adaptive" not in q_lower:
        return {
            "decision": "CLARIFY",
            "domain": "STEM",
            "reason": "ambiguous_acronym",
            "message": "Từ khóa 'CAT' có thể là Computerized Adaptive Testing hoặc các chủ đề khác. Vui lòng nêu rõ ngữ cảnh.",
            "academic_query": "",
            "perspectives": []
        }

    if re.search(r"\bchiến thuật bóng đá\b|\bbóng đá\b", q_lower):
        return {
            "decision": "CLARIFY",
            "domain": "NON_STEM",
            "reason": "sports_tactics",
            "message": "Bạn đang quan tâm đến chiến thuật bóng đá thông thường hay Phân tích Dữ liệu Thể thao (Sports Analytics) bằng AI?",
            "academic_query": "",
            "perspectives": []
        }

    if re.fullmatch(r"^(?:tìm\s+bài\s+hay|bài\s+hay|tài\s+liệu\s+hay|tài\s+liệu\s+tốt)$", q_lower):
        return {
            "decision": "REJECT", "domain": "NON_STEM", "reason": "empty_or_meaningless",
            "message": "Vui lòng nêu chủ đề nghiên cứu cụ thể.", "academic_query": "", "perspectives": [],
        }

    for pattern, domain_name in OUT_OF_DOMAIN_PATTERNS + NON_STEM_PATTERNS:
        if re.search(pattern, q_lower):
            # The scholarly index also covers topics outside arXiv's STEM catalogue.
            keywords = clean_academic_keywords(user_query) or user_query.strip()
            return {
                "decision": "SEARCH", "domain": "NON_STEM", "reason": "non_stem_scholarly",
                "message": f"Tìm tài liệu về {domain_name} trong chỉ mục học thuật.",
                "academic_query": keywords, "perspectives": _perspectives(keywords),
            }

    # 4. Standard STEM Search
    clean_kw = clean_academic_keywords(user_query) or "machine learning"
    return {
        "decision": "SEARCH",
        "domain": "STEM",
        "reason": "rule_fallback",
        "message": "Chủ đề học thuật hợp lệ.",
        "academic_query": clean_kw,
        "perspectives": _perspectives(clean_kw)
    }


def translate_to_academic_english(user_query: str) -> str:
    """Returns standardized English academic keywords extracted by Intent Engine."""
    analysis = analyze_intent(user_query)
    academic = analysis.get("academic_query", "").strip()
    return academic if academic else user_query.strip()


def generate_search_perspectives(user_query: str) -> list[dict[str, str]]:
    """Returns multi-angle search perspectives from Intent Engine."""
    analysis = analyze_intent(user_query)
    return analysis["perspectives"]
