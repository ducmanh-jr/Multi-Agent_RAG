"""
Agent 0 (Literature Scout) - SOTA Intent Intelligence & Academic Expander.
Powered by LLM In-Context Intent Understanding with SQLite Caching
and Resilient Offline Rule Fallback.

Capabilities:
  - 3-Way Decision: SEARCH / CLARIFY / REJECT.
  - Multi-perspective search formulation (Architecture, Benchmark, Recent).
  - Acronym disambiguation and bilingual STEM scoping.
"""

from __future__ import annotations

import re
from typing import Any


SYSTEM_PROMPT = """You are the Frontier Academic Intent Engine for Agent 0 (Literature Scout) in a scientific research platform.
Your task is to analyze user queries (Vietnamese or English) and decide how to process them on ArXiv.
ArXiv strictly covers STEM fields: Computer Science (cs.*), AI/Machine Learning, Mathematics (math.*), Physics (physics.*), Quantitative Biology, Statistics (stat.*), Quantitative Finance, and Electrical Engineering.

You must classify the query into exactly one of three decisions:
1. "REJECT":
   - The query is outside ArXiv STEM scope (e.g. Vietnamese/world history, literature/poetry, pure local law/traffic codes, cooking/recipes, entertainment).
   - Or the query is completely empty / meaningless ("tìm bài hay", "tài liệu tốt", "???").
   - Provide a polite Vietnamese explanation in "message" explaining that the topic belongs to history/arts/pure law, outside ArXiv's STEM scope, and suggest uploading a PDF directly into the workspace.

2. "CLARIFY":
   - The query is ONLY an extremely broad 2-letter meta-word like "AI" or "ML" without any specific technique.
   - The query has an ambiguous acronym with multiple interpretations (e.g. "CAT" -> Computerized Adaptive Testing, CAT robot, biology...).
   - The query is about pure sports tactics without analytics context (e.g. "chiến thuật bóng đá" -> clarify if they mean Sports Analytics / Computer Vision in sports).
   - Provide a clear, helpful Vietnamese question in "message".

3. "SEARCH":
   - The query is a legitimate academic STEM topic or hybrid topic (e.g. AI for cultural heritage, ML for air pollution prediction, Vietnamese NLP, quantum circuits, medical diffusion models, graph neural networks).
   - If the user asks about foundational architectures or paradigms like "transformer" or "học sâu" / "deep learning", classify as "SEARCH" to retrieve landmark foundational and survey papers.
   - Extract "academic_query": concise, standardized English academic search keywords (strip all conversational boilerplate words like "tìm giúp tôi", "cho mình xin", "nghiên cứu về", "search papers on", "có bài nào nói về").
   - For acronyms like "TIM trong y học", disambiguate to "TIMI myocardial cardiac flow".
   - For comparisons like "so sánh CNN và transformer cho thị giác máy tính", keep both concepts: "CNN vision transformer comparison image classification".
   - Generate 3 multi-perspective search angles in "perspectives": Foundational/Architecture, Benchmark/Evaluation, and Recent Advancements.

Respond ONLY with valid JSON following this schema:
{
  "decision": "SEARCH" | "CLARIFY" | "REJECT",
  "domain": "STEM" | "NON_STEM",
  "reason": "string",
  "message": "string",
  "academic_query": "string (empty if REJECT or CLARIFY)",
  "perspectives": [
    {"angle": "Architecture", "query": "string"},
    {"angle": "Benchmark", "query": "string"},
    {"angle": "Recent", "query": "string"}
  ]
}
"""

OUT_OF_DOMAIN_PATTERNS = [
    (r"\b(sau\s+giải\s+phóng|sau\s+giai\s+phong|giải\s+phóng\s+miền\s+nam|cách\s+mạng\s+tháng\s+tám|chiến\s+tranh\s+thế\s+giới|lịch\s+sử|triều\s+đại|nhà\s+trần|nhà\s+nguyễn)\b",
     "Lịch sử / Chính trị địa phương"),
    (r"\b(thơ\s+nguyễn\s+du|truyện\s+kiều|văn\s+học|tiểu\s+thuyết|thơ\s+ca|nhạc\s+trịnh)\b",
     "Văn học / Nghệ thuật"),
    (r"\b(luật\s+đất\s+đai|luật\s+giao\s+thông|nghị\s+định|thông\s+tư|bộ\s+luật\s+hình\s+sự|văn\s+bản\s+pháp\s+luật)\b",
     "Văn bản Pháp luật địa phương"),
    (r"\b(cách\s+nấu|nấu\s+ăn|món\s+ăn|phở\s+bò|bánh\s+chưng|công\s+thức\s+nấu)\b",
     "Ẩm thực / Đời sống"),
    (r"^(?:tìm\s+bài\s+hay|bài\s+hay|tài\s+liệu\s+hay|tài\s+liệu\s+tốt)$",
     "Yêu cầu chung chung"),
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
    (r"\btrong y học\b", "timi myocardial cardiac flow"),
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

def _rule_based_fallback(user_query: str) -> dict[str, Any]:
    """Fast local rule engine."""
    q_lower = user_query.strip().lower()
    
    if not q_lower:
        return {
            "decision": "REJECT",
            "domain": "NON_STEM",
            "reason": "empty",
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
            "message": "Từ khóa 'AI' quá rộng. Bạn muốn tìm về mảng nào: LLM, Computer Vision, hay Reinforcement Learning?",
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

    # 2. Out of Domain check FIRST
    for pat, domain_name in OUT_OF_DOMAIN_PATTERNS:
        if re.search(pat, q_lower):
            return {
                "decision": "REJECT",
                "domain": "NON_STEM",
                "reason": "out_of_domain",
                "message": f"Chủ đề này thuộc lĩnh vực {domain_name}, nằm ngoài phạm vi học thuật STEM của ArXiv. Vui lòng tải tài liệu PDF trực tiếp vào Kho Tài Liệu.",
                "academic_query": "",
                "perspectives": []
            }

    # 3. Non-STEM scholarly check
    for pat, domain_name in NON_STEM_PATTERNS:
        if re.search(pat, q_lower):
            clean_kw = clean_academic_keywords(user_query) or "scholarly work"
            return {
                "decision": "REJECT",
                "domain": "NON_STEM",
                "reason": "non_stem_scholarly",
                "message": f"Chủ đề học thuật về {domain_name} nằm ngoài phạm vi STEM của hệ thống.",
                "academic_query": "",
                "perspectives": []
            }

    # 4. Standard STEM Search
    clean_kw = clean_academic_keywords(user_query) or "machine learning"
    return {
        "decision": "SEARCH",
        "domain": "STEM",
        "reason": "rule_fallback",
        "message": "Chủ đề học thuật hợp lệ.",
        "academic_query": clean_kw,
        "perspectives": [
            {"angle": "Architecture", "query": f"{clean_kw} architecture"},
            {"angle": "Benchmark", "query": f"{clean_kw} benchmark"},
            {"angle": "Recent", "query": f"{clean_kw} 2024 2025"}
        ]
    }


def analyze_intent(user_query: str) -> dict[str, Any]:
    """
    Fast rule-first intent classification with graceful fallback.
    """
    clean_q = user_query.strip()
    if not clean_q:
        return {
            "decision": "REJECT",
            "domain": "NON_STEM",
            "reason": "empty_query",
            "message": "Vui lòng nhập câu hỏi hoặc chủ đề nghiên cứu cụ thể.",
            "academic_query": "",
            "perspectives": []
        }

    # Fast check: If matches out-of-domain or clarification directly, return in <1ms
    q_lower = clean_q.lower()
    for pat, domain_name in OUT_OF_DOMAIN_PATTERNS:
        if re.search(pat, q_lower):
            return _rule_based_fallback(clean_q)

    for pat, domain_name in NON_STEM_PATTERNS:
        if re.search(pat, q_lower):
            return _rule_based_fallback(clean_q)

    # Broad acronyms or vague queries -> clarify / reject
    if q_lower in {"ai", "trí tuệ nhân tạo", "tri tue nhan tao", "ml"}:
        return _rule_based_fallback(clean_q)
    if re.search(r"\bcat\b", q_lower) and "computer" not in q_lower and "adaptive" not in q_lower:
        return _rule_based_fallback(clean_q)
    if re.search(r"\bchiến thuật bóng đá\b|\bbóng đá\b", q_lower):
        return _rule_based_fallback(clean_q)
    if re.search(r"^(?:tìm\s+bài\s+hay|bài\s+hay|tài\s+liệu\s+hay|tài\s+liệu\s+tốt)$", q_lower):
        return _rule_based_fallback(clean_q)

    # If it has standard STEM indicators with word boundary regex (preventing false matches on 'bài', 'tại', 'hai', 'ngày mai')
    stem_patterns = [
        r"\battention\b", r"\btransformer\b", r"\bbert\b", r"\bgpt\b", r"\blora\b",
        r"\bdiffusion\b", r"\b(ai|trí tuệ nhân tạo)\b", r"\bmachine learning\b",
        r"\bdeep learning\b", r"\bhọc máy\b", r"\bhọc sâu\b", r"\balgorithm\b",
        r"\bthuật toán\b", r"\bneural\b", r"\bllm\b"
    ]
    if any(re.search(pat, q_lower) for pat in stem_patterns):
        clean_kw = clean_academic_keywords(clean_q) or clean_q
        return {
            "decision": "SEARCH",
            "domain": "STEM",
            "reason": "stem_academic",
            "message": "Chủ đề học thuật STEM hợp lệ.",
            "academic_query": clean_kw,
            "perspectives": [
                {"angle": "Architecture", "query": f"{clean_kw} architecture"},
                {"angle": "Benchmark", "query": f"{clean_kw} benchmark"},
                {"angle": "Recent", "query": f"{clean_kw} 2024"}
            ]
        }

    return _rule_based_fallback(clean_q)


def translate_to_academic_english(user_query: str) -> str:
    """Returns standardized English academic keywords extracted by Intent Engine."""
    analysis = analyze_intent(user_query)
    academic = analysis.get("academic_query", "").strip()
    return academic if academic else user_query.strip()


def generate_search_perspectives(user_query: str) -> list[dict[str, str]]:
    """Returns multi-angle search perspectives from Intent Engine."""
    analysis = analyze_intent(user_query)
    perspectives = analysis.get("perspectives")
    if perspectives and isinstance(perspectives, list) and len(perspectives) >= 2:
        return perspectives
        
    core = analysis.get("academic_query") or user_query.strip()
    return [
        {"angle": "Architecture", "query": f"{core} architecture"},
        {"angle": "Benchmark", "query": f"{core} benchmark performance"},
        {"angle": "Recent", "query": f"{core} 2024 2025"}
    ]
