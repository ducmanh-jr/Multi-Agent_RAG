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

import json
import re
import unicodedata
from typing import Any
from utils.llm_client import llm_client
from utils.logger import logger


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
]


def _rule_based_fallback(user_query: str) -> dict[str, Any]:
    """Fast local fallback if LLM is unreachable."""
    q_lower = user_query.strip().lower()
    
    if not q_lower or q_lower in {"ai", "trí tuệ nhân tạo", "tri tue nhan tao"}:
        return {
            "decision": "CLARIFY" if q_lower else "REJECT",
            "domain": "STEM" if q_lower else "NON_STEM",
            "reason": "too_broad" if q_lower else "empty",
            "message": "Từ khóa 'AI' quá rộng. Bạn muốn tìm về mảng nào: LLM, Computer Vision, hay Reinforcement Learning?",
            "academic_query": "",
            "perspectives": []
        }

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

    words = re.findall(r"[a-zA-Z0-9_-]+", user_query)
    clean_kw = " ".join(words[:4]) if words else "machine learning"
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
    Executes SOTA 3-way intent classification using LLM Intent Engine.
    Cached persistently in SQLite.
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

    # First check obvious out-of-domain patterns to save LLM calls
    q_lower = clean_q.lower()
    if not any(stem in q_lower for stem in ["ai", "machine learning", "deep learning", "học máy", "học sâu", "dữ liệu", "algorithm"]):
        for pat, domain_name in OUT_OF_DOMAIN_PATTERNS:
            if re.search(pat, q_lower):
                return {
                    "decision": "REJECT",
                    "domain": "NON_STEM",
                    "reason": "out_of_domain",
                    "message": f"Chủ đề về {domain_name} nằm ngoài phạm vi lưu trữ của ArXiv (Khoa học máy tính, AI, Toán học, Vật lý...). Bạn vui lòng tải trực tiếp tài liệu PDF vào Kho Tài Liệu bên trái.",
                    "academic_query": "",
                    "perspectives": []
                }

    prompt = f"Analyze this user query:\n\"{clean_q}\""
    try:
        raw = llm_client.generate(
            prompt=prompt, 
            system_instruction=SYSTEM_PROMPT, 
            json_mode=True, 
            temperature=0.1
        )
        raw_clean = raw.strip()
        if raw_clean.startswith("```json"):
            raw_clean = raw_clean[7:]
        if raw_clean.endswith("```"):
            raw_clean = raw_clean[:-3]
        parsed = json.loads(raw_clean.strip())
        
        decision = parsed.get("decision", "SEARCH").upper()
        if decision not in ("SEARCH", "CLARIFY", "REJECT"):
            decision = "SEARCH"
            
        parsed["decision"] = decision
        logger.info(f"[Scout Intent Engine] Query: '{clean_q}' -> Decision: {decision} | Domain: {parsed.get('domain')}")
        return parsed
    except Exception as e:
        logger.warning(f"[Scout Intent Engine] LLM call failed ({e}). Falling back to rule engine.")
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
