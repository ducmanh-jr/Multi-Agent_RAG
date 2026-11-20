"""
Agentic Supervisor, General-Purpose Intent Router & Multi-Hop Query Planner.
Provides:
  1. General-Purpose 9-Intent Classification (AGENT.md Sections 7, 8, 13):
       DOCUMENT_QA, GENERAL_QA, WEB_RESEARCH, MIXED_QA, TASK_ASSISTED_RESPONSE,
       TASK_EXECUTION, CLARIFICATION_REQUIRED, SAFETY_RESTRICTED, UNSUPPORTED_CAPABILITY.
  2. Legacy RAG sub-intent, Complexity (LOW / MEDIUM / HIGH), and Entity/Numeric analysis.
  3. Multi-Hop Sub-Query DAG Decomposition (rule-accelerated).
  4. Gap-Driven Query Rewriter when Inner Critic or Outer Fact-Checker requests more context.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from core.config import settings
from core.schemas import (
    InputStatus,
    PrimaryIntentType,
    QueryResolutionStatus,
    RouterPlanOutput,
)
from utils.logger import logger


_STOP_WORDS_VI_EN = {
    "và", "của", "cho", "với", "như", "thế", "nào", "được", "khi", "là", "gì",
    "ở", "đâu", "các", "những", "này", "đó", "ra", "vào", "từ", "về", "trong",
    "để", "có", "hãy", "trình", "bày", "cho", "biết", "tài", "liệu", "bài", "báo",
    "liệu", "theo", "giải", "so", "sánh", "phân", "tích", "tại", "sao", "vì", "làm",
    "điểm", "nêu", "mục", "tiêu", "nội", "dung", "tổng", "quan", "đánh", "giá",
    "thông", "tin", "chi", "tiết", "mô", "hình", "phương", "pháp", "kết", "quả",
    "the", "and", "for", "with", "what", "how", "why", "from", "this", "that",
    "are", "was", "were", "have", "has", "had", "into", "about", "paper", "document",
}


_SAFETY_PATTERNS = [
    r"\b(chế\s+tạo\s+bom|làm\s+bom|chất\s+nổ|vũ\s+khí\s+sinh\s+học|tự\s+tử)\b",
    r"\b(hack\s+mật\s+khẩu|ddos|ransomware|keylogger|exploit\s+zero-day)\b",
    r"\b(tiết\s+lộ|in\s+ra|show\s+me|dump|reveal)\b.{0,40}\b(api[_\s]?key|gemini_api_key|secret|password|token|\.env)\b",
    r"\b(ignore\s+(?:all\s+)?previous\s+instructions|bỏ\s+qua\s+mọi\s+quy\s+tắc)\b",
]

_AMBIGUOUS_EXACT = {
    "ai",
    "ml",
    "cat",
    "?",
    "???",
    "trí tuệ nhân tạo",
    "tri tue nhan tao",
    "kết quả thế nào",
    "kết quả thế nào?",
    "ket qua the nao",
    "cấu hình ra sao",
    "cấu hình ra sao?",
    "ưu điểm là gì",
    "ưu điểm là gì?",
    "tốt không",
    "tốt không?",
}

_TASK_EXECUTION_PATTERNS = [
    r"\b(chạy\s+(?:code|chương\s+trình|phần\s+mềm|script|lệnh|ứng\s+dụng)|thực\s+thi\s+(?:lệnh|script|tool|công\s+cụ|chương\s+trình)|run\s+(?:python|code|script|command|tool|program)|execute\s+(?:code|command|tool|script))\b",
    r"\b(xóa\s+file|tạo\s+file|chỉnh\s+sửa\s+file|gửi\s+email|triển\s+khai\s+server|deploy)\b",
    r"\b(tính\s+bằng\s+công\s+cụ|dùng\s+tool|gọi\s+tool|call\s+tool)\b",
]

_TASK_ASSISTED_PATTERNS = [
    r"\b(viết\s+(?:code|hàm|chương\s+trình|script|sql|regex|class|api)|lập\s+trình|code\s+python|write\s+(?:a\s+)?(?:python|code|script|function|sql))\b",
    r"\b(giải\s+thích\s+(?:lệnh|code|hàm|cú\s+pháp|thuật\s+toán|chương\s+trình|script)|explain\s+(?:command|code|syntax))\b",
    r"\b(hướng\s+dẫn\s+(?:cấu\s+hình|cài\s+đặt|triển\s+khai)|how\s+to\s+(?:configure|install|implement|build))\b",
    r"\b(dịch\s+(?:đoạn|câu|văn\s+bản|sang\s+tiếng)|translate\s+(?:this|to|into))\b",
    r"\b(đề\s+xuất\s+(?:phương\s+án|kiến\s+trúc|giải\s+pháp)|refactor\s+đoạn\s+code|debug\s+đoạn\s+code)\b",
]

_WEB_RESEARCH_PATTERNS = [
    r"\b(tìm(?:\s+kiếm)?\s+(?:các\s+)?(?:bài\s+báo|tài\s+liệu|nghiên\s+cứu|công\s+trình)|search\s+(?:for\s+)?papers?|find\s+papers?)\b",
    r"\b(nghiên\s+cứu\s+(?:mới(?:\s+nhất)?|mở\s+rộng|gần\s+đây|hiện\s+đại)|bài\s+báo\s+(?:mới(?:\s+nhất)?|gần\s+đây)|recent\s+(?:papers?|research)|latest\s+(?:papers?|research|release|version))\b",
    r"\b(phiên\s+bản\s+mới\s+nhất|tin\s+tức\s+mới\s+nhất|cập\s+nhật\s+mới\s+nhất|arxiv|openalex)\b",
]

_EXPLICIT_DOCUMENT_PATTERNS = [
    r"\b(tài\s+liệu\s+(?:này|trên|đã\s+chọn|đính\s+kèm)|trong\s+tài\s+liệu|theo\s+tài\s+liệu|dựa\s+vào\s+tài\s+liệu)\b",
    r"\b(pdf(?:\s+này)?|file\s+pdf|trong\s+pdf|tệp\s+này|file\s+này)\b",
    r"\b(bài\s+báo\s+này|trong\s+bài\s+báo|theo\s+bài\s+báo|this\s+paper|the\s+paper|the\s+document|this\s+document|in\s+the\s+pdf)\b",
    r"\b(trang\s+\d+|page\s+\d+|bảng\s+\d+|table\s+\d+|hình\s+\d+|figure\s+\d+)\b",
    r"^/(?:tom-tat|tomtat|so-sanh|dong-gop)\b",
    r"\b(tóm\s+tắt\s+(?:pdf|tài\s+liệu|bài\s+báo)|summarize\s+(?:this\s+)?(?:pdf|paper|document))\b",
]

_GENERAL_QA_PATTERNS = [
    r"^(?:[\wÀ-ỹ\s]{2,40})\s+là\s+gì\s*\??$",
    r"^(?:what\s+is|define|explain|giải\s+thích|khái\s+niệm|định\s+nghĩa)\b",
    r"^(?:xin\s+chào|chào\s+bạn|hello|hi|hey)\b",
]


class AgenticSupervisorPlanner:
    """
    Unified Supervisor, General-Purpose Intent Router & Query Planner used by all 6 official modes.
    Runs deterministically in < 2ms with zero external API calls.
    """

    def classify_primary_intent(
        self,
        query: str,
        has_pdf: bool = False,
        auto_scout: bool = False,
        requested_tool: Optional[str] = None,
    ) -> tuple[PrimaryIntentType, str, QueryResolutionStatus, InputStatus, bool, bool]:
        """
        Classifies user input into the 9 General-Purpose Assistant intents (AGENT.md Sections 7 & 8).
        Returns: (primary_intent, sub_intent, query_resolution, input_status, needs_scout, needs_retrieval).
        """
        clean_q = (query or "").strip()
        q_lower = clean_q.lower()

        # 0. Pre-check: Empty or ultra-ambiguous query -> CLARIFICATION_REQUIRED
        if not clean_q or q_lower in _AMBIGUOUS_EXACT:
            return ("CLARIFICATION_REQUIRED", "Ambiguous", "CLARIFICATION_REQUIRED", "OK", False, False)

        # 1. Pre-check: Safety & security boundary (P0)
        if any(re.search(pat, q_lower, re.IGNORECASE) for pat in _SAFETY_PATTERNS):
            return ("SAFETY_RESTRICTED", "Safety", "CLEAR", "OK", False, False)

        has_doc_ref = any(re.search(pat, q_lower, re.IGNORECASE) for pat in _EXPLICIT_DOCUMENT_PATTERNS)
        has_web_ref = any(re.search(pat, q_lower, re.IGNORECASE) for pat in _WEB_RESEARCH_PATTERNS)
        is_task_exec = bool(requested_tool) or any(
            re.search(pat, q_lower, re.IGNORECASE) for pat in _TASK_EXECUTION_PATTERNS
        )
        is_task_assisted = any(re.search(pat, q_lower, re.IGNORECASE) for pat in _TASK_ASSISTED_PATTERNS)
        is_general_pattern = any(re.search(pat, q_lower, re.IGNORECASE) for pat in _GENERAL_QA_PATTERNS)

        # 2. TASK_EXECUTION: Explicit tool/action execution request
        if is_task_exec and not has_doc_ref:
            tool_hint = (requested_tool or "").strip().lower()
            return ("TASK_EXECUTION", tool_hint or "Tool_Operation", "CLEAR", "OK", False, False)

        # 3. MIXED_QA: Combines document content AND external/recent research
        if (has_doc_ref or has_pdf) and has_web_ref:
            if not has_pdf and not auto_scout:
                return ("MIXED_QA", "Document_And_Research", "CLEAR", "NO_DOCUMENT_SELECTED", False, False)
            return ("MIXED_QA", "Document_And_Research", "CLEAR", "OK", bool(auto_scout), True)

        # 4. WEB_RESEARCH: External search / finding papers / latest versions (Scout only if enabled)
        if has_web_ref and not has_doc_ref:
            return ("WEB_RESEARCH", "Literature_Discovery", "CLEAR", "OK", bool(auto_scout), bool(auto_scout))

        # 5. TASK_ASSISTED_RESPONSE: Coding / config guidance / translation / drafting (no PDF required)
        if is_task_assisted and not has_doc_ref:
            sub = "Translation" if re.search(r"\b(dịch|translate)\b", q_lower) else "Coding_Or_Guidance"
            return ("TASK_ASSISTED_RESPONSE", sub, "CLEAR", "OK", False, False)

        # 6. DOCUMENT_QA: Explicit document reference
        if has_doc_ref:
            if has_pdf:
                return ("DOCUMENT_QA", "Document_Analysis", "CLEAR", "OK", False, True)
            if auto_scout:
                return ("DOCUMENT_QA", "Auto_Scout_Document", "CLEAR", "OK", True, True)
            return ("DOCUMENT_QA", "Document_Analysis", "CLEAR", "NO_DOCUMENT_SELECTED", False, False)

        # 7. Pure general-knowledge / explanation query (even if a PDF happens to be selected)
        if is_general_pattern:
            return ("GENERAL_QA", "Knowledge", "CLEAR", "OK", False, False)

        # 8. If user has selected a PDF and asks a non-general domain question -> DOCUMENT_QA
        if has_pdf:
            return ("DOCUMENT_QA", "Document_Analysis", "CLEAR", "OK", False, True)

        # 9. Default without PDF: GENERAL_QA (RAG is NOT the mandatory default!)
        return ("GENERAL_QA", "Knowledge", "CLEAR", "OK", False, False)

    def plan_query(
        self,
        query: str,
        evidence_gaps: Optional[list[dict[str, Any]]] = None,
        retry_count: int = 0,
        pdf_path: Optional[str] = None,
        pdf_paths: Optional[list[str]] = None,
        auto_scout: bool = False,
        requested_tool: Optional[str] = None,
    ) -> RouterPlanOutput:
        clean_q = (query or "").strip()
        q_lower = clean_q.lower()
        has_pdf = bool(pdf_path or (pdf_paths and any(pdf_paths)))

        primary_intent, sub_intent, query_resolution, input_status, needs_scout, needs_retrieval = (
            self.classify_primary_intent(
                query=clean_q,
                has_pdf=has_pdf,
                auto_scout=auto_scout,
                requested_tool=requested_tool,
            )
        )

        # 1. Detect Legacy Numeric, Comparative, Methodology, Summary, Limitation sub-intents
        is_numeric = any(
            kw in q_lower
            for kw in [
                "bao nhiêu", "mấy", "số lượng", "tỷ lệ", "phần trăm", "%", "năm nào",
                "ngày nào", "định lượng", "kết quả", "accuracy", "f1", "bleu", "rouge",
                "how many", "how much", "percentage", "score", "metric",
            ]
        )
        comparative_pattern = re.compile(
            r"\b(so sánh|khác nhau|giống nhau|ưu điểm|nhược điểm|tốt hơn|cao hơn|nhiều hơn|kém hơn|khác biệt|đối chiếu|so với|vs|versus|compare|comparison|difference|between)\b",
            re.IGNORECASE,
        )
        is_comparative = bool(comparative_pattern.search(q_lower))
        is_summary = any(
            kw in q_lower
            for kw in [
                "tóm tắt", "tổng quan", "nội dung chính", "đóng góp chính", "ý chính",
                "summary", "summarize", "overview", "main contribution",
            ]
        )
        is_methodology = any(
            kw in q_lower
            for kw in [
                "phương pháp", "kiến trúc", "quy trình", "thuật toán", "huấn luyện",
                "method", "methodology", "architecture", "pipeline", "algorithm",
            ]
        )
        is_limitation = any(
            kw in q_lower
            for kw in [
                "hạn chế", "nhược điểm", "thách thức", "rủi ro", "tương lai",
                "limitation", "weakness", "future work", "challenge",
            ]
        )

        if is_comparative:
            intent = "compare"
        elif is_numeric:
            intent = "results"
        elif is_methodology:
            intent = "methodology"
        elif is_limitation:
            intent = "limitation"
        elif is_summary:
            intent = "summarize"
        elif primary_intent in ("TASK_ASSISTED_RESPONSE", "TASK_EXECUTION"):
            intent = "code_request"
        elif primary_intent == "GENERAL_QA":
            intent = "general_question"
        else:
            intent = "academic_research"

        # 2. Extract Required Entities, Acronyms, Years, Numbers
        years_and_nums = re.findall(r"\b(?:19\d\d|20\d\d|\d+(?:[.,]\d+)?%?)\b", clean_q)
        capitalized = re.findall(r"\b[A-ZĐ][A-Za-zÀ-ỹ0-9_-]{1,}\b", clean_q)
        quoted = re.findall(r"[\"“”']([^\"“”']{2,40})[\"“”']", clean_q)
        required_entities: list[str] = []
        for item in years_and_nums + capitalized + quoted:
            it_clean = item.strip()
            if it_clean and it_clean.lower() not in _STOP_WORDS_VI_EN and it_clean not in required_entities:
                required_entities.append(it_clean)

        # 3. Decompose Multi-Hop Query into Sub-Queries
        sub_queries: list[str] = [clean_q]

        # If this is an outer-loop retry with evidence gaps, prioritize gap queries!
        if retry_count > 0 and evidence_gaps:
            gap_queries: list[str] = []
            for gap in evidence_gaps:
                if isinstance(gap, dict):
                    sq = str(gap.get("suggested_query") or gap.get("missing_aspect") or "").strip()
                else:
                    sq = str(gap).strip()
                if sq and sq not in gap_queries:
                    gap_queries.append(sq)
            if gap_queries:
                sub_queries = gap_queries[:3] + [clean_q]

        # Split compound questions ("A và B", "Compare X and Y", multi-clause questions)
        if len(sub_queries) == 1:
            clauses = re.split(
                r"(?:\?|\bđồng thời\b|\bvà cho biết\b|\bso với\b|\bcũng như\b|;)",
                clean_q,
                flags=re.IGNORECASE,
            )
            clauses = [c.strip() for c in clauses if len(c.strip().split()) >= 3]
            if len(clauses) >= 2:
                for c in clauses[:3]:
                    if c not in sub_queries:
                        sub_queries.append(c)

            if is_comparative and len(required_entities) >= 2:
                for ent in required_entities[:2]:
                    ent_q = f"{ent} {clean_q}"
                    if ent_q not in sub_queries:
                        sub_queries.append(ent_q)

            if is_summary and len(sub_queries) == 1:
                sub_queries.extend(
                    [
                        f"{clean_q} mục tiêu phương pháp chính",
                        f"{clean_q} kết quả thực nghiệm đóng góp",
                    ]
                )

        # Deduplicate and cap at 4 sub-queries
        dedup_subs: list[str] = []
        for sq in sub_queries:
            if sq and sq not in dedup_subs:
                dedup_subs.append(sq)
        dedup_subs = dedup_subs[:4]

        # 4. Determine Complexity & Optimal Hybrid alpha / top_k
        word_count = len(clean_q.split())
        if is_comparative or len(dedup_subs) >= 3 or word_count > 24 or retry_count > 0:
            complexity = "HIGH"
        elif is_numeric or is_methodology or is_limitation or word_count > 10:
            complexity = "MEDIUM"
        else:
            complexity = "LOW"

        # alpha = 1.0 favors Dense semantic search; alpha = 0.2 favors exact BM25 lexical match
        if is_numeric or len(required_entities) >= 2:
            alpha = 0.30
            top_k = 7
        elif is_summary:
            alpha = 0.65
            top_k = 8
        elif is_comparative:
            alpha = 0.50
            top_k = 8
        else:
            alpha = 0.50
            top_k = 6

        if retry_count > 0:
            top_k = min(10, top_k + 2 * retry_count)

        return RouterPlanOutput(
            intent=intent,
            complexity=complexity,  # type: ignore[arg-type]
            sub_queries=dedup_subs,
            required_entities=required_entities[:8],
            is_numeric_query=is_numeric,
            is_comparative_query=is_comparative,
            alpha=alpha,
            top_k=top_k,
            reasoning=(
                f"PrimaryIntent={primary_intent}, Intent={intent}, Complexity={complexity}, "
                f"Resolution={query_resolution}, InputStatus={input_status}, "
                f"Scout={needs_scout}, Retrieval={needs_retrieval}"
            ),
            primary_intent=primary_intent,
            sub_intent=sub_intent,
            query_resolution=query_resolution,
            input_status=input_status,
            needs_scout=needs_scout,
            needs_retrieval=needs_retrieval,
        )

    def build_gap_requeries(
        self,
        original_query: str,
        missing_points: list[str],
        unsupported_claims: list[str],
    ) -> list[str]:
        """Generates targeted follow-up search queries from Critic/Fact-Checker feedback."""
        follow_ups: list[str] = [original_query]
        for mp in missing_points[:3]:
            # Extract salient keywords rather than concatenating whole long strings
            tokens = [
                t
                for t in re.findall(r"\b[\wÀ-ỹ0-9.,%_-]+\b", str(mp))
                if t.lower() not in _STOP_WORDS_VI_EN and len(t) > 2
            ]
            if tokens:
                follow_ups.append(" ".join(tokens[:8]))
        for uc in unsupported_claims[:2]:
            # Extract numbers or key nouns from unsupported claim to verify against corpus
            tokens = [
                t
                for t in re.findall(r"\b[\wÀ-ỹ0-9.,%_-]+\b", str(uc))
                if t.lower() not in _STOP_WORDS_VI_EN and len(t) > 2
            ]
            if tokens:
                follow_ups.append(" ".join(tokens[:8]))
        return list(dict.fromkeys(follow_ups))[:4]


supervisor_planner = AgenticSupervisorPlanner()
