"""
Agentic Supervisor & Multi-Hop Query Planner (L3-L4).
Provides:
  1. Intent, Complexity (LOW / MEDIUM / HIGH), and Entity/Numeric analysis.
  2. Multi-Hop Sub-Query DAG Decomposition (rule-accelerated + optional LLM refinement).
  3. Gap-Driven Query Rewriter when Inner Critic or Outer Fact-Checker requests more context.
  4. Real-time Backend Capability Probing for Hybrid Mode (Cross-Model Routing).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from core.comparison_plan import ComparisonPlan, MAX_COMPARISON_DOCUMENTS
from core.config import settings
from core.schemas import RouterPlanOutput
from core.state import AgentState
from utils.json_repair import extract_and_parse_json
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


class AgenticSupervisorPlanner:
    """
    Unified Supervisor & Query Planner used by all 6 modes.
    Can run deterministically in < 2ms (for CPU Min / Hybrid fast-path) or refine via LLM.
    """

    def plan_document_comparison(self, state: AgentState) -> ComparisonPlan | None:
        paths = tuple(dict.fromkeys(state.get("pdf_paths") or []))
        if (
            state.get("is_comparative_query")
            and 2 <= len(paths) <= MAX_COMPARISON_DOCUMENTS
            and all(Path(path).suffix.lower() == ".pdf" for path in paths)
        ):
            return ComparisonPlan(paths)
        return None

    def plan_query_with_llm(
        self,
        query: str,
        *,
        evidence_gaps: Optional[list[dict[str, Any]]] = None,
        retry_count: int = 0,
        llm_fn: Any | None = None,
    ) -> RouterPlanOutput:
        """
        Refines the deterministic Supervisor plan with an LLM DAG decomposition when an LLM
        callable is supplied (e.g. Gemini or Kaggle Qwen Router), always failing open to
        the deterministic plan_query() on malformed JSON or provider errors.
        """
        base = self.plan_query(query, evidence_gaps=evidence_gaps, retry_count=retry_count)
        if llm_fn is None or not (query or "").strip():
            return base
        prompt = (
            "Decompose this academic research query into a structured search plan. "
            "Return ONLY a JSON object with keys: intent (compare|results|methodology|limitation|summarize|academic_research), "
            "complexity (LOW|MEDIUM|HIGH), sub_queries (1 to 4 concise focused search strings preserving exact entities/numbers), "
            "required_entities (array of up to 8 key terms/models/datasets), is_comparative_query (boolean), "
            "is_numeric_query (boolean), alpha (float 0.0..1.0), top_k (int 4..10).\n"
            f"Query: {query[:600]}"
        )
        try:
            raw = llm_fn(prompt)
            parsed = raw if isinstance(raw, dict) else extract_and_parse_json(str(raw or ""))
            if not isinstance(parsed, dict):
                return base
            raw_subs = parsed.get("sub_queries")
            llm_subs = [
                str(s).strip() for s in raw_subs
                if isinstance(s, str) and 3 <= len(s.strip()) <= 300
            ] if isinstance(raw_subs, list) else []
            merged_subs = list(dict.fromkeys([query.strip(), *llm_subs, *base.sub_queries]))[:4]
            raw_ents = parsed.get("required_entities")
            llm_ents = [
                str(e).strip() for e in raw_ents
                if isinstance(e, str) and 2 <= len(e.strip()) <= 80
            ] if isinstance(raw_ents, list) else []
            merged_ents = list(dict.fromkeys([*base.required_entities, *llm_ents]))[:8]

            intent_val = str(parsed.get("intent") or base.intent).strip().lower()
            valid_intents = {"compare", "results", "methodology", "limitation", "summarize", "academic_research"}
            intent = intent_val if intent_val in valid_intents else base.intent
            comp_val = str(parsed.get("complexity") or base.complexity).strip().upper()
            complexity = comp_val if comp_val in {"LOW", "MEDIUM", "HIGH"} else base.complexity
            is_comp = bool(parsed.get("is_comparative_query")) or base.is_comparative_query or intent == "compare"
            is_num = bool(parsed.get("is_numeric_query")) or base.is_numeric_query or intent == "results"
            try:
                alpha = max(0.0, min(1.0, float(parsed.get("alpha", base.alpha))))
            except (TypeError, ValueError):
                alpha = base.alpha
            try:
                top_k = max(4, min(10, int(parsed.get("top_k", base.top_k))))
            except (TypeError, ValueError):
                top_k = base.top_k

            return RouterPlanOutput(
                intent=intent,
                complexity=complexity,  # type: ignore[arg-type]
                sub_queries=merged_subs,
                required_entities=merged_ents,
                is_numeric_query=is_num,
                is_comparative_query=is_comp,
                alpha=alpha,
                top_k=top_k,
                reasoning=f"LLM-Refined Supervisor: Intent={intent}, Complexity={complexity}, SubQueries={len(merged_subs)}",
            )
        except Exception as exc:
            logger.debug("[SupervisorPlanner] LLM refinement fallback to deterministic plan: %s", exc)
            return base

    def plan_query(
        self,
        query: str,
        evidence_gaps: Optional[list[dict[str, Any]]] = None,
        retry_count: int = 0,
    ) -> RouterPlanOutput:
        clean_q = (query or "").strip()
        q_lower = clean_q.lower()

        # 1. Detect Numeric, Comparative, Methodology, Summary, Limitation intents
        is_numeric = any(
            kw in q_lower
            for kw in [
                "bao nhiêu", "mấy", "số lượng", "tỷ lệ", "phần trăm", "%", "năm nào",
                "ngày nào", "định lượng", "kết quả", "accuracy", "f1", "bleu", "rouge",
                "how many", "how much", "percentage", "score", "metric",
            ]
        )
        comparative_pattern = re.compile(
            r"\b(so sánh|khác nhau|giống nhau|tốt hơn|cao hơn|nhiều hơn|kém hơn|khác biệt|đối chiếu|so với|vs|versus|compare|comparison|difference|between)\b",
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
        elif is_limitation:
            intent = "limitation"
        elif is_methodology:
            intent = "methodology"
        elif is_summary:
            intent = "summarize"
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
                f"Intent={intent}, Complexity={complexity}, "
                f"SubQueries={len(dedup_subs)}, Entities={required_entities[:4]}, "
                f"alpha={alpha}, top_k={top_k}"
            ),
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
