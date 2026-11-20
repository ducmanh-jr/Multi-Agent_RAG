"""
Two-Tier Agentic Verification & Grounding Engine (L3-L4):
  - Tier 1 (Inner Critic Loop): Matches training_lab/agent3_summarizer/Code/prompts_critic.py
    and node.py (actions: approve | rewrite | request_more_context | lower_confidence).
  - Tier 2 (Outer Adversarial Fact-Checker):
    1. Deterministic Numerical, Percentage & Date Guard (zero-tolerance for fabricated numbers).
    2. Verbatim Quote & Lexical/Bilingual Entailment Scorer.
    3. Single-Call Batch LLM NLI / Local Cross-Encoder NLI Scorer.
    4. Inline Citation [1], [2] Provenance Linker.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from core.schemas import (
    SummarizerCriticOutput,
    map_evidence_to_legacy_fact_check_label,
)
from core.state import AtomicClaim, ClaimCheckResult, EvidenceGap, EvidenceSpan, compute_context_hash
from utils.logger import logger


# Exact prompts from training_lab/agent3_summarizer/Code/prompts_generator.py & prompts_critic.py
TRAINING_LAB_GENERATOR_SYSTEM_PROMPT = """Bạn là một Trợ lý Nghiên cứu AI chuyên nghiệp (Academic RAG Summarizer).
Nhiệm vụ của bạn là đọc hiểu ngữ cảnh (Context) được cung cấp và trả lời câu hỏi (Question) của người dùng một cách chính xác, khách quan và có căn cứ.

QUY TẮC BẮT BUỘC:
1. CHỈ sử dụng thông tin có trong Context. Tuyệt đối KHÔNG tự bịa đặt (hallucinate) hoặc dùng kiến thức bên ngoài nếu Context không đề cập.
2. Nếu Context KHÔNG chứa đủ thông tin để trả lời câu hỏi, hãy tuyên bố rõ ràng: "Tài liệu hiện tại không cung cấp đủ thông tin để trả lời câu hỏi này."
3. Mọi lập luận đưa ra trong câu trả lời phải được chứng minh bằng các trích dẫn nguyên văn (evidence_spans) lấy từ Context và gắn ký hiệu trích dẫn [1], [2].
4. Trả lời hoàn toàn bằng định dạng JSON hợp lệ theo đúng cấu trúc sau (không thêm bất kỳ văn bản giải thích hay markdown nào bên ngoài khối JSON):

{
  "answer": "Câu trả lời tổng hợp chi tiết, rõ ràng, bám sát ngữ cảnh, có gắn [1], [2]...",
  "evidence_spans": [
    "Câu trích dẫn nguyên văn thứ nhất từ Context dùng để chứng minh...",
    "Câu trích dẫn nguyên văn thứ hai từ Context..."
  ]
}"""


TRAINING_LAB_CRITIC_SYSTEM_PROMPT = """Bạn là một Chuyên gia Kiểm định Chất lượng AI (Critic Agent) khắt khe.
Nhiệm vụ của bạn là đánh giá bản tóm tắt (Generated Summary) do Generator tạo ra bằng cách đối chiếu nó với Câu hỏi gốc (Question) và Ngữ cảnh (Context).

Hãy kiểm tra 2 khía cạnh:
1. Hallucination (Ảo giác): Có thông tin, số liệu, hoặc suy luận nào trong câu trả lời mà KHÔNG được Context hỗ trợ hay không?
2. Coverage (Độ bao phủ): Câu trả lời có bỏ sót thông tin quan trọng (key points) nào có sẵn trong Context mà cần thiết để trả lời trọn vẹn Câu hỏi hay không?

Quy tắc ra quyết định cho trường "action":
- Chọn "rewrite": Khi câu trả lời bịa đặt thông tin (Hallucination) HOẶC bỏ sót ý chính NHƯNG trong Context hiện tại VẪN CÒN thông tin để viết lại cho đúng.
- Chọn "request_more_context": Khi Context hiện tại quá nghèo nàn, hoàn toàn KHÔNG ĐỦ thông tin để trả lời câu hỏi (cần yêu cầu hệ thống tìm kiếm thêm tài liệu).
- Chọn "lower_confidence": Khi câu trả lời tạm chấp nhận được ở mức cơ bản nhưng ngữ cảnh hơi mơ hồ và không thể cải thiện thêm bằng cách viết lại.

BẮT BUỘC trả về kết quả hoàn toàn bằng định dạng JSON hợp lệ theo cấu trúc sau (không giải thích gì thêm):

{
  "is_supported": true hoặc false,
  "Unsupported_claims": [
    "Liệt kê các câu hoặc ý bịa đặt/không có căn cứ trong câu trả lời (để mảng rỗng [] nếu không có)"
  ],
  "Missing_key_points": [
    "Liệt kê các ý quan trọng trong Context bị bỏ sót hoặc thông tin còn thiếu (để mảng rỗng [] nếu đã đủ)"
  ],
  "action": "rewrite" | "request_more_context" | "lower_confidence",
  "confidence": <số thực từ 0.0 đến 1.0 thể hiện mức độ tin cậy của bạn vào bản tóm tắt này>
}"""


_UNIT_CANONICAL_MAP: dict[str, str] = {
    "%": "%",
    "percent": "%",
    "pct": "%",
    "phần trăm": "%",
    "phan tram": "%",
    "km": "km",
    "kilometer": "km",
    "kilometers": "km",
    "kilomet": "km",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "mét": "m",
    "met": "m",
    "cm": "cm",
    "mm": "mm",
    "ms": "ms",
    "s": "s",
    "sec": "s",
    "second": "s",
    "seconds": "s",
    "giây": "s",
    "gb": "gb",
    "mb": "mb",
    "kb": "kb",
    "tb": "tb",
    "kg": "kg",
    "g": "g",
    "mg": "mg",
    "hz": "hz",
    "mhz": "mhz",
    "ghz": "ghz",
    "usd": "usd",
    "$": "usd",
    "vnd": "vnd",
}

_UNIT_PATTERN = re.compile(
    r"(?<!\d)(\d+(?:[.,]\d+)?)\s*(%|phần\s+trăm|phan\s+tram|percent|pct|"
    r"kilometers?|kilomet|km|meters?|mét|met|cm|mm|ms|seconds?|sec|giây|"
    r"ghz|mhz|hz|tb|gb|mb|kb|kg|mg|usd|vnd|\$|m\b|s\b|g\b)?",
    re.IGNORECASE,
)

_METRIC_ANCHOR_WORDS = {
    "accuracy", "f1", "bleu", "rouge", "precision", "recall", "latency",
    "throughput", "memory", "loss", "perplexity", "mrr", "ndcg", "auc",
    "bert", "roberta", "llama", "qwen", "gpt", "transformer", "lora",
}


def _canonicalize_unit(raw_unit: Optional[str]) -> str:
    if not raw_unit:
        return ""
    cleaned = re.sub(r"\s+", " ", raw_unit.strip().lower())
    return _UNIT_CANONICAL_MAP.get(cleaned, cleaned)


def extract_numbers_with_units(text: str) -> list[tuple[str, str, int]]:
    """
    Extracts (normalized_number, canonical_unit, char_index) tuples from text.
    Ignores citation brackets [1], page/section/table/figure indices, and arXiv IDs.
    """
    cleaned = re.sub(r"\[\d{1,3}\]", " ", text or "")
    cleaned = re.sub(r"(?i)\b(?:trang|page|mục|section|hình|figure|table|bảng)\s*\d+\b", " ", cleaned)
    cleaned = re.sub(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", " ", cleaned)

    results: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    for m in _UNIT_PATTERN.finditer(cleaned):
        raw_num = m.group(1)
        raw_unit = m.group(2)
        norm_num = raw_num.replace(",", ".")
        canon_unit = _canonicalize_unit(raw_unit)
        # Skip single-digit bullet/section indices (1..9) unless accompanied by a unit or decimal
        if len(norm_num) == 1 and norm_num.isdigit() and not canon_unit:
            continue
        key = (norm_num, canon_unit)
        if key not in seen:
            seen.add(key)
            results.append((norm_num, canon_unit, m.start()))
    return results


def extract_numbers_and_years(text: str) -> list[str]:
    """
    Extracts significant numbers, percentages, decimals, and 4-digit years from text.
    Ignores citation brackets [1], [2], page markers (Trang 1, Page 2), and arXiv version tags.
    """
    items = extract_numbers_with_units(text)
    normalized: list[str] = []
    for num, unit, _ in items:
        label = f"{num}%" if unit == "%" else (f"{num} {unit}" if unit else num)
        if label not in normalized:
            normalized.append(label)
    return normalized


def _extract_context_matches(full_context_text: str) -> list[tuple[str, str, int]]:
    """Extracts all (normalized_number, canonical_unit, char_index) occurrences in context (without dedup)."""
    ctx = full_context_text or ""
    out: list[tuple[str, str, int]] = []
    for m in _UNIT_PATTERN.finditer(ctx):
        norm_num = m.group(1).replace(",", ".")
        canon_unit = _canonicalize_unit(m.group(2))
        out.append((norm_num, canon_unit, m.start()))
    return out


def check_numerical_and_date_consistency(claim_text: str, full_context_text: str) -> tuple[bool, bool, list[str]]:
    """
    Unit-Aware & Context-Aware Deterministic Numerical & Date Guard:
      - `85%` matches `85%`, `85 percent`, `85 phần trăm`, NOT bare `85`.
      - `85 km` matches `85 km`, NOT `85 m`.
      - `2025` matches `2025`, NOT `2024`.
      - If claim mentions a known metric/model anchor that also exists in context,
        verifies the matched number appears within a local window of that anchor.
    Returns (numerical_passed, dates_passed, missing_numbers).
    """
    lower_claim = (claim_text or "").lower()
    if any(
        neg in lower_claim
        for neg in (
            "không tìm thấy",
            "không đề cập",
            "không chứa",
            "không có thông tin",
            "không tồn tại",
            "hoàn toàn không",
            "not mentioned",
            "does not mention",
            "not found",
            "absent from",
        )
    ):
        return True, True, []

    claim_items = extract_numbers_with_units(claim_text)
    if not claim_items:
        return True, True, []

    ctx_text = full_context_text or ""
    ctx_lower = ctx_text.lower()
    ctx_matches = _extract_context_matches(ctx_text)

    # Identify salient metric/model anchors present in BOTH claim and context
    claim_tokens = set(re.findall(r"[a-z0-9_]+", lower_claim))
    shared_anchors = {a for a in (_METRIC_ANCHOR_WORDS & claim_tokens) if a in ctx_lower}

    missing_nums: list[str] = []
    dates_passed = True
    numerical_passed = True

    for norm_num, claim_unit, _ in claim_items:
        candidates = [
            (c_num, c_unit, c_pos)
            for (c_num, c_unit, c_pos) in ctx_matches
            if c_num == norm_num
        ]

        unit_matched_positions: list[int] = []
        for _, c_unit, c_pos in candidates:
            if claim_unit:
                # Claim explicitly specifies a unit (e.g. %, km, ms) -> context MUST match canonical unit
                if c_unit == claim_unit:
                    unit_matched_positions.append(c_pos)
            else:
                # Claim has bare number -> match bare number in context (or non-% / non-conflicting?
                # bare number must NOT silently match a percentage if unit was dropped)
                if c_unit != "%":
                    unit_matched_positions.append(c_pos)

        found = False
        if unit_matched_positions:
            if not shared_anchors or len(ctx_text) <= 180:
                found = True
            else:
                # Verify at least one matched position sits near ALL shared metric/entity anchors (word-boundary)
                for pos in unit_matched_positions:
                    win_start = max(0, pos - 120)
                    win_end = min(len(ctx_lower), pos + 120)
                    window = ctx_lower[win_start:win_end]
                    if all(re.search(rf"\b{re.escape(anchor)}\b", window) for anchor in shared_anchors):
                        found = True
                        break

        if not found:
            display = f"{norm_num}%" if claim_unit == "%" else (f"{norm_num} {claim_unit}" if claim_unit else norm_num)
            missing_nums.append(display)
            if not claim_unit and re.match(r"^(?:19\d\d|20\d\d)$", norm_num):
                dates_passed = False
            else:
                numerical_passed = False

    return numerical_passed, dates_passed, missing_nums


def verify_evidence_span_provenance(
    span: dict[str, Any],
    source_units: list[dict[str, Any]],
) -> tuple[bool, str]:
    """
    Independent Citation Provenance Verifier:
    Validates that an EvidenceSpan actually matches a real SourceUnit (or one of its
    merged multi-source provenance records) on:
      1. doc_id
      2. page_no
      3. unit_id (when specified)
      4. quote substring presence in source text
      5. context_hash integrity (when provided)
    """
    if not isinstance(span, dict):
        return False, "Evidence span is not a dict."
    quote = str(span.get("quote") or span.get("text") or "").strip()
    doc_id = str(span.get("doc_id") or "").strip()
    unit_id = str(span.get("source_unit_id") or span.get("unit_id") or "").strip()
    raw_page = span.get("page_no")
    span_hash = str(span.get("context_hash") or "").strip()

    if not quote or not doc_id:
        return False, "Evidence span missing quote or doc_id."

    try:
        page_no = int(raw_page)
    except Exception:
        return False, f"Invalid page_no in evidence span: {raw_page!r}"

    if span_hash:
        expected_quote_hash = compute_context_hash(quote, doc_id, page_no)
        # Allow hash computed on quote OR on full source unit text (checked below)
    else:
        expected_quote_hash = ""

    norm_quote = re.sub(r"\s+", " ", quote).lower()

    for unit in source_units or []:
        if not isinstance(unit, dict):
            continue
        u_text = str(unit.get("text") or unit.get("content") or "")
        norm_u_text = re.sub(r"\s+", " ", u_text).lower()
        if norm_quote not in norm_u_text:
            continue

        # Build candidate provenance list: explicit provenance list + primary unit metadata
        prov_list = list(unit.get("provenance") or [])
        primary_prov = {
            "doc_id": str(unit.get("doc_id") or "").strip(),
            "page_no": unit.get("page_no"),
            "unit_id": str(unit.get("id") or "").strip(),
        }
        prov_list.append(primary_prov)

        for prov in prov_list:
            if not isinstance(prov, dict):
                continue
            p_doc = str(prov.get("doc_id") or "").strip()
            try:
                p_page = int(prov.get("page_no"))
            except Exception:
                continue
            p_uid = str(prov.get("unit_id") or "").strip()

            if p_doc != doc_id or p_page != page_no:
                continue
            if unit_id and unit_id not in ("unknown", "u1") and p_uid and p_uid != unit_id:
                # Allow matching either the primary unit id or the merged provenance unit_id
                if str(unit.get("id") or "").strip() != unit_id:
                    continue

            if span_hash:
                unit_hash = str(unit.get("context_hash") or "").strip()
                prov_unit_hash = compute_context_hash(u_text, p_doc, p_page)
                if span_hash not in (expected_quote_hash, unit_hash, prov_unit_hash):
                    return False, f"context_hash mismatch for doc={doc_id}, page={page_no}."

            return True, f"Verified provenance: doc={doc_id}, page={page_no}"

    return False, f"No matching source unit/provenance found for doc={doc_id}, page={page_no}."


def split_answer_into_atomic_claims(
    answer: str,
    source_units: list[dict[str, Any]],
    existing_claims: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """
    Extracts atomic claims from either existing structured claims or sentences in `answer`,
    automatically mapping each claim to the highest-overlap SourceUnit.
    """
    if existing_claims:
        normalized_claims: list[dict[str, Any]] = []
        for idx, c in enumerate(existing_claims):
            if isinstance(c, dict):
                txt = str(c.get("claim_text") or c.get("claim") or c.get("text") or "").strip()
                uid = c.get("unit_id") or c.get("evidence_unit_id")
                ev_ids = c.get("evidence_ids") or ([uid] if uid else [])
            else:
                txt = str(c).strip()
                ev_ids = []
            if len(txt) >= 15:
                if not ev_ids and source_units:
                    best_u = find_best_supporting_unit(txt, source_units)
                    ev_ids = [best_u["id"]] if best_u else [source_units[0]["id"]]
                normalized_claims.append(
                    AtomicClaim(
                        id=f"claim_{idx + 1}",
                        text=txt,
                        evidence_ids=ev_ids,
                        unit_id=ev_ids[0] if ev_ids else None,
                    ).to_dict()
                )
        if normalized_claims:
            return normalized_claims

    # Split answer sentences
    clean_ans = re.sub(r"<think>.*?</think>", "", answer or "", flags=re.DOTALL)
    raw_sents = re.split(r"(?<=[.!?])\s+|\n+", clean_ans)
    claims_out: list[dict[str, Any]] = []
    for s in raw_sents:
        s_clean = re.sub(r"^[#•\-\*\d.)\s]+", "", s).strip()
        if len(s_clean) < 25 or s_clean.startswith("```") or s_clean.endswith(":"):
            continue
        if any(
            skip in s_clean.lower()
            for skip in ["dựa trên tài liệu", "dưới đây là tóm tắt", "tài liệu nguồn không cung cấp", "các cơ chế & cấu hình", "phân tích chi tiết"]
        ):
            continue
        best_u = find_best_supporting_unit(s_clean, source_units)
        uid = best_u["id"] if best_u else (source_units[0]["id"] if source_units else "u1")
        claims_out.append(
            AtomicClaim(
                id=f"claim_{len(claims_out) + 1}",
                text=s_clean,
                evidence_ids=[uid],
                unit_id=uid,
            ).to_dict()
        )
        if len(claims_out) >= 8:
            break
    return claims_out


def find_best_supporting_unit(
    claim_text: str,
    source_units: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Finds the SourceUnit with highest lexical + numeric overlap with `claim_text`."""
    if not source_units:
        return None
    c_tokens = set(re.findall(r"\w+", (claim_text or "").lower()))
    c_nums = set(extract_numbers_and_years(claim_text))

    best_unit = source_units[0]
    best_score = -1.0
    for u in source_units:
        u_text = str(u.get("text") or u.get("content") or "")
        u_tokens = set(re.findall(r"\w+", u_text.lower()))
        if not c_tokens:
            continue
        overlap = len(c_tokens & u_tokens) / max(len(c_tokens), 1)
        u_nums = set(extract_numbers_and_years(u_text))
        num_bonus = 0.35 * len(c_nums & u_nums) if c_nums else 0.0
        score = overlap + num_bonus
        if score > best_score:
            best_score = score
            best_unit = u
    return best_unit


def deterministic_verify_claims(
    claims: list[dict[str, Any]],
    source_units: list[dict[str, Any]],
    query: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Fast, 100% deterministic claim verifier combining:
      1. Token overlap & bilingual synonym matching.
      2. Numerical & Date Exactness Guard.
      3. Automatic EvidenceGap generation for rejected claims.
    Returns (verified_claims, rejected_claims, evidence_gaps).
    """
    full_context = "\n".join(
        f"{u.get('doc_id', '')} Trang {u.get('page_no', 1)}: {str(u.get('text') or u.get('content') or '')}"
        for u in source_units
    )
    verified: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []

    stop_words = {
        "và", "của", "cho", "với", "như", "thế", "nào", "được", "khi", "là", "gì",
        "các", "những", "này", "đó", "trong", "để", "có", "từ", "về", "theo",
        "the", "and", "for", "with", "from", "this", "that", "are", "was",
    }

    for idx, c in enumerate(claims):
        c_text = str(c.get("claim_text") or c.get("claim") or c.get("text") or "").strip()
        c_id = str(c.get("id") or c.get("claim_id") or f"claim_{idx + 1}")
        if not c_text:
            continue

        best_unit = find_best_supporting_unit(c_text, source_units)
        unit_text = str(best_unit.get("text") or "") if best_unit else full_context

        c_tokens = {w for w in re.findall(r"\w+", c_text.lower()) if w not in stop_words and len(w) > 1}
        ctx_tokens = {w for w in re.findall(r"\w+", full_context.lower()) if w not in stop_words and len(w) > 1}
        overlap_ratio = len(c_tokens & ctx_tokens) / max(len(c_tokens), 1)

        # Cross-lingual check (Vietnamese claim over English PDF): match ASCII technical terms
        ascii_c_tokens = {
            w for w in c_tokens
            if w.isascii() and w not in {"trang", "page", "doc", "arxiv", "pdf", "nam", "theo", "trong"}
        }
        ascii_overlap = (
            len(ascii_c_tokens & ctx_tokens) / max(len(ascii_c_tokens), 1)
            if ascii_c_tokens
            else overlap_ratio
        )
        is_abstention_claim = any(
            neg in c_text.lower()
            for neg in (
                "không tìm thấy",
                "không đề cập",
                "không chứa",
                "không có thông tin",
                "không tồn tại",
                "hoàn toàn không",
                "not mentioned",
                "does not mention",
                "not found",
                "absent from",
            )
        )

        num_ok, date_ok, missing_nums = check_numerical_and_date_consistency(c_text, full_context)

        if not num_ok or not date_ok:
            res = ClaimCheckResult(
                claim_id=c_id,
                claim_text=c_text,
                status="NOT_ENOUGH_INFO",
                reason=f"Sai lệch số liệu/năm hoặc đơn vị không khớp tài liệu gốc: {', '.join(missing_nums)}",
                numerical_passed=num_ok,
                dates_passed=date_ok,
                confidence_score=0.25,
            ).to_dict()
            rejected.append(res)
            gaps.append(
                EvidenceGap(
                    missing_aspect=f"Kiểm chứng số liệu {', '.join(missing_nums)} trong mệnh đề: {c_text[:90]}",
                    suggested_query=f"{query} {' '.join(missing_nums)}".strip(),
                    perspective="numerical_verification",
                ).to_dict()
            )
        elif (
            (overlap_ratio >= 0.20 or ascii_overlap >= 0.30)
            and bool({w for w in re.findall(r"\w+", c_text.lower())}.intersection({"không", "không thể", "chưa", "chẳng", "not", "never", "cannot", "no"}))
            != bool({w for w in re.findall(r"\w+", unit_text.lower())}.intersection({"không", "không thể", "chưa", "chẳng", "not", "never", "cannot", "no"}))
        ):
            res = ClaimCheckResult(
                claim_id=c_id,
                claim_text=c_text,
                status="CONTRADICT",
                reason="Mâu thuẫn khẳng định/phủ định trực tiếp với tài liệu nguồn.",
                numerical_passed=num_ok,
                dates_passed=date_ok,
                confidence_score=0.88,
            ).to_dict()
            rejected.append(res)
            gaps.append(
                EvidenceGap(
                    missing_aspect=f"Xác minh mâu thuẫn: {c_text[:90]}",
                    suggested_query=f"{query} {c_text[:45]}".strip(),
                    perspective="contradiction_verification",
                ).to_dict()
            )
        elif (
            overlap_ratio >= 0.25
            or ascii_overlap >= 0.40
            or (len(c_tokens & ctx_tokens) >= 2 and overlap_ratio >= 0.20)
            or is_abstention_claim
            or (best_unit and c_text[:40].lower() in unit_text.lower())
        ):
            uid = best_unit.get("id", "u1") if best_unit else "u1"
            pno = best_unit.get("page_no", 1) if best_unit else 1
            res = ClaimCheckResult(
                claim_id=c_id,
                claim_text=c_text,
                status="SUPPORT",
                reason=f"Khớp ngữ cảnh nguồn [{uid}, Trang {pno}] (Độ phủ từ khóa: {overlap_ratio:.0%}, Số học: Đạt).",
                numerical_passed=True,
                dates_passed=True,
                confidence_score=min(0.98, round(0.55 + overlap_ratio * 0.45, 2)),
            ).to_dict()
            verified.append(res)
        else:
            res = ClaimCheckResult(
                claim_id=c_id,
                claim_text=c_text,
                status="INSUFFICIENT",
                reason=f"Độ phủ bằng chứng thấp ({overlap_ratio:.0%} < 32%). Cần truy xuất thêm ngữ cảnh.",
                numerical_passed=num_ok,
                dates_passed=date_ok,
                confidence_score=round(overlap_ratio, 2),
            ).to_dict()
            rejected.append(res)
            gaps.append(
                EvidenceGap(
                    missing_aspect=c_text[:100],
                    suggested_query=f"{query} {c_text[:60]}".strip(),
                    perspective="coverage_expansion",
                ).to_dict()
            )

    return verified, rejected, gaps


def evaluate_inner_critic_fast(
    query: str,
    draft_answer: str,
    source_units: list[dict[str, Any]],
    required_entities: Optional[list[str]] = None,
) -> SummarizerCriticOutput:
    """
    Fast deterministic Inner Critic (matches SummarizerCriticOutput schema) for CPU Min
    or fallback when LLM Critic is not invoked.
    """
    if not source_units:
        return SummarizerCriticOutput(
            is_supported=False,
            unsupported_claims=["Không có đoạn văn nguồn nào được truy xuất."],
            missing_key_points=[query],
            action="request_more_context",
            confidence=0.0,
            critique_rationale="Ngữ cảnh rỗng.",
        )

    claims = split_answer_into_atomic_claims(draft_answer, source_units)
    verified, rejected, _ = deterministic_verify_claims(claims, source_units, query=query)

    missing_points: list[str] = []
    full_ctx_lower = "\n".join(str(u.get("text") or "") for u in source_units).lower()
    ans_lower = (draft_answer or "").lower()

    for ent in required_entities or []:
        ent_l = ent.lower()
        if ent_l in full_ctx_lower and ent_l not in ans_lower:
            missing_points.append(f"Bỏ sót thực thể/số liệu quan trọng có trong tài liệu: '{ent}'")

    unsupported_texts = [r["claim_text"] for r in rejected]
    total = max(len(verified) + len(rejected), 1)
    support_ratio = len(verified) / total

    if support_ratio >= 0.75 and not missing_points:
        return SummarizerCriticOutput(
            is_supported=True,
            unsupported_claims=[],
            missing_key_points=[],
            action="approve",
            confidence=round(support_ratio, 2),
            critique_rationale="Bản nháp bám sát ngữ cảnh và bao phủ đầy đủ thực thể.",
        )

    if len(verified) == 0:
        return SummarizerCriticOutput(
            is_supported=False,
            unsupported_claims=unsupported_texts,
            missing_key_points=missing_points or [query],
            action="request_more_context",
            confidence=0.2,
            critique_rationale="Ngữ cảnh hiện tại không đủ hỗ trợ câu trả lời.",
        )

    return SummarizerCriticOutput(
        is_supported=False,
        unsupported_claims=unsupported_texts,
        missing_key_points=missing_points,
        action="rewrite",
        confidence=round(support_ratio, 2),
        critique_rationale="Cần loại bỏ mệnh đề thiếu căn cứ và bổ sung thực thể bị sót từ Context.",
    )


def ensure_inline_citations(
    answer: str,
    source_units: list[dict[str, Any]],
    evidence_spans: list[dict[str, Any]],
) -> str:
    """
    Ensures the synthesized answer contains explicit inline citations [1], [2]
    linked to source units, and appends a compact 'Nguồn trích dẫn' footer if missing.
    """
    if not answer or not source_units:
        return answer

    # Clean any dangling unclosed trailing bracket like ", [" or "[" from generation cutoff
    text_out = re.sub(r"[,;\s]*\[\s*$", "", answer.strip()).strip()

    # Normalize [u1], [u2] -> [1], [2]
    text_out = re.sub(r"\[u(\d+)\]", r"[\1]", text_out)
    # Normalize comma lists [1, 2] -> [1], [2]
    text_out = re.sub(r"\[(\d+(?:\s*,\s*\d+)+)\]", lambda m: ", ".join(f"[{n.strip()}]" for n in m.group(1).split(",")), text_out)

    has_brackets = bool(re.search(r"\[\d+\]", text_out))

    if not has_brackets and source_units:
        # Attach [1], [2] to bullet points or paragraphs based on best matching unit
        unit_id_to_idx = {u.get("id", f"u{i+1}"): i + 1 for i, u in enumerate(source_units[:8])}
        paragraphs = text_out.split("\n\n")
        updated_paras: list[str] = []
        for p in paragraphs:
            p_strip = p.strip()
            if len(p_strip) > 35 and not p_strip.startswith("#") and not p_strip.startswith(""):
                best_u = find_best_supporting_unit(p_strip, source_units[:8])
                if best_u:
                    idx = unit_id_to_idx.get(best_u.get("id", "u1"), 1)
                    if not p_strip.endswith(f"[{idx}]"):
                        p = f"{p_strip} [{idx}]"
            updated_paras.append(p)
        text_out = "\n\n".join(updated_paras)

    # Ensure numbered sections and bullet items start on fresh lines (never splitting headers like ## 1.)
    text_out = re.sub(r"([a-z0-9à-ỹ\.\)\]\>])[ \t]*([0-9]+\.\s+(?:\*\*|[A-ZÀ-ỸĐ]))", r"\1\n\n\2", text_out, flags=re.IGNORECASE)
    text_out = re.sub(r"([a-z0-9à-ỹ\.\)\]\>])[ \t]+([-\*•]\s+)", r"\1\n\2", text_out, flags=re.IGNORECASE)

    return re.sub(r"[,;\s]*\[\s*$", "", text_out).strip()


def finalize_fact_check_contract(
    *,
    query: str,
    verified_claims: list[dict[str, Any]],
    rejected_claims: list[dict[str, Any]],
    evidence_gaps: list[dict[str, Any]],
    evidence_spans: Optional[list[dict[str, Any]]] = None,
    source_units: Optional[list[dict[str, Any]]] = None,
    full_context_text: str = "",
    context_coverage: Optional[dict[str, Any]] = None,
    verifier_model_family: str = "unknown",
    reasoning_trace: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """
    Unified Fact-Checking Contract Enforcer shared across ALL 6 execution modes:
      1. Verifies Citation Provenance (`verify_evidence_span_provenance`) on all `evidence_spans`.
      2. Audits every claim for unit-aware numerical consistency, date consistency, and provenance.
      3. Downgrades ungrounded 'SUPPORT' claims to 'not_enough_info' (never conflating missing evidence
         or unverified numbers with 'contradict').
      4. Aggregates per-claim statuses into `evidence_status` (FOUND | PARTIAL | NOT_FOUND | CONFLICTING_EVIDENCE)
         and projects legacy `fact_check_label` (support | contradict | not_enough_info).
    """
    units = list(source_units or [])
    spans = [s for s in (evidence_spans or []) if isinstance(s, dict)]
    if not full_context_text:
        full_context_text = "\n".join(
            str(u.get("text") or u.get("content") or "") for u in units if isinstance(u, dict)
        )

    span_prov_results: list[tuple[dict[str, Any], bool, str]] = [
        (sp, *verify_evidence_span_provenance(sp, units)) for sp in spans
    ]

    final_verified: list[dict[str, Any]] = []
    final_rejected: list[dict[str, Any]] = []
    final_gaps: list[dict[str, Any]] = list(evidence_gaps or [])
    statuses: list[str] = []

    all_input_claims: list[tuple[dict[str, Any], str]] = [
        (c if isinstance(c, dict) else {"claim": str(c)}, "support")
        for c in (verified_claims or [])
    ] + [
        (c if isinstance(c, dict) else {"claim": str(c)}, "rejected")
        for c in (rejected_claims or [])
    ]

    for c_dict, bucket in all_input_claims:
        c_text = str(
            c_dict.get("claim_text") or c_dict.get("claim") or c_dict.get("text") or ""
        ).strip()
        if not c_text:
            continue

        raw_st = str(c_dict.get("status") or ("SUPPORT" if bucket == "support" else "NOT_ENOUGH_INFO")).upper()
        reason_str = str(c_dict.get("reason") or c_dict.get("rationale") or c_dict.get("reasoning") or "")

        num_ok, date_ok, missing_nums = check_numerical_and_date_consistency(c_text, full_context_text)
        if "numerical_passed" in c_dict and c_dict["numerical_passed"] is False:
            num_ok = False
        if "dates_passed" in c_dict and c_dict["dates_passed"] is False:
            date_ok = False

        linked_ids = set(str(x) for x in (c_dict.get("evidence_ids") or []))
        if c_dict.get("unit_id"):
            linked_ids.add(str(c_dict["unit_id"]))

        if span_prov_results:
            rel_spans = [
                r for r in span_prov_results
                if not linked_ids
                or str(r[0].get("id") or "") in linked_ids
                or str(r[0].get("source_unit_id") or r[0].get("unit_id") or "") in linked_ids
            ] or span_prov_results
            provenance_ok = any(r[1] for r in rel_spans)
            prov_msg = "Citation provenance verified." if provenance_ok else "; ".join(r[2] for r in rel_spans[:2])
        else:
            provenance_ok = bool(units) and not bool(spans)
            prov_msg = "Checked against source_units."

        # Determine canonical per-claim status: support | contradict | not_enough_info
        if bucket == "support" and "SUPPORT" in raw_st:
            status = "support"
        elif "CONTRA" in raw_st or "REFUT" in raw_st:
            # Guard against legacy callers that labeled numerical mismatch as CONTRADICT
            if not num_ok or not date_ok or "Sai lệch số liệu" in reason_str:
                status = "not_enough_info"
            else:
                status = "contradict"
        else:
            status = "not_enough_info"

        # Enforce hard guards on 'support' claims
        if not num_ok:
            if status == "support":
                status = "not_enough_info"
            reason_str = f"{reason_str} | Số liệu/đơn vị không khớp nguồn: {', '.join(missing_nums)}".strip(" |")
        if not date_ok:
            if status == "support":
                status = "not_enough_info"
            reason_str = f"{reason_str} | Năm/thời gian không khớp nguồn: {', '.join(missing_nums)}".strip(" |")
        if span_prov_results and not provenance_ok:
            if status == "support":
                status = "not_enough_info"
            reason_str = f"{reason_str} | {prov_msg}".strip(" |")

        enriched = dict(c_dict)
        enriched.update(
            {
                "claim": c_text,
                "claim_text": c_text,
                "text": c_text,
                "status": status,
                "numerical_passed": num_ok,
                "dates_passed": date_ok,
                "provenance_ok": provenance_ok,
                "numeric_valid": num_ok,
                "reason": reason_str,
                "rationale": reason_str,
                "reasoning": reason_str,
                "confidence": float(c_dict.get("confidence") or c_dict.get("confidence_score") or 0.9),
            }
        )
        statuses.append(status)
        if status == "support":
            final_verified.append(enriched)
        else:
            final_rejected.append(enriched)
            if not final_gaps:
                final_gaps.append(
                    EvidenceGap(
                        missing_aspect=f"Xác minh nhận định: {c_text[:80]}",
                        suggested_query=f"{query} {c_text[:50]}".strip(),
                    ).to_dict()
                )

    total = len(statuses)
    support_cnt = statuses.count("support")
    contradict_cnt = statuses.count("contradict")
    nei_cnt = statuses.count("not_enough_info")

    if total == 0:
        evidence_status = "NOT_FOUND"
    elif contradict_cnt > 0 and (support_cnt == 0 or contradict_cnt >= support_cnt):
        evidence_status = "CONFLICTING_EVIDENCE"
    elif support_cnt == total:
        evidence_status = "FOUND"
    elif support_cnt > 0:
        evidence_status = "PARTIAL"
    else:
        evidence_status = "NOT_FOUND"

    fact_check_label = map_evidence_to_legacy_fact_check_label(evidence_status)

    out: dict[str, Any] = {
        "verified_claims": final_verified,
        "rejected_claims": final_rejected,
        "evidence_gaps": final_gaps,
        "evidence_status": evidence_status,
        "processing_status": "SUCCESS",
        "fact_check_label": fact_check_label,
        "verifier_model_family": verifier_model_family,
    }
    if context_coverage is not None:
        out["context_coverage"] = context_coverage
    if reasoning_trace is not None:
        out["reasoning_trace"] = reasoning_trace
    return out
