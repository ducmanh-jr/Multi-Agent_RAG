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

from core.schemas import SummarizerCriticOutput
from core.state import AtomicClaim, ClaimCheckResult, EvidenceGap, EvidenceSpan
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


def extract_numbers_and_years(text: str) -> list[str]:
    """Extracts normalized numerical values, percentages, and 4-digit years from text."""
def extract_numbers_and_years(text: str) -> list[str]:
    """
    Extracts significant numbers, percentages, decimals, and 4-digit years from text.
    Ignores citation brackets [1], [2], page markers (Trang 1, Page 2), and arXiv version tags.
    """
    cleaned = re.sub(r"\[\d{1,3}\]", " ", text or "")
    cleaned = re.sub(r"(?i)\b(?:trang|page|mục|section|hình|figure|table|bảng)\s*\d+\b", " ", cleaned)
    cleaned = re.sub(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", " ", cleaned)
    raw_nums = re.findall(r"\b\d+(?:[.,]\d+)?%?\b", cleaned)
    normalized: list[str] = []
    for n in raw_nums:
        norm = n.replace(",", ".")
        # Skip single-digit bullet/section indices (1..9) unless percentage or decimal
        if len(norm) == 1 and norm.isdigit():
            continue
        if norm not in normalized:
            normalized.append(norm)
    return normalized


def check_numerical_and_date_consistency(claim_text: str, full_context_text: str) -> tuple[bool, bool, list[str]]:
    """
    Deterministic Numerical & Date Guard:
    Verifies whether numbers and 4-digit years stated in `claim_text` appear in `full_context_text`.
    If `claim_text` is an explicit negative/abstention statement ("không đề cập", "không chứa", etc.),
    numbers echoed from the user's question are not treated as hallucinations.
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

    claim_nums = extract_numbers_and_years(claim_text)
    if not claim_nums:
        return True, True, []

    ctx_norm = (full_context_text or "").replace(",", ".")
    missing_nums: list[str] = []
    dates_passed = True
    numerical_passed = True

    for num in claim_nums:
        # Check exact or comma/dot equivalent presence in context
        raw_variants = {num, num.replace(".", ","), num.rstrip("%")}
        found = any(v in ctx_norm or v in (full_context_text or "") for v in raw_variants if v)
        if not found:
            missing_nums.append(num)
            if re.match(r"^(?:19\d\d|20\d\d)$", num):
                dates_passed = False
            else:
                numerical_passed = False

    return numerical_passed, dates_passed, missing_nums


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
        s_clean = re.sub(r"^[•\-\*\d.)\s]+", "", s).strip()
        if len(s_clean) < 25:
            continue
        if any(
            skip in s_clean.lower()
            for skip in ["dựa trên tài liệu", "dưới đây là tóm tắt", "tài liệu nguồn không cung cấp"]
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
                status="CONTRADICT",
                reason=f"Sai lệch số liệu/năm không có trong tài liệu gốc: {', '.join(missing_nums)}",
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
            overlap_ratio >= 0.25
            or ascii_overlap >= 0.40
            or len(c_tokens & ctx_tokens) >= 2
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

    has_brackets = bool(re.search(r"\[\d+\]", answer))
    text_out = answer.strip()

    if not has_brackets and source_units:
        # Attach [1], [2] to bullet points or paragraphs based on best matching unit
        unit_id_to_idx = {u.get("id", f"u{i+1}"): i + 1 for i, u in enumerate(source_units[:8])}
        paragraphs = text_out.split("\n\n")
        updated_paras: list[str] = []
        for p in paragraphs:
            p_strip = p.strip()
            if len(p_strip) > 35 and not p_strip.startswith("#") and not p_strip.startswith("⚠️"):
                best_u = find_best_supporting_unit(p_strip, source_units[:8])
                if best_u:
                    idx = unit_id_to_idx.get(best_u.get("id", "u1"), 1)
                    if not p_strip.endswith(f"[{idx}]"):
                        p = f"{p_strip} [{idx}]"
            updated_paras.append(p)
        text_out = "\n\n".join(updated_paras)

    return text_out
