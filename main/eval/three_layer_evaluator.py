# -*- coding: utf-8 -*-
"""
Three-Layer Scientific Evaluation Engine (M0 Foundation):
  - Layer 1 (Deterministic Structural, Numeric, Derived-Calculation, Domain-Aware Math & Region Audit):
      * Validates abstention/clarification status AND displayed text consistency (blocks leaky refusals).
      * Enforces directed comparative subject-object roles ("A exceeds B" vs "B exceeds A").
      * Checks clause-level negation & polarity alignment (EN/VI).
      * Distinguishes percentage points (`pp` / `điểm phần trăm`) from relative percent (`%`),
        and verifies `derived_claim` operands, splits (Validation vs Test), units, and arithmetic.
      * Checks mathematical expression equivalence on a structure-preserving AST (`x^2` != `x_2`,
        inequalities preserved, and domain constraints like `x != 0` required before `x/x = 1`).
      * Computes bounding-box IoU and containment on `(document, page)` citations.
  - Layer 2 (Independent Cross-Lingual NLI Claim Scoring):
      * Evaluates atomic claims through a generator-independent NLI callable with calibrated
        direction-specific thresholds (`en->vi`, `vi->en`, `vi->vi`, `en->en`).
  - Layer 3 (Bilingual Human Calibration, Adjudication & Clopper-Pearson Confidence Bounds):
      * Computes Cohen's kappa inter-annotator agreement, calibrates Layer 2 thresholds on
        adjudicated bilingual human labels, and aggregates blind pairwise win/tie/loss with
        exact Clopper-Pearson confidence intervals.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from utils.bilingual import expand_bilingual_tokens


REFUSAL_STATUSES = frozenset({"NOT_FOUND_IN_DOCUMENT", "INSUFFICIENT_EVIDENCE"})
CLARIFICATION_STATUSES = frozenset({"CLARIFICATION_NEEDED"})
ANSWERED_STATUSES = frozenset({"ANSWERED", "ANSWERED_DIRECT_QUOTE"})

_REFUSAL_TEXT_PATTERNS = [
    r"không\s+(?:được\s+|hoàn\s+toàn\s+)*(?:đề\s+cập|nêu|nhắc\s+(?:đến|tới))",
    r"không\s+tìm\s+thấy",
    r"không\s+có\s+(?:thông\s+tin|dữ\s+liệu|bằng\s+chứng|số\s+liệu)",
    r"(?:không|chưa)\s+đủ\s+bằng\s+chứng",
    r"chưa\s+có\s+(?:dữ\s+liệu|bằng\s+chứng|thông\s+tin)",
    r"không\s+thể\s+(?:xác\s+định|kết\s+luận|gán)",
    r"tài\s+liệu.*không.*(?:đề\s+cập|chứa|nêu|cung\s+cấp|có)",
    r"insufficient\s+evidence",
    r"not\s+(?:mentioned|stated|reported|found|provided)",
    r"does\s+not\s+(?:mention|contain|provide|report|specify)",
    r"no\s+(?:information|evidence|data)\s+(?:is\s+|was\s+)?(?:available|provided|found|reported)",
    r"cannot\s+be\s+(?:determined|verified|concluded)",
]

_CLARIFICATION_TEXT_PATTERNS = [
    r"vui\s+lòng\s+(?:cho\s+biết|làm\s+rõ|cung\s+cấp\s+thêm|xác\s+định)",
    r"bạn\s+muốn\s+(?:xem|hỏi|so\s+sánh)",
    r"cần\s+làm\s+rõ",
    r"please\s+(?:clarify|specify|indicate)",
    r"which\s+(?:document|table|figure|experiment|dataset|split)",
    r"could\s+you\s+(?:clarify|specify)",
]

# Bilingual academic & experimental phrase lexicon for offline cross-lingual grounding
_EVAL_BILINGUAL_MAP: dict[str, tuple[str, ...]] = {
    "trial": ("thử nghiệm", "thí nghiệm"),
    "experiment": ("thí nghiệm", "thử nghiệm", "thực nghiệm"),
    "study": ("nghiên cứu", "khảo sát"),
    "lasted": ("kéo dài", "diễn ra"),
    "ran": ("kéo dài", "chạy", "diễn ra"),
    "exactly": ("đúng", "chính xác"),
    "consecutive": ("liên tiếp", "liên tục"),
    "days": ("ngày",),
    "day": ("ngày",),
    "hours": ("giờ", "tiếng"),
    "minutes": ("phút",),
    "seconds": ("giây",),
    "liters": ("lít", "lit"),
    "liter": ("lít", "lit"),
    "water": ("nước",),
    "drip": ("nhỏ giọt",),
    "irrigation": ("tưới", "tưới nước"),
    "accuracy": ("độ chính xác",),
    "precision": ("độ chính xác",),
    "recall": ("độ phủ", "độ nhạy"),
    "layers": ("lớp", "tầng"),
    "layer": ("lớp", "tầng"),
    "method": ("phương pháp",),
    "model": ("mô hình",),
    "orchard": ("vườn cây",),
    "greenhouse": ("nhà kính",),
    "test": ("kiểm thử",),
    "field": ("thực địa",),
    "higher": ("cao hơn", "lớn hơn", "vượt"),
    "lower": ("thấp hơn", "nhỏ hơn", "kém hơn"),
    "outperforms": ("vượt", "tốt hơn", "cao hơn"),
    "exceeds": ("vượt", "cao hơn", "lớn hơn"),
    "treatment": ("phương pháp điều trị", "điều trị", "nghiệm thức"),
    "works": ("hiệu quả", "có tác dụng", "hoạt động"),
    "reduce": ("giảm", "làm giảm", "hạ"),
    "blood": ("máu", "huyết"),
    "pressure": ("áp", "áp suất", "huyết áp"),
    "percentage points": ("điểm phần trăm",),
}

_WORD_NUMBERS: dict[str, str] = {
    "zero": "0", "không": "0",
    "one": "1", "một": "1",
    "two": "2", "hai": "2",
    "three": "3", "ba": "3",
    "four": "4", "bốn": "4", "tư": "4",
    "five": "5", "năm": "5",
    "six": "6", "sáu": "6",
    "seven": "7", "bảy": "7",
    "eight": "8", "tám": "8",
    "nine": "9", "chín": "9",
    "ten": "10", "mười": "10",
    "eleven": "11", "mười một": "11",
    "twelve": "12", "mười hai": "12",
    "thirteen": "13", "mười ba": "13",
    "fourteen": "14", "mười bốn": "14", "mười tư": "14",
    "fifteen": "15", "mười lăm": "15",
    "sixteen": "16", "mười sáu": "16",
    "seventeen": "17", "mười bảy": "17",
    "eighteen": "18", "mười tám": "18",
    "nineteen": "19", "mười chín": "19",
    "twenty": "20", "hai mươi": "20",
    "twenty-one": "21", "hai mươi mốt": "21",
    "thirty": "30", "ba mươi": "30",
    "thirty-five": "35", "ba mươi lăm": "35",
    "forty": "40", "bốn mươi": "40",
    "forty-two": "42", "bốn mươi hai": "42",
    "fifty": "50", "năm mươi": "50",
    "hundred": "100", "một trăm": "100",
}

# Ordered longest-first so `điểm phần trăm` / `percentage points` NEVER collide with `phần trăm` / `%`
_CANONICAL_UNITS_ORDERED: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(?:điểm\s+phần\s+trăm|diem\s+phan\s+tram|percentage\s+points?|pct\s+points?|pp)$", re.I), "pp"),
    (re.compile(r"^(?:%|phần\s+trăm|phan\s+tram|percent|pct)$", re.I), "%"),
    (re.compile(r"^(?:km/h|kmh|kph|km/giờ|km/gio)$", re.I), "km/h"),
    (re.compile(r"^(?:m/h|m/giờ|m/gio)$", re.I), "m/h"),
    (re.compile(r"^(?:m/s|mps|m/giây)$", re.I), "m/s"),
    (re.compile(r"^(?:km/s)$", re.I), "km/s"),
    (re.compile(r"^(?:mg/l|mg\s*/\s*l)$", re.I), "mg/l"),
    (re.compile(r"^(?:g/l|g\s*/\s*l)$", re.I), "g/l"),
    (re.compile(r"^(?:db/cm|db\s*/\s*cm)$", re.I), "db/cm"),
    (re.compile(r"^(?:mpa|megapascals?)$", re.I), "mpa"),
    (re.compile(r"^(?:tokens?/s|toks?/s)$", re.I), "tokens/s"),
    (re.compile(r"^(?:gb/s|gbps)$", re.I), "gb/s"),
    (re.compile(r"^(?:mb/s|mbps)$", re.I), "mb/s"),
    (re.compile(r"^(?:milliseconds?|ms)$", re.I), "ms"),
    (re.compile(r"^(?:seconds?|secs?|giây|s)$", re.I), "s"),
    (re.compile(r"^(?:minutes?|mins?|phút)$", re.I), "min"),
    (re.compile(r"^(?:hours?|hrs?|giờ|h)$", re.I), "h"),
    (re.compile(r"^(?:days?|ngày)$", re.I), "day"),
    (re.compile(r"^(?:liters?|litres?|lít|lit|l)$", re.I), "l"),
    (re.compile(r"^(?:milliliters?|ml)$", re.I), "ml"),
    (re.compile(r"^(?:kilometers?|kilomet|km)$", re.I), "km"),
    (re.compile(r"^(?:meters?|mét|met|m)$", re.I), "m"),
    (re.compile(r"^(?:cm)$", re.I), "cm"),
    (re.compile(r"^(?:mm)$", re.I), "mm"),
    (re.compile(r"^(?:kg)$", re.I), "kg"),
    (re.compile(r"^(?:mg)$", re.I), "mg"),
    (re.compile(r"^(?:g)$", re.I), "g"),
    (re.compile(r"^(?:gb)$", re.I), "gb"),
    (re.compile(r"^(?:mb)$", re.I), "mb"),
    (re.compile(r"^(?:layers?|lớp|tầng)$", re.I), "layer"),
]

_QUANTITY_RE = re.compile(
    r"(?<![\w.,])([−+-]?\d+(?:[.,]\d+)?)"
    r"(?:\s+(?:consecutive|clear|encoder|attention|total|full|additional))?\s*("
    r"điểm\s+phần\s+trăm|diem\s+phan\s+tram|percentage\s+points?|pct\s+points?|\bpp\b|"
    r"%|phần\s+trăm|phan\s+tram|\bpercent\b|\bpct\b|"
    r"km/h|kmh|kph|km/giờ|km/gio|m/h|m/giờ|m/gio|m/s|mps|m/giây|km/s|mg/l|g/l|db/cm|"
    r"tokens?/s|toks?/s|gb/s|gbps|mb/s|mbps|\bmpa\b|\bmegapascals?\b|"
    r"\bmilliseconds?\b|\bms\b|\bseconds?\b|\bsecs?\b|\bgiây\b|\bminutes?\b|\bmins?\b|\bphút\b|"
    r"\bhours?\b|\bhrs?\b|\bgiờ\b|\bdays?\b|\bngày\b|"
    r"\bliters?\b|\blitres?\b|\blít\b|\blit\b|\bmilliliters?\b|\bml\b|"
    r"\bkilometers?\b|\bkilomet\b|\bkm\b|\bmeters?\b|\bmét\b|\bmet\b|\bcm\b|\bmm\b|"
    r"\bkg\b|\bmg\b|\bgb\b|\bmb\b|\blayers?\b|\blớp\b|\btầng\b|"
    r"\bm\b|\bs\b|\bg\b|\bh\b|\bl\b"
    r")?",
    re.IGNORECASE,
)


def canonicalize_unit(raw_unit: str | None) -> str:
    if not raw_unit:
        return ""
    cleaned = " ".join(raw_unit.strip().split())
    for pat, canon in _CANONICAL_UNITS_ORDERED:
        if pat.match(cleaned):
            return canon
    return cleaned.lower()


def normalize_word_numbers(text: str) -> str:
    """Replace English and Vietnamese number words (sorted longest-first) with digits."""
    out = unicodedata.normalize("NFC", text or "")
    for phrase in sorted(_WORD_NUMBERS, key=len, reverse=True):
        digit = _WORD_NUMBERS[phrase]
        out = re.sub(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", digit, out, flags=re.IGNORECASE)
    return out


def _strip_citations_and_locators(text: str) -> str:
    cleaned = re.sub(r"(?<!\\)\[\d{1,3}\]", " ", text or "")
    cleaned = re.sub(
        r"(?i)\b(?:trang|page|mục|section|hình|figure|fig\.?|table|tab\.?|bảng|"
        r"equation|eq\.?|formula|công\s+thức|phương\s+trình)\s*\(?\d+(?:\.\d+)*\)?(?!\w)",
        " ",
        cleaned,
    )
    return cleaned


def extract_quantities(text: str) -> list[tuple[float, str, str]]:
    """
    Extract `(float_value, canonical_unit, raw_number_str)` from text after expanding
    English/Vietnamese number words and stripping citation/locator indices.
    """
    expanded = normalize_word_numbers(_strip_citations_and_locators(text))
    items: list[tuple[float, str, str]] = []
    for match in _QUANTITY_RE.finditer(expanded):
        raw_num = match.group(1).replace("−", "-").replace(",", ".")
        unit = canonicalize_unit(match.group(2))
        try:
            val = float(raw_num)
        except ValueError:
            continue
        items.append((val, unit, raw_num))
    return items


# =====================================================================
# 1. Abstention, Clarification & Displayed-Content Consistency
# =====================================================================

def has_refusal_phrasing(text: str) -> bool:
    lower = unicodedata.normalize("NFC", (text or "")).lower()
    return any(re.search(pat, lower) for pat in _REFUSAL_TEXT_PATTERNS)


def has_clarification_phrasing(text: str) -> bool:
    lower = unicodedata.normalize("NFC", (text or "")).lower()
    return any(re.search(pat, lower) for pat in _CLARIFICATION_TEXT_PATTERNS)


def audit_abstention_and_display(
    *,
    answer: str,
    answer_status: str | None,
    answerability: str = "answerable",
    forbidden_entities: Sequence[str] = (),
    forbidden_numbers: Sequence[str | float] = (),
) -> dict[str, Any]:
    """
    Jointly audits `answer_status` and the actual text displayed to the user.
    Catches silent mismatches where status claims refusal (`INSUFFICIENT_EVIDENCE` /
    `NOT_FOUND_IN_DOCUMENT`) while the text states fabricated numbers or trap entities as fact.
    """
    norm_status = (answer_status or "").strip().upper()
    clean_ans = (answer or "").strip()
    refusal_text = has_refusal_phrasing(clean_ans)
    clarify_text = has_clarification_phrasing(clean_ans)

    leaked_entities: list[str] = []
    lower_ans = clean_ans.lower()
    for ent in forbidden_entities or ():
        e_clean = str(ent or "").strip().lower()
        if e_clean and re.search(r"(?<!\w)" + re.escape(e_clean) + r"(?!\w)", lower_ans):
            # Mentioning the queried term inside "thông tin về X không được đề cập" is acceptable ONLY when refusal_text is True
            if not refusal_text:
                leaked_entities.append(str(ent))

    extracted_vals = [val for val, _, _ in extract_quantities(clean_ans)]
    leaked_numbers: list[str] = []
    for fnum in forbidden_numbers or ():
        try:
            target_f = float(str(fnum).replace(",", "."))
        except ValueError:
            continue
        if any(math.isclose(v, target_f, abs_tol=1e-6) for v in extracted_vals):
            leaked_numbers.append(str(fnum))

    status_is_refusal = norm_status in REFUSAL_STATUSES
    status_is_clarify = norm_status in CLARIFICATION_STATUSES
    status_is_answered = norm_status in ANSWERED_STATUSES

    # Detect true status-vs-display contradiction:
    # e.g., status says NOT_FOUND_IN_DOCUMENT / INSUFFICIENT_EVIDENCE, while displayed text lacks refusal phrasing
    # or leaks forbidden numbers/entities; or status says ANSWERED while displayed text is a pure refusal.
    status_display_mismatch = False
    if status_is_refusal and clean_ans and (not refusal_text or bool(leaked_numbers) or bool(leaked_entities)):
        status_display_mismatch = True
    elif status_is_answered and refusal_text and not extracted_vals:
        status_display_mismatch = True

    effective_abstained = (
        (status_is_refusal and (not clean_ans or refusal_text))
        or (not norm_status and bool(clean_ans) and refusal_text)
    ) and not status_display_mismatch and not leaked_numbers and not leaked_entities

    effective_clarified = (
        (status_is_clarify and (not clean_ans or clarify_text))
        or (not norm_status and bool(clean_ans) and clarify_text)
    ) and not leaked_numbers

    target_mode = (answerability or "answerable").strip().lower()
    if target_mode in {"unanswerable", "insufficient_evidence", "adversarial_trap"}:
        passed = effective_abstained
    elif target_mode == "ambiguous":
        passed = effective_clarified or effective_abstained
    else:
        passed = bool(clean_ans) and not effective_abstained and not effective_clarified and not status_display_mismatch

    return {
        "passed": passed,
        "effective_abstained": effective_abstained,
        "effective_clarified": effective_clarified,
        "status_display_mismatch": status_display_mismatch,
        "leaked_entities": leaked_entities,
        "leaked_numbers": leaked_numbers,
    }


# =====================================================================
# 2. Negation Polarity & Directed Comparative Subject-Object Guard
# =====================================================================

_NEGATION_CUE_RE = re.compile(
    r"(?:\bit\s+is\s+(?:false|not\s+true|incorrect)\s+that\b"
    r"|\bcontrary\s+to\s+the\s+claim\s+that\b"
    r"|\b(?:does|did|do|is|are|was|were|can|could|will|would|has|have)\s+not\b"
    r"|\b(?:doesn't|didn't|isn't|aren't|wasn't|weren't|cannot|can't|won't)\b"
    r"|\bnever\b|\bno\s+(?:significant\s+)?(?:effect|improvement|gain|benefit|evidence|difference)\b"
    r"|\bfailed\s+to\b"
    r"|\bsai\s+khi\s+(?:nói|cho)\s+rằng\b|\bkhông\s+phải\s+(?:là\s+)?"
    r"|\bhoàn\s+toàn\s+không\b|\bchưa\s+hề\b|\bkhông\s+(?:có\s+tác\s+dụng|hiệu\s+quả|cải\s+thiện|đạt|tăng|vượt|làm)\b)",
    re.IGNORECASE,
)


def has_negation(text: str) -> bool:
    return bool(_NEGATION_CUE_RE.search(unicodedata.normalize("NFC", text or "")))


def check_polarity_alignment(candidate: str, reference: str) -> tuple[bool, str]:
    """Verify candidate does not invert matrix/clause negation relative to reference."""
    cand_neg = has_negation(candidate)
    ref_neg = has_negation(reference)
    if cand_neg != ref_neg:
        return False, "polarity_inverted"
    return True, "ok"


_NON_ENTITY_MODIFIERS = frozenset({
    "cao", "thấp", "lớn", "nhỏ", "nhanh", "chậm", "tốt", "kém", "ít", "nhiều", "mạnh", "yếu",
    "hơn", "nhất", "bằng", "vượt", "kết", "quả", "giá", "trị", "tỷ", "lệ", "điểm", "mức",
    "higher", "lower", "greater", "smaller", "faster", "slower", "better", "worse", "less", "more",
    "than", "is", "was", "are", "were", "the", "and", "or", "by", "in", "on", "at", "of",
})

_ENTITY_TOKEN = (
    r"(?<!\w)(?:(?:phương\s+pháp|mô\s+hình|nhóm|hệ\s+thống|nghiệm\s+thức|method|model|group|system|variant|config|treatment)\s+)?"
    r"[A-Za-z0-9][A-Za-z0-9._-]*(?!\w)"
)

_GREATER_PATTERNS = [
    re.compile(
        rf"(?P<lhs>{_ENTITY_TOKEN})\s+(?:cao\s+hơn|lớn\s+hơn|nhanh\s+hơn|tốt\s+hơn|mạnh\s+hơn|vượt(?:\s+trội)?(?:\s+hơn)?|"
        rf"(?<!thấp\s)(?<!nhỏ\s)(?<!chậm\s)(?<!kém\s)(?<!ít\s)(?<!yếu\s)hơn|"
        rf"outperforms?|exceeds?|surpasses?|beats?|is\s+(?:higher|greater|larger|faster|better)\s+than|>)\s+(?P<rhs>{_ENTITY_TOKEN})",
        re.IGNORECASE,
    ),
]
_LESS_PATTERNS = [
    re.compile(
        rf"(?P<lhs>{_ENTITY_TOKEN})\s+(?:thấp\s+hơn|nhỏ\s+hơn|chậm\s+hơn|kém\s+hơn|ít\s+hơn|"
        rf"underperforms?|lags?\s+behind|is\s+(?:lower|smaller|slower|worse|less)\s+than|<)\s+(?P<rhs>{_ENTITY_TOKEN})",
        re.IGNORECASE,
    ),
]


def _normalize_entity_label(raw: str) -> str:
    cleaned = re.sub(
        r"^(?:phương\s+pháp|mô\s+hình|nhóm|hệ\s+thống|nghiệm\s+thức|method|model|group|system|variant|config|treatment)\s+",
        "",
        (raw or "").strip(),
        flags=re.IGNORECASE,
    )
    norm = cleaned.strip(".,;:() ").lower()
    return "" if norm in _NON_ENTITY_MODIFIERS else norm


def extract_canonical_greater_pairs(text: str) -> list[tuple[str, str]]:
    """
    Extract canonical `(higher_entity, lower_entity)` pairs from comparative clauses.
    Normalizes `"A thấp hơn B"` and `"B cao hơn A"` to the same canonical pair `("b", "a")`.
    """
    norm = unicodedata.normalize("NFC", text or "")
    if has_negation(norm):
        return []
    pairs: list[tuple[str, str]] = []
    for pat in _GREATER_PATTERNS:
        for m in pat.finditer(norm):
            lhs = _normalize_entity_label(m.group("lhs"))
            rhs = _normalize_entity_label(m.group("rhs"))
            if lhs and rhs and lhs != rhs:
                pairs.append((lhs, rhs))
    for pat in _LESS_PATTERNS:
        for m in pat.finditer(norm):
            lhs = _normalize_entity_label(m.group("lhs"))
            rhs = _normalize_entity_label(m.group("rhs"))
            if lhs and rhs and lhs != rhs:
                pairs.append((rhs, lhs))
    return list(dict.fromkeys(pairs))


def check_directed_relation(
    candidate: str,
    *,
    higher_entity: str,
    lower_entity: str,
) -> tuple[bool, str]:
    """
    Verify that `candidate` states `higher_entity > lower_entity` (or `lower_entity < higher_entity`)
    and NEVER reverses the subject-object direction.
    """
    hi = _normalize_entity_label(higher_entity)
    lo = _normalize_entity_label(lower_entity)
    pairs = extract_canonical_greater_pairs(candidate)
    if (lo, hi) in pairs:
        return False, f"subject_swapped:{lo}>{hi}"
    if (hi, lo) in pairs:
        return True, "ok"
    return False, f"missing_directed_relation:{hi}>{lo}"


# =====================================================================
# 3. Derived Numerical Claim Verification (`derived_claim`)
# =====================================================================

def verify_derived_calculation(
    candidate_answer: str,
    derivation_spec: Mapping[str, Any],
    source_passages: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Verifies a derived numerical claim against its structured operand specification:
      - Checks all operands share the same evaluation split/condition (unless `allow_cross_split=True`).
      - Checks each operand's entity, value, and unit exist in the supplied source passages (when provided).
      - Computes the deterministic result (`diff_pp`, `rel_change_pct`, `sub`, `add`, `ratio`, `mean`).
      - Verifies the candidate answer states the derived value with the exact canonical unit
        (e.g., `11 pp` vs `13.58 %`) and preserves comparative direction when two entities are compared.
    """
    op = str(derivation_spec.get("op") or "sub").strip().lower()
    operands = list(derivation_spec.get("operands") or [])
    allow_cross_split = bool(derivation_spec.get("allow_cross_split", False))
    tolerance = float(derivation_spec.get("tolerance", 0.05))

    if len(operands) < 2 and op != "mean":
        return {"passed": False, "reason": "insufficient_operands"}

    splits = {str(item.get("split") or "").strip().lower() for item in operands if item.get("split")}
    if len(splits) > 1 and not allow_cross_split:
        return {"passed": False, "reason": f"split_mismatch:{sorted(splits)}"}

    metrics = {str(item.get("metric") or "").strip().lower() for item in operands if item.get("metric")}
    if len(metrics) > 1:
        return {"passed": False, "reason": f"metric_mismatch:{sorted(metrics)}"}

    # Verify each operand against source passages if supplied
    if source_passages:
        by_id = {str(p.get("id") or p.get("source_id") or ""): p for p in source_passages if isinstance(p, Mapping)}
        all_text = "\n".join(
            str(p.get("text") or p.get("content") or p.get("passage") or "")
            for p in source_passages
            if isinstance(p, Mapping)
        )
        if all_text.strip():
            all_bilingual = expand_cross_lingual_tokens(all_text)
            for item in operands:
                sid = str(item.get("source_id") or "")
                unit_obj = by_id.get(sid) if sid else None
                target_text = str(
                    (unit_obj.get("text") or unit_obj.get("content") or unit_obj.get("passage") or "")
                    if isinstance(unit_obj, Mapping)
                    else all_text
                ) or all_text
                ent = str(item.get("entity") or "").strip()
                val = float(item["value"])
                unit = canonicalize_unit(item.get("unit"))
                if ent and ent.lower() not in target_text.lower():
                    ent_tokens = {w for w in re.findall(r"[a-z0-9à-ỹ]+", ent.lower()) if len(w) >= 2}
                    if not ent_tokens or not (ent_tokens <= (expand_cross_lingual_tokens(target_text) | all_bilingual)):
                        return {"passed": False, "reason": f"operand_entity_missing:{ent}"}
                q_list = extract_quantities(target_text)
                if not any(math.isclose(qv, val, abs_tol=1e-4) and (not unit or qu == unit) for qv, qu, _ in q_list):
                    return {"passed": False, "reason": f"operand_value_missing:{val}{unit}"}

    vals = [float(item["value"]) for item in operands]
    units = [canonicalize_unit(item.get("unit")) for item in operands]

    if op == "diff_pp":
        if any(u != "%" for u in units):
            return {"passed": False, "reason": "diff_pp_requires_percent_operands"}
        computed_val = vals[0] - vals[1]
        expected_unit = "pp"
    elif op == "rel_change_pct":
        if math.isclose(vals[1], 0.0, abs_tol=1e-12):
            return {"passed": False, "reason": "division_by_zero"}
        computed_val = ((vals[0] - vals[1]) / abs(vals[1])) * 100.0
        expected_unit = "%"
    elif op == "sub":
        if len(set(units)) > 1:
            return {"passed": False, "reason": f"incompatible_operand_units:{units}"}
        computed_val = vals[0] - vals[1]
        expected_unit = canonicalize_unit(derivation_spec.get("expected_unit")) or units[0]
    elif op == "add":
        if len(set(units)) > 1:
            return {"passed": False, "reason": f"incompatible_operand_units:{units}"}
        computed_val = sum(vals)
        expected_unit = canonicalize_unit(derivation_spec.get("expected_unit")) or units[0]
    elif op == "ratio":
        if math.isclose(vals[1], 0.0, abs_tol=1e-12):
            return {"passed": False, "reason": "division_by_zero"}
        computed_val = vals[0] / vals[1]
        expected_unit = canonicalize_unit(derivation_spec.get("expected_unit"))
    elif op == "mean":
        computed_val = sum(vals) / len(vals)
        expected_unit = canonicalize_unit(derivation_spec.get("expected_unit")) or (units[0] if units else "")
    else:
        return {"passed": False, "reason": f"unsupported_op:{op}"}

    if "expected_value" in derivation_spec and derivation_spec["expected_value"] is not None:
        spec_val = float(derivation_spec["expected_value"])
        if not math.isclose(computed_val, spec_val, abs_tol=tolerance):
            return {"passed": False, "reason": f"spec_arithmetic_inconsistent:{computed_val}!={spec_val}"}

    cand_quantities = extract_quantities(candidate_answer)
    matched_val_and_unit = any(
        math.isclose(qv, computed_val, abs_tol=tolerance)
        and (qu == expected_unit if expected_unit else True)
        for qv, qu, _ in cand_quantities
    )
    if not matched_val_and_unit:
        # Check if candidate computed the right number with the wrong unit (e.g., 11% instead of 11 pp)
        wrong_unit_hit = next(
            (qu for qv, qu, _ in cand_quantities if math.isclose(qv, computed_val, abs_tol=tolerance)),
            None,
        )
        if wrong_unit_hit is not None and expected_unit:
            return {"passed": False, "reason": f"unit_mismatch:{wrong_unit_hit}_expected_{expected_unit}", "computed_value": computed_val}
        return {"passed": False, "reason": f"derived_value_missing:{round(computed_val, 4)}{expected_unit}", "computed_value": computed_val}

    # Verify comparative direction if two named entities are compared in diff_pp / sub / rel_change_pct
    if op in {"diff_pp", "sub", "rel_change_pct"} and len(operands) >= 2:
        ent0 = str(operands[0].get("entity") or "").strip()
        ent1 = str(operands[1].get("entity") or "").strip()
        if ent0 and ent1 and not math.isclose(vals[0], vals[1], abs_tol=1e-9):
            hi, lo = (ent0, ent1) if vals[0] > vals[1] else (ent1, ent0)
            dir_ok, dir_reason = check_directed_relation(candidate_answer, higher_entity=hi, lower_entity=lo)
            if not dir_ok:
                return {"passed": False, "reason": dir_reason, "computed_value": computed_val}

    return {
        "passed": True,
        "reason": "ok",
        "computed_value": round(computed_val, 4),
        "expected_unit": expected_unit,
    }


# =====================================================================
# 4. Domain-Preserving Mathematical Expression AST Audit
# =====================================================================

_UNICODE_SUPER_TO_LATEX = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
})
_UNICODE_SUB_TO_LATEX = str.maketrans({
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
})

_REL_OP_MAP = {
    "\\leq": "<=", "\\le": "<=", "≤": "<=",
    "\\geq": ">=", "\\ge": ">=", "≥": ">=",
    "\\neq": "!=", "\\ne": "!=", "≠": "!=",
    "=": "=", "<": "<", ">": ">", "<=": "<=", ">=": ">=", "!=": "!=",
}
_FLIP_REL = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "=": "=", "!=": "!="}


_REL_OP_RE = re.compile(r"\\leq\b|\\le\b|≤|\\geq\b|\\ge\b|≥|\\neq\b|\\ne\b|≠|<=|>=|!=|=|<|>")


def _canonicalize_math_surface(expr: str) -> str:
    """
    Normalizes Unicode math symbols into explicit structural tokens while STRICTLY
    preserving `^` (superscript) vs `_` (subscript), `\frac{a}{b}`, and relation signs.
    """
    s = unicodedata.normalize("NFC", expr or "").strip()
    s = s.replace("$$", " ").replace("$", " ")
    s = re.sub(
        r"([⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
        lambda m: "^(" + m.group(1).translate(_UNICODE_SUPER_TO_LATEX) + ")",
        s,
    )
    s = re.sub(
        r"([₀₁₂₃₄₅₆₇₈₉]+)",
        lambda m: "_(" + m.group(1).translate(_UNICODE_SUB_TO_LATEX) + ")",
        s,
    )
    s = s.replace("−", "-").replace("×", "*").replace("⋅", "*").replace("\\cdot", "*").replace("\\times", "*")
    s = _REL_OP_RE.sub(lambda m: f" {_REL_OP_MAP[m.group(0)]} ", s)
    # Expand \frac{A}{B} iteratively (innermost first) into ((A)/(B))
    frac_re = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    while frac_re.search(s):
        s = frac_re.sub(r"((\1)/(\2))", s)
    # Normalize braces to parentheses for grouping after \frac expansion
    s = s.replace("{", "(").replace("}", ")")
    return " ".join(s.split())


@dataclass(frozen=True)
class MathNode:
    kind: str
    value: str = ""
    args: tuple["MathNode", ...] = ()


class _MathParser:
    """Recursive-descent parser producing a domain-preserving structural AST."""

    _TOKEN_RE = re.compile(
        r"\s*(<=|>=|!=|=|<|>|\+|-|\*|/|\^|_|,|\(|\)|\\[A-Za-z]+|[A-Za-zΑ-Ωα-ω]+|\d+(?:\.\d+)?)"
    )

    def __init__(self, text: str):
        norm = _canonicalize_math_surface(text)
        self.tokens = [m.group(1) for m in self._TOKEN_RE.finditer(norm)]
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: str | None = None) -> str:
        tok = self.peek()
        if tok is None or (expected is not None and tok != expected):
            raise ValueError(f"Expected {expected!r}, got {tok!r}")
        self.pos += 1
        return tok

    def parse_relation(self) -> MathNode:
        lhs = self.parse_add()
        if self.peek() in {"=", "!=", "<", ">", "<=", ">="}:
            op = self.consume()
            rhs = self.parse_add()
            return MathNode("Rel", op, (lhs, rhs))
        return lhs

    def parse_add(self) -> MathNode:
        node = self.parse_mul()
        while self.peek() in {"+", "-"}:
            op = self.consume()
            rhs = self.parse_mul()
            if op == "+":
                node = MathNode("Add", "+", (node, rhs))
            else:
                node = MathNode("Sub", "-", (node, rhs))
        return node

    def parse_mul(self) -> MathNode:
        node = self.parse_unary()
        while True:
            nxt = self.peek()
            if nxt in {"*", "/"}:
                op = self.consume()
                rhs = self.parse_unary()
                node = MathNode("Mul" if op == "*" else "Div", op, (node, rhs))
            elif nxt is not None and (
                nxt == "(" or re.match(r"^(?:\\[A-Za-z]+|[A-Za-zΑ-Ωα-ω]+)$", nxt)
            ):
                # Implicit multiplication (e.g. 2x or a(b+c)), unless preceding node is a function call handled in atom
                rhs = self.parse_unary()
                node = MathNode("Mul", "*", (node, rhs))
            else:
                break
        return node

    def parse_unary(self) -> MathNode:
        if self.peek() == "-":
            self.consume("-")
            return MathNode("Neg", "-", (self.parse_unary(),))
        if self.peek() == "+":
            self.consume("+")
            return self.parse_unary()
        return self.parse_postfix()

    def parse_postfix(self) -> MathNode:
        node = self.parse_atom()
        # Support chained/paired subscript and superscript (canonicalize Sub before Pow: x_i^2 == x^2_i)
        sub_node: MathNode | None = None
        pow_node: MathNode | None = None
        while self.peek() in {"_", "^"}:
            op = self.consume()
            arg = self.parse_atom()
            if op == "_":
                sub_node = arg
            else:
                pow_node = arg
        if sub_node is not None:
            node = MathNode("Subscript", "_", (node, sub_node))
        if pow_node is not None:
            node = MathNode("Pow", "^", (node, pow_node))
        return node

    def parse_atom(self) -> MathNode:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of math expression")
        if tok == "(":
            self.consume("(")
            inner = self.parse_relation()
            if self.peek() == ")":
                self.consume(")")
            return inner
        self.consume()
        if re.match(r"^\d+(?:\.\d+)?$", tok):
            num_str = str(float(tok)).rstrip("0").rstrip(".") if "." in tok else tok
            return MathNode("Num", num_str)
        # Check explicit function call f(x, y) for standard math functions
        if tok.lower() in {"sin", "cos", "tan", "exp", "log", "ln", "sqrt", "\\sqrt", "softmax", "max", "min"} and self.peek() == "(":
            self.consume("(")
            args = [self.parse_relation()]
            while self.peek() == ",":
                self.consume(",")
                args.append(self.parse_relation())
            if self.peek() == ")":
                self.consume(")")
            return MathNode("Func", tok.lstrip("\\").lower(), tuple(args))
        return MathNode("Sym", tok)


def _canonicalize_ast(node: MathNode, active_nonzero_syms: frozenset[str] = frozenset()) -> MathNode:
    """
    Canonicalizes commutative Add/Mul ordering and flipped inequalities WITHOUT performing
    domain-destroying algebraic cancellations unless explicitly licensed by `active_nonzero_syms`.
    """
    c_args = tuple(_canonicalize_ast(a, active_nonzero_syms) for a in node.args)
    if node.kind in {"Add", "Mul"}:
        flat: list[MathNode] = []
        for a in c_args:
            if a.kind == node.kind:
                flat.extend(a.args)
            else:
                flat.append(a)
        flat.sort(key=repr)
        return MathNode(node.kind, node.value, tuple(flat))
    if node.kind == "Div" and len(c_args) == 2:
        num, den = c_args
        # Domain-guarded cancellation: X / X -> 1 ONLY when X is explicitly declared non-zero!
        if num == den and num.kind == "Sym" and num.value in active_nonzero_syms:
            return MathNode("Num", "1")
        return MathNode("Div", "/", (num, den))
    if node.kind == "Rel" and len(c_args) == 2:
        lhs, rhs = c_args
        op = node.value
        if op in {">", ">="}:
            return MathNode("Rel", _FLIP_REL[op], (rhs, lhs))
        if op in {"=", "!="} and repr(rhs) < repr(lhs):
            return MathNode("Rel", op, (rhs, lhs))
        return MathNode("Rel", op, (lhs, rhs))
    return MathNode(node.kind, node.value, c_args)


def extract_domain_constraints(text: str) -> set[str]:
    """
    Extracts explicit mathematical domain conditions from text, canonicalized as e.g. `x!=0`, `t>=0`.
    """
    norm = _canonicalize_math_surface(text)
    constraints: set[str] = set()
    for m in re.finditer(r"\b([A-Za-zΑ-Ωα-ω])\s*(!=|<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)\b", norm):
        var, op, bound = m.group(1), m.group(2), m.group(3)
        if op == ">":
            constraints.add(f"{var}>{bound}")
        elif op == ">=":
            constraints.add(f"{var}>={bound}")
        elif op == "<":
            constraints.add(f"{var}<{bound}")
        elif op == "<=":
            constraints.add(f"{var}<={bound}")
        elif op == "!=":
            constraints.add(f"{var}!={bound}")
    # Natural-language Vietnamese/English non-zero conditions ("với x khác 0", "for x non-zero")
    for m in re.finditer(r"\b([A-Za-z])\s+(?:khác\s+0|is\s+non-?zero|not\s+equal\s+to\s+0|not\s+zero)\b", text or "", re.I):
        constraints.add(f"{m.group(1)}!=0")
    return constraints


def check_math_equivalence_with_domain(
    candidate_text: str,
    gold_formula: str,
    *,
    required_domain_constraints: Sequence[str] = (),
) -> tuple[bool, str]:
    """
    Verifies mathematical equivalence preserving superscripts (`x^2`) vs subscripts (`x_2`),
    inequality signs (`x < 3`, `y > 2`), fractions (`y = a/b`), and explicit domain conditions (`x != 0`).
    """
    req_constraints = {c.replace(" ", "") for c in required_domain_constraints if c and c.strip()}
    cand_constraints = extract_domain_constraints(candidate_text)
    missing_constraints = sorted(req_constraints - cand_constraints)
    if missing_constraints:
        return False, "missing_domain_constraint:" + ",".join(missing_constraints)

    nonzero_vars = frozenset(
        c.split("!=")[0] for c in cand_constraints if c.endswith("!=0")
    )

    try:
        gold_parts = [p.strip() for p in re.split(r"\band\b|\bvà\b|;", gold_formula) if p.strip()]
        gold_asts = [_canonicalize_ast(_MathParser(p).parse_relation(), nonzero_vars) for p in gold_parts]
    except Exception as exc:
        return False, f"invalid_gold_formula:{exc}"

    # Extract candidate math clauses (inline $...$, display $$...$$, or equation-bearing segments)
    _ALLOWED_MATH_WORDS = {
        "sin", "cos", "tan", "exp", "log", "ln", "sqrt", "softmax", "max", "min", "lim", "sum", "prod", "frac",
        "alpha", "beta", "gamma", "delta", "theta", "lambda", "pi", "sigma", "omega",
    }
    cand_segments: list[str] = []
    for m in re.finditer(r"\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$", candidate_text or ""):
        seg = (m.group(1) or m.group(2) or "").strip()
        if seg:
            cand_segments.append(seg)

    # Strip prose words (>= 2 letters not in _ALLOWED_MATH_WORDS and not preceded by backslash) to isolate bare formulas
    prose_stripped = re.sub(r"(?<!\\)\[\d{1,3}\]", " ", candidate_text or "")
    prose_clauses = re.split(r"\band\b|\bvà\b|\bwith\b|\bvới\b|\bwhen\b|\bkhi\b|\bfor\b|\bwhere\b|[.;,\n]", prose_stripped)
    for cl in prose_clauses:
        if not any(sym in cl for sym in ("=", "<", ">", "≤", "≥", "≠", "\\le", "\\ge", "\\ne", "^", "_", "/")):
            continue
        cleaned_cl = re.sub(
            r"(?<!\\)\b([A-Za-zÀ-ỹ]{2,})\b",
            lambda w: w.group(1) if w.group(1).lower() in _ALLOWED_MATH_WORDS else " ",
            cl,
        )
        cleaned_cl = " ".join(cleaned_cl.split()).strip()
        if cleaned_cl and any(op in cleaned_cl for op in ("=", "<", ">", "≤", "≥", "≠", "\\le", "\\ge", "\\ne")):
            cand_segments.append(cleaned_cl)

    cand_asts: list[MathNode] = []
    for seg in cand_segments:
        for sub_seg in re.split(r"\band\b|\bvà\b|;", seg):
            sub_clean = sub_seg.strip()
            if not sub_clean:
                continue
            try:
                parser = _MathParser(sub_clean)
                ast_node = parser.parse_relation()
                cand_asts.append(_canonicalize_ast(ast_node, nonzero_vars))
            except Exception:
                continue

    if not cand_asts:
        return False, "no_parseable_math_in_candidate"

    for g_ast in gold_asts:
        if g_ast not in cand_asts:
            return False, f"formula_mismatch:{g_ast}"

    return True, "ok"


# =====================================================================
# 5. Citation Region Bounding-Box IoU & Language Check
# =====================================================================

def compute_bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> tuple[float, float]:
    """
    Returns `(iou, containment_ratio)` for two `[x0, y0, x1, y1]` bounding boxes.
    """
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0, 0.0
    ax0, ay0, ax1, ay1 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
    bx0, by0, bx1, by1 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])
    if ax1 <= ax0 or ay1 <= ay0 or bx1 <= bx0 or by1 <= by0:
        return 0.0, 0.0

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0, 0.0

    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    containment = inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
    return round(iou, 4), round(containment, 4)


def audit_citation_regions(
    predicted_regions: Sequence[Mapping[str, Any]],
    gold_regions: Sequence[Mapping[str, Any]],
    *,
    min_iou: float = 0.30,
    min_containment: float = 0.70,
) -> dict[str, Any]:
    if not gold_regions:
        return {"passed": True, "region_recall": None, "region_precision": None}
    if not predicted_regions:
        return {"passed": False, "region_recall": 0.0, "region_precision": 0.0}

    def _doc_norm(v: Any) -> str:
        return str(v or "").replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".pdf").lower()

    matched_gold: set[int] = set()
    matched_pred: set[int] = set()

    for p_idx, pred in enumerate(predicted_regions):
        p_doc = pred.get("sha256") or _doc_norm(pred.get("document") or pred.get("doc_id"))
        p_page = int(pred.get("page") or pred.get("page_no") or 0)
        p_bbox = pred.get("bbox") or ()
        for g_idx, gold in enumerate(gold_regions):
            g_doc = gold.get("sha256") or _doc_norm(gold.get("document") or gold.get("doc_id"))
            g_page = int(gold.get("page") or gold.get("page_no") or 0)
            g_bbox = gold.get("bbox") or ()
            if p_doc != g_doc or p_page != g_page:
                continue
            if not g_bbox:
                matched_gold.add(g_idx)
                matched_pred.add(p_idx)
                continue
            if not p_bbox:
                continue
            iou, cont = compute_bbox_iou(p_bbox, g_bbox)
            if iou >= min_iou or cont >= min_containment:
                matched_gold.add(g_idx)
                matched_pred.add(p_idx)

    precision = len(matched_pred) / len(predicted_regions)
    recall = len(matched_gold) / len(gold_regions)
    return {
        "passed": recall == 1.0 and precision >= 1.0,
        "region_precision": round(precision, 4),
        "region_recall": round(recall, 4),
    }


_VI_DIACRITIC_RE = re.compile(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", re.I)
_VI_FUNCTION_WORDS_RE = re.compile(
    r"\b(?:không|chưa|trong|của|được|nhưng|hoặc|tài\s+liệu|bằng\s+chứng|kết\s+quả|trang|phương\s+pháp|"
    r"ket\s+qua|tai\s+lieu|bao\s+nhieu|trang\s+nao|khong\s+co)\b",
    re.I,
)
_EN_FUNCTION_WORDS_RE = re.compile(
    r"\b(?:the|was|were|are|is|not|from|with|according|evidence|document|reported|trial|method|higher|lower)\b",
    re.I,
)


def detect_response_language(text: str) -> str:
    stripped = re.sub(r"\$\$[\s\S]*?\$\$|\$[^$\n]*\$|`[^`\n]*`", " ", text or "")
    if _VI_DIACRITIC_RE.search(stripped) or _VI_FUNCTION_WORDS_RE.search(stripped):
        return "vi"
    if _EN_FUNCTION_WORDS_RE.search(stripped):
        return "en"
    return "unknown"


# =====================================================================
# 6. Layer 2: Cross-Lingual Atomic Claim Scorer & Thresholds
# =====================================================================

DEFAULT_NLI_THRESHOLDS: dict[str, float] = {
    "en->en": 0.50,
    "vi->vi": 0.50,
    "en->vi": 0.45,
    "vi->en": 0.45,
}


def expand_cross_lingual_tokens(text: str) -> set[str]:
    norm = normalize_word_numbers(unicodedata.normalize("NFC", text or "")).lower()
    expanded_phrases: list[str] = [norm]
    for src, targets in _EVAL_BILINGUAL_MAP.items():
        if re.search(r"(?<!\w)" + re.escape(src) + r"(?!\w)", norm):
            expanded_phrases.extend(targets)
        for tgt in targets:
            if re.search(r"(?<!\w)" + re.escape(tgt) + r"(?!\w)", norm):
                expanded_phrases.append(src)
                expanded_phrases.extend(targets)
    joined = " ".join(expanded_phrases)
    raw_tokens = {w for w in re.findall(r"[a-z0-9à-ỹ]+", joined) if len(w) >= 2 or w.isdigit()}
    return expand_bilingual_tokens(joined, raw_tokens) | raw_tokens


def score_claim_support_offline(candidate_answer: str, atomic_claim: str) -> float:
    """
    Deterministic cross-lingual entailment score in `[0.0, 1.0]` used when no external NLI
    classifier is injected: enforces negation alignment, directed entity order, quantity+unit
    preservation, and bilingual content coverage.
    """
    if not (candidate_answer or "").strip() or not (atomic_claim or "").strip():
        return 0.0
    pol_ok, _ = check_polarity_alignment(candidate_answer, atomic_claim)
    if not pol_ok:
        return 0.0

    claim_pairs = extract_canonical_greater_pairs(atomic_claim)
    if claim_pairs:
        cand_pairs = extract_canonical_greater_pairs(candidate_answer)
        for hi, lo in claim_pairs:
            if (lo, hi) in cand_pairs or (hi, lo) not in cand_pairs:
                return 0.0

    claim_quants = extract_quantities(atomic_claim)
    if claim_quants:
        cand_quants = extract_quantities(candidate_answer)
        for c_val, c_unit, _ in claim_quants:
            if not any(
                math.isclose(q_val, c_val, abs_tol=0.05) and (not c_unit or q_unit == c_unit)
                for q_val, q_unit, _ in cand_quants
            ):
                return 0.0

    stopwords = {
        "the", "and", "for", "with", "that", "this", "was", "were", "are", "is", "in", "on", "of", "to", "by",
        "của", "và", "cho", "với", "trong", "trên", "các", "những", "được", "là", "có", "từ", "theo", "đã",
    }
    claim_raw = {
        w for w in re.findall(r"[a-z0-9à-ỹ]+", normalize_word_numbers(atomic_claim).lower())
        if (len(w) >= 2 or w.isdigit()) and w not in stopwords
    }
    if not claim_raw:
        return 1.0
    cand_expanded = expand_cross_lingual_tokens(candidate_answer) - stopwords
    matched = len(claim_raw & cand_expanded)
    return round(matched / len(claim_raw), 4)


def evaluate_layer2_nli(
    candidate_answer: str,
    atomic_claims: Sequence[str | Mapping[str, Any]],
    *,
    source_language: str = "en",
    answer_language: str = "vi",
    nli_scorer: Callable[[list[tuple[str, str]]], list[float]] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    claims_text = [
        str(c.get("text") or c.get("claim") or "").strip() if isinstance(c, Mapping) else str(c).strip()
        for c in (atomic_claims or ())
    ]
    claims_text = [c for c in claims_text if c]
    if not claims_text:
        return {"claim_recall": None, "supported_claims": 0, "total_claims": 0, "scores": []}

    direction = f"{source_language.lower()}->{answer_language.lower()}"
    active_thresholds = {**DEFAULT_NLI_THRESHOLDS, **(thresholds or {})}
    cutoff = active_thresholds.get(direction, 0.50)

    if nli_scorer is not None:
        pairs = [(candidate_answer, cl) for cl in claims_text]
        raw_scores = [float(s) for s in nli_scorer(pairs)]
    else:
        raw_scores = [score_claim_support_offline(candidate_answer, cl) for cl in claims_text]

    # Even when an external neural NLI scorer is supplied, polarity & directed-relation inversions veto false positives
    final_scores: list[float] = []
    supported = 0
    for cl, sc in zip(claims_text, raw_scores):
        pol_ok, _ = check_polarity_alignment(candidate_answer, cl)
        rel_pairs = extract_canonical_greater_pairs(cl)
        rel_ok = True
        if rel_pairs:
            cand_pairs = extract_canonical_greater_pairs(candidate_answer)
            rel_ok = all((hi, lo) in cand_pairs and (lo, hi) not in cand_pairs for hi, lo in rel_pairs)
        adj = sc if (pol_ok and rel_ok) else 0.0
        final_scores.append(round(adj, 4))
        if adj >= cutoff:
            supported += 1

    return {
        "direction": direction,
        "threshold": cutoff,
        "supported_claims": supported,
        "total_claims": len(claims_text),
        "claim_recall": round(supported / len(claims_text), 4),
        "scores": final_scores,
    }


# =====================================================================
# 7. Layer 3: Human Calibration, Cohen's Kappa & Clopper-Pearson Bounds
# =====================================================================

def compute_cohens_kappa(labels_a: Sequence[int | bool], labels_b: Sequence[int | bool]) -> float:
    """Compute Cohen's kappa agreement between two binary human annotators."""
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("Annotator label sequences must be non-empty and equal length")
    n = len(labels_a)
    a_bin = [int(bool(x)) for x in labels_a]
    b_bin = [int(bool(x)) for x in labels_b]
    p_o = sum(1 for x, y in zip(a_bin, b_bin) if x == y) / n
    p_a1 = sum(a_bin) / n
    p_b1 = sum(b_bin) / n
    p_e = p_a1 * p_b1 + (1.0 - p_a1) * (1.0 - p_b1)
    if math.isclose(1.0 - p_e, 0.0, abs_tol=1e-12):
        return 1.0
    return round((p_o - p_e) / (1.0 - p_e), 4)


def calibrate_nli_thresholds(
    calibration_rows: Sequence[Mapping[str, Any]],
    *,
    max_false_positive_rate: float = 0.05,
) -> dict[str, float]:
    """
    Calibrate Layer-2 NLI thresholds per `direction` (`en->vi`, `vi->en`, `vi->vi`, `en->en`)
    using adjudicated bilingual human binary labels (`human_label` in {0, 1} and `nli_score` in [0, 1]),
    maximizing F1 subject to `FPR <= max_false_positive_rate`.
    """
    by_dir: dict[str, list[tuple[float, int]]] = {}
    for row in calibration_rows:
        direction = str(row.get("direction") or "en->vi").lower()
        score = float(row["nli_score"])
        label = int(bool(row["human_label"]))
        by_dir.setdefault(direction, []).append((score, label))

    calibrated = dict(DEFAULT_NLI_THRESHOLDS)
    for direction, items in by_dir.items():
        positives = sum(lbl for _, lbl in items)
        negatives = len(items) - positives
        candidates = sorted({round(sc, 4) for sc, _ in items} | {0.35, 0.45, 0.50, 0.60, 0.70, 0.80})
        best_thr = calibrated.get(direction, 0.50)
        best_f1 = -1.0
        for thr in candidates:
            tp = sum(1 for sc, lbl in items if sc >= thr and lbl == 1)
            fp = sum(1 for sc, lbl in items if sc >= thr and lbl == 0)
            fn = positives - tp
            fpr = (fp / negatives) if negatives > 0 else 0.0
            if negatives > 0 and fpr > max_false_positive_rate:
                continue
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        calibrated[direction] = round(best_thr, 4)
    return calibrated


def clopper_pearson_lower_bound(successes: int, trials: int, *, alpha: float = 0.05) -> float:
    """
    Exact one-sided `(1 - alpha)` Clopper-Pearson lower bound on binomial success probability.
    Note: for `60/60` at `alpha=0.05`, closed-form exact bound is `0.05**(1/60) ≈ 0.9513` (95.1%).
    Uses bisection on the exact binomial CDF for `0 < successes < trials`.
    """
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Invalid binomial counts")
    if successes == 0:
        return 0.0
    if successes == trials:
        return round(alpha ** (1.0 / trials), 4)

    def _upper_tail(p: float) -> float:
        # P(X >= successes | n=trials, p)
        total = 0.0
        for i in range(successes, trials + 1):
            total += math.comb(trials, i) * (p ** i) * ((1.0 - p) ** (trials - i))
        return total

    lo, hi = 0.0, 1.0
    for _ in range(52):
        mid = 0.5 * (lo + hi)
        if _upper_tail(mid) > alpha:
            hi = mid
        else:
            lo = mid
    return round(0.5 * (lo + hi), 4)


def summarize_blind_pairwise(comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate blind pairwise judgments (`winner` in `{'a', 'b', 'tie'}`) per dimension."""
    dims = ("accuracy", "completeness", "fluency", "citation_quality")
    out: dict[str, Any] = {"total_pairs": len(comparisons), "dimensions": {}}
    for dim in dims:
        votes = [str(row.get(dim) or "").strip().lower() for row in comparisons if row.get(dim)]
        if not votes:
            continue
        n = len(votes)
        wins_a = sum(1 for v in votes if v == "a")
        wins_b = sum(1 for v in votes if v == "b")
        ties = sum(1 for v in votes if v == "tie")
        out["dimensions"][dim] = {
            "count": n,
            "win_rate_a": round(wins_a / n, 4),
            "win_rate_b": round(wins_b / n, 4),
            "tie_rate": round(ties / n, 4),
            "non_loss_lower_95_a": clopper_pearson_lower_bound(wins_a + ties, n),
        }
    return out


# =====================================================================
# 8. Unified Three-Layer Case Evaluation Entrypoint
# =====================================================================

def evaluate_three_layer_case(
    case: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    nli_scorer: Callable[[list[tuple[str, str]]], list[float]] | None = None,
    calibrated_thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """
    Runs Layer 1 (deterministic structural/numeric/derived/math/region/language) +
    Layer 2 (cross-lingual atomic claim scoring under Layer-3 calibrated thresholds)
    on a single benchmark case and returns granular verdicts.
    """
    report_data = prediction.get("written_report_data") or {}
    answer = str(
        report_data.get("answer")
        or prediction.get("answer")
        or prediction.get("final_report")
        or ""
    ).strip()
    answer_status = prediction.get("answer_status") or report_data.get("status")
    answerability = str(
        case.get("answerability")
        or ("unanswerable" if case.get("is_unanswerable") else "answerable")
    )

    abstention_audit = audit_abstention_and_display(
        answer=answer,
        answer_status=answer_status,
        answerability=answerability,
        forbidden_entities=case.get("forbidden_entities") or (),
        forbidden_numbers=case.get("forbidden_numbers") or (),
    )

    failures: list[str] = []
    if not abstention_audit["passed"]:
        failures.append(
            "status_display_mismatch" if abstention_audit["status_display_mismatch"] else "abstention_or_answerability_failed"
        )

    # Language check (when answer_language is specified)
    expected_lang = str(case.get("answer_language") or case.get("language") or "").strip().lower()
    detected_lang = detect_response_language(answer)
    lang_ok = True
    if expected_lang in {"vi", "en"} and answer:
        lang_ok = detected_lang in {expected_lang, "unknown"}
        if detected_lang in {"vi", "en"} and detected_lang != expected_lang:
            lang_ok = False
            failures.append(f"wrong_language:{detected_lang}_expected_{expected_lang}")

    # For unanswerable / ambiguous cases, abstention_audit + language_ok complete the verdict
    if answerability.lower() in {"unanswerable", "insufficient_evidence", "adversarial_trap", "ambiguous"}:
        return {
            "passed": len(failures) == 0,
            "failures": failures,
            "abstention_audit": abstention_audit,
            "language_ok": lang_ok,
            "detected_language": detected_lang,
            "derived_audit": None,
            "math_audit": None,
            "region_audit": None,
            "layer2_nli": None,
        }

    # Polarity check against gold_answer if present
    gold_answer = str(case.get("gold_answer") or "").strip()
    if gold_answer:
        pol_ok, pol_reason = check_polarity_alignment(answer, gold_answer)
        if not pol_ok:
            failures.append(pol_reason)

    # Directed relation check if specified
    if isinstance(case.get("directed_relation"), Mapping):
        rel = case["directed_relation"]
        rel_ok, rel_reason = check_directed_relation(
            answer,
            higher_entity=str(rel["higher"]),
            lower_entity=str(rel["lower"]),
        )
        if not rel_ok:
            failures.append(rel_reason)

    # Exact quantity + unit check if `gold_quantities` is specified
    for gq in case.get("gold_quantities") or ():
        g_val = float(gq["value"])
        g_unit = canonicalize_unit(gq.get("unit"))
        tol = float(gq.get("tolerance", 0.02))
        cand_qs = extract_quantities(answer)
        if not any(math.isclose(v, g_val, abs_tol=tol) and u == g_unit for v, u, _ in cand_qs):
            failures.append(f"quantity_or_unit_mismatch:{g_val}{g_unit}")

    # Derived calculation check
    derived_audit = None
    if isinstance(case.get("derivation_spec"), Mapping):
        derived_audit = verify_derived_calculation(
            answer,
            case["derivation_spec"],
            source_passages=prediction.get("source_units") or case.get("gold_sources"),
        )
        if not derived_audit["passed"]:
            failures.append(str(derived_audit["reason"]))

    # Domain-preserving math formula check
    math_audit = None
    if case.get("gold_formula"):
        m_ok, m_reason = check_math_equivalence_with_domain(
            answer,
            str(case["gold_formula"]),
            required_domain_constraints=case.get("domain_constraints") or (),
        )
        math_audit = {"passed": m_ok, "reason": m_reason}
        if not m_ok:
            failures.append(m_reason)

    # Citation region IoU check
    region_audit = None
    if case.get("gold_regions"):
        pred_regions = prediction.get("cited_regions") or (report_data.get("references") or [])
        region_audit = audit_citation_regions(pred_regions, case["gold_regions"])
        if not region_audit["passed"]:
            failures.append("citation_region_mismatch")

    # Layer 2 NLI atomic claim check
    layer2 = None
    if case.get("atomic_claims"):
        layer2 = evaluate_layer2_nli(
            answer,
            case["atomic_claims"],
            source_language=str(case.get("source_language") or "en"),
            answer_language=expected_lang or "vi",
            nli_scorer=nli_scorer,
            thresholds=calibrated_thresholds,
        )
        if (layer2.get("claim_recall") or 0.0) < 1.0:
            failures.append(f"atomic_claim_recall_below_1:{layer2['claim_recall']}")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "abstention_audit": abstention_audit,
        "language_ok": lang_ok,
        "detected_language": detected_lang,
        "derived_audit": derived_audit,
        "math_audit": math_audit,
        "region_audit": region_audit,
        "layer2_nli": layer2,
    }
