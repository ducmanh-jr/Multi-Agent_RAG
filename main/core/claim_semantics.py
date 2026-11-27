"""
Deterministic Semantic, Mathematical, Negation, Comparative-Direction & Derived-Claim Guards.

Shared across Reader, Summarizer, FactChecker (`verifier.py`, `judge.py`), Citation Audit,
and Post-Writer Verification (`write_with_citation_audit`) so every execution mode (`cpu_min`,
`cpu_max`, `gemini`, `gpu_finetune`, `gpu_max`, `hybrid`) enforces identical zero-hallucination
invariants:
  1. Safe HTML tag removal that NEVER strips `<` or `>` mathematical inequalities (`x < 3 and y > 2`).
  2. Structural math normalization preserving superscripts (`^`) vs subscripts (`_`), signs,
     inequalities (`<`, `>`, `<=`, `>=`, `!=`), and domain constraints (`x != 0`, `t >= 0`).
  3. Enclosing-sentence negation/polarity, conditional framing, universal quantifier, and
     directed comparative relation (`A > B` vs `B > A`) verification (EN + VI).
  4. Deterministic `derived_claim` operand & arithmetic verification (`92% - 81% = 11 pp`,
     distinguishing percentage points `pp` from relative change `%`, and blocking cross-split subtraction).
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Mapping, Optional, Sequence

from utils.bilingual import expand_bilingual_tokens

# =====================================================================
# 1. Safe HTML Tag Stripping & Structural Math Normalization
# =====================================================================

_KNOWN_HTML_TAG_RE = re.compile(
    r"</?(?:br|p|div|span|b|i|u|em|strong|table|thead|tbody|tfoot|tr|td|th|"
    r"ul|ol|li|a|img|code|pre|blockquote|h[1-6]|hr|font|small|big|mark|del|ins|"
    r"caption|section|article|header|footer|nav|main|figure|figcaption|think)"
    r"(?:\s+[^<>]{0,200})?/?>",
    re.IGNORECASE,
)
_HTML_SUP_RE = re.compile(r"<sup(?:\s+[^<>]{0,80})?>(.*?)</sup>", re.IGNORECASE | re.DOTALL)
_HTML_SUB_RE = re.compile(r"<sub(?:\s+[^<>]{0,80})?>(.*?)</sub>", re.IGNORECASE | re.DOTALL)

_UNICODE_SUPER_MAP = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁼": "=", "⁽": "(", "⁾": ")",
    "ⁿ": "n", "ⁱ": "i",
})
_UNICODE_SUB_MAP = str.maketrans({
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "₊": "+", "₋": "-", "₌": "=", "₍": "(", "₎": ")",
    "ₐ": "a", "ₑ": "e", "ₕ": "h", "ᵢ": "i", "ⱼ": "j",
    "ₖ": "k", "ₗ": "l", "ₘ": "m", "ₙ": "n", "ₒ": "o",
    "ₚ": "p", "ᵣ": "r", "ₛ": "s", "ₜ": "t", "ᵤ": "u",
    "ᵥ": "v", "ₓ": "x",
})
_UNICODE_SUPER_RE = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ]+")
_UNICODE_SUB_RE = re.compile(r"[₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ]+")


def strip_html_tags_preserve_math(text: str) -> str:
    """
    Removes genuine HTML tags (`<br>`, `<p>`, `<div>`, etc.) while converting `<sup>`/`<sub>`
    to explicit `^{...}`/`_{...}` and NEVER deleting mathematical inequalities like `x < 3 and y > 2`.
    """
    if not text:
        return ""
    s = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    s = _HTML_SUP_RE.sub(lambda m: f"^{{{m.group(1).strip()}}}", s)
    s = _HTML_SUB_RE.sub(lambda m: f"_{{{m.group(1).strip()}}}", s)
    s = _KNOWN_HTML_TAG_RE.sub(" ", s)
    return s


def expand_unicode_scripts_preserving_role(text: str) -> str:
    """
    Expands Unicode superscripts (`x²` -> `x^2`) and subscripts (`x₂` -> `x_2`) BEFORE any
    NFKC normalization so `^` and `_` are never conflated.
    """
    if not text:
        return ""
    s = _UNICODE_SUPER_RE.sub(
        lambda m: "^" + m.group(0).translate(_UNICODE_SUPER_MAP),
        text,
    )
    s = _UNICODE_SUB_RE.sub(
        lambda m: "_" + m.group(0).translate(_UNICODE_SUB_MAP),
        s,
    )
    return s


def normalize_math_preserving_structure(text: str) -> str:
    """
    Normalizes PDF ligatures (`ﬁ` -> `fi`), Unicode math operators (`−` -> `-`, `×` -> `*`,
    `≤` -> `<=`, `≥` -> `>=`, `≠` -> `!=`), and braces (`^{2}` -> `^2`, `_{2}` -> `_2`)
    while strictly keeping superscripts (`^`) and subscripts (`_`) distinct.
    """
    if not text:
        return ""
    expanded = expand_unicode_scripts_preserving_role(text)
    nfkc = unicodedata.normalize("NFKC", expanded)
    nfkc = (
        nfkc.replace("−", "-")
        .replace("–", "-")
        .replace("×", "*")
        .replace("⋅", "*")
        .replace("∗", "*")
        .replace("\\cdot", "*")
        .replace("\\times", "*")
        .replace("\\leq", "<=")
        .replace("\\le", "<=")
        .replace("≤", "<=")
        .replace("\\geq", ">=")
        .replace("\\ge", ">=")
        .replace("≥", ">=")
        .replace("\\neq", "!=")
        .replace("\\ne", "!=")
        .replace("≠", "!=")
    )
    # Normalize single-token braces x^{2} -> x^2 and x_{2} -> x_2 while keeping ^ vs _
    nfkc = re.sub(r"\^\s*\{\s*([A-Za-z0-9+-]+)\s*\}", r"^\1", nfkc)
    nfkc = re.sub(r"_\s*\{\s*([A-Za-z0-9+-]+)\s*\}", r"_\1", nfkc)
    nfkc = re.sub(r"([\\^_])\s+([A-Za-z0-9])", r"\1\2", nfkc)
    return " ".join(nfkc.split())


_SCRIPT_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-zΑ-Ωα-ω]+)\s*([\^_])\s*\{?\s*([A-Za-z0-9+-]+)\s*\}?")
_DOMAIN_CONSTRAINT_RE = re.compile(
    r"(?:\b(?:for|with|where|when|provided|given|assuming|if|với|khi|điều\s+kiện)\s+"
    r"[A-Za-zΑ-Ωα-ω0-9_]+\s*(?:!=|≠|\\neq|\\ne|>|<|>=|≤|≥|<=|\bkhác\s+0\b|\bnon-?zero\b))"
    r"|(?:,\s*[A-Za-zΑ-Ωα-ω0-9_]+\s*(?:!=|≠|\\neq|\\ne)\s*0\b)",
    re.IGNORECASE,
)
_UNCONDITIONAL_DOMAIN_CLAIM_RE = re.compile(
    r"\b(?:for\s+all|for\s+every|for\s+any|với\s+mọi|cho\s+mọi|luôn\s+đúng\s+với\s+mọi)\b",
    re.IGNORECASE,
)
_RELATION_EXPR_RE = re.compile(
    r"(?<![A-Za-z0-9_\\^])([A-Za-zΑ-Ωα-ω](?:[\^_][A-Za-z0-9]+)?)\s*(<=|>=|!=|=|<|>)\s*([−+-]?\d+(?:\.\d+)?)(?![A-Za-z0-9_])"
)


def check_math_and_domain_consistency(claim_text: str, evidence_text: str) -> tuple[bool, str]:
    """
    Verifies mathematical formulas, subscripts vs superscripts (`x_2` vs `x^2`),
    inequality directions (`x < 3` vs `x > 3`), signs (`-4` vs `4`), and explicit
    domain constraints (`x / x = 1 for x != 0`).
    """
    if not claim_text or not evidence_text:
        return True, "ok"

    c_norm = normalize_math_preserving_structure(claim_text.replace("$$", " ").replace("$", " "))
    e_norm = normalize_math_preserving_structure(evidence_text.replace("$$", " ").replace("$", " "))

    # 1. Subscript vs superscript role swap on the same base variable & index (e.g., x_2 vs x^2)
    c_scripts = {(m.group(1), m.group(2), m.group(3)) for m in _SCRIPT_TOKEN_RE.finditer(c_norm)}
    e_scripts = {(m.group(1), m.group(2), m.group(3)) for m in _SCRIPT_TOKEN_RE.finditer(e_norm)}
    for base, role, idx in c_scripts:
        if (base, role, idx) in e_scripts:
            continue
        opposite = "_" if role == "^" else "^"
        if (base, opposite, idx) in e_scripts:
            return False, f"math_subscript_superscript_mismatch:{base}{role}{idx}_vs_{base}{opposite}{idx}"

    # 2. Variable relation / inequality / sign conflict (e.g., x > 3 when source says x < 3, or x = -4 vs x = 4)
    c_rels = [(m.group(1), m.group(2), float(m.group(3))) for m in _RELATION_EXPR_RE.finditer(c_norm)]
    e_rels = [(m.group(1), m.group(2), float(m.group(3))) for m in _RELATION_EXPR_RE.finditer(e_norm)]
    if c_rels and e_rels:
        for c_var, c_op, c_val in c_rels:
            same_var = [(e_op, e_val) for e_var, e_op, e_val in e_rels if e_var == c_var]
            if not same_var:
                continue
            # Direct sign flip on same variable & magnitude (e.g. x = 4 vs x = -4)
            if any(
                e_op == c_op and abs(c_val) > 0 and math.isclose(c_val, -e_val, abs_tol=1e-9)
                for e_op, e_val in same_var
            ):
                return False, f"math_sign_mismatch:{c_var}{c_op}{c_val}"
            # Direct inequality reversal on same variable & bound (e.g. x > 3 vs x < 3)
            same_bound = [e_op for e_op, e_val in same_var if math.isclose(c_val, e_val, abs_tol=1e-9)]
            if same_bound and c_op not in same_bound:
                incompatible = {
                    "<": {">", ">=", "="},
                    "<=": {">"},
                    ">": {"<", "<=", "="},
                    ">=": {"<"},
                    "=": {"!=", "<", ">"},
                    "!=": {"="},
                }
                if any(e_op in incompatible.get(c_op, set()) for e_op in same_bound):
                    return False, f"math_operator_mismatch:{c_var}{c_op}{c_val}"

    # 3. Domain constraint preservation on rational/singular identities (e.g., x / x = 1 requires x != 0)
    has_rational_or_log = bool(re.search(r"(?:/|\\frac|\blog\b|\bln\b|\bsqrt\b|√)", c_norm)) and "=" in c_norm
    if has_rational_or_log:
        matching_sents = _find_matching_sentences(claim_text, evidence_text)
        for sent in matching_sents:
            if _DOMAIN_CONSTRAINT_RE.search(sent):
                if _UNCONDITIONAL_DOMAIN_CLAIM_RE.search(claim_text) or not _DOMAIN_CONSTRAINT_RE.search(claim_text):
                    return False, "missing_domain_constraint"

    return True, "ok"


# =====================================================================
# 2. Negation, Conditional Framing, Quantifiers & Directed Relations
# =====================================================================

_MASK_NON_NEGATION_RE = re.compile(
    r"\b(?:"
    r"không\s+(?:khí|gian|hàng|lực|phận|độ|chỉ|những)|chân\s+không|"
    r"not\s+only|non-?zero|khác\s+0|"
    r"(?:khi|nếu|nhánh|trường\s+hợp|if|when|on|branch)\s+['\"`]?(?:no|false|sai|không|khong|fail|invalid|reject)"
    r")\b"
    r"|\[condition:\s*[^\[\]]{1,32}\]"
    r"|\[(?:no|false|sai|không|khong|fail|invalid|reject)\]"
    r"|(?:!=|≠|\\neq|\\ne)\s*0\b",
    re.IGNORECASE,
)

_NEGATION_CUE_RE = re.compile(
    r"\b(?:"
    r"it\s+is\s+(?:false|not\s+true|untrue|incorrect)\s+that|"
    r"false\s+that|not\s+true\s+that|untrue\s+that|contrary\s+to\s+the\s+(?:claim|belief|view)\s+that|"
    r"no\s+evidence\s+that|refutes?\s+that|disproves?\s+that|"
    r"not|no|never|neither|nor|cannot|can't|couldn't|wouldn't|shouldn't|won't|"
    r"didn't|doesn't|don't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|"
    r"without|lack|lacks|lacking|fail|fails|failed|unable|ineffective|insignificant|"
    r"sai\s+khi\s+(?:nói|cho)\s+rằng|không\s+(?:đúng|phải)(?:\s+là|\s+rằng)?|"
    r"bác\s+bỏ(?:\s+việc|\s+rằng)?|phủ\s+nhận|không|chưa|chẳng|vô\s+hiệu|thất\s+bại"
    r")\b",
    re.IGNORECASE,
)

_CONDITIONAL_FRAMING_RE = re.compile(
    r"(?:^|[,;]\s*)\b(?:"
    r"if|unless|whether|assuming\s+that|suppose\s+that|hypothesize\s+that|in\s+case|"
    r"nếu|trừ\s+khi|giả\s+sử|giả\s+định\s+rằng|liệu\s+rằng|liệu"
    r")\b",
    re.IGNORECASE,
)

_UNIVERSAL_QUANTIFIER_RE = re.compile(
    r"\b(?:all|every|always|universally|in\s+all\s+cases|tất\s+cả|mọi|luôn\s+luôn|trong\s+mọi\s+trường\s+hợp)\b",
    re.IGNORECASE,
)
_EXISTENTIAL_HEDGE_RE = re.compile(
    r"\b(?:some|sometimes|occasionally|in\s+one\s+case|certain|một\s+số|đôi\s+khi|thỉnh\s+thoảng|vài)\b",
    re.IGNORECASE,
)

_NON_ENTITY_MODIFIERS = frozenset({
    "cao", "thấp", "lớn", "nhỏ", "nhanh", "chậm", "tốt", "kém", "ít", "nhiều", "mạnh", "yếu",
    "hơn", "nhất", "bằng", "vượt", "kết", "quả", "giá", "trị", "tỷ", "lệ", "điểm", "mức",
    "higher", "lower", "greater", "smaller", "faster", "slower", "better", "worse", "less", "more",
    "than", "is", "was", "are", "were", "the", "and", "or", "by", "in", "on", "at", "of",
    "theo", "trong", "với", "cho", "của", "đạt", "có",
})

_ENTITY_PHRASE = (
    r"(?<!\w)(?:(?:phương\s+pháp|mo\s+hinh|mô\s+hình|phuong\s+phap|nhóm|nhom|hệ\s+thống|nghiệm\s+thức|"
    r"method|model|group|system|variant|config|baseline|treatment|approach)\s+)?"
    r"[A-Za-z0-9][A-Za-z0-9._-]*(?!\w)"
)

_GREATER_REL_RE = re.compile(
    rf"(?P<lhs>{_ENTITY_PHRASE})\s+(?:cao\s+hơn|cao\s+hon|lớn\s+hơn|lon\s+hon|nhanh\s+hơn|nhanh\s+hon|"
    rf"tốt\s+hơn|tot\s+hon|mạnh\s+hơn|vượt(?:\s+trội)?(?:\s+hơn)?|vuot(?:\s+troi)?(?:\s+hon)?|"
    rf"(?<!thấp\s)(?<!thap\s)(?<!nhỏ\s)(?<!nho\s)(?<!chậm\s)(?<!cham\s)(?<!kém\s)(?<!kem\s)(?<!ít\s)(?<!it\s)(?<!yếu\s)hơn|"
    rf"outperforms?|exceeds?|surpasses?|beats?|"
    rf"(?:is|was|are|were)\s+(?:higher|greater|larger|faster|better|superior)\s+than)\s+(?P<rhs>{_ENTITY_PHRASE})",
    re.IGNORECASE,
)
_LESS_REL_RE = re.compile(
    rf"(?P<lhs>{_ENTITY_PHRASE})\s+(?:thấp\s+hơn|thap\s+hon|nhỏ\s+hơn|nho\s+hon|chậm\s+hơn|cham\s+hon|"
    rf"kém\s+hơn|kem\s+hon|ít\s+hơn|it\s+hon|yếu\s+hơn|"
    rf"underperforms?|lags?\s+behind|"
    rf"(?:is|was|are|were)\s+(?:lower|smaller|slower|worse|less|inferior)\s+than)\s+(?P<rhs>{_ENTITY_PHRASE})",
    re.IGNORECASE,
)


def has_clause_negation(text: str) -> bool:
    if not text:
        return False
    masked = _MASK_NON_NEGATION_RE.sub(" ", unicodedata.normalize("NFC", text))
    return bool(_NEGATION_CUE_RE.search(masked))


def normalize_entity_key(raw: str) -> str:
    cleaned = re.sub(
        r"^(?:phương\s+pháp|phuong\s+phap|mô\s+hình|mo\s+hinh|nhóm|nhom|hệ\s+thống|nghiệm\s+thức|"
        r"method|model|group|system|variant|config|baseline|treatment|approach)\s+",
        "",
        (raw or "").strip(),
        flags=re.IGNORECASE,
    )
    norm = cleaned.strip(".,;:()[] ").lower()
    return "" if norm in _NON_ENTITY_MODIFIERS else norm


def extract_directed_entity_pairs(text: str) -> list[tuple[str, str]]:
    """Returns canonical `(higher_or_superior_entity, lower_or_inferior_entity)` pairs."""
    norm = unicodedata.normalize("NFC", text or "")
    if has_clause_negation(norm):
        return []
    pairs: list[tuple[str, str]] = []
    for m in _GREATER_REL_RE.finditer(norm):
        lhs = normalize_entity_key(m.group("lhs"))
        rhs = normalize_entity_key(m.group("rhs"))
        if lhs and rhs and lhs != rhs:
            pairs.append((lhs, rhs))
    for m in _LESS_REL_RE.finditer(norm):
        lhs = normalize_entity_key(m.group("lhs"))
        rhs = normalize_entity_key(m.group("rhs"))
        if lhs and rhs and lhs != rhs:
            pairs.append((rhs, lhs))
    return list(dict.fromkeys(pairs))


def _split_source_sentences(source_text: str) -> list[str]:
    if not source_text:
        return []
    parts = re.split(
        r"(?<!\bEq\.)(?<!\bFig\.)(?<!\bTab\.)(?<!\bSec\.)(?<!\bal\.)(?<!\bvs\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<=[.!?])\s+|\n+",
        source_text,
    )
    return [p.strip() for p in parts if p and p.strip()]


def _find_matching_sentences(claim_text: str, source_text: str) -> list[str]:
    sents = _split_source_sentences(source_text)
    if not sents:
        return [source_text] if source_text else []
    c_clean = re.sub(r"\[\d+\]", " ", claim_text or "").strip().lower().rstrip(".,;:!?")
    if not c_clean:
        return sents[:1]
    exact_sents = [s for s in sents if c_clean in s.lower()]
    if exact_sents:
        return exact_sents
    c_words = {w for w in re.findall(r"[a-zA-Z0-9À-ỹ]+", c_clean) if len(w) >= 3}
    c_expanded = expand_bilingual_tokens(claim_text, c_words)
    scored: list[tuple[float, str]] = []
    for s in sents:
        s_words = {w for w in re.findall(r"[a-zA-Z0-9À-ỹ]+", s.lower()) if len(w) >= 3}
        overlap = len(c_expanded & s_words)
        if overlap > 0:
            scored.append((overlap / max(len(c_words), 1), s))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [s for _, s in scored[:2]] if scored else sents[:1]


def check_sentence_entailment_guards(
    claim_text: str,
    source_text: str,
    *,
    qualitative_comparatives: bool = True,
) -> tuple[bool, str, str]:
    """
    Checks sentence-level semantic invariants that substring/bag-of-words matchers miss:
      1. Polarity / matrix negation (`"It is false that treatment works."` vs `"Treatment works."`).
      2. Counterfactual / conditional framing (`"If treatment works..."` vs `"Treatment works."`).
      3. Existential -> universal quantifier inflation (`"Some samples..."` vs `"All samples..."`).
      4. Directed comparative subject-object reversal (`"Model A exceeds Model B"` vs `"Model B exceeds Model A"`).

    Returns `(passed, failure_verdict, reason)`.
    """
    if not claim_text or not source_text:
        return True, "SUPPORT", "ok"

    # Exemption: explicit meta-refusal claims ("not mentioned in document")
    c_low = claim_text.lower()
    if any(
        neg in c_low
        for neg in (
            "không được đề cập", "không đề cập", "không tìm thấy", "chưa đủ bằng chứng",
            "not mentioned", "does not mention", "insufficient evidence",
        )
    ):
        return True, "SUPPORT", "ok"

    matching_sents = _find_matching_sentences(claim_text, source_text)
    claim_neg = has_clause_negation(claim_text)
    claim_cond = bool(_CONDITIONAL_FRAMING_RE.search(claim_text))

    for sent in matching_sents:
        sent_neg = has_clause_negation(sent)
        if claim_neg != sent_neg:
            # Verify they actually share the core predicate/subject tokens (not two unrelated clauses)
            c_toks = {w for w in re.findall(r"[a-zA-Z0-9À-ỹ]+", c_low) if len(w) >= 3}
            c_exp = expand_bilingual_tokens(claim_text, c_toks)
            s_toks = {w for w in re.findall(r"[a-zA-Z0-9À-ỹ]+", sent.lower()) if len(w) >= 3}
            if len(c_exp & s_toks) >= 1 or c_low.strip(" .!?") in sent.lower():
                return False, "CONTRADICT", f"polarity_mismatch: source sentence negates/inverts claim ({sent[:120]})"

        if not claim_cond and _CONDITIONAL_FRAMING_RE.search(sent):
            # If the claim is an unconditional substring lifted out of an 'if/unless/whether/giả sử' clause
            c_bare = re.sub(r"\[\d+\]", " ", c_low).strip(" .!?")
            if c_bare and c_bare in sent.lower():
                return False, "NOT_ENOUGH_INFO", f"conditional_clause_unconfirmed: ({sent[:120]})"

        if _UNIVERSAL_QUANTIFIER_RE.search(claim_text) and _EXISTENTIAL_HEDGE_RE.search(sent) and not _UNIVERSAL_QUANTIFIER_RE.search(sent):
            return False, "CONTRADICT", f"quantifier_overgeneralization: ({sent[:120]})"

    # Directed comparative check against explicit comparative sentences (when qualitative_comparatives=True)
    # and always against explicit numeric bindings in source (e.g. A = 81%, B = 92%)
    claim_pairs = extract_directed_entity_pairs(claim_text)
    if claim_pairs:
        source_pairs = extract_directed_entity_pairs(source_text) if qualitative_comparatives else []
        bindings = extract_entity_quantity_bindings(source_text)
        is_qualitative_superiority = bool(
            re.search(
                r"\b(?:better|faster|superior|outperforms?|surpasses?|beats?|worse|slower|inferior|underperforms?|"
                r"tốt\s+hơn|nhanh\s+hơn|vượt\s+trội|kém\s+hơn|chậm\s+hơn)\b",
                claim_text,
                re.I,
            )
        )
        claim_mentions_lower_better = bool(
            re.search(r"\b(?:latency|delay|error|loss|perplexity|wer|cer|độ\s+trễ|lỗi|mất\s+mát|thời\s+gian\s+chạy)\b", claim_text, re.I)
        )
        claim_split_hint = _normalize_split_name(claim_text)
        claim_qs = extract_structured_quantities(claim_text)
        claim_unit_hints = {
            ("%" if qu == "pp" else qu)
            for _, qu, _ in claim_qs
            if qu
        }
        filtered_bindings = [
            b for b in bindings
            if (not claim_split_hint or not b["split"] or b["split"] == claim_split_hint)
            and (not claim_unit_hints or not b["unit"] or b["unit"] in claim_unit_hints)
        ]
        active_bindings = filtered_bindings if len(filtered_bindings) >= 2 else bindings
        for i in range(len(active_bindings)):
            for j in range(i + 1, len(active_bindings)):
                b1, b2 = active_bindings[i], active_bindings[j]
                if b1["unit"] == b2["unit"] and b1["split"] == b2["split"] and b1["entity_key"] != b2["entity_key"]:
                    if not math.isclose(b1["value"], b2["value"], abs_tol=1e-9):
                        col_is_lower_better = bool(
                            re.search(r"\b(?:latency|delay|error|loss|perplexity|wer|cer|độ\s+trễ|lỗi|mất\s+mát|thời\s+gian\s+chạy)\b", str(b1.get("split") or ""), re.I)
                        )
                        use_lower_is_better = is_qualitative_superiority and (claim_mentions_lower_better or col_is_lower_better)
                        if not use_lower_is_better:
                            hi, lo = (b1["entity_key"], b2["entity_key"]) if b1["value"] > b2["value"] else (b2["entity_key"], b1["entity_key"])
                        else:
                            hi, lo = (b1["entity_key"], b2["entity_key"]) if b1["value"] < b2["value"] else (b2["entity_key"], b1["entity_key"])
                        if (hi, lo) not in source_pairs:
                            source_pairs.append((hi, lo))

        for c_hi, c_lo in claim_pairs:
            if (c_lo, c_hi) in source_pairs and (c_hi, c_lo) not in source_pairs:
                return False, "CONTRADICT", f"directed_relation_reversed:{c_hi}>{c_lo}"

    return True, "SUPPORT", "ok"


# =====================================================================
# 3. Canonical Units & Deterministic Derived-Claim Verification
# =====================================================================

_ORDERED_UNIT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(?:điểm\s+phần\s+trăm|diem\s+phan\s+tram|percentage\s+points?|pct\s+points?|pp)$", re.I), "pp"),
    (re.compile(r"^(?:%|phần\s+trăm|phan\s+tram|percent|pct)$", re.I), "%"),
    (re.compile(r"^(?:km/h|kmh|kph|km/giờ|km/gio)$", re.I), "km/h"),
    (re.compile(r"^(?:m/h|m/giờ|m/gio)$", re.I), "m/h"),
    (re.compile(r"^(?:m/s|mps|m/giây)$", re.I), "m/s"),
    (re.compile(r"^(?:km/s)$", re.I), "km/s"),
    (re.compile(r"^(?:tokens?/s|toks?/s)$", re.I), "tokens/s"),
    (re.compile(r"^(?:gb/s|gbps)$", re.I), "gb/s"),
    (re.compile(r"^(?:mb/s|mbps)$", re.I), "mb/s"),
    (re.compile(r"^(?:milliseconds?|ms)$", re.I), "ms"),
    (re.compile(r"^(?:seconds?|secs?|giây|s)$", re.I), "s"),
    (re.compile(r"^(?:minutes?|mins?|phút)$", re.I), "min"),
    (re.compile(r"^(?:hours?|hrs?|giờ|h)$", re.I), "h"),
    (re.compile(r"^(?:days?|ngày|ngay)$", re.I), "day"),
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
    (re.compile(r"^(?:layers?|lớp|lop|tầng|tang)$", re.I), "layer"),
]

_QUANTITY_WITH_UNIT_RE = re.compile(
    r"(?<![\w.,])([−+-]?\d+(?:[.,]\d+)?)"
    r"(?:\s+(?:consecutive|clear|encoder|attention|total|full|additional))?\s*("
    r"điểm\s+phần\s+trăm|diem\s+phan\s+tram|percentage\s+points?\b|pct\s+points?\b|\bpp\b|"
    r"%|phần\s+trăm|phan\s+tram|\bpercent\b|\bpct\b|"
    r"km/h|kmh|kph|km/giờ|km/gio|m/h|m/giờ|m/gio|m/s|mps|m/giây|km/s|"
    r"tokens?/s|toks?/s|gb/s|gbps|mb/s|mbps|"
    r"\bmilliseconds?\b|\bms\b|\bseconds?\b|\bsecs?\b|\bgiây\b|\bminutes?\b|\bmins?\b|\bphút\b|"
    r"\bhours?\b|\bhrs?\b|\bgiờ\b|\bdays?\b|\bngày\b|\bngay\b|"
    r"\bliters?\b|\blitres?\b|\blít\b|\blit\b|\bmilliliters?\b|\bml\b|"
    r"\bkilometers?\b|\bkilomet\b|\bkm\b|\bmeters?\b|\bmét\b|\bmet\b|\bcm\b|\bmm\b|"
    r"\bkg\b|\bmg\b|\bgb\b|\bmb\b|\blayers?\b|\blớp\b|\blop\b|\btầng\b|\btang\b|"
    r"\bm\b|\bs\b|\bg\b|\bh\b|\bl\b"
    r")?",
    re.IGNORECASE,
)

_LOCATOR_CLEAN_RE = re.compile(
    r"\[\d+\]|\b(?:trang|page|mục|section|hình|figure|fig\.?|table|tab\.?|bảng|"
    r"equation|eq\.?|formula|công\s+thức|phương\s+trình|biểu\s+thức)\s*\(?\d+(?:\.\d+)*\)?(?!\w)",
    re.IGNORECASE,
)

_SPLIT_TAG_RE = re.compile(r"\b(validation|val|dev|development|test|testing|train|training|holdout)\b", re.IGNORECASE)


def canonicalize_unit_strict(raw_unit: Optional[str]) -> str:
    if not raw_unit:
        return ""
    cleaned = " ".join(raw_unit.strip().split())
    for pat, canon in _ORDERED_UNIT_RULES:
        if pat.match(cleaned):
            return canon
    return cleaned.lower()


def extract_structured_quantities(text: str) -> list[tuple[float, str, str]]:
    """Returns `(float_value, canonical_unit, raw_number_str)` tuples from `text`."""
    cleaned = _LOCATOR_CLEAN_RE.sub(" ", expand_unicode_scripts_preserving_role(text or ""))
    # Ignore exponent/subscript numbers like ^2 or _2 when extracting measurement quantities
    cleaned = re.sub(r"[\^_]\s*\{?[A-Za-z0-9+-]+\}?", " ", cleaned)
    out: list[tuple[float, str, str]] = []
    for m in _QUANTITY_WITH_UNIT_RE.finditer(cleaned):
        raw_num = m.group(1).replace("−", "-").replace(",", ".")
        try:
            val = float(raw_num)
        except ValueError:
            continue
        unit = canonicalize_unit_strict(m.group(2))
        out.append((val, unit, raw_num))
    return out


def _normalize_split_name(text: str) -> str:
    m = _SPLIT_TAG_RE.search(text or "")
    if not m:
        return ""
    tok = m.group(1).lower()
    if tok in {"val", "validation", "dev", "development"}:
        return "validation"
    if tok in {"test", "testing"}:
        return "test"
    if tok in {"train", "training"}:
        return "train"
    return tok


_ENTITY_BINDING_RE = re.compile(
    rf"(?P<ent>{_ENTITY_PHRASE})\s*(?:=|:|\bis\b|\bwas\b|\bachieved\b|\breached\b|\bscored\b|\byielded\b|"
    rf"\bđạt\b|\blà\b|\bcó\s+độ\s+chính\s+xác\b|\bcó\s+giá\s+trị\b)\s*"
    rf"(?P<num>[−+-]?\d+(?:[.,]\d+)?)\s*(?P<unit>điểm\s+phần\s+trăm|percentage\s+points?|pp|%|phần\s+trăm|percent|ms|s|gb|mb|days?|ngày|lít|liters?)?",
    re.IGNORECASE,
)


def extract_entity_quantity_bindings(
    text: str,
    *,
    source_id: str = "",
) -> list[dict[str, Any]]:
    """Extracts `(entity, entity_key, value, unit, split, source_id)` bindings from prose and Markdown pipe tables."""
    if not text:
        return []
    bindings: list[dict[str, Any]] = []

    # 1. Parse Markdown pipe tables (`| Entity | Col1 | Col2 |`) preserving column split/unit headers
    pipe_rows: list[list[str]] = []
    for ln in text.splitlines():
        st = ln.strip()
        if st.count("|") >= 2:
            cells = [c.strip() for c in st.strip("|").split("|")]
            if all(re.match(r"^[-:\s]*$", c) for c in cells):
                continue
            if len(cells) >= 2:
                pipe_rows.append(cells)
    if len(pipe_rows) >= 2:
        headers = pipe_rows[0]
        header_units: list[str] = []
        header_splits: list[str] = []
        for h in headers:
            um = re.search(r"\(([^()]{1,24})\)|\[([^\[\]]{1,24})\]", h)
            header_units.append(canonicalize_unit_strict((um.group(1) or um.group(2)).strip()) if um else "")
            header_splits.append(_normalize_split_name(h) or re.sub(r"\s*[\(\[].*?[\)\]]", "", h).strip().lower())
        headers_low = [h.lower() for h in headers]
        for drow in pipe_rows[1:]:
            if [c.lower() for c in drow] == headers_low:
                continue
            ent_raw = drow[0].strip()
            ent_key = normalize_entity_key(ent_raw)
            if not ent_key:
                continue
            for c_idx in range(1, min(len(drow), len(headers))):
                cell_str = drow[c_idx].strip()
                qs = extract_structured_quantities(cell_str)
                if qs:
                    val, c_unit, _ = qs[0]
                    eff_unit = c_unit or header_units[c_idx]
                    bindings.append({
                        "entity": ent_raw,
                        "entity_key": ent_key,
                        "value": val,
                        "unit": eff_unit,
                        "split": header_splits[c_idx],
                        "source_id": source_id,
                    })

    # 2. Parse narrative / inline comparative clauses
    for clause in re.split(r"[;\n]|(?<=[.!?])\s+", text):
        if not clause.strip() or clause.strip().startswith("|"):
            continue
        clause_split = _normalize_split_name(clause)
        sub_clauses = [sc.strip() for sc in re.split(r",\s*(?=[A-Za-zÀ-ỹ])", clause) if sc.strip()]
        for sub in sub_clauses:
            local_split = _normalize_split_name(sub) or clause_split
            for m in _ENTITY_BINDING_RE.finditer(sub):
                ent_raw = m.group("ent").strip()
                ent_key = normalize_entity_key(ent_raw)
                if not ent_key or ent_key in {"accuracy", "f1", "precision", "recall", "score", "loss", "value", "kết", "tỷ"}:
                    continue
                try:
                    val = float(m.group("num").replace("−", "-").replace(",", "."))
                except ValueError:
                    continue
                unit = canonicalize_unit_strict(m.group("unit"))
                bindings.append({
                    "entity": ent_raw,
                    "entity_key": ent_key,
                    "value": val,
                    "unit": unit,
                    "split": local_split,
                    "source_id": source_id,
                })
    return bindings


def infer_and_bind_derived_claim(
    claim_text: str,
    sources: Sequence[Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    """
    Attempts to bind an implicit comparative difference / relative-change claim
    (e.g., `"B cao hơn A 11 điểm phần trăm"` against `"A = 81%, B = 92%"`) into a formal
    `derivation` specification so it can be deterministically verified.
    """
    if not claim_text or not sources:
        return None
    pairs = extract_directed_entity_pairs(claim_text)
    if not pairs:
        return None
    quantities = extract_structured_quantities(claim_text)
    if not quantities:
        return None

    hi_key, lo_key = pairs[0]
    all_bindings: list[dict[str, Any]] = []
    for src in sources:
        if not isinstance(src, Mapping):
            continue
        sid = str(src.get("id") or src.get("source_unit_id") or src.get("unit_id") or "")
        txt = str(src.get("quote") or src.get("text") or src.get("content") or "")
        all_bindings.extend(extract_entity_quantity_bindings(txt, source_id=sid))

    hi_matches = [b for b in all_bindings if b["entity_key"] == hi_key]
    lo_matches = [b for b in all_bindings if b["entity_key"] == lo_key]
    if not hi_matches or not lo_matches:
        return None

    # Prefer operand pair matching claim's explicit split first, otherwise sharing the same split
    preferred_split = _normalize_split_name(claim_text)
    chosen_pair: tuple[dict[str, Any], dict[str, Any]] | None = None
    if preferred_split:
        for hm in hi_matches:
            for lm in lo_matches:
                if hm["split"] == preferred_split and lm["split"] == preferred_split and hm["unit"] == lm["unit"]:
                    chosen_pair = (hm, lm)
                    break
            if chosen_pair:
                break
    if chosen_pair is None:
        for hm in hi_matches:
            for lm in lo_matches:
                if hm["split"] == lm["split"] and hm["unit"] == lm["unit"]:
                    chosen_pair = (hm, lm)
                    break
            if chosen_pair:
                break
    if chosen_pair is None:
        chosen_pair = (hi_matches[0], lo_matches[0])

    op_hi, op_lo = chosen_pair
    is_rel_pct = bool(
        re.search(r"\b(?:tăng\s+tương\s+đối|phần\s+trăm\s+tương\s+đối|relative\s+(?:increase|change|gain|improvement))\b", claim_text, re.I)
    )
    if any(u == "pp" for _, u, _ in quantities):
        op_name = "diff_pp"
    elif is_rel_pct and op_hi["unit"] == "%" and op_lo["unit"] == "%":
        op_name = "rel_change_pct"
    elif op_hi["unit"] == "%" and op_lo["unit"] == "%":
        # Default subtraction of two percentage metrics requires percentage points ('pp'), NOT bare '%'
        op_name = "diff_pp"
    else:
        op_name = "sub"

    return {
        "op": op_name,
        "operands": [
            {
                "entity": op_hi["entity"],
                "split": op_hi["split"],
                "value": op_hi["value"],
                "unit": op_hi["unit"],
                "evidence_id": op_hi["source_id"],
            },
            {
                "entity": op_lo["entity"],
                "split": op_lo["split"],
                "value": op_lo["value"],
                "unit": op_lo["unit"],
                "evidence_id": op_lo["source_id"],
            },
        ],
    }


def verify_derived_claim(
    claim_text: str,
    derivation: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Deterministically verifies a `derived_claim` against linked `SourceUnit`s or `EvidenceSpan`s:
      1. Every operand's `(entity, value, unit)` must appear in the supplied `sources`.
      2. All operands must share the same `split` and `metric` (unless `allow_cross_split=True`).
      3. Python computes the exact result and required unit (`diff_pp` -> `pp`, `rel_change_pct` -> `%`, etc.).
      4. `claim_text` must state the computed value with the exact canonical unit, preserve
         subject-object direction (`higher > lower`), and contain no ungrounded extra numbers!
    """
    if not isinstance(derivation, Mapping):
        return {"passed": False, "reason": "invalid_derivation_spec"}
    op = str(derivation.get("op") or "sub").strip().lower()
    operands = [o for o in (derivation.get("operands") or []) if isinstance(o, Mapping)]
    allow_cross_split = bool(derivation.get("allow_cross_split", False))
    tolerance = float(derivation.get("tolerance", 0.05))

    if len(operands) < 2 and op != "mean":
        return {"passed": False, "reason": "insufficient_operands"}
    if not sources:
        return {"passed": False, "reason": "no_sources_for_derivation"}

    splits = {str(item.get("split") or "").strip().lower() for item in operands if item.get("split")}
    if len(splits) > 1 and not allow_cross_split:
        return {"passed": False, "reason": f"split_mismatch:{sorted(splits)}"}

    metrics = {str(item.get("metric") or "").strip().lower() for item in operands if item.get("metric")}
    if len(metrics) > 1:
        return {"passed": False, "reason": f"metric_mismatch:{sorted(metrics)}"}

    by_id: dict[str, Mapping[str, Any]] = {}
    for src in sources:
        if isinstance(src, Mapping):
            for k in ("id", "source_unit_id", "unit_id", "source_id"):
                val_id = src.get(k)
                if isinstance(val_id, str) and val_id and val_id not in by_id:
                    by_id[val_id] = src

    all_text = "\n".join(
        str(src.get("quote") or src.get("text") or src.get("content") or "")
        for src in sources
        if isinstance(src, Mapping)
    )
    if not all_text.strip():
        return {"passed": False, "reason": "empty_source_text"}

    used_source_ids: list[str] = []
    operand_vals: list[float] = []
    operand_units: list[str] = []

    for item in operands:
        try:
            val = float(item["value"])
        except (KeyError, TypeError, ValueError):
            return {"passed": False, "reason": "invalid_operand_value"}
        unit = canonicalize_unit_strict(item.get("unit"))
        ent = str(item.get("entity") or "").strip()
        ref_id = str(item.get("evidence_id") or item.get("source_id") or item.get("unit_id") or "").strip()

        target_src = by_id.get(ref_id) if ref_id else None
        target_text = (
            str(target_src.get("quote") or target_src.get("text") or target_src.get("content") or "")
            if target_src is not None
            else all_text
        )

        # Also check if source text mentions a conflicting split for this operand
        if not allow_cross_split and not item.get("split"):
            inferred_bindings = extract_entity_quantity_bindings(target_text)
            ent_key = normalize_entity_key(ent)
            for b in inferred_bindings:
                if b["entity_key"] == ent_key and math.isclose(b["value"], val, abs_tol=1e-4) and b["split"]:
                    splits.add(b["split"])
            if len(splits) > 1:
                return {"passed": False, "reason": f"split_mismatch:{sorted(splits)}"}

        if ent:
            ent_key = normalize_entity_key(ent)
            t_low = target_text.lower()
            if ent.lower() not in t_low and (not ent_key or not re.search(rf"(?<!\w){re.escape(ent_key)}(?!\w)", t_low)):
                ent_toks = {w for w in re.findall(r"[a-zA-Z0-9À-ỹ]+", ent.lower()) if len(w) >= 2}
                t_expanded = expand_bilingual_tokens(target_text, set(re.findall(r"[a-zA-Z0-9À-ỹ]+", t_low)))
                if not ent_toks or not (ent_toks <= t_expanded):
                    return {"passed": False, "reason": f"operand_entity_missing:{ent}"}

        src_quantities = extract_structured_quantities(target_text)
        if not any(
            math.isclose(qv, val, abs_tol=1e-4) and (qu == unit if unit else True)
            for qv, qu, _ in src_quantities
        ):
            return {"passed": False, "reason": f"operand_value_missing:{val}{unit}"}

        if target_src is not None and isinstance(target_src.get("id"), str):
            used_source_ids.append(target_src["id"])
        operand_vals.append(val)
        operand_units.append(unit)

    if op in {"diff_pp", "pp_diff"}:
        if any(u != "%" for u in operand_units):
            return {"passed": False, "reason": "diff_pp_requires_percent_operands"}
        computed = operand_vals[0] - operand_vals[1]
        expected_unit = "pp"
    elif op == "rel_change_pct":
        if math.isclose(operand_vals[1], 0.0, abs_tol=1e-12):
            return {"passed": False, "reason": "division_by_zero"}
        computed = ((operand_vals[0] - operand_vals[1]) / abs(operand_vals[1])) * 100.0
        expected_unit = "%"
    elif op in {"sub", "diff"}:
        if len(set(operand_units)) > 1:
            return {"passed": False, "reason": f"incompatible_units:{operand_units}"}
        # Subtracting two percentage values without relative-change framing must use percentage points ('pp'), not '%'
        if operand_units[0] == "%" and not derivation.get("expected_unit"):
            expected_unit = "pp"
        else:
            expected_unit = canonicalize_unit_strict(derivation.get("expected_unit") or derivation.get("unit")) or operand_units[0]
        computed = operand_vals[0] - operand_vals[1]
    elif op in {"add", "sum"}:
        if len(set(operand_units)) > 1:
            return {"passed": False, "reason": f"incompatible_units:{operand_units}"}
        computed = sum(operand_vals)
        expected_unit = canonicalize_unit_strict(derivation.get("expected_unit") or derivation.get("unit")) or operand_units[0]
    elif op == "ratio":
        if math.isclose(operand_vals[1], 0.0, abs_tol=1e-12):
            return {"passed": False, "reason": "division_by_zero"}
        computed = operand_vals[0] / operand_vals[1]
        expected_unit = canonicalize_unit_strict(derivation.get("expected_unit") or derivation.get("unit"))
    elif op == "mean":
        computed = sum(operand_vals) / len(operand_vals)
        expected_unit = canonicalize_unit_strict(derivation.get("expected_unit") or derivation.get("unit")) or (operand_units[0] if operand_units else "")
    else:
        return {"passed": False, "reason": f"unsupported_op:{op}"}

    explicit_res = derivation.get("result") if derivation.get("result") is not None else derivation.get("expected_value")
    if explicit_res is not None:
        try:
            if not math.isclose(computed, float(explicit_res), abs_tol=tolerance):
                return {"passed": False, "reason": f"derivation_result_mismatch:{computed}!={explicit_res}"}
        except (TypeError, ValueError):
            return {"passed": False, "reason": "invalid_expected_value"}

    claim_quantities = extract_structured_quantities(claim_text)
    matched_derived = False
    allowed_source_quantities = extract_structured_quantities(all_text)

    for qv, qu, raw_str in claim_quantities:
        if math.isclose(qv, abs(computed) if computed < 0 and _LESS_REL_RE.search(claim_text) else computed, abs_tol=tolerance):
            if expected_unit and qu != expected_unit:
                return {"passed": False, "reason": f"derived_unit_mismatch:{qu or 'none'}_expected_{expected_unit}"}
            matched_derived = True
            continue
        # Any OTHER number appearing in the claim must exist directly in the cited sources (no extra fabricated numbers!)
        if not any(
            math.isclose(qv, sv, abs_tol=1e-4) and (qu == su if qu else su != "%")
            for sv, su, _ in allowed_source_quantities
        ):
            return {"passed": False, "reason": f"ungrounded_extra_number:{raw_str}{qu}"}

    if not matched_derived:
        return {"passed": False, "reason": f"derived_value_missing_in_claim:{round(computed, 4)}_{expected_unit}"}

    # Check comparative direction (e.g. "B cao hơn A 11 điểm phần trăm" vs swapped "A cao hơn B 11 điểm phần trăm")
    if op in {"diff_pp", "pp_diff", "sub", "diff", "rel_change_pct"} and len(operands) >= 2:
        ent0 = normalize_entity_key(str(operands[0].get("entity") or ""))
        ent1 = normalize_entity_key(str(operands[1].get("entity") or ""))
        if ent0 and ent1 and not math.isclose(operand_vals[0], operand_vals[1], abs_tol=1e-9):
            true_hi, true_lo = (ent0, ent1) if operand_vals[0] > operand_vals[1] else (ent1, ent0)
            claim_pairs = extract_directed_entity_pairs(claim_text)
            if (true_lo, true_hi) in claim_pairs:
                return {"passed": False, "reason": f"directed_relation_reversed:{true_lo}>{true_hi}"}
            if has_clause_negation(claim_text):
                return {"passed": False, "reason": "polarity_mismatch_on_derived_claim"}

    return {
        "passed": True,
        "reason": "verified_derived_claim",
        "computed_value": round(computed, 4),
        "expected_unit": expected_unit,
        "used_source_ids": list(dict.fromkeys(used_source_ids)),
    }


# =====================================================================
# 5. M2b Diagram Node–Edge Directed Path & Branch Condition Verifier
# =====================================================================

_DIRECT_EDGE_CLAIM_CUES = re.compile(
    r"\b(?:đi\s+thẳng\s+(?:vào|tới|sang)|nối\s+trực\s+tiếp\s+(?:vào|tới|với)|chuyển\s+trực\s+tiếp\s+sang|"
    r"đầu\s+ra\s+của\s+.+?\s+(?:đi\s+vào|nối\s+vào|đi\s+tới|chuyển\s+sang)|"
    r"connects?\s+directly\s+to|feeds?\s+directly\s+into|routes?\s+directly\s+to|"
    r"output\s+of\s+.+?\s+(?:goes|feeds|connects|routes)\s+(?:directly\s+)?(?:to|into))\b",
    re.IGNORECASE,
)

_FLOW_FORWARD_CUES = re.compile(
    r"(?:-->|->|→|⟶|⇒|"
    r"\b(?:đi\s+(?:thẳng\s+)?(?:vào|tới|sang)|nối\s+(?:trực\s+tiếp\s+)?(?:vào|tới|với)|chuyển\s+(?:trực\s+tiếp\s+)?(?:sang|tới|về)|dẫn\s+(?:thẳng\s+)?(?:tới|đến)|"
    r"feeds?\s+(?:directly\s+)?into|connects?\s+(?:directly\s+)?to|routes?\s+(?:directly\s+)?to|goes\s+(?:directly\s+)?to|leads?\s+(?:directly\s+)?to|precedes?)\b)",
    re.IGNORECASE,
)

_FLOW_REVERSE_CUES = re.compile(
    r"\b(?:theo\s+sau|diễn\s+ra\s+sau|nhận\s+đầu\s+vào\s+từ|follows?|receives?\s+input\s+from|comes?\s+after)\b",
    re.IGNORECASE,
)

_BRANCH_COND_SYNONYMS: dict[str, set[str]] = {
    "yes": {"yes", "true", "đúng", "dung", "hợp lệ", "valid", "pass", "ok", "accept"},
    "no": {"no", "false", "sai", "không", "khong", "không hợp lệ", "invalid", "fail", "reject"},
}


def _canonical_branch_cond(raw: str) -> str:
    s = (raw or "").strip().lower()
    if not s:
        return ""
    for canon, group in _BRANCH_COND_SYNONYMS.items():
        if s in group:
            return canon
    return s


def check_diagram_graph_consistency(
    claim_text: str,
    units: Sequence[Mapping[str, Any]],
) -> tuple[bool, str, str, Optional[dict[str, Any]]]:
    """
    Deterministically verifies directed diagram flow & conditional branch claims against
    extracted `diagram_metadata` (`nodes`, `edges`, `adjacency`) on `SourceUnit`s:
      - Returns `(passed, status, reason, matched_unit)`:
        * `(True, 'SUPPORT', 'verified_diagram_...', unit)` when claim's directed edge/path & branch condition match the diagram graph!
        * `(False, 'CONTRADICT', 'diagram_flow_reversed:...', None)` when claim reverses a directed flow (`B -> A` vs `A -> B`).
        * `(False, 'CONTRADICT', 'diagram_direct_edge_absent:...', None)` when claim asserts a direct edge (`A -> C`) that skips intermediate nodes (`A -> B -> C`) or does not exist.
        * `(False, 'CONTRADICT', 'diagram_branch_condition_mismatch:...', None)` when claim swaps branch conditions (`Yes` vs `No` / `Đúng` vs `Sai`).
        * `(True, 'NOT_APPLICABLE', 'no_diagram_claim', None)` when no diagram units or flow cues apply.
    """
    if not claim_text or not units or has_clause_negation(claim_text):
        return True, "NOT_APPLICABLE", "no_diagram_claim", None

    diagram_units: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    for u in units:
        if not isinstance(u, Mapping):
            continue
        dmeta = u.get("diagram_metadata") or ((u.get("metadata") or {}).get("diagram_metadata") if isinstance(u.get("metadata"), Mapping) else None)
        if isinstance(dmeta, dict) and dmeta.get("nodes") and dmeta.get("edges"):
            diagram_units.append((u, dmeta))

    if not diagram_units:
        return True, "NOT_APPLICABLE", "no_diagram_units", None

    has_fwd = bool(_FLOW_FORWARD_CUES.search(claim_text))
    has_rev = bool(_FLOW_REVERSE_CUES.search(claim_text))
    if not (has_fwd or has_rev):
        return True, "NOT_APPLICABLE", "no_flow_relation_cue", None

    requires_direct = bool(_DIRECT_EDGE_CLAIM_CUES.search(claim_text)) or any(sym in claim_text for sym in ("->", "-->", "→", "⟶", "⇒"))
    c_low = unicodedata.normalize("NFC", claim_text).lower()

    # Check if claim specifies an explicit branch condition (`khi Đúng`, `nếu Sai`, `if Yes`, `when No`, `[Yes]`)
    claimed_cond = ""
    m_cond = re.search(
        r"(?:\b(?:khi|nếu|nhánh|trường\s+hợp|if|when|on|branch)\s+['\"`\[]?(yes|no|true|false|đúng|sai|pass|fail|valid|invalid|timeout)['\"`\]]?|"
        r"\[(yes|no|true|false|đúng|sai|pass|fail|valid|invalid|timeout)\])",
        c_low,
        re.IGNORECASE,
    )
    if m_cond:
        claimed_cond = _canonical_branch_cond(m_cond.group(1) or m_cond.group(2) or "")

    for unit_obj, dmeta in diagram_units:
        nodes = list(dmeta.get("nodes") or [])
        edges = list(dmeta.get("edges") or [])
        # Find all diagram nodes mentioned in `claim_text`, ordered by their earliest appearance in `claim_text`
        mentioned: list[tuple[int, dict[str, Any]]] = []
        for n in nodes:
            lbl = str(n.get("label") or "").strip()
            if not lbl:
                continue
            pos = c_low.find(lbl.lower())
            if pos >= 0:
                # Avoid matching a short node label that is merely a substring of a longer matched node label
                mentioned.append((pos, n))
        # Filter substring overlaps (keep longer label when spans overlap)
        filtered_mentions: list[tuple[int, dict[str, Any]]] = []
        for pos, n in sorted(mentioned, key=lambda it: (it[0], -len(str(it[1].get("label") or "")))):
            lbl_len = len(str(n.get("label") or ""))
            if any(p0 <= pos < p0 + len(str(n0.get("label") or "")) for p0, n0 in filtered_mentions):
                continue
            filtered_mentions.append((pos, n))

        if len(filtered_mentions) < 2:
            continue

        # Determine claimed `(src_node, dst_node)` pairs in order
        ordered_nodes = [n for _, n in sorted(filtered_mentions, key=lambda it: it[0])]
        if has_rev and not has_fwd:
            # "Y follows X" / "Y diễn ra sau X" -> X -> Y
            claimed_pairs = [(ordered_nodes[i + 1], ordered_nodes[i]) for i in range(len(ordered_nodes) - 1)]
        else:
            claimed_pairs = [(ordered_nodes[i], ordered_nodes[i + 1]) for i in range(len(ordered_nodes) - 1)]

        # Build graph lookup structures
        direct_edges: dict[tuple[str, str], list[str]] = {}
        adj_ids: dict[str, set[str]] = {}
        for e in edges:
            u_id, v_id = str(e.get("source") or ""), str(e.get("target") or "")
            cond_c = _canonical_branch_cond(str(e.get("condition") or ""))
            direct_edges.setdefault((u_id, v_id), []).append(cond_c)
            adj_ids.setdefault(u_id, set()).add(v_id)

        def _has_path(start_id: str, goal_id: str) -> bool:
            visited: set[str] = set()
            stack = list(adj_ids.get(start_id, ()))
            while stack:
                cur = stack.pop()
                if cur == goal_id:
                    return True
                if cur not in visited:
                    visited.add(cur)
                    stack.extend(adj_ids.get(cur, ()))
            return False

        all_pairs_ok = True
        for u_node, v_node in claimed_pairs:
            uid, vid = str(u_node["id"]), str(v_node["id"])
            ulbl, vlbl = str(u_node["label"]), str(v_node["label"])
            has_direct = (uid, vid) in direct_edges
            has_reachable = _has_path(uid, vid)
            has_reverse = _has_path(vid, uid)

            if has_direct:
                if claimed_cond:
                    edge_conds = [c for c in direct_edges[(uid, vid)] if c]
                    # Check if another outgoing edge from `uid` actually owns `claimed_cond`
                    other_with_cond = [
                        e for e in edges
                        if str(e.get("source")) == uid
                        and str(e.get("target")) != vid
                        and _canonical_branch_cond(str(e.get("condition") or "")) == claimed_cond
                    ]
                    if (edge_conds and claimed_cond not in edge_conds) or (not edge_conds and other_with_cond):
                        return (
                            False,
                            "CONTRADICT",
                            f"diagram_branch_condition_mismatch:{ulbl}->[{claimed_cond}]->{vlbl}",
                            dict(unit_obj),
                        )
                continue

            if requires_direct:
                if (vid, uid) in direct_edges or (has_reverse and not has_reachable):
                    return False, "CONTRADICT", f"diagram_flow_reversed:{ulbl}->{vlbl}", dict(unit_obj)
                return False, "CONTRADICT", f"diagram_direct_edge_absent:{ulbl}->{vlbl}", dict(unit_obj)

            if has_reachable:
                continue

            if has_reverse:
                return False, "CONTRADICT", f"diagram_flow_reversed:{ulbl}->{vlbl}", dict(unit_obj)
            return False, "CONTRADICT", f"diagram_path_absent:{ulbl}->{vlbl}", dict(unit_obj)

        if all_pairs_ok:
            return True, "SUPPORT", "verified_diagram_graph", dict(unit_obj)

    return True, "NOT_APPLICABLE", "diagram_nodes_not_matched", None
