"""
Unified Vietnamese/English Language Detection, Separation & Localized Status Messages.

Separates:
  - `query_language`: Language of the user's question (`"vi"` or `"en"`).
  - `answer_language`: Requested output language (explicit user override, explicit in-query
    request such as "answer in English" / "trả lời bằng tiếng Việt", or `query_language`).
  - `source_language`: Dominant language of the selected source documents (`"vi"` or `"en"`).

Handles:
  - Accented Vietnamese without false positives on ASCII `'s'` (no Unicode range `[à-ỹ]`
    with `re.IGNORECASE`, which case-folds `'s'` to `U+017F` `ſ` and spans Greek `α-ω`).
  - Unaccented Vietnamese queries (e.g., `"Ket qua o trang nao?"`, `"La gi?"`, `"Bao nhieu?"`).
  - Mixed technical identifiers, LaTeX math, and Greek symbols (`α, β, γ, π, θ, λ, σ`).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Literal, Optional

LanguageCode = Literal["vi", "en"]

# Exact Vietnamese characters (both cases listed explicitly; NEVER use a cross-block Unicode range)
_VI_UNIQUE_CHARS = (
    "ăằắẳẵặâầấẩẫậđêềếểễệôồốổỗộơờớởỡợưừứửữự"
    "ảãạẻẽẹỉĩịỏõọủũụỳýỷỹỵ"
    "ĂẰẮẲẴẶÂẦẤẨẪẬĐÊỀẾỂỄỆÔỒỐỔỖỘƠỜỚỞỠỢƯỪỨỬỮỰ"
    "ẢÃẠẺẼẸỈĨỊỎÕỌỦŨỤỲÝỶỸỴ"
)
_VI_SHARED_ACCENT_CHARS = "àáèéìíòóùúÀÁÈÉÌÍÒÓÙÚ"
_VI_ALL_DIACRITIC_CHARS = _VI_UNIQUE_CHARS + _VI_SHARED_ACCENT_CHARS

VI_UNIQUE_DIACRITIC_RE = re.compile(
    f"[{_VI_UNIQUE_CHARS}]|[A-Za-z][\u0309\u0303\u0323\u031b]"
)
VI_ANY_DIACRITIC_RE = re.compile(
    f"[{_VI_ALL_DIACRITIC_CHARS}]|[A-Za-z][\u0300\u0301\u0309\u0303\u0323\u031b]"
)

# Common European loanwords in English academic prose that carry grave/acute accents
_EUROPEAN_LOANWORDS = frozenset({
    "cafe", "café", "naive", "naïve", "resume", "résumé", "elite", "élite",
    "regime", "régime", "role", "rôle", "etude", "étude", "vis-à-vis", "voilà",
})

# High-precision unaccented Vietnamese multi-word expressions (short queries, questions, comparatives)
_UNACCENTED_VI_PHRASES: tuple[str, ...] = (
    "ket qua", "o trang", "trang nao", "muc nao", "phan nao", "hinh nao", "bang nao",
    "o dau", "bao nhieu", "may lop", "may tang", "may nhom", "may buoc", "nhu the nao",
    "the nao", "la gi", "nghia la gi", "tai sao", "vi sao", "khi nao", "nam nao",
    "ngay nao", "bao gio", "phan tram", "diem phan tram", "so sanh", "doi chieu",
    "tom tat", "tong quan", "noi dung chinh", "tai lieu", "bai bao", "phuong trinh",
    "cong thuc", "bang so lieu", "mo hinh", "phuong phap", "do chinh xac", "huan luyen",
    "tham so", "trong bang", "trong hinh", "trong tai lieu", "trong bai bao",
    "dong gop", "uu nhuoc", "cho biet", "hay cho", "hay neu", "vui long",
    "cao hon", "thap hon", "nhanh hon", "cham hon", "tot hon", "kem hon", "lon hon",
    "nho hon", "nhieu hon", "it hon", "vuot troi", "so voi", "gia tri", "toc do",
    "thoi gian", "thi nghiem", "nghien cuu", "thuat toan", "ngon ngu", "du lieu",
    "tap du lieu", "khong co", "khong duoc", "chua du", "bang chứng", "bang chung",
)

# Unambiguous unaccented Vietnamese tokens (excluding English collisions like a, in, on, to, is, an, do, so, ban, con, may)
_UNACCENTED_VI_WORDS = frozenset({
    "ket", "qua", "trang", "nao", "nhieu", "nhung", "duoc", "khong", "trong", "cua",
    "voi", "tren", "duoi", "giua", "hoac", "vay", "thuat", "toan", "ngon", "ngu",
    "phuong", "phap", "trinh", "nghien", "cuu", "thuc", "nghiem", "danh", "hieu",
    "nang", "chinh", "xac", "diem", "phan", "tram", "luong", "thoi", "gian", "ngay",
    "thang", "tuan", "buoc", "nhom", "muc", "tieu", "gioi", "khoang", "chua", "vui",
    "hay", "neu", "biet", "hinh", "bang", "lieu", "bao", "tom", "tat",
})

_ENGLISH_FUNCTION_WORDS = frozenset({
    "the", "this", "that", "these", "those", "what", "which", "where", "when", "why",
    "how", "who", "whom", "whose", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "does", "did", "will", "would", "could", "should", "can",
    "with", "from", "about", "into", "between", "within", "without", "through",
    "during", "before", "after", "above", "below", "than", "and", "for", "not",
    "reported", "paper", "document", "study", "trial", "results", "accuracy", "test",
})

_EXPLICIT_ANSWER_EN_RE = re.compile(
    r"(?:\b(?:answer|respond|reply|write|explain|summarize)\s+in\s+english\b"
    r"|\bin\s+english\s+please\b"
    r"|\bbằng\s+tiếng\s+anh\b|\bbang\s+tieng\s+anh\b"
    r"|\btiếng\s+anh\s+nhé\b|\bdịch\s+sang\s+tiếng\s+anh\b|\bdich\s+sang\s+tieng\s+anh\b)",
    re.IGNORECASE,
)
_EXPLICIT_ANSWER_VI_RE = re.compile(
    r"(?:\b(?:answer|respond|reply|write|explain|summarize)\s+in\s+vietnamese\b"
    r"|\bin\s+vietnamese\s+please\b"
    r"|\bbằng\s+tiếng\s+việt\b|\bbang\s+tieng\s+viet\b"
    r"|\bdịch\s+sang\s+tiếng\s+việt\b|\bdich\s+sang\s+tieng\s+viet\b)",
    re.IGNORECASE,
)


def has_vietnamese_diacritics(text: str) -> bool:
    """True when `text` contains genuine Vietnamese diacritics (excluding European loanwords)."""
    if not text:
        return False
    nfc = unicodedata.normalize("NFC", text)
    if VI_UNIQUE_DIACRITIC_RE.search(nfc):
        return True
    if not VI_ANY_DIACRITIC_RE.search(nfc):
        return False
    # Only grave/acute accents found: check they aren't solely European loanwords like 'café'
    accented_tokens = [
        tok.lower()
        for tok in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", nfc)
        if VI_ANY_DIACRITIC_RE.search(tok)
    ]
    non_loan = [tok for tok in accented_tokens if tok not in _EUROPEAN_LOANWORDS]
    return bool(non_loan)


def normalize_language_code(value: Any) -> Optional[LanguageCode]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if cleaned in {"vi", "vie", "vn", "vietnamese", "tiếng việt", "tieng viet"}:
        return "vi"
    if cleaned in {"en", "eng", "english", "tiếng anh", "tieng anh"}:
        return "en"
    return None


def detect_explicit_answer_language(query: str) -> Optional[LanguageCode]:
    """Detects when a user explicitly asks for the answer in Vietnamese or English."""
    if not query:
        return None
    nfc = unicodedata.normalize("NFC", query)
    if _EXPLICIT_ANSWER_EN_RE.search(nfc):
        return "en"
    if _EXPLICIT_ANSWER_VI_RE.search(nfc):
        return "vi"
    return None


def detect_language(text: str, *, default: LanguageCode = "en") -> LanguageCode:
    """
    Detects whether `text` is Vietnamese (`"vi"`) or English (`"en"`).
    Handles accented Vietnamese, unaccented Vietnamese, short queries, mixed STEM/code terms,
    and Greek mathematical notation (`α, β, π, θ, ∑`) without misclassifying English sentences
    such as `"This is a test"`.
    """
    if not text or not text.strip():
        return default
    nfc = unicodedata.normalize("NFC", text.strip())

    # Strip fenced code blocks and URLs before lexical inspection
    cleaned = re.sub(r"```[\s\S]*?```", " ", nfc)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)

    if cleaned.startswith("/") and re.match(
        r"^/(?:tom[-_]?tat|tóm[-_]?tắt|so[-_]?sanh|so[-_]?sánh|dong[-_]?gop|đóng[-_]?góp|"
        r"phuong[-_]?phap|phương[-_]?pháp|kiem[-_]?chung|kiểm[-_]?chứng|so[-_]?lieu|số[-_]?liệu|"
        r"thuat[-_]?ngu|thuật[-_]?ngữ|trich[-_]?dan|trích[-_]?dẫn|khoang[-_]?trong|khoảng[-_]?trống|"
        r"cau[-_]?hoi|câu[-_]?hỏi|quet|quét)\b",
        cleaned,
        re.IGNORECASE,
    ):
        return "vi"

    if VI_UNIQUE_DIACRITIC_RE.search(cleaned):
        return "vi"

    ascii_words = re.findall(r"[a-z]+", cleaned.lower())
    joined_ascii = " ".join(ascii_words)

    phrase_hits = sum(
        1
        for phrase in _UNACCENTED_VI_PHRASES
        if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", joined_ascii)
    )
    vi_word_hits = sum(1 for w in ascii_words if w in _UNACCENTED_VI_WORDS)
    en_word_hits = sum(1 for w in ascii_words if w in _ENGLISH_FUNCTION_WORDS)

    if has_vietnamese_diacritics(cleaned):
        # Shared acute/grave accents: confirm with VI words or absence of dominant English prose
        if vi_word_hits >= 1 or phrase_hits >= 1 or en_word_hits == 0:
            return "vi"

    if phrase_hits >= 1 and phrase_hits * 2 + vi_word_hits > en_word_hits:
        return "vi"
    if vi_word_hits >= 2 and vi_word_hits > en_word_hits:
        return "vi"
    if en_word_hits >= 1:
        return "en"
    if vi_word_hits >= 1 and len(ascii_words) <= 4:
        return "vi"
    return default


def is_vietnamese_text(text: str) -> bool:
    """Returns True when `text` is Vietnamese (accented or unaccented)."""
    return detect_language(text, default="en") == "vi"


def detect_corpus_language(
    units_or_docs: Iterable[Any] | None,
    *,
    default: LanguageCode = "en",
) -> LanguageCode:
    """Estimates dominant language (`"vi"` or `"en"`) of a document/unit collection."""
    if not units_or_docs:
        return default
    samples: list[str] = []
    for item in units_or_docs:
        if isinstance(item, dict):
            txt = str(item.get("text") or item.get("content") or item.get("quote") or "")
        else:
            txt = str(getattr(item, "text", "") or "")
        if txt.strip():
            samples.append(txt[:800])
        if len(samples) >= 12:
            break
    if not samples:
        return default
    combined = "\n".join(samples)
    nfc = unicodedata.normalize("NFC", combined)
    vi_unique_count = len(VI_UNIQUE_DIACRITIC_RE.findall(nfc))
    alpha_count = max(1, len(re.findall(r"[A-Za-zÀ-ỹ]", nfc)))
    if vi_unique_count >= 10 and (vi_unique_count / alpha_count) >= 0.015:
        return "vi"
    vi_votes = sum(1 for s in samples if detect_language(s, default="en") == "vi")
    return "vi" if (vi_votes / len(samples)) >= 0.4 else "en"


def resolve_language_bundle(
    query: str,
    *,
    explicit_answer_language: Any = None,
    units_or_docs: Iterable[Any] | None = None,
    existing_source_language: Any = None,
) -> dict[str, LanguageCode]:
    """
    Computes independent `query_language`, `answer_language`, and `source_language`.
    Priority for `answer_language`:
      1. Explicit API/state override (`explicit_answer_language` in `{"vi", "en"}`)
      2. Explicit user instruction inside `query` (e.g., `"trả lời bằng tiếng Anh"`)
      3. `query_language`
    """
    q_lang = detect_language(query, default="en")
    ans_lang = (
        normalize_language_code(explicit_answer_language)
        or detect_explicit_answer_language(query)
        or q_lang
    )
    src_lang = (
        detect_corpus_language(units_or_docs, default=normalize_language_code(existing_source_language) or "en")
        if units_or_docs
        else (normalize_language_code(existing_source_language) or "en")
    )
    return {
        "query_language": q_lang,
        "answer_language": ans_lang,
        "source_language": src_lang,
    }


def localized_status_message(
    status: str,
    *,
    language: str = "vi",
    query: str = "",
) -> str:
    """Standardized bilingual status/refusal/clarification messages."""
    lang = normalize_language_code(language) or detect_language(query, default="vi")
    clean_q = (query or "").strip()
    key = (status or "").upper()

    if lang == "en":
        if key == "NOT_FOUND_IN_DOCUMENT":
            if clean_q:
                return f'Based on the selected documents, information about "{clean_q}" is not mentioned in the provided sources.'
            return "Based on the selected documents, the requested information is not mentioned in the provided sources."
        if key == "CLARIFICATION_NEEDED":
            return "Please specify which document, section, or evaluation condition you would like clarified."
        if key == "WAITING_FOR_DOCUMENT":
            return "Please select a document from the library or enable Auto-Scout to begin research."
        if key == "LOW_CONFIDENCE_BANNER":
            return "> **Low confidence** (`LOW_CONFIDENCE · Độ tin cậy thấp`). Available evidence is insufficient to confirm every part of the answer."
        return "Insufficient evidence in the selected documents to verify the answer."

    # Vietnamese ("vi")
    if key == "NOT_FOUND_IN_DOCUMENT":
        if clean_q:
            return f'Dựa trên các tài liệu đã chọn, thông tin về "{clean_q}" không được đề cập.'
        return "Dựa trên các tài liệu đã chọn, thông tin được hỏi không được đề cập."
    if key == "CLARIFICATION_NEEDED":
        return "Vui lòng cho biết tài liệu hoặc khía cạnh cần làm rõ."
    if key == "WAITING_FOR_DOCUMENT":
        return "Vui lòng chọn một tài liệu trong kho hoặc bật Auto-Scout để bắt đầu nghiên cứu."
    if key == "LOW_CONFIDENCE_BANNER":
        return "> **Độ tin cậy thấp.** Bằng chứng hiện có chưa đủ để xác nhận toàn bộ câu trả lời."
    return "Chưa đủ bằng chứng trong tài liệu đã chọn để xác nhận câu trả lời."
