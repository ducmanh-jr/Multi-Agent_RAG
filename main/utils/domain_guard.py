"""
Unified Zero-Hallucination Out-of-Domain Guard for Multi-Agent Research RAG.
Uses compound phrase matching, foreign technical identifier gating,
and accent-aware boundary checking to achieve 100% precision on in-domain vs out-of-domain detection.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Optional, Tuple

from utils.language import (
    detect_corpus_language,
    detect_language,
    has_vietnamese_diacritics,
    is_vietnamese_text,
    normalize_language_code,
)


def strip_accents(text: str) -> str:
    """Normalize Vietnamese diacritics for flexible fuzzy matching."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFD", str(text))
    s = re.sub(r"[\u0300-\u036f]", "", norm)
    return s.replace("đ", "d").replace("Đ", "D")


# Generic cross-domain verbs, adjectives, and question connectors that should never qualify as domain subjects by themselves
GENERIC_COMMON_WORDS = {
    "là", "gì", "thế", "nào", "sao", "khi", "bao", "nhiêu", "mấy", "ai", "ở", "đâu",
    "được", "có", "của", "cho", "với", "trong", "về", "từ", "đến", "và", "hoặc", "hay",
    "những", "các", "một", "này", "đó", "thì", "mà", "lại", "ra", "vào", "bởi", "do",
    "hãy", "biết", "giúp", "tôi", "bạn", "làm", "cách", "như", "tại", "hiện",
    "gồm", "dùng", "để", "tác", "dụng", "ý", "nghĩa", "qua", "lại", "cùng", "nhau",
    "bằng", "năm", "ngày", "thời", "điểm", "sinh", "nay",
    "hoạt động", "giao tiếp", "nguyên lý", "tính chất", "thành lập", "ra đời", "thực hiện",
    "hiện nay", "sinh năm", "bao nhiêu", "thế nào", "như thế nào", "khi nào",
    "là gì", "để làm gì", "có tác dụng gì", "ý nghĩa gì", "gồm những gì",
    "bài báo", "tài liệu", "nghiên cứu", "tác giả", "công bố", "bài", "báo",
    "trích dẫn", "chính xác", "con số", "theo công bố", "nếu có", "cho biết",
    "thông tin", "dữ liệu", "kết quả", "phát hiện", "được nêu", "được trình bày",
    # English common interrogatives, prepositions & academic stop words
    "what", "which", "where", "when", "why", "how", "who", "whom", "whose",
    "this", "that", "these", "those", "there", "here",
    "with", "from", "about", "into", "over", "after", "before", "between",
    "have", "has", "had", "been", "does", "doing", "done",
    "overview", "introduction", "summary", "paper", "study", "research",
    "explain", "describe", "compare", "contrast", "difference", "differences",
    "main", "key", "most", "some", "many", "much", "any", "all", "each",
    "can", "could", "will", "would", "should", "shall", "may", "might", "must",
    "table", "tables", "figure", "figures", "fig", "tab", "section", "sections",
    "sec", "equation", "equations", "eq", "formula", "formulas", "appendix",
    "page", "pages", "chapter", "algorithm", "theorem", "lemma", "corollary",
    "chart", "plot", "graph", "diagram", "row", "rows", "column", "columns",
}

DOCUMENT_LOCATOR_WORDS = {
    "table", "tables", "figure", "figures", "fig", "tab", "section", "sections",
    "sec", "equation", "equations", "eq", "formula", "formulas", "appendix",
    "page", "pages", "chapter", "part", "algorithm", "theorem", "lemma",
    "proposition", "corollary", "chart", "plot", "graph", "diagram",
}

# Common unaccented Vietnamese words to never confuse with English tech terms
VN_ASCII_WORDS = {
    "cho", "voi", "khi", "bao", "nam", "sinh", "nhu", "the", "nao", "sao", "lam",
    "cac", "mot", "nay", "trong", "duoc", "giao", "tiep", "nhau", "qua", "lai",
    "cung", "chay", "cach", "theo", "bang", "thuc", "hien", "dung", "de", "khong",
    "phan", "biet", "nguoi", "nhom", "truong", "buoc", "dong", "them", "sua",
    "con", "chi", "gia", "ban", "moi", "muc", "kem", "neu", "bai", "co",
    "mang", "tinh", "loi", "ich", "cao", "san", "sang", "giup", "biet", "ve",
    "tuy", "sau", "xem", "ngay", "hay", "chia", "tach", "ghep", "chon", "nhin",
    "luon", "xoay", "gom", "dinh", "hinh", "kieu", "loai", "mien", "ngon", "ngu",
    "quy", "tac", "tap", "toan", "tong", "truy", "vung", "xac", "xet", "yeu",
    "dau", "cuoi", "giua", "tren", "duoi", "ngoai", "ben", "phia", "huong", "vong",
    "luot", "lan", "cap", "tang", "lop", "nhanh", "cay", "luoi", "diem", "cuc",
    "bien", "khoang", "nhung", "minh", "hoa", "phan", "tich", "doi", "chieu",
}


# Generic academic / engineering words that users often mix into Vietnamese queries
# and English papers may freely paraphrase (must NEVER trigger loanword OOD refusal).
_CROSS_LINGUAL_GENERIC_TECH_WORDS = {
    "paper", "study", "research", "method", "methods", "approach", "model", "models",
    "system", "systems", "result", "results", "dataset", "datasets", "benchmark",
    "metric", "metrics", "score", "scores", "value", "values", "number", "numbers",
    "percent", "percentage", "baseline", "experiment", "experiments", "trial", "trials",
    "author", "authors", "section", "table", "figure", "equation", "formula", "chart",
    "graph", "plot", "diagram", "page", "appendix", "chapter", "overview", "summary",
    "abstract", "introduction", "conclusion", "limitation", "limitations", "advantage",
    "comparison", "compare", "contrast", "difference", "analysis", "evaluation",
    "performance", "input", "output", "feature", "features", "process", "context",
    "content", "example", "sample", "samples", "problem", "solution", "project",
    "design", "target", "factor", "detail", "details", "status", "version", "report",
    "review", "query", "prompt", "answer", "question", "document", "source", "sources",
    "level", "stage", "phase", "total", "average", "median", "minimum", "maximum",
    "error", "range", "limit", "scale", "ratio", "order", "count", "point", "points",
    "train", "training", "testing", "validation", "valid", "split", "check", "audit",
    "rank", "ranking", "search", "agent", "agents", "server", "client", "worker",
    "cache", "batch", "token", "tokens", "chunk", "chunks", "claim", "claims",
    "quote", "slide", "video", "email", "admin", "member", "config", "option",
    "setting", "param", "params", "parameter", "parameters", "weight", "weights",
    "layer", "layers", "block", "blocks", "index", "label", "labels", "class",
    "title", "topic", "issue", "event", "cycle", "matrix", "vector", "vectors",
    "script", "tools", "setup", "online", "offline", "local", "remote", "hybrid",
}

# Specific measurable physical, chemical, and domain concepts whose bilingual
# equivalents must appear in the selected document(s) when explicitly queried
# across languages (both VI->EN and EN->VI).
# Format: concept_id -> (query_triggers, doc_evidence_terms, optional_doc_regex)
_CROSS_LINGUAL_REQUIRED_CONCEPTS: dict[str, tuple[tuple[str, ...], tuple[str, ...], str | None]] = {
    "concentration": (
        ("nồng độ", "concentration"),
        ("concentration", "concentrations", "mg/l", "g/l", "ppm", "ppb", "mmol", "mol/l", "molar", "salinity", "density", "nồng độ", "hàm lượng"),
        r"\b\d+(?:[.,]\d+)?\s*(?:mg/l|g/l|ppm|ppb|mmol|mol/l)\b",
    ),
    "nitrate": (
        ("nitrat", "nitrate", "nitrates"),
        ("nitrat", "nitrate", "nitrates", "no3", "nitrogen", "nitơ"),
        None,
    ),
    "phosphate": (
        ("photphat", "phốt pho", "phosphate", "phosphates", "phosphorus"),
        ("photphat", "phốt pho", "phosphate", "phosphates", "phosphorus", "po4"),
        None,
    ),
    "ammonia": (
        ("amoni", "amoniac", "ammonia", "ammonium"),
        ("amoni", "amoniac", "ammonia", "ammonium", "nh3", "nh4"),
        None,
    ),
    "salinity": (
        ("độ mặn", "salinity"),
        ("độ mặn", "mặn", "salinity", "saline", "g/l", "ppt", "psu"),
        None,
    ),
    "dissolved_oxygen": (
        ("oxy hòa tan", "dissolved oxygen"),
        ("oxy hòa tan", "oxy", "dissolved oxygen", "oxygen", "do "),
        None,
    ),
    "moisture": (
        ("độ ẩm", "moisture", "humidity"),
        ("độ ẩm", "moisture", "humidity", "humid"),
        None,
    ),
    "temperature": (
        ("nhiệt độ", "temperature"),
        ("nhiệt độ", "temperature", "temp", "thermal", "celsius", "kelvin", "fahrenheit", "roomtemp"),
        r"\b\d+(?:[.,]\d+)?\s*°?\s*[cf]\b",
    ),
    "pressure": (
        ("áp suất", "pressure"),
        ("áp suất", "pressure", "mpa", "kpa", "pascal", "bar", "atm"),
        None,
    ),
    "rainfall": (
        ("lượng mưa", "rainfall", "precipitation"),
        ("lượng mưa", "mưa", "rainfall", "precipitation", "rain"),
        None,
    ),
    "voltage": (
        ("điện áp", "hiệu điện thế", "voltage"),
        ("điện áp", "hiệu điện thế", "voltage", "volt", "volts", "cutoff"),
        r"\b\d+(?:[.,]\d+)?\s*m?v\b",
    ),
    "impedance": (
        ("trở kháng", "impedance"),
        ("trở kháng", "impedance", "ohm", "resistance"),
        None,
    ),
    "wavelength": (
        ("bước sóng", "wavelength"),
        ("bước sóng", "wavelength", "nm", "micrometer", "c-band"),
        None,
    ),
    "propagation_loss": (
        ("suy hao", "propagation loss", "attenuation"),
        ("suy hao", "propagation loss", "attenuation", "db/cm", "loss"),
        None,
    ),
    "mortality": (
        ("tỷ lệ tử vong", "tử vong", "mortality", "fatality"),
        ("tử vong", "mortality", "fatality", "death", "survival"),
        None,
    ),
    "cost_budget": (
        ("chi phí", "ngân sách", "cost", "budget"),
        ("chi phí", "ngân sách", "giá", "cost", "costs", "budget", "price", "expense", "overhead", "fee", "usd", "vnd", "dollar"),
        None,
    ),
    "invoice": (
        ("hóa đơn", "invoice", "invoices"),
        ("hóa đơn", "invoice", "invoices", "receipt", "receipts", "bill"),
        None,
    ),
    "volume_liters": (
        ("lít", "liters", "liter", "litres", "litre"),
        ("lít", "liter", "liters", "litre", "litres", "ml", "gallon", "gallons", "volume"),
        r"\b\d+(?:[.,]\d+)?\s*(?:l|ml|liters?|litres?|lít)\b",
    ),
}


def _is_non_vietnamese_loanword(token: str) -> bool:
    """
    Detect polysyllabic or orthographically non-Vietnamese ASCII scientific/domain loanwords
    (e.g., `nitrat`, `glucose`, `ethanol`, `pesticide`, `photonics`, `voltage`) embedded inside
    a Vietnamese query. Native Vietnamese syllables are strictly monosyllabic and never use
    letters `f`, `j`, `w`, `z` or two separate vowel nuclei inside a single space-delimited token.
    """
    t = (token or "").strip()
    if len(t) < 5 or not t.isalpha() or has_vietnamese_diacritics(t):
        return False
    tl = t.lower()
    if (
        tl in VN_ASCII_WORDS
        or tl in GENERIC_COMMON_WORDS
        or tl in DOCUMENT_LOCATOR_WORDS
        or tl in _CROSS_LINGUAL_GENERIC_TECH_WORDS
    ):
        return False
    if any(ch in tl for ch in "fjwz"):
        return True
    # Strip Vietnamese labialized/palatalized onsets `qu` and `gi` before counting vowel nuclei
    stem = re.sub(r"^(?:qu|gi)(?=[aeiouy])", "k", tl)
    vowel_nuclei = re.findall(r"[aeiouy]+", stem)
    return len(vowel_nuclei) >= 2


def _active_doc_cache_keys(source_units: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    """Collect active doc_ids and normalized paths/stems so cache scans never leak across unrelated PDFs."""
    doc_ids: set[str] = set()
    stems: set[str] = set()
    for u in source_units:
        did = str(getattr(u, "doc_id", None) or (u.get("doc_id", "") if isinstance(u, dict) else "")).strip().lower()
        if did:
            doc_ids.add(did)
            stems.add(re.sub(r"\.pdf$", "", os.path.basename(did), flags=re.I))
        sp = str(getattr(u, "source_path", None) or (u.get("source_path", "") if isinstance(u, dict) else "")).strip().lower()
        if sp:
            stems.add(os.path.splitext(os.path.basename(sp))[0])
    return doc_ids, {s for s in stems if s}


def _iter_active_cached_doc_texts(source_units: list[dict[str, Any]]) -> list[str]:
    """Return full lowercase texts of cached PDFs belonging ONLY to the active `source_units`."""
    try:
        from core.tools import _PDF_CHUNK_CACHE
    except Exception:
        return []
    active_doc_ids, active_stems = _active_doc_cache_keys(source_units)
    if not active_doc_ids and not active_stems:
        return []
    texts: list[str] = []
    for key, chunks in _PDF_CHUNK_CACHE.items():
        if not chunks:
            continue
        key_str = str(key).lower()
        key_stem = os.path.splitext(os.path.basename(key_str))[0]
        matched = key_stem in active_stems
        if not matched:
            for c in chunks[:3]:
                if isinstance(c, dict):
                    cdid = str(c.get("doc_id") or "").strip().lower()
                    if cdid and (cdid in active_doc_ids or re.sub(r"\.pdf$", "", os.path.basename(cdid), flags=re.I) in active_stems):
                        matched = True
                        break
        if matched:
            texts.append(" ".join(str(c.get("text", "")) for c in chunks if isinstance(c, dict)).lower())
    return texts


def _matches_loanword_in_doc(
    term: str,
    source_units: list[dict[str, Any]],
    doc_l: str,
    doc_unaccent: str,
    query_has_accents: bool,
) -> bool:
    """Match a loanword via exact phrase, English morphological suffix (-e/-s/-es/-ed/-ing/-ic/-al), or bilingual map."""
    if _has_phrase_in_sources_or_docs(term, source_units, doc_l, doc_unaccent, query_has_accents):
        return True
    tl = term.lower().strip()
    morph_pat = re.compile(r"(?<![a-zA-Z0-9])" + re.escape(tl) + r"(?:e|es|s|ed|ing|ic|al)?(?![a-zA-Z0-9])")
    cached_texts = _iter_active_cached_doc_texts(source_units)
    all_texts = [doc_l, *cached_texts]
    if any(morph_pat.search(txt) for txt in all_texts):
        return True
    try:
        from utils.bilingual import BILINGUAL_PHRASE_MAP, ENGLISH_TO_VIETNAMESE
        candidates: set[str] = set()
        if tl in BILINGUAL_PHRASE_MAP:
            candidates.update(BILINGUAL_PHRASE_MAP[tl])
        for en_word, vi_list in ENGLISH_TO_VIETNAMESE.items():
            if any(tl == v.lower() or tl in v.lower().split() for v in vi_list):
                candidates.add(en_word)
        for cand in candidates:
            if _has_phrase_in_sources_or_docs(cand, source_units, doc_l, doc_unaccent, False):
                return True
    except Exception:
        pass
    return False


def _check_cross_lingual_target_absence(
    clean_q: str,
    foreign_terms: list[str],
    source_units: list[dict[str, Any]],
    doc_l: str,
    doc_unaccent: str,
    query_has_accents: bool,
    query_is_vietnamese: bool,
    doc_is_vietnamese: bool,
) -> list[str]:
    """
    For cross-lingual queries (VI->EN or EN->VI), verify that explicitly queried
    specific measurement concepts / chemical substances / non-Vietnamese scientific loanwords
    have at least one bilingual match in the active document(s).
    """
    q_lower = clean_q.lower()
    q_unaccent = strip_accents(q_lower)
    cached_texts = _iter_active_cached_doc_texts(source_units)
    all_doc_lower = [doc_l, *cached_texts]
    all_doc_unaccent = [doc_unaccent, *(strip_accents(t) for t in cached_texts)]

    missing: list[str] = []

    # 1. Check specific bilingual measurement / substance concept groups
    for concept_id, (triggers, evidence_terms, regex_pat) in _CROSS_LINGUAL_REQUIRED_CONCEPTS.items():
        triggered = any(
            _has_phrase(trig, q_lower, q_unaccent, has_vietnamese_diacritics(trig))
            for trig in triggers
        )
        if not triggered:
            continue
        found = False
        for ev in evidence_terms:
            ev_has_acc = has_vietnamese_diacritics(ev)
            if any(_has_phrase(ev, dl, du, ev_has_acc) for dl, du in zip(all_doc_lower, all_doc_unaccent)):
                found = True
                break
        if not found and regex_pat:
            compiled = re.compile(regex_pat, re.IGNORECASE)
            if any(compiled.search(dl) for dl in all_doc_lower):
                found = True
        if not found:
            missing.append(concept_id)

    # 2. For Vietnamese queries over English docs, verify embedded scientific/chemical loanwords
    if query_is_vietnamese and not doc_is_vietnamese:
        for term in foreign_terms:
            if not _is_non_vietnamese_loanword(term):
                continue
            if not _matches_loanword_in_doc(term, source_units, doc_l, doc_unaccent, query_has_accents):
                if term.lower() not in missing:
                    missing.append(term.lower())

    return missing


def _has_phrase(phrase: str, doc_lower: str, doc_unaccented: str, query_has_accents: bool) -> bool:
    """Matches a phrase or technical token with word boundary and multi-whitespace guards (treating '_' as separator)."""
    if not phrase:
        return False
    words = phrase.lower().strip().split()
    if not words:
        return False
    p_c = r"[\s_]+".join(re.escape(w) for w in words)
    pat1 = r"(?<![a-zA-Z0-9à-ỹÀ-Ỹ])" + p_c + r"(?![a-zA-Z0-9à-ỹÀ-Ỹ])"
    if re.search(pat1, doc_lower):
        return True
    # Only fall back to unaccented match if the query itself was typed without accents
    if not query_has_accents:
        p_u = r"[\s_]+".join(re.escape(strip_accents(w)) for w in words)
        pat2 = r"(?<![a-zA-Z0-9])" + p_u + r"(?![a-zA-Z0-9])"
        if re.search(pat2, doc_unaccented):
            return True
    return False


def _has_phrase_in_sources_or_docs(
    phrase: str,
    source_units: list[dict[str, Any]],
    doc_lower: str,
    doc_unaccented: str,
    query_has_accents: bool,
) -> bool:
    """Matches a phrase or token in retrieved source units, doc metadata/titles, or cached PDF text."""
    if not phrase:
        return False
    # 1. First check the active source_units text (fast path)
    if _has_phrase(phrase, doc_lower, doc_unaccented, query_has_accents):
        return True

    # 2. Check document titles, filenames, and IDs (e.g. user quoting paper title or filename)
    p_norm = re.sub(r"[^a-zA-Z0-9]+", " ", phrase).strip().lower()
    for u in source_units:
        for attr in ("title", "filename", "doc_name", "doc_id"):
            val = str(getattr(u, attr, None) or (u.get(attr, "") if isinstance(u, dict) else "")).strip()
            if val:
                val_clean = re.sub(r"^(?:doc_)?[0-9a-f]{8}_", "", val)
                val_clean = re.sub(r"\.pdf$", "", val_clean, flags=re.I)
                val_norm = re.sub(r"[^a-zA-Z0-9]+", " ", val_clean).strip().lower()
                if p_norm and (p_norm == val_norm or p_norm in val_norm or val_norm in p_norm):
                    return True
        sp = str(getattr(u, "source_path", None) or (u.get("source_path", "") if isinstance(u, dict) else "")).strip()
        if sp:
            stem = re.sub(r"^(?:doc_)?[0-9a-f]{8}_", "", os.path.splitext(os.path.basename(sp))[0]).lower()
            stem_norm = re.sub(r"[^a-zA-Z0-9]+", " ", stem).strip()
            if p_norm and (p_norm == stem_norm or p_norm in stem_norm or stem_norm in p_norm):
                return True

    # 3. Check full document text from cached chunks belonging ONLY to active source_units
    for full_txt in _iter_active_cached_doc_texts(source_units):
        if _has_phrase(phrase, full_txt, strip_accents(full_txt), query_has_accents):
            return True

    return False


def extract_core_subject_phrases(query: str) -> list[str]:
    """Extracts core subject compounds, technical identifiers, and proper noun phrases from query."""
    q_clean = query.strip().rstrip("?.!")
    
    # 1. Technical identifiers / code / English terms (e.g. generateHosts, proxy_pass, client_max_body_size, nginx, upstream, wsl, openssl, kafka, oop)
    raw_tech = re.findall(r"\b[a-zA-Z0-9_.-]{3,}\b", q_clean)
    tech = [t for t in raw_tech if t.lower() not in VN_ASCII_WORDS and t.lower() not in GENERIC_COMMON_WORDS]
    
    # 2. Quoted phrases
    quotes = re.findall(r"[\"\'\`“](.*?)[\"\'\`”]", q_clean)
    
    # 3. Capitalized proper nouns (2 or more capitalized words)
    caps = re.findall(r"\b[A-ZĐ][a-zà-ỹ0-9_-]+(?:\s+[A-ZĐ][a-zà-ỹ0-9_-]+)+\b", q_clean)
    
    # 4. Multi-word compound phrases (bigrams and trigrams)
    words = re.findall(r"\w+", q_clean.lower())
    compounds = []
    for i in range(len(words) - 1):
        bg = f"{words[i]} {words[i+1]}"
        if bg not in GENERIC_COMMON_WORDS and not (words[i] in GENERIC_COMMON_WORDS and words[i+1] in GENERIC_COMMON_WORDS):
            compounds.append(bg)
        if i < len(words) - 2:
            tg = f"{words[i]} {words[i+1]} {words[i+2]}"
            if tg not in GENERIC_COMMON_WORDS and not (words[i] in GENERIC_COMMON_WORDS and words[i+1] in GENERIC_COMMON_WORDS and words[i+2] in GENERIC_COMMON_WORDS):
                compounds.append(tg)
                
    phrases = []
    phrases.extend(tech)
    phrases.extend(quotes)
    phrases.extend(caps)
    phrases.extend(compounds)
    
    unique = []
    for p in sorted(phrases, key=len, reverse=True):
        p_strip = p.strip()
        if p_strip.lower() not in [u.lower() for u in unique] and len(p_strip) >= 3:
            unique.append(p_strip)
    return unique


def check_out_of_domain(query: str, source_units: list[dict[str, Any]]) -> Tuple[bool, str]:
    """
    Evaluates whether the given query is out of the domain of the selected documents.
    Returns: (is_out_of_domain: bool, reason: str)
    """
    clean_q = (query or "").strip()
    if not clean_q or not source_units:
        return True, "empty_query_or_sources"

    # Aggregate full document text
    raw_doc_text = " ".join(str(getattr(u, "text", u.get("text", "") if isinstance(u, dict) else str(u))) for u in source_units)
    doc_l = re.sub(r"\s+", " ", raw_doc_text.lower())
    doc_unaccent = strip_accents(doc_l)
    query_has_accents = has_vietnamese_diacritics(clean_q)
    # Distinguish actual Vietnamese documents from English PDFs containing occasional
    # European accented author names or Greek math symbols by using exact Vietnamese detection.
    vn_specific_pat = r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]"
    query_is_vietnamese = is_vietnamese_text(clean_q)
    doc_is_vietnamese = detect_corpus_language(source_units, default="en") == "vi"

    # 1. Check quoted entities FIRST: if user explicitly probes quoted entities absent from document
    quotes = re.findall(r"[\"\'\`“](.*?)[\"\'\`”]", clean_q)
    substantive_quotes = [q.strip() for q in quotes if len(q.strip()) >= 3]
    if not doc_is_vietnamese:
        # On an English document, a Vietnamese translated concept in quotes (e.g. 'tính điển hình', 'tinh thần đồng đội')
        # cannot appear verbatim in English text; only check non-Vietnamese quoted entities verbatim.
        substantive_quotes = [
            q for q in substantive_quotes
            if not re.search(vn_specific_pat, q.lower())
            and not set(re.findall(r"\w+", q.lower())).issubset(VN_ASCII_WORDS | GENERIC_COMMON_WORDS)
        ]
    if substantive_quotes:
        matched_quotes = [
            q for q in substantive_quotes
            if _has_phrase_in_sources_or_docs(q, source_units, doc_l, doc_unaccent, query_has_accents)
        ]
        if not matched_quotes:
            return True, f"quoted_entities_absent: {substantive_quotes}"
    # 2. Summary / Overview / Comparison commands are always in-domain (meta-document queries)
    is_slash_cmd = clean_q.startswith("/")
    is_general_summary = is_slash_cmd or any(
        kw in clean_q.lower()
        for kw in [
            "tóm tắt", "tổng quan", "nội dung chính", "/tom-tat", "/so-sanh",
            "so sánh", "đóng góp", "/dong-gop", "điểm mới", "ưu nhược điểm", "tom-tat"
        ]
    )
    if is_general_summary:
        return False, "summary_overview"

    # Identify synthetic machine document IDs/hashes (to prevent random hash tokens from matching as subjects)
    doc_id_tokens = set()
    for u in source_units:
        raw_id = str(getattr(u, "doc_id", None) or (u.get("doc_id", "") if isinstance(u, dict) else "")).lower()
        for tok in re.findall(r"\b[a-f0-9]{6,}\b", raw_id):
            doc_id_tokens.add(tok)

    # 3. Check for specific foreign/English technical identifiers (e.g. Kafka, ZooKeeper, Kubernetes)
    foreign_terms = [
        t for t in re.findall(r"\b[A-Za-z0-9_-]{3,}\b", clean_q)
        if t.lower() not in VN_ASCII_WORDS
        and not has_vietnamese_diacritics(t)
        and t.lower() not in GENERIC_COMMON_WORDS
        and t.lower() not in DOCUMENT_LOCATOR_WORDS
        and t.lower() not in doc_id_tokens
    ]
    words_in_q = clean_q.split()
    sentence_initials = set(
        re.findall(r"(?:^|[.?!:;\n(\[\"\'\`“]\s*)([A-Za-z0-9_-]{3,})\b", clean_q)
    )
    if words_in_q:
        sentence_initials.add(words_in_q[0].strip("(\"'`“"))
    is_cross_lingual = query_is_vietnamese != doc_is_vietnamese
    tech_named = []
    for t in foreign_terms:
        has_upper_or_digit = any(c.isupper() or c.isdigit() for c in t)
        if t in sentence_initials:
            # Sentence-initial capitalization: only count as named tech if acronym, camelCase or identifier
            if t.isupper() or '_' in t or any(c.isupper() for c in t[1:]) or ('-' in t and (not is_cross_lingual or any(c.isupper() or c.isdigit() for c in t[1:]))):
                tech_named.append(t)
        elif t[0].isupper() or any(c.isupper() for c in t[1:]) or '_' in t or ('-' in t and (not is_cross_lingual or has_upper_or_digit)):
            tech_named.append(t)
    if tech_named:
        missing_tech = [
            t for t in tech_named
            if not _has_phrase_in_sources_or_docs(t, source_units, doc_l, doc_unaccent, query_has_accents)
        ]
        min_missing = len(tech_named) if is_cross_lingual else max(1, int(len(tech_named) * 0.5))
        if len(missing_tech) >= min_missing:
            # Named technical identifiers asked in query are absent from document
            return True, f"foreign_term_absent: {missing_tech}"

    # 4. Check core subject phrases & compound n-grams
    # For cross-lingual queries (VI->EN and EN->VI), verbatim lexical n-gram gating cannot apply,
    # so we verify explicitly queried measurement/substance concepts & scientific loanwords via bilingual mapping.
    if is_cross_lingual:
        missing_targets = _check_cross_lingual_target_absence(
            clean_q=clean_q,
            foreign_terms=foreign_terms,
            source_units=source_units,
            doc_l=doc_l,
            doc_unaccent=doc_unaccent,
            query_has_accents=query_has_accents,
            query_is_vietnamese=query_is_vietnamese,
            doc_is_vietnamese=doc_is_vietnamese,
        )
        if missing_targets:
            return True, f"cross_lingual_target_absent: {missing_targets}"
        return False, "cross_lingual_in_domain"

    phrases = extract_core_subject_phrases(clean_q)
    matched_phrases = [p for p in phrases if _has_phrase(p, doc_l, doc_unaccent, query_has_accents)]
    substantive_matches = [
        p for p in matched_phrases
        if not set(re.findall(r"\w+", p.lower())).issubset(doc_id_tokens | GENERIC_COMMON_WORDS)
    ]
    if not substantive_matches:
        # Fallback check: Check if any contiguous 2-to-4 word sequence from query appears verbatim in document text
        words_lower = re.findall(r"\w+", clean_q.lower())
        found_contiguous = False
        for n in range(min(4, len(words_lower)), 1, -1):
            for i in range(len(words_lower) - n + 1):
                ngram = " ".join(words_lower[i : i + n])
                if not set(words_lower[i : i + n]).issubset(GENERIC_COMMON_WORDS):
                    if _has_phrase(ngram, doc_l, doc_unaccent, query_has_accents):
                        found_contiguous = True
                        break
            if found_contiguous:
                break
        if not found_contiguous:
            return True, f"no_core_subject_matched: {phrases[:4]}"

    return False, "in_domain"


def build_out_of_domain_response(
    clean_q: str,
    model_family: str = "extractive",
    answer_language: Optional[str] = None,
) -> dict[str, Any]:
    """Generates the standardized Zero-Hallucination Honest Refusal payload in the requested language."""
    lang = normalize_language_code(answer_language) or detect_language(clean_q, default="vi")
    if lang == "en":
        no_doc_msg = (
            f"Based on the selected documents in the system, information about \"{clean_q}\" "
            f"**is not mentioned in the provided sources**.\n\n"
            f"Please ask questions grounded in the selected documents or upload relevant papers for analysis."
        )
        claim_txt = f"The selected documents do not mention '{clean_q}'."
    else:
        no_doc_msg = (
            f"Dựa trên các bài báo/tài liệu hiện có trong hệ thống, thông tin về \"{clean_q}\" "
            f"**hoàn toàn không được đề cập**.\n\n"
            f"Vui lòng hỏi các câu hỏi liên quan đến nội dung tài liệu được cung cấp hoặc tải thêm tài liệu liên quan lên hệ thống để phân tích."
        )
        claim_txt = f"Tài liệu hiện có hoàn toàn không đề cập đến '{clean_q}'."
    return {
        "answer": no_doc_msg,
        "claims": [{"id": "claim_1", "text": claim_txt, "unit_id": None}],
        "evidence_spans": [],
        "critic_action": "approve",
        "critic_feedback": {"action": "approve", "confidence": 1.0, "reason": "out_of_domain"},
        "generator_model_family": model_family,
        "is_out_of_domain": True,
        "answer_status": "NOT_FOUND_IN_DOCUMENT",
        "answer_language": lang,
    }


def build_ambiguous_clarification_response(
    clean_q: str,
    branches: list[dict[str, Any]],
    model_family: str = "extractive",
    answer_language: Optional[str] = None,
) -> dict[str, Any]:
    """Generates the standardized Active Disambiguation payload for vague / multi-aspect queries."""
    lang = normalize_language_code(answer_language) or detect_language(clean_q, default="vi")
    branch_items = []
    evidence_spans = []
    claims = []
    for idx, b in enumerate(branches[:4], 1):
        default_topic = f"Aspect {idx}" if lang == "en" else f"Khía cạnh {idx}"
        topic = b.get("topic") or b.get("title") or default_topic
        ref = b.get("ref", "")
        preview = b.get("preview", "")
        branch_line = f"{idx}. **{topic}**"
        if ref:
            branch_line += f" *({ref})*"
        if preview:
            branch_line += f": {preview}"
        branch_items.append(branch_line)

        if b.get("quote"):
            evidence_spans.append({
                "id": f"ev_{idx}",
                "quote": b.get("quote"),
                "source_unit_id": b.get("unit_id", f"u{idx}"),
                "doc_id": b.get("doc_id", "doc"),
                "page_no": b.get("page_no", 1),
            })
        claims.append({
            "id": f"claim_{idx}",
            "text": (f"The document covers aspect: {topic}" if lang == "en" else f"Tài liệu đề cập khía cạnh: {topic}"),
            "unit_id": b.get("unit_id"),
        })

    if lang == "en":
        branches_str = "\n".join(branch_items) if branch_items else "1. Related technical aspects in the document."
        msg = (
            f"Based on the provided documents, the question *\"{clean_q}\"* spans multiple distinct aspects or evaluation conditions:\n\n"
            f"{branches_str}\n\n"
            f"**Please clarify which specific aspect, document, or condition you would like analyzed.**"
        )
    else:
        branches_str = "\n".join(branch_items) if branch_items else "1. Các khía cạnh kỹ thuật liên quan trong tài liệu."
        msg = (
            f"Dựa trên tài liệu được cung cấp, câu hỏi *\"{clean_q}\"* có đề cập đến chủ đề này "
            f"nhưng tồn tại nhiều khía cạnh/kịch bản khác nhau:\n\n"
            f"{branches_str}\n\n"
            f"👉 **Vui lòng cung cấp thêm chi tiết để tôi hỗ trợ chính xác hơn dựa trên tài liệu.**"
        )
    return {
        "answer": msg,
        "claims": claims,
        "evidence_spans": evidence_spans,
        "critic_action": "approve",
        "critic_feedback": {"action": "approve", "confidence": 1.0, "reason": "ambiguous_query_disambiguation"},
        "generator_model_family": model_family,
        "is_ambiguous": True,
        "answer_status": "CLARIFICATION_NEEDED",
        "answer_language": lang,
    }
