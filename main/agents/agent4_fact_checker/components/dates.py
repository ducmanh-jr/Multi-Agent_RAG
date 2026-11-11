"""
Bilingual Date & Temporal Consistency Verifier.
Validates explicit calendar dates (English, Vietnamese, ISO format) to prevent temporal hallucination.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

_MONTH_NAMES = (
    "january jan",
    "february feb",
    "march mar",
    "april apr",
    "may",
    "june jun",
    "july jul",
    "august aug",
    "september sept sep",
    "october oct",
    "november nov",
    "december dec",
)
_MONTHS = {name: month for month, names in enumerate(_MONTH_NAMES, 1) for name in names.split()}
_MONTH = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_PATTERNS = [
    re.compile(
        rf"(?<![\w+−-])(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+"
        rf"(?P<month>{_MONTH})\s*,?\s*(?P<year>\d{{4}})(?![\w%％])",
        re.I,
    ),
    re.compile(
        rf"(?<!\w)(?P<month>{_MONTH})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?"
        r"(?:\s*,\s*|\s+)(?P<year>\d{4})(?![\w%％])",
        re.I,
    ),
    re.compile(
        r"(?<![\w+−-])(?:ngày\s+)?(?P<day>\d{1,2})\s+tháng\s+(?P<month>\d{1,2})"
        r"\s+(?:năm\s+)?(?P<year>\d{4})(?![\w%％])",
        re.I,
    ),
    re.compile(r"(?<![\w/+−-])(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})(?![\w/%％-])"),
    re.compile(r"\b(?:in|năm|year|vào năm|during)?\s*(?P<year>(?:18|19|20)\d{2})\b", re.I),
]
_NEGATED = re.compile(
    r"(?:\bnot|\bnever|rather than|instead of|không phải|không đạt)"
    r"(?:\s+(?:on|ngày))?\s*$",
    re.I,
)


@dataclass(frozen=True)
class DateMention:
    start: int
    end: int
    raw: str
    year: int
    value: Optional[date]
    negated: bool


def extract_date_mentions(text: str) -> list[DateMention]:
    """Extracts date mentions without cross-quote assembly."""
    mentions: list[DateMention] = []
    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            if any(match.start() < m.end and m.start < match.end() for m in mentions):
                continue
            # Guard against technical parameters: d_model = 2048, port: 2020, 1920x1080
            prefix = text[max(0, match.start() - 10) : match.start()]
            suffix = text[match.end() : min(len(text), match.end() + 10)]
            if re.search(r"[=:xX]\s*$", prefix) or re.search(r"^\s*(?:x\d|px|dim|tokens|steps|bytes|kb|mb|gb)", suffix, re.I):
                continue
            year = int(match.groupdict().get("year", 0))
            month = 1
            day = 1
            val = None
            if "month" in match.groupdict() and "day" in match.groupdict():
                m_str = match.group("month").lower().rstrip(".")
                month = int(m_str) if m_str.isdecimal() else _MONTHS.get(m_str, 1)
                day = int(match.group("day"))
                try:
                    val = date(year, month, day)
                except ValueError:
                    val = None

            mentions.append(
                DateMention(
                    start=match.start(),
                    end=match.end(),
                    raw=match.group(0),
                    year=year,
                    value=val,
                    negated=bool(_NEGATED.search(text[max(0, match.start() - 40) : match.start()])),
                )
            )
    return mentions


def verify_dates_in_claim(claim: str, context: str) -> tuple[bool, str]:
    """
    Verifies that explicit dates or years in the claim exist in the context.
    """
    c_mentions = extract_date_mentions(claim)
    if not c_mentions:
        return True, "Không có mốc thời gian cần kiểm chứng."

    ctx_mentions = extract_date_mentions(context)
    ctx_years = {m.year for m in ctx_mentions}

    for cm in c_mentions:
        if cm.year and cm.year not in ctx_years:
            # Also check literal year string in context with word boundary
            if not re.search(rf"\b{cm.year}\b", context):
                return False, f"Mốc thời gian (năm {cm.year}) không tồn tại trong tài liệu nguồn."

    return True, "Toàn bộ mốc thời gian khớp với tài liệu nguồn."
