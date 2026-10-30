"""
Deterministic Numerical Verifier.
Validates that numeric metrics, percentages, and counts in a claim are substantiated by the source context.
Eliminates hallucinated quantitative data.
"""

from __future__ import annotations

import re


def extract_numbers(text: str) -> list[str]:
    """Extract numbers, floats, and percentages."""
    # Matches integers, decimals, and percentages like 40.97%, 18,587, 1024, 3.5
    pattern = r"\b\d+(?:[.,]\d+)*(?:%|\b)"
    return [m.strip().replace(",", "") for m in re.findall(pattern, text)]


def verify_numbers_in_claim(claim: str, context: str) -> tuple[bool, str]:
    """
    Checks whether all critical numbers mentioned in the claim appear in the context.
    Returns:
      - is_valid: True if numbers are grounded or no numbers exist
      - reason: detailed explanation
    """
    claim_nums = extract_numbers(claim)
    if not claim_nums:
        return True, "Không có số liệu định lượng cần kiểm chứng."

    context_nums = set(extract_numbers(context))
    missing = []
    for num in claim_nums:
        clean_num = num.rstrip("%")
        found = any(clean_num in c_num for c_num in context_nums)
        if not found:
            missing.append(num)

    if missing:
        return False, f"Số liệu không có trong tài liệu nguồn: {', '.join(missing)}"
    return True, "Toàn bộ số liệu định lượng khớp với tài liệu nguồn."
