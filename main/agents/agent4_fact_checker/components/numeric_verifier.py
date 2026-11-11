"""
Deterministic Numerical Verifier with Unit & Percentage Normalization.
Validates that numeric metrics, percentages, counts, and quantitative units
in a claim are substantiated by the source context with rounding tolerance.
Eliminates hallucinated quantitative data and false positive contradictions.
"""

from __future__ import annotations

import re
from typing import Optional


def _normalize_number_token(token: str) -> Optional[float]:
    """Parses a numeric token to float, properly distinguishing Vietnamese decimal commas vs thousand separators."""
    clean = token.rstrip("%").strip().replace("−", "-")
    if not clean:
        return None
    # If both . and , exist: 1,234.56 or 1.234,56
    if "." in clean and "," in clean:
        if clean.rfind(".") > clean.rfind(","):
            clean = clean.replace(",", "")
        else:
            clean = clean.replace(".", "").replace(",", ".")
    elif "," in clean:
        # Check if comma is thousand separator (e.g. 1,000 or 1,000,000) vs decimal separator (1,5 or 0,75)
        parts = clean.split(",")
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]) and len(parts[0]) <= 3 and len(parts[0]) > 0:
            clean = clean.replace(",", "")
        else:
            clean = clean.replace(",", ".")
    try:
        return float(clean)
    except ValueError:
        return None


def extract_numbers(text: str) -> list[str]:
    """Extract raw numbers, integers, negative values, floats, and percentages."""
    pattern = r"(?<!\w)[−-]??\d+(?:[.,]\d+)*(?:%|\b)"
    raw_matches = re.findall(pattern, text)
    result = []
    for m in raw_matches:
        clean_m = m.replace("−", "-")
        # Normalize thousand-separator commas e.g. 18,587 -> 18587
        parts = clean_m.rstrip("%").split(",")
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            clean_m = clean_m.replace(",", "")
        result.append(clean_m)
    return result


def _parse_numeric_value(token: str) -> Optional[float]:
    """Parses a numeric token to float supporting negative values and decimal commas."""
    return _normalize_number_token(token)


def _extract_quantities_with_units(text: str) -> list[tuple[float, str]]:
    """
    Extracts quantities with common scientific & computing units.
    E.g. ('1.5', 'gb') -> (1.5, 'gb'), ('40.97', '%')
    """
    quantities = []
    pattern = re.compile(
        r"(?P<val>[−-]??\d+(?:[.,]\d+)*)\s*(?P<unit>%|gb|mb|tb|kb|triệu|tỷ|nghìn|[kmb])?(?=[^\w]|$)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        val_str = m.group("val")
        unit = (m.group("unit") or "").lower()
        val = _normalize_number_token(val_str)
        if val is not None:
            quantities.append((val, unit))
    return quantities


def _numbers_match_with_tolerance(val_claim: float, unit_claim: str, val_ctx: float, unit_ctx: str) -> bool:
    """Checks if two quantities match with unit normalization and rounding tolerance."""
    # 1. Exact or near float match
    if abs(val_claim - val_ctx) < 1e-5:
        return True

    # 2. Percentage vs Ratio: 40.97% vs 0.4097
    if unit_claim == "%" and unit_ctx == "":
        if abs((val_claim / 100.0) - val_ctx) < 0.01:
            return True
    if unit_claim == "" and unit_ctx == "%":
        if abs(val_claim - (val_ctx / 100.0)) < 0.01:
            return True

    # 3. Unit normalization for memory (GB <-> MB)
    if unit_claim == "gb" and unit_ctx == "mb":
        if abs(val_claim * 1000 - val_ctx) / max(val_ctx, 1.0) <= 0.05 or abs(val_claim * 1024 - val_ctx) / max(val_ctx, 1.0) <= 0.05:
            return True
    if unit_claim == "mb" and unit_ctx == "gb":
        if abs(val_claim - val_ctx * 1000) / max(val_claim, 1.0) <= 0.05 or abs(val_claim - val_ctx * 1024) / max(val_claim, 1.0) <= 0.05:
            return True

    # 4. Magnitudes: k/thousand (10k <-> 10000), m/million
    scale_claim = 1000.0 if unit_claim in ("k", "nghìn") else (1000000.0 if unit_claim in ("m", "triệu") else (1000000000.0 if unit_claim in ("b", "tỷ") else 1.0))
    scale_ctx = 1000.0 if unit_ctx in ("k", "nghìn") else (1000000.0 if unit_ctx in ("m", "triệu") else (1000000000.0 if unit_ctx in ("b", "tỷ") else 1.0))
    normalized_claim = val_claim * scale_claim
    normalized_ctx = val_ctx * scale_ctx
    if abs(normalized_claim - normalized_ctx) / max(normalized_ctx, 1e-6) <= 0.02:
        return True

    # 5. Rounding tolerance: e.g. 41% vs 40.97%
    if max(val_claim, val_ctx) > 0:
        rel_diff = abs(val_claim - val_ctx) / max(val_claim, val_ctx)
        if rel_diff <= 0.02:  # within 2% rounding margin
            return True

    return False


def verify_numbers_in_claim(claim: str, context: str) -> tuple[bool, str]:
    """
    Checks whether all critical numbers mentioned in the claim are substantiated in the context.
    Supports unit normalization (% vs decimal, GB vs MB) and rounding tolerance (<= 2%).
    """
    claim_nums = extract_numbers(claim)
    if not claim_nums:
        return True, "Không có số liệu định lượng cần kiểm chứng."

    context_nums = set(extract_numbers(context))
    claim_quantities = _extract_quantities_with_units(claim)
    context_quantities = _extract_quantities_with_units(context)

    missing = []
    for num_str in claim_nums:
        clean_num = num_str.rstrip("%")
        # 1. Quick verbatim / substring match
        if any(clean_num in c_num for c_num in context_nums):
            continue

        # 2. Advanced semantic & unit normalization match
        val_claim = _normalize_number_token(num_str)
        if val_claim is None:
            missing.append(num_str)
            continue

        unit_claim = "%" if "%" in num_str else ""
        for v, u in claim_quantities:
            if abs(v - val_claim) < 1e-4:
                unit_claim = u
                break

        matched = False
        for val_ctx, unit_ctx in context_quantities:
            if _numbers_match_with_tolerance(val_claim, unit_claim, val_ctx, unit_ctx):
                matched = True
                break

        if not matched:
            missing.append(num_str)

    if missing:
        return False, f"Số liệu không có trong tài liệu nguồn: {', '.join(missing)}"
    return True, "Toàn bộ số liệu định lượng khớp với tài liệu nguồn (đã chuẩn hóa đơn vị)."
