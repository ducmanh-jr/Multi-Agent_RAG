"""
Senior Peer-Reviewer Statistical Auditor for Agent 4 (Fact-Checker).
Audits empirical scientific claims for statistical significance, sample sizes,
error bars, multi-seed validation, hardware baseline parity, and cherry-picking bias.
"""

from __future__ import annotations

import re
from typing import Any
from utils.logger import logger


# Patterns indicating statistical rigor
MULTI_SEED_PATTERNS = [
    r"\b(random\s*seed|multiple\s*seeds|3\s*seeds|5\s*seeds)\b",
    r"\b(standard\s*deviation|std\s*dev|\bstd\b|\bvariance\b)\b",
    r"(±|\+/-|\+\-)\s*\d+(\.\d+)?",
    r"\b(confidence\s*interval|95%\s*ci|p\s*<\s*0\.0\d+)\b",
    r"\b(statistically\s*significant|paired\s*t-test|wilcoxon)\b",
]

SAMPLE_SIZE_PATTERNS = [
    r"\b\d+([,\.]\d+)?\s*(samples|examples|tokens|parameters|instances|sequences|questions)\b",
    r"\b(n\s*=\s*\d+|dataset\s*size|benchmark\s*split)\b",
]

HARDWARE_PATTERNS = [
    r"\b(a100|h100|v100|rtx\s*3090|rtx\s*4090|tesla\s*t4|tpu\s*v[2-5])\b",
    r"\b(gpu\s*hours|compute\s*budget|batch\s*size|vram)\b",
]


def audit_statistical_methodology(claims: list[str], corpus_text: str) -> dict[str, Any]:
    """
    Performs a senior peer-reviewer audit of the paper's statistical rigor.
    Returns composite score (0-10), findings checklist, and integrity rating.
    """
    corpus_lower = corpus_text.lower()
    full_text = " ".join(claims).lower() + " " + corpus_lower

    # 1. Check Variance & Multi-seed Reporting
    multi_seed_hits = []
    for pat in MULTI_SEED_PATTERNS:
        matches = re.findall(pat, full_text, flags=re.IGNORECASE)
        if matches:
            multi_seed_hits.extend(matches)
    has_variance = len(multi_seed_hits) > 0

    # 2. Check Sample Size Transparency
    sample_size_hits = []
    for pat in SAMPLE_SIZE_PATTERNS:
        matches = re.findall(pat, full_text, flags=re.IGNORECASE)
        if matches:
            sample_size_hits.extend(matches)
    has_sample_size = len(sample_size_hits) > 0

    # 3. Check Hardware Parity
    hardware_hits = []
    for pat in HARDWARE_PATTERNS:
        matches = re.findall(pat, full_text, flags=re.IGNORECASE)
        if matches:
            hardware_hits.extend(matches)
    has_hardware = len(hardware_hits) > 0

    # 4. Compute Senior Reviewer Integrity Score (0.0 to 10.0)
    score = 6.5  # Baseline score for peer-reviewed academic preprints
    findings: list[str] = []

    if has_variance:
        score += 2.0
        findings.append("[ĐẠT] Báo cáo sai số thống kê / độ lệch chuẩn (±σ) hoặc nhiều random seeds độc lập.")
    else:
        score -= 1.0
        findings.append("[LƯU Ý] Thiếu độ lệch chuẩn (±σ) rõ ràng giữa các lần chạy; nguy cơ kết quả là lần chạy tốt nhất (cherry-picking).")

    if has_sample_size:
        score += 1.0
        findings.append("[ĐẠT] Cỡ mẫu (sample size / tokens / sequences) được định lượng minh bạch.")
    else:
        findings.append("[THÔNG TIN] Chưa tìm thấy mô tả chi tiết cỡ mẫu cụ thể trên tập benchmark.")

    if has_hardware:
        score += 0.5
        findings.append("[ĐẠT] Cấu hình phần cứng (GPU/VRAM) được công bố rõ ràng.")
    else:
        findings.append("[THÔNG TIN] Không nêu rõ thông số card đồ họa cụ thể trong trích đoạn hiện tại.")

    final_score = max(1.0, min(10.0, round(score, 1)))

    if final_score >= 8.0:
        integrity_rating = "HIGH_RIGOR (Độ tin cậy phương pháp rất cao)"
    elif final_score >= 6.0:
        integrity_rating = "MODERATE_RIGOR (Độ tin cậy phương pháp đạt chuẩn)"
    else:
        integrity_rating = "CAUTION_POTENTIAL_BIAS (Cần lưu ý kiểm chứng độc lập)"

    logger.info(f"[Statistical Auditor] Score: {final_score}/10.0 | Rating: {integrity_rating}")

    return {
        "score": final_score,
        "rating": integrity_rating,
        "has_variance_reported": has_variance,
        "has_sample_size_reported": has_sample_size,
        "has_hardware_specified": has_hardware,
        "findings": findings,
    }
