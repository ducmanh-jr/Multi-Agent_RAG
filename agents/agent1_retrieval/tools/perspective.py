"""
Multi-Perspective Query Decomposition (Inspired by Stanford STORM).
Decomposes complex scientific inquiries into distinct angles (Theoretical, Methodological, Comparative, Limitations).
"""

from __future__ import annotations

from typing import Any
from utils.llm_client import llm_client
from utils.logger import logger


def generate_research_perspectives(query: str) -> tuple[list[str], list[str]]:
    """
    Returns:
      - perspectives: list of viewpoints (e.g. ['Theoretical foundation', 'Empirical benchmark'])
      - sub_queries: focused search queries representing each perspective.
    """
    prompt = f"""Bạn là một chuyên gia nghiên cứu khoa học cấp cao (Stanford STORM Multi-Perspective Decomposer).
Hãy phân rã câu hỏi nghiên cứu sau đây thành:
1. 3 góc nhìn khoa học cốt lõi (perspectives): ví dụ như Cơ sở lý thuyết, Phương pháp thực nghiệm, So sánh đối chuẩn/Giới hạn.
2. 3 đến 4 truy vấn tìm kiếm chuyên biệt (sub_queries) nhằm thu thập bằng chứng đầy đủ nhất cho các góc nhìn này.

Câu hỏi nghiên cứu:
"{query}"

YÊU CẦU TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON (không kèm markdown):
{{
  "perspectives": ["Góc nhìn 1", "Góc nhìn 2", "Góc nhìn 3"],
  "sub_queries": ["truy vấn 1", "truy vấn 2", "truy vấn 3"]
}}
"""
    try:
        data = llm_client.generate_json(prompt)
        perspectives = data.get("perspectives", ["Tổng quan phương pháp", "Kết quả thực nghiệm"])
        sub_queries = data.get("sub_queries", [query])
        if query not in sub_queries:
            sub_queries.insert(0, query)
        return perspectives, sub_queries
    except Exception as e:
        logger.warning(f"Failed to generate perspectives with LLM ({e}), using default decomposition.")
        return ["Tổng quan phương pháp", "Đánh giá thực nghiệm"], [query]
