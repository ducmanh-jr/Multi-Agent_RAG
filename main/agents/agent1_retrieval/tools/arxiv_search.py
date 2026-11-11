"""
Live ArXiv Academic Search Tool for Agent 1.
Connects directly to the official ArXiv REST API (export.arxiv.org)
to retrieve real-time, peer-reviewed scientific research papers
when queries are not covered by the local offline corpus.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Optional

from utils.logger import logger


def clean_query_for_arxiv(query: str) -> str:
    """Prepares and normalizes query keywords for the ArXiv API."""
    # Common Vietnamese scientific terms mapping for ArXiv
    vn_term_map = {
        "mô hình ngôn ngữ": "language model",
        "mô hình": "model",
        "thị giác máy tính": "computer vision",
        "thị giác": "vision",
        "học máy": "machine learning",
        "học sâu": "deep learning",
        "mạng nơ-ron": "neural network",
        "cơ chế chú ý": "attention mechanism",
        "tự chú ý": "self-attention",
        "so sánh": "comparison",
        "hiệu năng": "performance",
        "bộ nhớ": "memory",
        "tinh chỉnh": "finetuning",
        "thẩm định sự thật": "fact verification",
        "truy xuất": "retrieval",
        "khoa học": "scientific",
    }
    
    q_lower = query.lower()
    for vn, en in vn_term_map.items():
        if vn in q_lower:
            q_lower = q_lower.replace(vn, en)

    # Strip special punctuation
    cleaned = re.sub(r"[^\w\s-]", " ", q_lower)
    words = [w.strip() for w in cleaned.split() if len(w.strip()) > 1]
    
    # Avoid query being too long (ArXiv API works best with 2-5 core terms)
    return "+".join(words[:5]) if words else "deep+learning"


def search_arxiv(query: str, max_results: int = 3, timeout: int = 8) -> list[dict[str, Any]]:
    """
    Queries ArXiv API and returns structured scientific paper records.
    """
    clean_kw = clean_query_for_arxiv(query)
    api_url = f"https://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote_plus(clean_kw)}&start=0&max_results={max_results}"

    logger.info(f"[ArXiv Live Search] Querying ArXiv on Internet: '{clean_kw}'...")
    req = urllib.request.Request(
        api_url,
        headers={"User-Agent": "MultiAgentResearchRAG/1.0 (academic-research-tool)"}
    )

    results: list[dict[str, Any]] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            xml_data = resp.read()
            root = ET.fromstring(xml_data)
            
            # Namespace for Atom
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)

            for idx, entry in enumerate(entries):
                title_el = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                id_el = entry.find("atom:id", ns)
                published_el = entry.find("atom:published", ns)
                
                title = title_el.text.strip().replace("\n", " ") if title_el is not None and title_el.text else "Tài liệu ArXiv"
                summary = summary_el.text.strip().replace("\n", " ") if summary_el is not None and summary_el.text else ""
                raw_id = id_el.text.strip() if id_el is not None and id_el.text else f"arxiv_{idx}"
                clean_id = raw_id.split("/")[-1].replace(".", "_")
                
                # Find PDF link
                pdf_url = f"https://arxiv.org/pdf/{raw_id.split('/')[-1]}.pdf"
                for link in entry.findall("atom:link", ns):
                    if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                        pdf_url = link.attrib.get("href", pdf_url)
                        break

                pub_year = published_el.text[:4] if published_el is not None and published_el.text else "2024"

                results.append({
                    "id": f"arxiv_live_{clean_id}",
                    "doc_id": f"arxiv_{clean_id}",
                    "title": f"[{pub_year}] {title}",
                    "text": summary,
                    "source": "arxiv_internet",
                    "pdf_url": pdf_url,
                    "url": pdf_url,
                    "page_no": 1,
                    "rrf_score": round(1.0 / (60.0 + idx + 1), 5),
                })

        logger.info(f"[ArXiv Live Search] Found {len(results)} live scientific papers from ArXiv!")
    except Exception as e:
        logger.warning(f"[ArXiv Live Search] Failed to fetch from ArXiv ({e}). Falling back.")

    return results
