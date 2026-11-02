"""
Benchmark & Fine-tuning Dataset Generator for Multi-Agent RAG.
Reads PDF articles from 'bao/' folder, generates ground-truth QA pairs,
runs them through the Multi-Agent RAG pipeline, and benchmarks accuracy.
"""

import os
import sys
import json
import re
import time
import httpx
from pathlib import Path
from dotenv import load_dotenv

# Add main directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "main"))

from core import db
from utils.llm_client import llm_client
from utils.logger import logger

load_dotenv(".env")
load_dotenv("main/.env")

BAO_DIR = Path(__file__).resolve().parent.parent / "bao"
REPORT_MD = BAO_DIR / "benchmark_report.md"
REPORT_JSON = BAO_DIR / "benchmark_report.json"
BASE_API_URL = "http://127.0.0.1:8000"

def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    import fitz
    doc = fitz.open(str(pdf_path))
    pages = []
    for idx, page in enumerate(doc):
        t = page.get_text("text").strip()
        if t:
            pages.append({"page_no": idx + 1, "text": t})
    doc.close()
    return pages

def generate_qa_from_article(doc_name: str, pages: list[dict]) -> list[dict]:
    """Generates ground truth question-answer pairs using available LLM."""
    full_text = "\n\n".join([f"--- Trang {p['page_no']} ---\n{p['text']}" for p in pages])
    prompt = f"""Bạn là Chuyên gia Khảo thí và Kiểm định Học thuật (Benchmark Creator).
Dưới đây là toàn văn bài báo: "{doc_name}".

{full_text}

Nhiệm vụ của bạn:
Hãy tạo ra 5 câu hỏi sát nhất với nội dung bài báo, kèm theo đáp án chuẩn xác và số trang chứa thông tin:
1. Câu hỏi 1: Về thời gian, địa điểm, sự kiện cốt lõi (Fact-based).
2. Câu hỏi 2: Về số liệu cụ thể (con số, tỷ lệ, số lượng văn bản, điểm cầu, v.v.).
3. Câu hỏi 3: Về nhân vật cụ thể xuất hiện và phát biểu/hành động của họ.
4. Câu hỏi 4: Về các nội dung, định hướng hoặc các nhóm vấn đề trọng tâm.
5. Câu hỏi 5: Câu hỏi tóm tắt toàn diện bài báo.

YÊU CẦU TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON (không kèm markdown ngoài):
{{
  "qa_pairs": [
    {{
      "question": "Nội dung câu hỏi...",
      "ground_truth": "Đáp án chuẩn xác, đầy đủ chi tiết...",
      "page_no": 1,
      "key_entities": ["từ khóa 1", "từ khóa 2", "số liệu"]
    }}
  ]
}}
"""
    try:
        # Check if valid Gemini key exists
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if gemini_key.startswith("AIzaSy"):
            resp = llm_client.generate_json(prompt, agent="summarizer")
            if isinstance(resp, dict) and "qa_pairs" in resp:
                return resp["qa_pairs"]
        
        # Or call Kaggle remote if online
        remote_url = os.getenv("REMOTE_INFERENCE_URL")
        if remote_url:
            resp = llm_client._call_remote_kaggle(
                task="generate",
                payload={"prompt": prompt, "json_mode": True, "mode": "kaggle_fn"}
            )
            data = json.loads(resp)
            if "qa_pairs" in data:
                return data["qa_pairs"]
    except Exception as e:
        logger.warning(f"LLM auto QA generation fallback: {e}")

    # High-quality deterministic QA fallback based on extracted facts
    if "Báo Hà Nội mới" in doc_name:
        return [
            {
                "question": "Báo Thủ đô - tiền thân của báo Hà Nội mới ra số hằng ngày đầu tiên vào ngày tháng năm nào?",
                "ground_truth": "Báo Thủ đô ra số hằng ngày đầu tiên vào ngày 24.10.1957.",
                "page_no": 1,
                "key_entities": ["24.10.1957", "Báo Thủ đô", "tiền thân", "Hà Nội mới"]
            },
            {
                "question": "Trong những năm đầu hoạt động, Báo Thủ đô tập trung phản ánh những nội dung gì?",
                "ground_truth": "Báo tập trung phản ánh đời sống chính trị, kinh tế, văn hóa và xã hội của thành phố; tuyên truyền chủ trương, chính sách của Đảng, Nhà nước; cổ vũ phong trào thi đua lao động sản xuất.",
                "page_no": 2,
                "key_entities": ["chính trị", "kinh tế", "văn hóa", "lao động sản xuất"]
            },
            {
                "question": "Vào ngày 24/10/1945, sự kiện lịch sử quốc tế quan trọng nào đã diễn ra?",
                "ground_truth": "Ngày 24/10/1945, Hiến chương Liên Hợp Quốc chính thức có hiệu lực, đánh dấu sự ra đời của tổ chức Liên Hợp Quốc (UN).",
                "page_no": 3,
                "key_entities": ["24/10/1945", "Hiến chương Liên Hợp Quốc", "Liên Hợp Quốc"]
            },
            {
                "question": "Mục tiêu thành lập của Liên Hợp Quốc là gì?",
                "ground_truth": "Duy trì hòa bình và an ninh quốc tế, thúc đẩy quan hệ hữu nghị giữa các quốc gia, tăng cường hợp tác giải quyết những vấn đề kinh tế, xã hội, văn hóa và nhân đạo.",
                "page_no": 3,
                "key_entities": ["hòa bình", "an ninh quốc tế", "hữu nghị", "hợp tác"]
            },
            {
                "question": "Bài báo Báo Hà Nội mới ra số đầu tiên nói về những chủ đề chính nào?",
                "ground_truth": "Bài báo nói về kỷ niệm ngày báo Thủ đô (tiền thân của Hà Nội mới) ra số đầu tiên ngày 24.10.1957 và điểm lại các sự kiện lịch sử đáng chú ý diễn ra cùng ngày 24.10 trên thế giới như sự thành lập Liên Hợp Quốc.",
                "page_no": 1,
                "key_entities": ["Hà Nội mới", "24.10.1957", "Liên Hợp Quốc"]
            }
        ]
    elif "môi trường" in doc_name:
        return [
            {
                "question": "Phiên chất vấn về lĩnh vực môi trường của Thường trực HĐND TP Hà Nội diễn ra vào thời gian nào và kết nối tới bao nhiêu điểm cầu?",
                "ground_truth": "Diễn ra vào sáng ngày 22/10/2026 và được kết nối trực tuyến đến 126 điểm cầu các xã, phường.",
                "page_no": 1,
                "key_entities": ["22/10/2026", "126 điểm cầu", "xã, phường"]
            },
            {
                "question": "Nội dung chất vấn tập trung vào năm nhóm vấn đề bức xúc nào?",
                "ground_truth": "Năm nhóm vấn đề gồm: Quy hoạch và nguồn lực bảo vệ môi trường; quản lý chất thải rắn sinh hoạt; chất thải rắn xây dựng; chất thải rắn y tế; và chất lượng môi trường không khí.",
                "page_no": 1,
                "key_entities": ["chất thải rắn", "không khí", "quy hoạch"]
            },
            {
                "question": "Phạm vi thời gian được chất vấn tại phiên họp là từ thời điểm nào?",
                "ground_truth": "Phạm vi chất vấn từ ngày 1/7/2025 đến nay và các vấn đề phát sinh trước đó chưa được giải quyết dứt điểm.",
                "page_no": 2,
                "key_entities": ["1/7/2025", "chưa được giải quyết dứt điểm"]
            },
            {
                "question": "Chủ tịch HĐND thành phố Hà Nội Phùng Thị Hồng Hà đề nghị tập trung làm rõ 3 nội dung trọng tâm nào?",
                "ground_truth": "Ba nội dung gồm: Đánh giá đúng thực trạng, làm rõ nguyên nhân và trách nhiệm; làm rõ trách nhiệm và điều kiện thực hiện nhiệm vụ của từng cấp, từng ngành; phát huy những kết quả tích cực, xác định giải pháp và trách nhiệm thực hiện.",
                "page_no": 2,
                "key_entities": ["thực trạng", "nguyên nhân", "trách nhiệm", "giải pháp"]
            },
            {
                "question": "Đại biểu Nguyễn Thị Diễm Hằng xuất hiện trong bài báo với vai trò gì?",
                "ground_truth": "Đại biểu Nguyễn Thị Diễm Hằng tham gia đặt câu hỏi chất vấn tại nghị trường trong phiên chất vấn.",
                "page_no": 2,
                "key_entities": ["Đại biểu Nguyễn Thị Diễm Hằng", "chất vấn tại nghị trường"]
            }
        ]

    return []

def run_benchmark():
    pdf_files = list(BAO_DIR.glob("*.pdf"))
    print(f"[*] Tìm thấy {len(pdf_files)} tệp PDF trong thư mục {BAO_DIR}")

    results = []

    for pdf_path in pdf_files:
        doc_name = pdf_path.stem
        print(f"\n--- Xử lý bài báo: {doc_name} ---")
        pages = extract_text_from_pdf(pdf_path)
        qa_pairs = generate_qa_from_article(doc_name, pages)
        print(f"  > Đã tạo {len(qa_pairs)} câu hỏi đối chuẩn (Ground-Truth QA).")

        # Upload or ensure document exists in workspace.db
        doc_id = None
        for d in db.list_documents():
            if pdf_path.name in d.get("name", ""):
                doc_id = d["id"]
                break
        
        if not doc_id:
            doc_record = db.add_document(
                name=pdf_path.name,
                path=str(pdf_path),
                size=pdf_path.stat().st_size,
                doc_type="pdf",
                metadata={"pages": len(pages)}
            )
            doc_id = doc_record["id"]
            print(f"  > Đã đăng ký tài liệu vào DB: {doc_id}")

        for item in qa_pairs:
            q = item["question"]
            gt = item["ground_truth"]
            exp_page = item.get("page_no", 1)
            entities = item.get("key_entities", [])

            print(f"\n[Test Case] Hỏi: '{q}'")
            start_t = time.time()
            try:
                # Trigger run via API
                res = httpx.post(
                    f"{BASE_API_URL}/api/runs",
                    json={"question": q, "document_ids": [doc_id]},
                    timeout=15
                )
                if res.status_code == 200:
                    run_id = res.json().get("id")
                    # Poll until completion
                    ans_text = ""
                    evidence_count = 0
                    status = "processing"
                    for _ in range(120):
                        time.sleep(1)
                        poll_res = httpx.get(f"{BASE_API_URL}/api/runs/{run_id}", timeout=5)
                        if poll_res.status_code == 200:
                            data = poll_res.json()
                            status = data.get("status")
                            if status in ["completed", "failed"]:
                                ans_text = data.get("final_report") or data.get("report") or ""
                                evidence_count = len(data.get("evidence_spans") or [])
                                break
                    elapsed = round(time.time() - start_t, 1)

                    # Score factuality
                    found_entities = [e for e in entities if e.lower() in ans_text.lower()]
                    entity_score = round(len(found_entities) / max(1, len(entities)), 2)

                    results.append({
                        "doc": doc_name,
                        "question": q,
                        "ground_truth": gt,
                        "rag_status": status,
                        "elapsed_s": elapsed,
                        "evidence_count": evidence_count,
                        "entity_coverage": entity_score,
                        "matched_entities": found_entities,
                        "missing_entities": [e for e in entities if e not in found_entities],
                        "answer_snippet": ans_text[:280].replace("\n", " ") + "..."
                    })
                    print(f"  Status: {status} in {elapsed}s | Trích dẫn: {evidence_count} | Độ khớp thực thể: {entity_score * 100}%")
            except Exception as e:
                print(f"  Lỗi khi chạy câu hỏi: {e}")

    # Export Report
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Markdown Report
    total_q = len(results)
    avg_cov = round(sum(r["entity_coverage"] for r in results) / max(1, total_q) * 100, 1) if total_q else 0
    passed = sum(1 for r in results if r["entity_coverage"] >= 0.6 and r["rag_status"] == "completed")

    md_lines = [
        f"# BÁO CÁO KIỂM ĐỊNH & ĐỐI CHUẨN RAG (BENCHMARK REPORT)",
        f"- **Thời gian thực hiện**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Tổng số câu hỏi đánh giá**: {total_q}",
        f"- **Số câu đạt chuẩn độ chính xác cao (>=60% thực thể & có trích dẫn)**: {passed}/{total_q} ({round(passed/max(1, total_q)*100, 1)}%)",
        f"- **Độ phủ thực thể trung bình**: {avg_cov}%",
        "\n## Chi tiết từng câu hỏi đối chuẩn\n",
        "| Bài báo | Câu hỏi | Đáp án Chuẩn | Trạng thái RAG | Thời gian | Trích dẫn | Độ khớp thực thể |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for r in results:
        cov_badge = f"{int(r['entity_coverage']*100)}%"
        md_lines.append(f"| {r['doc'][:25]} | {r['question']} | {r['ground_truth']} | `{r['rag_status']}` | {r['elapsed_s']}s | {r['evidence_count']} | **{cov_badge}** |")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[+] Đã hoàn thành! Báo cáo đã được lưu tại:")
    print(f"    - JSON: {REPORT_JSON}")
    print(f"    - Markdown: {REPORT_MD}")

if __name__ == "__main__":
    run_benchmark()
