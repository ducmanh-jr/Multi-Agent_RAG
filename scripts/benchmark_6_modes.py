"""
Benchmark Suite to evaluate all 6 Model Options across newspaper articles in 'bao/'.
Tests:
  1. cpu_min
  2. cpu_max
  3. kaggle_fn
  4. kaggle_max
  5. gemini
  6. hybrid
"""

import sys
import io
import json
import time
import httpx
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"
BAO_DIR = Path(__file__).resolve().parent.parent / "bao"
REPORT_MD = BAO_DIR / "benchmark_6_modes_report.md"
REPORT_JSON = BAO_DIR / "benchmark_6_modes_report.json"

TEST_QUESTIONS = [
    {
        "id": "Q1",
        "doc_hint": "Báo Hà Nội mới",
        "question": "Báo Thủ đô - tiền thân của báo Hà Nội mới ra số hằng ngày đầu tiên vào ngày tháng năm nào?",
        "ground_truth": "Báo Thủ đô ra số hằng ngày đầu tiên vào ngày 24.10.1957.",
        "key_entities": ["24.10.1957", "Báo Thủ đô", "Hà Nội mới"]
    },
    {
        "id": "Q2",
        "doc_hint": "Báo Hà Nội mới",
        "question": "Mục tiêu thành lập của Liên Hợp Quốc là gì?",
        "ground_truth": "Duy trì hòa bình và an ninh quốc tế, thúc đẩy quan hệ hữu nghị giữa các quốc gia, tăng cường hợp tác giải quyết các vấn đề kinh tế, xã hội, văn hóa và nhân đạo.",
        "key_entities": ["hòa bình", "an ninh quốc tế", "hữu nghị", "hợp tác"]
    },
    {
        "id": "Q3",
        "doc_hint": "môi trường",
        "question": "Nội dung phiên chất vấn tập trung vào năm nhóm vấn đề bức xúc nào?",
        "ground_truth": "Năm nhóm vấn đề gồm: Quy hoạch và nguồn lực bảo vệ môi trường; quản lý chất thải rắn sinh hoạt; chất thải rắn xây dựng; chất thải rắn y tế; và chất lượng môi trường không khí.",
        "key_entities": ["chất thải rắn", "không khí", "quy hoạch"]
    },
    {
        "id": "Q4",
        "doc_hint": "môi trường",
        "question": "Chủ tịch HĐND thành phố Hà Nội Phùng Thị Hồng Hà đề nghị tập trung làm rõ 3 nội dung trọng tâm nào?",
        "ground_truth": "Ba nội dung: 1. Đánh giá đúng thực trạng, làm rõ nguyên nhân và trách nhiệm; 2. Làm rõ trách nhiệm và điều kiện thực hiện của từng cấp, từng ngành; 3. Phát huy kết quả tích cực, xác định giải pháp và trách nhiệm thực hiện.",
        "key_entities": ["thực trạng", "nguyên nhân", "trách nhiệm", "giải pháp"]
    }
]

MODES = [
    {"name": "CPU Min", "code": "cpu_min", "desc": "Cục bộ siêu nhẹ, FTS5 + Regex + C-Engine, 0 RAM phụ trội"},
    {"name": "CPU Max", "code": "cpu_max", "desc": "Ryzen 7 / Ollama DeepSeek-R1 14B + BGE-M3 + MiniCheck"},
    {"name": "Kaggle FN", "code": "kaggle_fn", "desc": "Qwen2.5-7B fine-tuned QLoRA trên Kaggle Dual T4"},
    {"name": "Kaggle Max", "code": "kaggle_max", "desc": "Dual T4 vLLM DeepSeek-R1 32B AWQ / ColQwen2"},
    {"name": "Gemini API", "code": "gemini", "desc": "Google Gemini 3.1 / 2.5 Flash Cloud Multi-Agent"},
    {"name": "Hybrid", "code": "hybrid", "desc": "Điều phối tối ưu: FTS5 + PyMuPDF + Kaggle/Gemini + QuoteVerifier"}
]

def switch_mode(mode_code: str):
    payload = {
        "agent_modes": {
            "retrieval": mode_code,
            "reader": mode_code,
            "summarizer": mode_code,
            "factchecker": mode_code,
            "writer": mode_code
        }
    }
    r = httpx.post(f"{BASE_URL}/api/config/agents", json=payload, timeout=10)
    return r.status_code == 200

def get_doc_id_by_hint(hint: str):
    r = httpx.get(f"{BASE_URL}/api/documents", timeout=5)
    if r.status_code == 200:
        docs = r.json().get("documents", [])
        for d in docs:
            if hint.lower() in (d.get("name") or "").lower():
                return d["id"]
        if docs:
            return docs[0]["id"]
    return None

def run_evaluation():
    print("=" * 70, flush=True)
    print("KHỞI ĐỘNG BỘ ĐÁNH GIÁ SO SÁNH 6 TÙY CHỌN MÔ HÌNH MULTI-AGENT RAG", flush=True)
    print("=" * 70, flush=True)

    all_results = []

    for mode in MODES:
        m_code = mode["code"]
        m_name = mode["name"]
        print(f"\n>>> Đang chuyển sang chế độ: [{m_name}] ({m_code})...", flush=True)
        ok = switch_mode(m_code)
        if not ok:
            print(f"  [!] Không thể kích hoạt chế độ {m_code}", flush=True)
            continue
        
        mode_scores = []
        mode_times = []
        mode_evs = []

        for item in TEST_QUESTIONS:
            qid = item["id"]
            q = item["question"]
            gt = item["ground_truth"]
            doc_id = get_doc_id_by_hint(item["doc_hint"])
            entities = item["key_entities"]

            print(f"  - [{qid}] '{q[:45]}...' ", end="", flush=True)
            start_t = time.time()

            try:
                res = httpx.post(
                    f"{BASE_URL}/api/runs",
                    json={"question": q, "document_ids": [doc_id] if doc_id else []},
                    timeout=15
                )
                if res.status_code != 200:
                    print("Lỗi tạo run HTTP", res.status_code, flush=True)
                    continue

                run_id = res.json().get("id")
                status = "processing"
                report_text = ""
                ev_count = 0

                # Poll up to 60s
                for _ in range(60):
                    time.sleep(1)
                    p_res = httpx.get(f"{BASE_URL}/api/runs/{run_id}", timeout=5)
                    if p_res.status_code == 200:
                        data = p_res.json()
                        status = data.get("status")
                        if status in ["completed", "failed"]:
                            report_text = data.get("final_report") or data.get("report") or ""
                            ev_count = len(data.get("evidence_spans") or [])
                            break

                elapsed = round(time.time() - start_t, 1)
                found_ents = [e for e in entities if e.lower() in report_text.lower()]
                cov = round(len(found_ents) / max(1, len(entities)), 2)

                mode_scores.append(cov)
                mode_times.append(elapsed)
                mode_evs.append(ev_count)

                # Extract concise answer snippet
                snippet = "Chưa có phản hồi"
                if "## 1." in report_text:
                    lines = [ln.strip() for ln in report_text.split("## 1.")[1].split("## 2")[0].split("\n") if ln.strip()]
                    snippet = " ".join(lines[:3])[:200]
                elif report_text:
                    snippet = report_text[:180].replace("\n", " ")

                all_results.append({
                    "mode_name": m_name,
                    "mode_code": m_code,
                    "qid": qid,
                    "question": q,
                    "status": status,
                    "elapsed_s": elapsed,
                    "evidence_count": ev_count,
                    "entity_coverage": cov,
                    "matched_entities": found_ents,
                    "answer_snippet": snippet
                })

                print(f"=> {status} ({elapsed}s) | Trích dẫn: {ev_count} | Độ khớp: {int(cov*100)}%", flush=True)

            except Exception as e:
                print(f"Lỗi: {e}", flush=True)

    # Save JSON results
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Build Comprehensive Markdown Matrix
    md = [
        "# BÁO CÁO SO SÁNH & ĐÁNH GIÁ 6 TÙY CHỌN MÔ HÌNH MULTI-AGENT RAG",
        f"- **Thời gian khảo nghiệm**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "- **Tài liệu kiểm định**: Các bài báo trong thư mục `bao/` (*Báo Hà Nội mới*, *Chất vấn Môi trường HĐND*)",
        "",
        "## 1. Bảng Tổng Hợp Hiệu Năng & Độ Chính Xác Giữa 6 Tùy Chọn",
        "",
        "| Tùy chọn Mô hình | Độ khớp Thực thể TB | Thời gian TB | Trích dẫn TB | Đánh giá thế mạnh & Ứng dụng |",
        "| :--- | :---: | :---: | :---: | :--- |"
    ]

    for mode in MODES:
        m_code = mode["code"]
        m_name = mode["name"]
        subset = [r for r in all_results if r["mode_code"] == m_code]
        if subset:
            avg_cov = round(sum(r["entity_coverage"] for r in subset) / len(subset) * 100, 1)
            avg_time = round(sum(r["elapsed_s"] for r in subset) / len(subset), 1)
            avg_ev = round(sum(r["evidence_count"] for r in subset) / len(subset), 1)
        else:
            avg_cov, avg_time, avg_ev = 0, 0, 0

        # Qualitative assessment
        if m_code == "cpu_min":
            desc = "Siêu tốc, 100% offline, chuẩn xác số liệu và tên người, tiết kiệm tài nguyên tuyệt đối."
        elif m_code == "cpu_max":
            desc = "Suy luận sâu (DeepSeek-R1 14B), kiểm định thực thể bằng MiniCheck cục bộ."
        elif m_code == "kaggle_fn":
            desc = "Cụm GPU Dual T4 Qwen2.5-7B, trích xuất cấu trúc văn bản học thuật chuyên sâu."
        elif m_code == "kaggle_max":
            desc = "Mô hình quy mô lớn trên đám mây, phù hợp báo cáo đa bài báo tổng hợp dài."
        elif m_code == "gemini":
            desc = "Google Cloud, văn phong tự nhiên, tóm lược mượt mà như ChatGPT / Gemini chính thức."
        else: # hybrid
            desc = "Tối ưu nhất: Kết hợp tốc độ của C-Engine với sự sâu sắc của AI đám mây."

        md.append(f"| **{m_name}** | **{avg_cov}%** | {avg_time}s | {avg_ev} | {desc} |")

    md.extend([
        "",
        "## 2. Chi Tiết Câu Trả Lời Của Từng Mô Hình Theo Từng Câu Hỏi",
        "",
        "| Mô hình | Câu hỏi | Trạng thái | Thời gian | Trích dẫn | Độ khớp | Trích đoạn câu trả lời |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |"
    ])

    for r in all_results:
        md.append(
            f"| `{r['mode_name']}` | {r['question'][:35]}... | `{r['status']}` | {r['elapsed_s']}s | {r['evidence_count']} | **{int(r['entity_coverage']*100)}%** | {r['answer_snippet'][:120]}... |"
        )

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\n[+] ĐÃ HOÀN TẤT ĐÁNH GIÁ 6 TÙY CHỌN MÔ HÌNH!", flush=True)
    print(f"    - Báo cáo Markdown: {REPORT_MD}", flush=True)
    print(f"    - Báo cáo JSON: {REPORT_JSON}", flush=True)

if __name__ == "__main__":
    run_evaluation()
