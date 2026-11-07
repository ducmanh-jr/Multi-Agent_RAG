# 🖥️ Thư Mục Chính Dự Án (`main/`)

`main/` là trung tâm điều phối của toàn bộ hệ thống **Multi-Agent Research RAG**, bao gồm máy chủ FastAPI, đồ thị trạng thái LangGraph, 6 chế độ thực thi (`modes/`), giao diện người dùng (`web_demo2/`) và bộ kiểm thử đánh giá tự động (`eval/`).

---

## 🗂️ Cấu Trúc Thư Mục `main/`

| Thư mục / Tệp | Chức năng chi tiết |
| :--- | :--- |
| [`main.py`](main.py) | Điểm khởi chạy ứng dụng (`--serve` để mở Web Server cổng `8000`, hoặc chạy truy vấn CLI). |
| [`core/`](core/) | **Agentic Engine L3–L4**: Đồ thị LangGraph (`graph.py`), hợp đồng trạng thái (`state.py`, `schemas.py`), bộ lập kế hoạch (`supervisor.py`), công cụ đọc PDF/toán học (`tools.py`), và bộ kiểm chứng (`verifier.py`). |
| [`modes/`](modes/) | **6 Chế độ Thực thi**: `cpu_min`, `cpu_max`, `gemini`, `gpu_ft`, `gpu_max`, và `hybrid` cùng bộ đăng ký trung tâm (`registry.py`). |
| [`agents/`](agents/) | **Các thành phần Agent cơ sở**: Tìm kiếm học thuật (`agent0_scout`), FTS5/BM25/Hybrid RRF (`agent1_retrieval`), phân tích bố cục PDF (`agent2_reader`), tổng hợp (`agent3_summarizer`), kiểm chứng (`agent4_fact_checker`), xuất báo cáo (`agent5_writer`). |
| [`api/`](api/) | **FastAPI Backend** (`app.py`): Quản lý phiên làm việc, tải lên PDF, cấu hình động từng Agent, kiểm tra trạng thái phần cứng (`/api/capabilities`). |
| [`web_demo2/`](web_demo2/) | **Giao diện Web Nghiên cứu chính**: Hỗ trợ chọn tài liệu PDF, xem trích dẫn trực tiếp, chọn chế độ thực thi nhanh hoặc tùy biến từng Agent. |
| [`eval/`](eval/) | **Bộ Benchmark 120 câu hỏi song ngữ**: Chứa `benchmark_120.jsonl`, `metrics.py`, `run_benchmark.py` và báo cáo `benchmark_report.md`. |
| [`utils/`](utils/) | Tiện ích hệ thống: `llm_client.py` (kết nối Ollama Vulkan, Gemini, Kaggle), `local_vulkan_setup.py` (tăng tốc Radeon 780M iGPU), `bilingual.py`. |
| `uploads/` & `data/` | Thư mục lưu trữ tài liệu PDF người dùng tải lên và cơ sở dữ liệu SQLite (`workspace.db`, `corpus.jsonl`). |

---

## 🚀 Các Lệnh Vận Hành Chính

### 1. Khởi chạy Web Workspace (`http://localhost:8000/demo2/`)
```bash
cd main
py -3.10 -X utf8 main.py --serve
```

### 2. Chạy truy vấn trực tiếp từ dòng lệnh (CLI)
```bash
cd main
py -3.10 -X utf8 main.py --query "Tóm tắt đóng góp chính của bài báo FlashAttention" --mode hybrid
```

### 3. Chạy bộ kiểm thử đánh giá 6 chế độ (`eval/`)
```bash
cd main
py -3.10 -X utf8 eval/run_benchmark.py --limit-per-cat 2
```
