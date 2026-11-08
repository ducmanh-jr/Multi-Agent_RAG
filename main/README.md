# 🖥️ Thư Mục Chính Dự Án (`main/`)

`main/` là trung tâm điều phối của toàn bộ hệ thống **Multi-Agent Research RAG**, bao gồm máy chủ FastAPI, đồ thị trạng thái LangGraph, **7 chế độ thực thi độc lập** (`modes/`), giao diện người dùng (`web_demo2/`) và bộ kiểm thử đánh giá tự động (`eval/`, `tests/`).

---

## 🗂️ Cấu Trúc Thư Mục `main/`

| Thư mục / Tệp | Chức năng chi tiết |
| :--- | :--- |
| [`main.py`](main.py) | Điểm khởi chạy ứng dụng (`--serve` để mở Web Server cổng `8000`, hoặc chạy truy vấn CLI với cờ `--mode`). |
| [`core/`](core/) | **Agentic Engine L3–L4**: Đồ thị LangGraph (`graph.py`), hợp đồng trạng thái (`state.py`, `schemas.py`), bộ lập kế hoạch (`supervisor.py`), công cụ đọc PDF/toán học (`tools.py`), và bộ kiểm chứng (`verifier.py`). |
| [`modes/`](modes/) | **7 Chế độ Thực thi Độc lập**: `gpu_finetune` (Toàn diện 7 mô hình tinh chỉnh), `hybrid` (Kiến trúc đa tác tử cân bằng tự phục hồi), `gpu_ft` (Kiến trúc chuẩn 3 adapter LoRA), `gpu_max` (32B AWQ Tensor Parallelism), `gemini` (Cloud Foundation API), `cpu_min` (C-Engine Siêu nhẹ), `cpu_max` (Trạm làm việc Ryzen/Vulkan) cùng bộ đăng ký trung tâm ([`registry.py`](modes/registry.py)). |
| [`agents/`](agents/) | **Các thành phần Agent cơ sở**: Tìm kiếm học thuật ([`agent0_scout`](agents/agent0_scout/)), FTS5/BM25/Hybrid RRF ([`agent1_retrieval`](agents/agent1_retrieval/)), phân tích bố cục PDF ([`agent2_reader`](agents/agent2_reader/)), tổng hợp ([`agent3_summarizer`](agents/agent3_summarizer/)), kiểm chứng ([`agent4_fact_checker`](agents/agent4_fact_checker/)), xuất báo cáo ([`agent5_writer`](agents/agent5_writer/)). |
| [`models/`](models/) | **Mô hình Trọng số Cục bộ Offline**: Chứa `multilingual-e5-small` (~493 MB) phục vụ riêng cho Agent 0 (Scout Paper Ranking) chạy 100% offline. |
| [`api/`](api/) | **FastAPI Backend** (`app.py`): Quản lý phiên làm việc, tải lên PDF, cấu hình động từng Agent, kiểm tra trạng thái phần cứng (`/api/capabilities`). |
| [`web_demo2/`](web_demo2/) | **Giao diện Web Nghiên cứu chính**: Hỗ trợ chọn tài liệu PDF, xem trích dẫn trực tiếp, chọn chế độ thực thi nhanh hoặc tùy biến từng Agent. |
| [`eval/`](eval/) | **Bộ Benchmark 120 câu hỏi song ngữ**: Chứa `benchmark_120.jsonl`, `metrics.py`, `run_benchmark.py` và báo cáo `benchmark_report.md`. |
| [`tests/`](tests/) | **Bộ Kiểm Thử Toàn Diện**: Chứa `system_sanity_check.py` (kiểm tra 35 cặp agent), `test_hybrid_resilience.py` (kiểm tra 4 kịch bản chịu lỗi) và `test_gpu_ft_live.py`. |
| [`utils/`](utils/) | Tiện ích hệ thống: `llm_client.py` (kết nối Ollama Vulkan, Gemini, Kaggle), `local_vulkan_setup.py` (thăm dò tài nguyên phần cứng), `bilingual.py`. |
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
# Thực thi chế độ Tinh chỉnh Toàn diện 7 Mô hình (gpu_finetune):
py -3.10 -X utf8 main/main.py --query "Tóm tắt đóng góp chính của bài báo FlashAttention" --mode gpu_finetune

# Thực thi chế độ Đa tác tử Cân bằng Tự chủ (hybrid):
py -3.10 -X utf8 main/main.py --query "Deep learning for image recognition overview" --mode hybrid

# Thực thi chế độ Cục bộ Siêu tốc Không phụ thuộc GPU/Mạng (cpu_min):
py -3.10 -X utf8 main/main.py --query "What is quantum computing overview?" --mode cpu_min
```

### 3. Chạy bộ kiểm thử tự phục hồi & ổn định hệ thống
```bash
# Kiểm tra tính toàn vẹn và khả năng khởi tạo của toàn bộ 7 chế độ:
python main/tests/system_sanity_check.py

# Kiểm tra 4 kịch bản tự phục hồi lỗi và suy thoái mềm của chế độ Hybrid:
python main/tests/test_hybrid_resilience.py
```
