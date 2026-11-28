# 🖥️ Thư Mục Chính Dự Án (`main/`)

`main/` là trung tâm điều phối của toàn bộ hệ thống **Multi-Agent Research RAG**, bao gồm máy chủ FastAPI, đồ thị trạng thái LangGraph, **6 chế độ thực thi độc lập** (`modes/`, trong đó `gpu_ft` là alias của `gpu_finetune`), giao diện người dùng (`web_demo2/`) và bộ kiểm thử đánh giá tự động (`eval/`, `tests/`).

---

## 🗂️ Cấu Trúc Thư Mục `main/`

| Thư mục / Tệp | Chức năng chi tiết |
| :--- | :--- |
| [`main.py`](main.py) | Điểm khởi chạy ứng dụng (`--serve` để mở Web Server cổng `8000`, hoặc chạy truy vấn CLI với cờ `--mode`). |
| [`core/`](core/) | **Multi-Agent Hub Engine (L3–L4)**: Đồ thị LangGraph Hub-and-Spoke mặc định (`graph.py`, `supervisor_hub.py`, `supervisor_mode="hub"|"pipeline"`), hợp đồng tác tử (`agent_protocol.py`, `agent_decisions.py`, `coordination.py`), vòng tool có chặn (`base_agent.py`), trạng thái (`state.py`, `schemas.py`), bộ lập kế hoạch thống nhất (`supervisor.py`), công cụ đọc PDF/toán học (`tools.py`), và bộ kiểm chứng (`verifier.py`). |
| [`modes/`](modes/) | **6 Chế độ Thực thi Độc lập**: `gpu_finetune` (alias `gpu_ft`, cụm mô hình Kaggle), `hybrid` (Kiến trúc đa tác tử lai tự phục hồi giữa Kaggle/CPU/Gemini), `gpu_max` (suy luận CoT trên GPU lớn), `gemini` (Cloud Foundation API), `cpu_min` (C-Engine Siêu nhẹ), `cpu_max` (Trạm làm việc Ryzen/Vulkan) cùng bộ đăng ký trung tâm ([`registry.py`](modes/registry.py)). |
| [`agents/`](agents/) | **Các thành phần Agent cơ sở**: Tìm kiếm học thuật ([`agent0_scout`](agents/agent0_scout/)), FTS5/BM25/Hybrid RRF ([`agent1_retrieval`](agents/agent1_retrieval/)), phân tích bố cục PDF ([`agent2_reader`](agents/agent2_reader/)), tổng hợp ([`agent3_summarizer`](agents/agent3_summarizer/)), kiểm chứng ([`agent4_fact_checker`](agents/agent4_fact_checker/)), xuất báo cáo ([`agent5_writer`](agents/agent5_writer/)). |
| [`models/`](models/) | **Mô hình Trọng số Cục bộ Offline**: Chứa `multilingual-e5-small` (~493 MB) phục vụ riêng cho Agent 0 (Scout Paper Ranking) chạy 100% offline. |
| [`api/`](api/) | **FastAPI Backend** (`app.py`): Quản lý phiên làm việc, tải lên PDF, cấu hình động từng Agent, chuyển đổi `POST /api/config/supervisor-mode`, tự động đăng ký Kaggle Tunnel (`POST /api/config/remote-url/register`), checkpoint/resume (`/api/runs/{id}/checkpoints`, `/resume`), và kiểm tra trạng thái phần cứng (`/api/capabilities`). |
| [`web_demo2/`](web_demo2/) | **Giao diện Web Nghiên cứu chính**: Hỗ trợ chọn tài liệu PDF, xem trích dẫn trực tiếp, chọn chế độ thực thi nhanh hoặc tùy biến từng Agent. |
| [`eval/`](eval/) | **Bộ Benchmark & Đánh giá Trung thực**: Chứa `benchmark_120.jsonl`, cổng M0 (`m0_regression_gate.py`), bộ NLI held-out 40 cặp song ngữ EN+VI (`nli_holdout_eval.py`, `nli_holdout_40.json`), ablation `pipeline` vs `hub` và từng agent (`ablation_study.py`). |
| [`tests/`](tests/) | **Kiểm thử Research V2** gồm các ca kiểm tra hợp đồng và các ca tích hợp graph chạy offline. |
| [`utils/`](utils/) | Tiện ích hệ thống: `llm_client.py` (kết nối Ollama Vulkan, Gemini, Kaggle), `local_vulkan_setup.py` (thăm dò tài nguyên phần cứng), `bilingual.py`. |
| `uploads/` & `data/` | Thư mục lưu trữ tài liệu PDF người dùng tải lên và cơ sở dữ liệu SQLite (`workspace.db`, `corpus.jsonl`). |

---

## 🚀 Các Lệnh Vận Hành Chính

### 1. Khởi chạy Web Workspace (`http://localhost:8000/demo2/`)

Đặt `APP_AUTH_PASSWORD` trong `.env` rồi đăng nhập tại `/demo2/`. Trình duyệt dùng cookie phiên để xem PDF, hình và tải báo cáo. API client lấy token từ `POST /api/auth/login` và gửi `Authorization: Bearer <token>`. Phiên hết hạn sau 12 giờ và được lưu trong tiến trình API. Khi chạy local, `APP_DEV_MODE=true` cho phép truy cập không mật khẩu qua loopback nếu `APP_AUTH_PASSWORD` đang trống.

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

### 3. Tái chạy baseline Research V2

Chạy từ thư mục gốc của repo với Python 3.11.9 trên Windows. CI dùng bộ phiên bản trong [`requirements-baseline.txt`](../requirements-baseline.txt). Các lệnh kiểm thử dùng PDF fixture và provider giả lập khi cần.

```powershell
python -m pip install -r requirements.txt -c requirements-baseline.txt
$env:PYTHON_DOTENV_DISABLED = '1'
python -m pytest -q -m "not integration"
python -m pytest -q -m integration
```

Pilot tự tạo PDF từ fixture và chạy graph thật ở `cpu_min`. Bốn câu dùng để kiểm tra đường chạy và tái hiện lỗi đã biết.

```powershell
python main/eval/run_benchmark.py --pilot --modes cpu_min --limit-per-cat 0 --output-dir "$env:TEMP\research-baseline-pilot"
```

CI chạy hai nhóm test và lưu báo cáo Markdown cùng kết quả từng câu dạng JSON làm artifact. [Workflow Research V2 baseline](../.github/workflows/research-baseline.yml)

### 4. Vòng đời research run

`POST /api/runs` nhận `document_ids` có thứ tự và `auto_scout`. Khi thiếu tài liệu hợp lệ, bật Auto-Scout để tìm nguồn và tiếp tục kiểm chứng trong cùng run. Ba API `/api/runs`, `/api/research` và `/api/research/stream` lưu kết quả qua cùng service. API legacy mặc định bật Scout khi thiếu tài liệu. Client có thể truyền `auto_scout: false` để tắt.

`GET /api/runs/{id}` trả về `document_ids` và danh sách `documents` có tên và SHA-256 đã ghi nhận. SQLite tự tạo bảng `run_documents` khi khởi động. Run cũ giữ ID tài liệu chính đã lưu. `POST /api/runs/{id}/cancel` trả về trạng thái thực tế của run. Các bước đã lưu được giữ lại và trạng thái kết thúc ghi trước được bảo toàn.
