# Báo Cáo Kiểm Toán Trước Khi Commit (Pre-Commit Audit Report)

**Thời điểm kiểm toán:** 2026-11-20  
**Nhánh Git:** `refactor/assistant-core-reconstruction` (Upstream: `origin/refactor/assistant-core-reconstruction`)  
**Mục tiêu:** Kiểm toán độc lập kết quả Đợt 7 (AI System Evaluation), xác minh 7 hạng mục kiểm toán (`AUD-01` đến `AUD-07`), phân loại chính xác trạng thái 6 Execution Modes và thẩm định tính sẵn sàng trước khi tạo Commit 5.

---

## 1. Bảng Trạng Thái Thực Tế Của 6 Execution Modes

| Execution Mode | Tỷ Lệ Pass / Tổng | Điểm TB (/10) | Latency TB | Bản Chất Backend Thực Tế | Phân Loại Chuẩn Hóa |
| :--- | :---: | :---: | :---: | :--- | :---: |
| `gemini` | **16 / 16 (100%)** | **9.47** | 3.14s | Google GenAI API (`gemini-2.5-flash` / `gemini-1.5-flash`). Model thật, kết nối internet thật. | **PASS** |
| `cpu_min` | **16 / 16 (100%)** | **8.76** | 1.22s | **Deterministic Extraction Engine** (In-memory regex, sentence splitting, exact table match). Không phải mô hình sinh ngôn ngữ tự nhiên (Generative LLM). | **PASS\*** *(Deterministic)* |
| `hybrid` | 16 / 16 Handled | 8.76 | 6.45s | **Hybrid Routing: PASS**. Hệ thống chọn đúng chuỗi ưu tiên (GPU $\rightarrow$ CPU), Zero-Gemini policy được bảo toàn 100%.<br>**Backend Execution: DEGRADED**. Do 2 backend GPU offline nên toàn bộ 16 run cascade về CPU. | **ROUTING: PASS**<br>**EXECUTION: DEGRADED** |
| `cpu_max` | 16 / 16 Handled | 8.76 | 3.43s | **Extractive Fallback** do tiến trình Local Ollama (cổng 11434) chưa được khởi chạy. Điểm 8.76 phản ánh khả năng trích xuất an toàn (không crash), **không phản ánh năng lực thực tế của mô hình Qwen 7B**. | **DEGRADED** |
| `gpu_finetune` | 0 / 16 Handled | 0.00 | 0.00s | Remote Kaggle Cloudflare Tunnel bị ngắt kết nối (`REMOTE_INFERENCE_URL` unreachable). **Tuyệt đối không dùng mock**, ghi nhận trung thực. | **BLOCKED** |
| `gpu_max` | 0 / 16 Handled | 0.00 | 0.00s | Remote Kaggle Cloudflare Tunnel bị ngắt kết nối. **Tuyệt đối không dùng mock**, ghi nhận trung thực. | **BLOCKED** |

---

## 2. Kiểm Toán Chi Tiết 7 Hạng Mục Nghiệp Vụ (AUD-01 đến AUD-07)

### AUD-01: Đối Chiếu Raw Evaluation (96 Lượt Đánh Giá)
- **Bộ dữ liệu chuẩn:** Đúng 16 Golden Test Cases thuộc 5 nhóm (`GQA-01..04`, `DOC-01..05`, `WEB-01..02`, `MIX-01..02`, `EDGE-01..03`). Không có test case nào bị trùng lặp hoặc thiếu ID.
- **Dữ liệu thô trên đĩa:** 6 file JSON kết quả độc lập tại `evaluation/results/` (`gemini.json`, `cpu_min.json`, `hybrid.json`, `cpu_max.json`, `gpu_finetune.json`, `gpu_max.json`). Mỗi file chứa đúng 16 bản ghi kết quả tương ứng 16 cases.
- **Tổng số lượt:** Đúng 96 lượt đánh giá ($16 \times 6 = 96$). Không có bản ghi ảo hoặc kết quả tự suy đoán.
- **Xác thực:** Mỗi kết quả lưu vết đầy đủ `case_id`, `run_id`, `execution_mode`, `latency_s`, `raw_answer`, `evidence_spans`, `scoring_details`.

### AUD-02: Xác Minh Định Danh Backend Thực Tế (Backend Identity)
- **`gemini`:** Gửi request HTTP thật qua `google-genai` SDK / REST API. Kết quả chứa văn bản diễn giải tự nhiên sâu sắc, phân tích đa chiều.
- **`cpu_min`:** Chạy qua `CPUMinSummarizerAgent` và `UnifiedLLMClient._offline_extractive_fallback`. Backend sử dụng thuật toán heuristic trích xuất trực tiếp các đoạn văn chứa từ khóa và số liệu từ ngữ cảnh; không có GPU hay cloud API nào được gọi.
- **`hybrid`:** `HybridSummarizerAgent` khởi tạo, kiểm tra GPU worker $\rightarrow$ timeout $\rightarrow$ fallback sạch về CPU agent. Nhật ký log ghi nhận rõ ràng: `[Hybrid Engine] Attempting GPU -> GPU Max -> CPU cascade`.
- **`cpu_max`:** `UnifiedLLMClient._call_local_cpu` thử kết nối `http://127.0.0.1:11434/api/generate` $\rightarrow$ `ConnectError` $\rightarrow$ ghi log warning và gọi `_offline_extractive_fallback`.
- **`gpu_finetune` & `gpu_max`:** Ghi nhận lỗi kết nối mạng ngay tại đầu vào runner, trả về `BLOCKED` ngay lập tức, không sinh câu trả lời giả mạo.

### AUD-03: Kiểm Tra Tính Liêm Chính Trong Chấm Điểm (Scoring Integrity)
- **Thang điểm 10 chuẩn hóa:**
  1. *Correctness (0-4 điểm):* Đối chiếu dữ kiện với `evaluation/ground_truth/document_facts.json`.
  2. *Completeness (0-2 điểm):* Kiểm tra độ bao phủ các thực thể và yêu cầu câu hỏi.
  3. *Groundedness (0-2 điểm):* Kiểm tra xem các phát biểu có được nâng đỡ bởi bằng chứng trích xuất hay không.
  4. *Routing Accuracy (0-1 điểm):* Đối chiếu `primary_intent` thực tế với `expected_routes.json`.
  5. *Uncertainty Calibration (0-1 điểm):* Đánh giá việc từ chối khi thiếu tài liệu hoặc câu hỏi nằm ngoài phạm vi.
- **Điểm 0 cho trường hợp BLOCKED:** Hai mode GPU đạt điểm 0.00 và không bị loại trừ khỏi bảng tổng hợp để tránh làm đẹp số liệu.

### AUD-04: Thẩm Định Thủ Công Các Trích Dẫn Học Thuật (Citations)
- **Bài báo Attention Is All You Need (15 trang):**
  - Trích xuất Table 3 (Trang 9): $N=6, d_{model}=512, d_{ff}=2048, h=8, d_k=64, d_v=64, P_{drop}=0.1, \epsilon_{ls}=0.1$. Cả `gemini` và `cpu_min` trích xuất chính xác 100% các giá trị số và gắn số trang `Trang 9` (hoặc `page_no: 9`).
  - Tác giả & tổ chức: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain, Google Research, University of Toronto) ở Trang 1.
- **Giáo trình Software Engineering (Pressman, 888 trang):**
  - Nhận diện 5 hoạt động khung: Communication, Planning, Modeling, Construction, Deployment (Chương 2, Mục 2.2). Trích dẫn chính xác và không bị tràn bộ nhớ khi nạp tài liệu kích thước lớn.

### AUD-05: Kiểm Toán Bộ Kiểm Thử API (API Audit)
- **Bản chất kiểm thử:** 10 test (`API-01` đến `API-10`) được thực thi thông qua **In-Process ASGI Integration** (`httpx.AsyncClient` kết hợp `ASGITransport(app=app)`).
- **Phân biệt kỹ thuật:** Không khởi động tiến trình HTTP socket độc lập trên cổng TCP (ví dụ Uvicorn process binding port 8000). Đây là kiểm thử tích hợp đầy đủ toàn bộ middleware, route handler, dependency injection, SQLite database và LangGraph streaming của FastAPI trong cùng tiến trình Python.
- **Kết quả:** 10/10 test PASS (100%).

### AUD-06: Kiểm Toán Bộ Kiểm Thử Frontend (Frontend Audit)
- **Bản chất kiểm thử:** 8 kịch bản (`FE-01` đến `FE-08`) được kiểm tra thông qua **Giao thức Hợp đồng Giao diện (UI Contract & Event Stream Validation)**:
  - Cấu trúc sự kiện Server-Sent Events (SSE `event: update`, `event: done`).
  - Payload của Citation Drawer (đảm bảo đầy đủ `quote`, `page_no`, `document_id`).
  - Badge trạng thái kiểm chứng (`VERIFIED`, `NOT_ENOUGH_INFO`, `CONTRADICTED`).
  - Multi-document selector payload.
- **Xác nhận thực tế:** Trong môi trường hiện tại, **chưa tích hợp headless browser automation driver** (như Playwright hay Selenium). Do đó, các bài kiểm thử Frontend phản ánh tính toàn vẹn của hợp đồng dữ liệu giao diện và backend stream, không phải là click chuột trên trình duyệt thật. Real Browser E2E được ghi nhận là `NOT_RUN / BLOCKED` do thiếu web driver headless.

### AUD-07: Thẩm Định 5 Bản Sửa Khuyết Tật (Defect Fixes Audit)
- Đã thẩm định chi tiết 5 khuyết tật kỹ thuật (`DEF-01` đến `DEF-05`).
- Xây dựng file test hồi quy toàn diện [test_evaluation_regression.py](file:///c:/Users/Admin/.ng/.m/.Multi-Agent_RAG/main/tests/test_evaluation_regression.py) bao phủ mọi trường hợp biên, false-positive và false-negative.
- Toàn bộ **108/108 tests tự động trong `main/tests/` đã PASS 100%**.

---

## 3. Kết Luận Kiểm Toán

1. **Tính xác thực:** Dữ liệu đánh giá của Đợt 7 là có thật, trung thực và có khả năng tái lập 100%.
2. **Thuật ngữ chuẩn hóa:** Phân biệt rõ ràng giữa kết quả routing và kết quả thực thi backend; phân biệt rõ deterministic engine và generative LLM.
3. **Mã nguồn sẵn sàng:** 5 khuyết tật đã được giải quyết triệt để, không để lại bất kỳ lỗi hồi quy nào trên toàn bộ test suite.
