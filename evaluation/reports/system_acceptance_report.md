# Báo cáo Nghiệm thu Hệ thống Toàn diện (System Acceptance Report — Đợt 7)

**Mã báo cáo:** `SYS_ACCEPTANCE_PHASE_E6`  
**Dự án:** Multi-Agent Research RAG (`ducmanh-jr/Multi-Agent_RAG`)  
**Nhánh kiểm thử:** `refactor/assistant-core-reconstruction` (HEAD: `1af1a1262a`)  
**Quyết định đề xuất:** **GIỮ LẠI TRÊN DRAFT PR — CHƯA MERGE VÀO DEVELOP** (Cho đến khi kiểm toán môi trường Remote GPU và hoàn thiện bộ hồi quy mã nguồn mới).

---

## 1. Bảng Tổng kết 4 Tầng Kiểm thử Hệ thống

| Tầng Kiểm thử | Quy mô thực hiện | Kết quả Nghiệm thu | Đánh giá Trạng thái |
| :--- | :--- | :---: | :--- |
| **Level 1 — Graph & Contract** | 103 bài kiểm thử tự động pytest | **103 / 103 PASSED (100%)** | Đạt xuất sắc hợp đồng trạng thái, reducer và router |
| **Level 2 — Live Model Matrix** | 96 lượt đánh giá (16 Golden Cases × 6 Modes) | **48 PASS, 16 DEGRADED, 32 BLOCKED** | Gemini & CPU Min & Hybrid đạt điểm giỏi (8.76 - 9.47/10) |
| **Level 3 — API End-to-End** | 10 kịch bản HTTP API (API-01 đến API-10) | **10 / 10 PASSED (100%)** | Đạt hợp đồng upload, session, stream, policy |
| **Level 4 — Reliability & UI** | 10 kịch bản chịu lỗi (REL) + 8 UI journeys (FE) | **18 / 18 PASSED (100%)** | Chống sập, fail-closed, cách ly session, SSE streaming |

---

## 2. Bảng Phân loại Khuyết tật & Vấn đề (Defect Log)

| Mã Lỗi | Mức độ | Thành phần | Mô tả Hiện tượng | Nguyên nhân gốc | Trạng thái xử lý |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **DEF-01** | `Medium` | `llm_client.py` | Cache va chạm giữa các execution modes | `cache_key_str` ban đầu không chứa `target_mode`, dẫn đến mode này đọc nhầm cache fallback của mode khác. | **ĐÃ KHẮC PHỤC TRIỆT ĐỂ** (Thêm `target_mode` vào cache key). |
| **DEF-02** | `Medium` | `modes/gemini` | Lỗi JSON parse khi Gemini sinh công thức LaTeX | Gemini xuất công thức toán (`\sqrt`, `\left`) khiến regex sanitizer biến thành escape không hợp lệ. | **ĐÃ KHẮC PHỤC TRIỆT ĐỂ** (Cố gắng `json.loads` trực tiếp trước khi sanitize lambda). |
| **DEF-03** | `Low` | `supervisor.py` | Câu hỏi dịch thuật bị nhận nhầm thành `DOCUMENT_QA` | Pattern `_EXPLICIT_DOCUMENT_PATTERNS` khớp từ 'tài liệu' trong cụm 'tài liệu kỹ thuật'. | **ĐÃ KHẮC PHỤC TRIỆT ĐỂ** (Chuẩn hóa pattern thành 'tài liệu này/trên/đã chọn'). |
| **DEF-04** | `Low` | `supervisor.py` | Yêu cầu chạy chương trình bị xếp vào `GENERAL_QA` | Thiếu từ khóa 'chương trình', 'phần mềm' trong `_TASK_EXECUTION_PATTERNS`. | **ĐÃ KHẮC PHỤC TRIỆT ĐỂ** (Mở rộng đầy đủ pattern lệnh hệ thống). |
| **DEF-05** | `High` | `app.py` | API chặn các câu hỏi General QA khi không upload PDF | `create_run` trong `app.py` còn sót logic legacy RAG cũ, tự ngắt với `WAITING_FOR_DOCUMENT` trước khi vào Graph. | **ĐÃ KHẮC PHỤC TRIỆT ĐỂ** (Cho phép mọi query vào Supervisor để phân loại). |
| **ENV-01** | `High` | Môi trường | Remote Kaggle GPU Worker không thể kết nối | Cloudflare tunnel của Kaggle worker bị ngắt kết nối. | **GHI NHẬN MINH BẠCH BLOCKED** (Không dùng mock giả mạo). |
| **ENV-02** | `Medium` | Môi trường | Local Ollama daemon không chạy trên cổng 11434 | Chưa khởi động tiến trình Ollama cục bộ. | **GHI NHẬN MINH BẠCH DEGRADED** (Chạy extractive fallback). |

---

## 3. Những Tính năng Hoạt động Xuất sắc (Proven Capabilities)

1. **Kiến trúc 9 Ý định (Agentic Supervisor):**
   - Phân luồng hoàn hảo giữa câu hỏi thông thường (`GENERAL_QA`), câu hỏi lập trình/dịch thuật (`TASK_ASSISTED_RESPONSE`), câu hỏi tài liệu (`DOCUMENT_QA`), nghiên cứu tài liệu ngoài (`WEB_RESEARCH`), và từ chối câu hỏi mơ hồ/lệnh nguy hiểm (`CLARIFICATION_REQUIRED`, `UNSUPPORTED_CAPABILITY`).
2. **Xử lý Tài liệu Thực tế (Real Document Intelligence):**
   - Đọc, trích xuất và liên kết số trang chính xác cho bài báo khoa học `Vaswani_2017_Attention.pdf` (15 trang) và giáo trình đồ sộ `Software Engineering` của Pressman (888 trang).
   - Trích xuất bảng tham số Transformer Base (Table 3) chính xác từng tham số ($N=6, d_{model}=512, d_{ff}=2048, h=8$).
3. **Nghiên cứu Học thuật Thực tế (OpenAlex Scout):**
   - Tự động gọi API OpenAlex, tìm kiếm các bài báo khoa học thật về RAG (năm 2024+), tải PDF và trích xuất siêu dữ liệu học thuật chính xác.
4. **Bảo vệ Ranh giới Nhà cung cấp (Zero-Gemini Provider Policy):**
   - Mode `gemini` được kích hoạt đầy đủ với Google GenAI API.
   - Các mode `cpu_min`, `cpu_max`, `hybrid`, `gpu_finetune`, `gpu_max` được khóa cứng tuyệt đối: không bao giờ gọi sang Gemini dưới bất kỳ tình huống lỗi nào.
5. **Cơ chế Fail-Closed Verification:**
   - Khi Critic gặp sự cố hoặc JSON không hợp lệ, hệ thống tự động chuyển sang `VERIFICATION_FAILED` và gán nhãn `refute` / `not_enough_info`, tuyệt đối không bao giờ phê duyệt bừa bãi.

---

## 4. Những Tính năng Cần Cải thiện (Areas for Improvement)

1. **Khởi động Local Ollama Daemon:** Để kiểm thử trọn vẹn mode `cpu_max` với mô hình LLM cục bộ 7B thật (thay vì extractive fallback).
2. **Khôi phục Remote Kaggle Worker:** Thiết lập lại Cloudflare tunnel hoặc Kaggle session để đánh giá trực tiếp năng lực 2 mode `gpu_finetune` và `gpu_max`.

---

## 5. Khuyến nghị Nghiệm thu (Release Gate Recommendation)

> [!IMPORTANT]
> **Khuyến nghị chính thức:** **CHƯA MERGE VÀO NHÁNH DEVELOP TẠI THỜI ĐIỂM NÀY.**
> 
> **Lý do:**
> 1. Nhánh `refactor/assistant-core-reconstruction` đã đạt kết quả xuất sắc trên các mode khả dụng (`gemini`: 9.47/10, `cpu_min`: 8.76/10, `hybrid`: 8.76/10, API: 10/10, Reliability: 10/10).
> 2. Tuy nhiên, hai mode GPU (`gpu_finetune`, `gpu_max`) đang ở trạng thái `BLOCKED` do đường truyền Kaggle offline, và `cpu_max` đang ở trạng thái `DEGRADED`.
> 3. Trong quá trình đánh giá Đợt 7, đã phát hiện và khắc phục 5 khuyết tật kỹ thuật (`DEF-01` đến `DEF-05` trong `app.py`, `supervisor.py`, `llm_client.py`). Những thay đổi này hiện đang nằm trong Working Tree và cần được kiểm toán, bổ sung test hồi quy độc lập trước khi tạo commit mới.
> 
> **Hành động tiếp theo:** Giữ PR ở trạng thái **DRAFT**, báo cáo minh bạch cho User duyệt bảng điểm và kết quả kiểm thử.
