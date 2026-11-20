# Báo Cáo Thẩm Định 5 Bản Sửa Khuyết Tật (Defect Validation Report)

**Thời điểm:** 2026-11-20  
**Phạm vi:** Thẩm định kỹ thuật sâu 5 lỗi phát hiện trong Đợt 7 và các bài kiểm thử hồi quy tương ứng.

---

## 1. DEF-01: Cross-Mode Semantic Cache Partitioning

### A. Vấn đề Phát Hiện
Trong `main/utils/llm_client.py`, khóa cache ngữ nghĩa SHA-256 (`cache_key_str`) trước đây không chứa thông tin về `target_mode` và các tham số sinh (`temperature`, `max_tokens`). Khi một câu hỏi được chạy lần đầu dưới mode `cpu_max` (rơi vào extractive fallback), kết quả trích xuất dạng văn bản thô được lưu vào SQLite cache. Khi câu hỏi đó được chạy lại dưới mode `gemini`, hàm `generate()` tìm thấy cache hit và trả về ngay kết quả fallback của CPU trong 0.0s, làm nhiễm bẩn và triệt tiêu hoàn toàn năng lực của mô hình Gemini.

### B. Giải Pháp Triển Khai
Tại [main/utils/llm_client.py](file:///c:/Users/Admin/.ng/.m/.Multi-Agent_RAG/main/utils/llm_client.py#L53-L60), cấu trúc `cache_key_str` được mở rộng thành chuỗi định danh đa chiều:
```python
agent_str = agent or "default"
cache_key_str = (
    f"mode={target_mode}___agent={agent_str}___temp={temperature:.2f}___"
    f"max_tokens={max_tokens}___json={json_mode}___"
    f"prompt={prompt}___sys={system_instruction or ''}"
)
cache_key = hashlib.sha256(cache_key_str.encode("utf-8")).hexdigest()
```

### C. Xác Minh Hồi Quy & Các Chiều Phân Lập
Hàm kiểm thử `test_def01_cache_key_partitioning_by_all_dimensions()` trong [test_evaluation_regression.py](file:///c:/Users/Admin/.ng/.m/.Multi-Agent_RAG/main/tests/test_evaluation_regression.py#L29) chứng minh:
- Cùng một prompt nhưng khác `execution_mode` (`gemini` vs `cpu_max` vs `cpu_min`) $\rightarrow$ tạo ra 3 mã băm SHA-256 hoàn toàn khác biệt.
- Cùng mode nhưng khác `temperature` (0.2 vs 0.7) $\rightarrow$ sinh ra mã băm khác biệt.
- Cùng mode nhưng khác `max_tokens` (1024 vs 2048) $\rightarrow$ sinh ra mã băm khác biệt.
- Cùng mode nhưng khác `agent` (`summarizer` vs `fact_checker` vs `default`) $\rightarrow$ sinh ra mã băm khác biệt.
- Cùng mode nhưng khác `json_mode` (True vs False) $\rightarrow$ sinh ra mã băm khác biệt.
**Kết luận:** Tuyệt đối không còn hiện tượng rò rỉ hoặc ô nhiễm bộ nhớ đệm chéo giữa các chế độ thực thi.

---

## 2. DEF-02: LaTeX JSON Escaping & Robust Parsing

### A. Vấn đề Phát Hiện
Khi mô hình Gemini sinh JSON chứa công thức toán học LaTeX (ví dụ: `$$\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$`):
1. Các ký tự thoát không hợp lệ trong JSON như `\left`, `\right`, `\sqrt`, `\alpha` khiến bộ giải mã chuẩn `json.loads()` văng ngoại lệ `json.decoder.JSONDecodeError: Invalid \escape`.
2. Nếu dùng regex thay thế mù quáng `r"\\(?![/\"\\bfnrtu]|u[0-9a-fA-F]{4})"`:
   - Các lệnh LaTeX bắt đầu bằng ký tự trùng với ký tự thoát chuẩn JSON (`\text`, `\frac`, `\beta`, `\rho`, `\nu`) sẽ bị `json.loads()` nuốt chửng thành các ký tự điều khiển ASCII: `\text` biến thành `<TAB>ext`, `\frac` biến thành `<FORMFEED>rac`, `\beta` biến thành `<BACKSPACE>eta`, phá hủy hoàn toàn công thức toán học!
   - Ký tự thoát hợp lệ đã có sẵn `\\` hoặc `\n` có thể bị gập đôi ngoài ý muốn thành `\\\` dẫn đến lỗi cú pháp mới.

### B. Giải Pháp Triển Khai
Tách logic thành module độc lập chất lượng cao [main/utils/json_repair.py](file:///c:/Users/Admin/.ng/.m/.Multi-Agent_RAG/main/utils/json_repair.py) với bộ quét ký tự từng bước (`sanitize_json_string` & `extract_and_parse_json`):
1. **Fast-path:** Thử giải mã `json.loads` trực tiếp trước. Nếu JSON vốn đã chuẩn 100%, bảo toàn nguyên vẹn 100%.
2. **Scanner Context:** Quét chuỗi theo ngữ cảnh bên trong dấu ngoặc kép (`"`).
3. **Phân biệt chẵn/lẻ Backslash:** Các cặp dấu gạch chéo ngược đã thoát (`\\`) được giữ nguyên không đổi.
4. **Bảo tồn Escape chuẩn:** Giữ nguyên `\"`, `\/`, `\\`, các ký tự unicode `\uXXXX` (4 chữ số hex), và các ngắt dòng `\n`, `\t` thật của văn bản.
5. **Escape có chọn lọc cho LaTeX:** Chỉ bổ sung escape cho các lệnh LaTeX (`\text`, `\frac`, `\beta`, `\rho`, `\nu`...) và các ký tự không hợp lệ trong JSON (`\left`, `\right`, `\sqrt`, `\alpha`, `\{`, `\}`).

### C. Xác Minh Hồi Quy
Hàm `test_def02_latex_escaping_and_valid_json_preservation()` kiểm thử thành công:
- Công thức phức tạp: `$$\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$` và `\alpha + \beta = \gamma`.
- Giữ nguyên các ký tự thoát JSON chuẩn: `\n`, `\t`, chuỗi unicode `\u00e9tudiant \u03b1` ($\rightarrow$ "étudiant α"), đường dẫn Windows `C:\\Windows\\System32`.
- Mã lệnh Python lồng trong JSON: `print(\"hello\\nworld\")`.
- Từ chối an toàn (raise exception) với chuỗi JSON bị vỡ cấu trúc không thể phục hồi.

---

## 3. DEF-03: Translation vs Document QA Disambiguation

### A. Vấn đề Phát Hiện
Trước đây, regex nhận diện tài liệu chứa từ khóa đơn lẻ `"tài liệu"`. Khi người dùng nhập: *"Dịch đoạn tài liệu kỹ thuật sau từ tiếng Anh sang tiếng Việt: 'Attention is all you need'"*, hệ thống bắt nhầm vào `DOCUMENT_QA`, nhưng vì không có file PDF nào đính kèm nên đã trả về thông báo lỗi: `NO_DOCUMENT_SELECTED`.

### B. Giải Pháp Triển Khai
Tại [main/core/supervisor.py](file:///c:/Users/Admin/.ng/.m/.Multi-Agent_RAG/main/core/supervisor.py#L84):
1. Thu hẹp `_EXPLICIT_DOCUMENT_PATTERNS` chỉ khớp các cụm từ chỉ định tài liệu cụ thể:
   `r"\b(tài\s+liệu\s+(?:này|trên|đã\s+chọn|đính\s+kèm)|trong\s+tài\s+liệu|theo\s+tài\s+liệu|dựa\s+vào\s+tài\s+liệu)\b"`
2. Đưa cụm từ dịch thuật vào `_TASK_ASSISTED_PATTERNS`:
   `r"\b(dịch\s+(?:đoạn|câu|văn\s+bản|sang\s+tiếng)|translate\s+(?:this|to|into))\b"`

### C. Xác Minh Hồi Quy
Hàm `test_def03_translation_and_document_routing_disambiguation()` kiểm thử 6 kịch bản đối ứng:
- `"Dịch đoạn tài liệu kỹ thuật sau..."` (không kèm PDF) $\rightarrow$ `TASK_ASSISTED_RESPONSE` (sub-intent: `Translation`, `needs_retrieval=False`, `input_status="OK"`).
- `"Tài liệu PDF này nói gì?"` (không kèm PDF) $\rightarrow$ `DOCUMENT_QA` (`input_status="NO_DOCUMENT_SELECTED"`).
- `"Tài liệu PDF này nói gì?"` (có kèm PDF) $\rightarrow$ `DOCUMENT_QA` (`input_status="OK"`, `needs_retrieval=True`).
- `"Dịch nội dung trang 5 của PDF"` (không kèm PDF) $\rightarrow$ `DOCUMENT_QA` (`NO_DOCUMENT_SELECTED`).
- `"Dịch nội dung trang 5 của PDF"` (có kèm PDF) $\rightarrow$ `DOCUMENT_QA` (`OK`).
- `"Tài liệu là gì?"` hoặc `"Cách viết tài liệu kỹ thuật tốt?"` $\rightarrow$ `GENERAL_QA` (`needs_retrieval=False`).

---

## 4. DEF-04: Task Execution vs Assistance Disambiguation

### A. Vấn đề Phát Hiện
Khi người dùng hỏi câu hỏi giải thích hoặc yêu cầu hỗ trợ viết mã có chứa các từ khóa như `chương trình`, `phần mềm`, `lệnh`, hệ thống có thể bị nhầm lẫn giữa **Yêu cầu thực thi công cụ nguy hiểm (`TASK_EXECUTION`)** và **Câu hỏi hỗ trợ/giải thích thông thường (`GENERAL_QA` hoặc `TASK_ASSISTED_RESPONSE`)**.

### B. Giải Pháp Triển Khai
Tại [main/core/supervisor.py](file:///c:/Users/Admin/.ng/.m/.Multi-Agent_RAG/main/core/supervisor.py#L65-L77):
1. Khóa chặt `_TASK_EXECUTION_PATTERNS`: Chỉ kích hoạt khi có động từ hành động chỉ định thực thi:
   `r"\b(chạy\s+(?:code|chương\s+trình|phần\s+mềm|script|lệnh|ứng\s+dụng)|thực\s+thi\s+(?:lệnh|script|tool|công\s+cụ|chương\s+trình)|run\s+(?:python|code|script|command|tool|program)|execute\s+(?:code|command|tool|script))\b"`
2. Bổ sung cụm từ giải thích câu lệnh vào `_TASK_ASSISTED_PATTERNS`:
   `r"\b(giải\s+thích\s+(?:lệnh|code|hàm|cú\s+pháp|thuật\s+toán|chương\s+trình|script)|explain\s+(?:command|code|syntax))\b"`

### C. Xác Minh Hồi Quy
Hàm `test_def04_execution_keywords_do_not_leak_to_task_execution()` kiểm thử thành công:
- `"Phần mềm là gì?"` $\rightarrow$ `GENERAL_QA` (Không yêu cầu quyền thực thi).
- `"Giải thích lệnh ls"` $\rightarrow$ `TASK_ASSISTED_RESPONSE` (Coding_Or_Guidance, không kích hoạt tool execution).
- `"Viết chương trình Python tính tổng hai số"` $\rightarrow$ `TASK_ASSISTED_RESPONSE` (Hỗ trợ viết mã, không thực thi trên OS).
- `"Chạy lệnh ls trên máy tính của tôi"` $\rightarrow$ `TASK_EXECUTION` (Chuyển sang nhánh Permission Guard).
- `"Chạy chương trình này trên máy tính của bạn và đọc file hệ thống."` $\rightarrow$ `TASK_EXECUTION` (Chuyển sang nhánh Permission Guard).

---

## 5. DEF-05: API Document Handling & Session Isolation

### A. Vấn đề Phát Hiện
Trước đây trong `main/api/app.py`, hàm `create_run` chứa đoạn mã kiểm tra sớm: nếu `target_doc_ids` rỗng, API lập tức abort hoặc đặt trạng thái chờ tài liệu, khiến cho các câu hỏi tổng quát (`GENERAL_QA`, `TASK_ASSISTED_RESPONSE`, `WEB_RESEARCH`) không thể đi vào LangGraph để Supervisor phân luồng.

### B. Giải Pháp Triển Khai
Tại [main/api/app.py](file:///c:/Users/Admin/.ng/.m/.Multi-Agent_RAG/main/api/app.py#L2035):
1. Cho phép `doc_id = None` và `pdf_path = None` khi `target_doc_ids` rỗng.
2. Khởi tạo `AgentState` đầy đủ và gửi vào `active_graph.stream()` để Supervisor Graph tự quyết định định tuyến theo đúng thiết kế Agentic.
3. Đảm bảo `session_id`, `chat_history`, quyền truy cập file và các trường payload cũ (`question` vs `query`, `document_id` vs `document_ids`) hoàn toàn tương thích ngược.

### C. Xác Minh Hồi Quy
Hàm `test_def05_api_document_handling_and_session_isolation()` kiểm thử thành công:
- Câu hỏi kiến thức chung không kèm tài liệu: Run được tạo thành công, `status="processing"`, nhận diện `id` và phân luồng vào `DirectAssistant`.
- Tương thích ngược: Truyền trường cũ `question` và `document_id` hoạt động hoàn hảo.
- Cô lập phiên: `session_id` được ghi nhận chính xác trong cơ sở dữ liệu SQLite và trả về trong phản hồi API.
- Câu hỏi tài liệu nhưng không kèm PDF: Vẫn tạo run thành công, sau đó Supervisor phát hiện `NO_DOCUMENT_SELECTED` và sinh thông báo mẫu hướng dẫn người dùng.
