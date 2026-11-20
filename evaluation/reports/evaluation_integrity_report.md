# Báo Cáo Tính Liêm Chính Đánh Giá (Evaluation Integrity Report)

**Thời điểm:** 2026-11-20  
**Tài liệu tham chiếu:** `evaluation/dataset/`, `evaluation/ground_truth/`, `evaluation/results/`  
**Mục tiêu:** Chứng minh tính xác thực, tính bất biến và độ tin cậy của toàn bộ bộ dữ liệu và kết quả đánh giá Đợt 7.

---

## 1. Tính Bất Biến Của Bộ Dữ Liệu Chuẩn (Golden Dataset)

Bộ dữ liệu chuẩn bao gồm đúng **16 Golden Test Cases** độc lập, được tổ chức theo cấu trúc JSON phân nhóm chuẩn hóa:

| Nhóm Đánh Giá | Mã File | Số Lượng Case | Mã Case | Mục Tiêu Thẩm Định |
| :--- | :--- | :---: | :--- | :--- |
| **Nhóm A: General Assistant** | `general_qa.json` | 4 | `GQA-01` .. `GQA-04` | Kiến thức chung, lập trình Python, dịch thuật kỹ thuật, chào hỏi xã giao. Không dùng RAG. |
| **Nhóm B: Document Intelligence** | `document_qa.json` | 5 | `DOC-01` .. `DOC-05` | Đọc hiểu PDF, trích xuất bảng số liệu chính xác (Table 3 Vaswani), so sánh đa tài liệu, tài liệu đồ sộ (Pressman 888 trang). |
| **Nhóm C: Web Research (OpenAlex)** | `web_research.json` | 2 | `WEB-01` .. `WEB-02` | Khám phá nghiên cứu RAG mới từ OpenAlex API (năm 2024+), trích xuất DOI và tóm tắt học thuật. |
| **Nhóm D: Mixed QA** | `mixed_qa.json` | 2 | `MIX-01` .. `MIX-02` | Tổng hợp đối chiếu thông tin từ PDF nội bộ và công trình nghiên cứu bên ngoài. |
| **Nhóm E: Edge Cases & Abstention** | `edge_cases.json` | 3 | `EDGE-01` .. `EDGE-03` | Câu hỏi mơ hồ (`CLARIFICATION_REQUIRED`), câu hỏi không có trong tài liệu (`ZERO_HALLUCINATION`), lệnh nguy hiểm (`TASK_EXECUTION`). |

- **Kiểm tra Schema:** Cả 16 test cases đã được xác thực qua [validate_dataset.py](file:///c:/Users/Admin/.ng\.m\.Multi-Agent_RAG/evaluation/schemas/validate_dataset.py) với [test_case.schema.json](file:///c:/Users/Admin/.ng\.m\.Multi-Agent_RAG/evaluation/schemas/test_case.schema.json) và đạt tỷ lệ hợp lệ **100% (16/16)**.
- **Tính Bất Biến:** Toàn bộ kết quả E7 ban đầu được giữ nguyên trong `evaluation/results/` làm mốc tham chiếu lịch sử (baseline).

---

## 2. Dữ Kiện Kiểm Chứng Nền Tảng (Ground Truth Facts)

Toàn bộ thang điểm và đáp án đối chiếu được trích xuất trực tiếp từ các tài liệu gốc tại `evaluation/ground_truth/`:

1. **`document_facts.json`:**
   - *Vaswani et al. (2017) Table 3 (Trang 9):* Bảng cấu hình Transformer Base:
     - Số lớp encoder/decoder: $N = 6$
     - Kích thước vector biểu diễn: $d_{model} = 512$
     - Kích thước tầng ẩn feed-forward: $d_{ff} = 2048$
     - Số đầu cơ chế chú ý: $h = 8$
     - Kích thước từng đầu: $d_k = 64, d_v = 64$
     - Dropout rate: $P_{drop} = 0.1$
     - Label smoothing: $\epsilon_{ls} = 0.1$
   - *Pressman (2001) Chương 2:* 5 hoạt động khung công nghệ phần mềm: Communication, Planning, Modeling, Construction, Deployment.
2. **`expected_citations.json`:** Quy định bắt buộc số trang và câu trích dẫn cho từng luận điểm kiểm chứng.
3. **`expected_routes.json`:** Quy định nhãn `primary_intent` kỳ vọng và cờ kích hoạt (`needs_scout`, `needs_retrieval`).

---

## 3. Thang Điểm Chuẩn Hóa 10 Điểm (Scoring Rubric)

Mỗi lần chạy được chấm điểm tự động và độc lập theo công thức:

$$\text{Total Score} = \text{Correctness} (4) + \text{Completeness} (2) + \text{Groundedness} (2) + \text{Routing} (1) + \text{Calibration} (1)$$

- **Correctness (0 - 4):**
  - $4.0$: Đúng toàn bộ dữ kiện tham chiếu, số liệu khớp từng giá trị.
  - $3.0$: Đúng ý chính nhưng thiếu một chi tiết nhỏ không cốt lõi.
  - $2.0$: Đúng một phần, có dữ kiện chưa chuẩn xác.
  - $0.0$: Sai lệch dữ kiện hoặc bị ảo giác (hallucination).
- **Completeness (0 - 2):**
  - $2.0$: Bao phủ đầy đủ tất cả khía cạnh và thực thể được yêu cầu.
  - $1.0$: Trả lời được khía cạnh chính nhưng bỏ sót câu hỏi phụ.
  - $0.0$: Trả lời cụt ngủn hoặc không đáp ứng câu hỏi.
- **Groundedness & Citation (0 - 2):**
  - $2.0$: Trích dẫn đúng số trang và có đoạn trích dẫn chứng minh.
  - $1.0$: Có trích dẫn nhưng thiếu số trang hoặc trích dẫn mơ hồ.
  - $0.0$: Khẳng định không có căn cứ hoặc trích dẫn sai số trang.
- **Routing Accuracy (0 - 1):**
  - $1.0$: Định tuyến đúng `primary_intent` và cấu hình retrieval.
  - $0.0$: Định tuyến sai luồng xử lý.
- **Uncertainty Calibration (0 - 1):**
  - $1.0$: Từ chối chính xác khi thiếu tài liệu hoặc yêu cầu nguy hiểm.
  - $0.0$: Tự tiện bịa đặt hoặc thực thi ngoài thẩm quyền.

---

## 4. Nguyên Tắc Minh Bạch Với Backend Offline (Zero-Mock Principle)

Hệ thống tuân thủ nghiêm ngặt nguyên tắc **Không dùng Mock để làm giả kết quả**:
1. **`gpu_finetune` & `gpu_max`:** Ghi nhận trung thực `BLOCKED` với điểm $0.00$. Tuyệt đối không dùng mock object giả lập để báo cáo kết quả PASS giả.
2. **`cpu_max`:** Được phân loại chính xác là `DEGRADED`. Điểm số $8.76$ phản ánh năng lực của bộ trích xuất nội dung an toàn (in-memory extractive engine), tuyệt đối không được gán ghép thành năng lực của mô hình LLM Qwen 7B.
3. **`cpu_min`:** Được ghi nhận là `PASS*` với chú thích rõ ràng: đây là deterministic engine trích xuất câu và bảng số liệu, không phải là mô hình sinh ngôn ngữ tự nhiên.
4. **`hybrid`:** Được tách bạch rõ ràng: `ROUTING: PASS` (chọn luồng đúng thiết kế), nhưng `EXECUTION: DEGRADED` (phải cascade về CPU vì backend GPU vắng mặt).
