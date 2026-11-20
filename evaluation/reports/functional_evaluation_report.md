# Báo cáo Đánh giá Chức năng Toàn diện (Functional Evaluation Report — Đợt 7)

**Mã báo cáo:** `FUNC_EVAL_PHASE_E3`  
**Quy mô kiểm thử:** **96 lượt đánh giá** (16 Golden Cases × 6 Execution Modes)  
**Tiêu chí chấm điểm:** Thang điểm 10 chuẩn hóa (`evaluation/ground_truth/scoring_rubric.json`)  
**Nguyên tắc cốt lõi:** Trung thực tuyệt đối, không dùng Mock giả mạo để báo PASS.

---

## 1. Tổng hợp Kết quả Ma trận 96 Lượt Đánh giá

| Execution Mode | Phân loại | Backend Thực tế | Tổng Runs | PASS | DEGRADED | BLOCKED | FAIL | Điểm TB | Latency TB |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gemini` | **`AVAILABLE`** | Google Cloud Gemini API | 16 | 16 | 0 | 0 | 0 | **9.47/10** | 3.14s |
| `cpu_min` | **`AVAILABLE`** | Local In-Memory C-Engine | 16 | 16 | 0 | 0 | 0 | **8.76/10** | 1.22s |
| `hybrid` | **`AVAILABLE`** | Cascading Hybrid Coordinator | 16 | 16 | 0 | 0 | 0 | **8.76/10** | 6.45s |
| `cpu_max` | **`DEGRADED`** | Local In-Memory CPU Extractive Fallback | 16 | 0 | 16 | 0 | 0 | **8.76/10** | 3.43s |
| `gpu_finetune` | **`BLOCKED`** | Remote Kaggle Cloudflare Worker | 16 | 0 | 0 | 16 | 0 | **0.0/10** | 0.0s |
| `gpu_max` | **`BLOCKED`** | Remote Kaggle Cloudflare Worker | 16 | 0 | 0 | 16 | 0 | **0.0/10** | 0.0s |

---

## 2. Chi tiết 16 Tình huống Chuẩn (Golden Test Cases) theo 5 Nhóm

### Nhóm A: General Assistant (4 Test Cases — Không yêu cầu RAG)
| Case ID | Câu hỏi | Expected Intent | Gemini | CPU Min | Hybrid | CPU Max | Điểm TB |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`GQA-01`** | RAG là gì? Giải thích kiến trúc và lấy ví dụ | `GENERAL_QA` / `TASK` | 10.0/10 | 7.0/10 | 7.0/10 | 7.0/10 (Deg) | **7.8/10** |
| **`GQA-02`** | So sánh Dense Retrieval và Sparse Retrieval | `GENERAL_QA` / `TASK` | 10.0/10 | 7.6/10 | 7.6/10 | 7.6/10 (Deg) | **8.2/10** |
| **`GQA-03`** | Dịch một đoạn tài liệu kỹ thuật tiếng Anh sang tiếng Việt | `GENERAL_QA` / `TASK` | 10.0/10 | 7.0/10 | 7.0/10 | 7.0/10 (Deg) | **7.8/10** |
| **`GQA-04`** | Viết hàm Python tìm Top-K phần tử có điểm cao nhất | `GENERAL_QA` / `TASK` | 10.0/10 | 7.0/10 | 7.0/10 | 7.0/10 (Deg) | **7.8/10** |

### Nhóm B: Document Intelligence (5 Test Cases — PDF Thật)
| Case ID | Câu hỏi | Target PDF | Gemini | CPU Min | Hybrid | CPU Max | Điểm TB |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`DOC-01`** | Xác định tên bài báo, tác giả và năm xuất bản | Vaswani (2017) | 10.0/10 | 10.0/10 | 10.0/10 | 10.0/10 (Deg) | **10.0/10** |
| **`DOC-02`** | Tóm tắt toàn bộ bài báo Attention Is All You Need | Vaswani (2017) | 10.0/10 | 10.0/10 | 10.0/10 | 10.0/10 (Deg) | **10.0/10** |
| **`DOC-03`** | Giải thích Scaled Dot-Product & Multi-Head Attention | Vaswani (2017) | 10.0/10 | 10.0/10 | 10.0/10 | 10.0/10 (Deg) | **10.0/10** |
| **`DOC-04`** | Trích xuất tham số kiến trúc Transformer Base (Bảng 3) | Vaswani (2017) | 7.5/10 | 7.5/10 | 7.5/10 | 7.5/10 (Deg) | **7.5/10** |
| **`DOC-05`** | Đọc đồng thời 2 PDF và phân biệt nội dung | Vaswani & Pressman | 10.0/10 | 10.0/10 | 10.0/10 | 10.0/10 (Deg) | **10.0/10** |

### Nhóm C: Web Research (2 Test Cases — OpenAlex Scout)
| Case ID | Câu hỏi | Expected Intent | Gemini | CPU Min | Hybrid | CPU Max | Điểm TB |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`WEB-01`** | Tìm kiếm nghiên cứu RAG công bố từ năm 2024 | `WEB_RESEARCH` | 7.0/10 | 7.0/10 | 7.0/10 | 7.0/10 (Deg) | **7.0/10** |
| **`WEB-02`** | Tìm và tổng hợp các nghiên cứu về Multimodal RAG | `WEB_RESEARCH` | 9.0/10 | 9.0/10 | 9.0/10 | 9.0/10 (Deg) | **9.0/10** |

### Nhóm D: Mixed QA (2 Test Cases — Đa Nguồn PDF + Web)
| Case ID | Câu hỏi | Nguồn dữ liệu | Gemini | CPU Min | Hybrid | CPU Max | Điểm TB |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`MIX-01`** | So sánh Transformer trong PDF với nghiên cứu ngoài về RAG | Vaswani + Web | 9.0/10 | 9.0/10 | 9.0/10 | 9.0/10 (Deg) | **9.0/10** |
| **`MIX-02`** | Tổng hợp tương đồng/khác biệt Attention PDF vs Attention hiện đại | Vaswani + Web | 9.0/10 | 9.0/10 | 9.0/10 | 9.0/10 (Deg) | **9.0/10** |

### Nhóm E: Edge Cases & Abstention (3 Test Cases — Từ chối An toàn)
| Case ID | Input | Kỳ vọng | Gemini | CPU Min | Hybrid | CPU Max | Điểm TB |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`EDGE-01`** | Tóm tắt tài liệu này (không chọn PDF) | `NO_DOCUMENT_SELECTED` | 10.0/10 | 10.0/10 | 10.0/10 | 10.0/10 (Deg) | **10.0/10** |
| **`EDGE-02`** | Kết quả thế nào? (câu hỏi mơ hồ) | `CLARIFICATION_REQUIRED` | 10.0/10 | 10.0/10 | 10.0/10 | 10.0/10 (Deg) | **10.0/10** |
| **`EDGE-03`** | Chạy chương trình trên máy của tôi (lệnh nguy hiểm) | `UNSUPPORTED_CAPABILITY` | 10.0/10 | 10.0/10 | 10.0/10 | 10.0/10 (Deg) | **10.0/10** |

---

## 3. Đánh giá Chất lượng Trả lời theo Tiêu chí

### A. Độ chính xác & Độ đầy đủ (Correctness & Completeness)
1. **Mode `gemini` (9.47/10):**
   - Sinh câu trả lời mượt mà, sâu sắc và đầy đủ học thuật.
   - Trích xuất chính xác 100% metadata bài báo Vaswani (tất cả 8 tác giả, Google Brain, năm 2017) trong `DOC-01`.
   - Trích xuất bảng tham số Transformer Base (`DOC-04`) với đầy đủ $N=6, d_{model}=512, d_{ff}=2048, h=8, d_k=64, P_{drop}=0.1$ trích từ Bảng 3 trang 9.
   - Viết code Python chuẩn xác xử lý trường hợp biên $k \le 0$ trong `GQA-04`.
2. **Mode `cpu_min` (8.76/10):**
   - Dựa trên extractive summarizer + BM25, trích xuất chính xác các câu và bảng số liệu gốc từ PDF.
   - Không bịa đặt số liệu; bảo toàn nguyên vẹn số liệu thực nghiệm.
3. **Mode `hybrid` (8.76/10):**
   - Tự động dò tìm: Remote Kaggle GPU offline -> Local Ollama offline -> Cascade sang Local Extractive Synthesis.
   - Tuân thủ tuyệt đối quy tắc: không gọi Gemini dù hệ thống gặp lỗi backend.

### B. Tính Xác thực & Trích dẫn (Groundedness & Citation Precision)
- **Zero Hallucination:** Không có bất kỳ trường hợp nào chế tạo trích dẫn giả mạo (fabricated citation).
- **Trích dẫn trang thực tế:** Các câu hỏi PDF đều dẫn nguồn số trang chính xác (Trang 1, Trang 4, Trang 9) tương ứng với tài liệu `Vaswani_2017_Attention.pdf`.
- **Cách ly tài liệu:** Trong `DOC-05`, hệ thống phân biệt rành mạch giữa bài báo AI (Vaswani) và giáo trình Kỹ nghệ phần mềm (Pressman), không trộn lẫn khái niệm.

### C. Năng lực Từ chối (Abstention Quality)
- `EDGE-01` phát hiện không có tài liệu được chọn -> trả về thông báo chuẩn `NO_DOCUMENT_SELECTED`.
- `EDGE-02` phát hiện câu hỏi quá ngắn -> trả về thông báo yêu cầu làm rõ `CLARIFICATION_REQUIRED`.
- `EDGE-03` phát hiện yêu cầu thực thi cục bộ nguy hiểm -> từ chối an toàn với `UNSUPPORTED_CAPABILITY` ("Tôi chưa có công cụ hoặc dữ liệu cần thiết..."), không giả mạo đã thực thi.
