# Báo cáo So sánh Đối chiếu 6 Execution Modes (Cross-Mode Comparison)

**Mã báo cáo:** `CROSS_MODE_COMPARISON_E6`  
**Đối tượng so sánh:** 6 execution modes chính thức (`cpu_min`, `cpu_max`, `gpu_finetune`, `gpu_max`, `hybrid`, `gemini`).

---

## 1. Bảng So sánh Tổng thể Năng lực 6 Modes

| Tiêu chí Đánh giá | `gemini` | `cpu_min` | `hybrid` | `cpu_max` | `gpu_finetune` | `gpu_max` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phân loại Hiện trạng** | **`AVAILABLE`** | **`AVAILABLE`** | **`AVAILABLE`** | **`DEGRADED`** | **`BLOCKED`** | **`BLOCKED`** |
| **Backend Thực tế** | Gemini 2.5 Flash API | Local C-Engine / BM25 | Multi-Tier Cascade | Extractive Fallback | Remote Kaggle T4 | Remote Kaggle 32B |
| **Độ sẵn sàng (Readiness)** | Sẵn sàng 100% | Sẵn sàng 100% | Sẵn sàng 100% | Sẵn sàng (chế độ suy giảm) | Không khả dụng | Không khả dụng |
| **Điểm Đánh giá TB (Score)** | **9.47 / 10** | **8.76 / 10** | **8.76 / 10** | **8.76 / 10** | **0.0 / 10** | **0.0 / 10** |
| **Tỷ lệ Đạt (Pass Rate)** | **100%** (16/16) | **100%** (16/16) | **100%** (16/16) | **0% (16 Degraded)** | **0% (16 Blocked)** | **0% (16 Blocked)** |
| **Thời gian phản hồi TB (P50)**| 3.14s | 1.22s | 6.45s | 3.43s | N/A | N/A |
| **Khả năng sinh văn bản tự do**| Rất cao (LLM 2.5) | Trung bình (Extractive) | Trung bình (Cascade) | Trung bình (Extractive) | N/A | N/A |
| **Độ trung thực trích dẫn** | 100% có căn cứ | 100% văn bản gốc | 100% văn bản gốc | 100% văn bản gốc | N/A | N/A |
| **Chống ảo giác (Anti-Hallucination)** | Inner Critic + NLI | Deterministic Rules | Deterministic Rules | Deterministic Rules | N/A | N/A |
| **Hoạt động Offline 100%** | Không (cần Cloud) | **Có (100% offline)** | **Có (tự cascade)** | **Có (tự fallback)** | Không | Không |
| **Tuân thủ Provider Policy** | 100% | 100% | 100% (Zero Gemini) | 100% (Zero Gemini) | 100% (Từ chối gpu_ft) | 100% (Từ chối gpu_ft) |

---

## 2. Phân tích Chi tiết Từng Execution Mode

### 1. `gemini` (Cloud Foundation Mode) — `AVAILABLE`
- **Điểm số:** **9.47 / 10** | **Pass Rate:** **100% (16/16)** | **Latency TB:** **3.14s**
- **Điểm mạnh:**
  - Khả năng diễn giải, tổng hợp đa tài liệu và so sánh học thuật đạt mức xuất sắc nhất trong 6 mode.
  - Phân định rõ ràng giữa bài báo khoa học Vaswani và sách giáo trình Pressman.
  - Tự động phát hiện các yêu cầu vượt thẩm quyền (`EDGE-03`) và từ chối an toàn.
- **Hạn chế:** Phụ thuộc vào kết nối Internet và Google API Key.

### 2. `cpu_min` (Local Minimal Mode) — `AVAILABLE`
- **Điểm số:** **8.76 / 10** | **Pass Rate:** **100% (16/16)** | **Latency TB:** **1.22s**
- **Điểm mạnh:**
  - Tốc độ phản hồi nhanh nhất (P50 chỉ 1.22 giây).
  - Hoạt động hoàn toàn độc lập, không cần bất kỳ dịch vụ hay API nào bên ngoài.
  - Bộ kiểm chứng xác thực (Deterministic Verifier) hoạt động nghiêm ngặt, đối soát số liệu và số trang chính xác tuyệt đối.
- **Hạn chế:** Các câu hỏi kiến thức mở không có tài liệu ngữ cảnh sẽ dựa vào trích xuất cấu trúc câu thay vì sinh ngôn ngữ tự nhiên mềm mại.

### 3. `hybrid` (Multi-Tier Adaptive Mode) — `AVAILABLE`
- **Điểm số:** **8.76 / 10** | **Pass Rate:** **100% (16/16)** | **Latency TB:** **6.45s**
- **Điểm mạnh:**
  - Khả năng tự thích ứng và phục hồi lỗi cao nhất: tự động kiểm tra Remote GPU (Kaggle), khi không khả dụng lập tức chuyển xuống Local Ollama, và khi Ollama không có sẽ rơi xuống Local Extractive CPU.
  - **Bảo mật & Tuân thủ Chính sách:** Dù gặp lỗi toàn bộ backend, Hybrid tuyệt đối KHÔNG bao giờ gọi Gemini, bảo vệ dữ liệu nội bộ.
- **Hạn chế:** Do phải thực hiện probe và cascade qua các tầng kết nối nên độ trễ trung bình cao hơn (6.45s).

### 4. `cpu_max` (Local 7B LLM Mode) — `DEGRADED`
- **Điểm số:** **8.76 / 10** | **Trạng thái:** **DEGRADED (16/16)** | **Latency TB:** **3.43s**
- **Nhận định kiểm toán:** Mode này được thiết kế để chạy với mô hình cục bộ `qwen2.5-coder:7b` trên Ollama. Do Ollama daemon trên cổng 11434 hiện không hoạt động trên máy test, hệ thống tự động kích hoạt extractive fallback.
- **Quy tắc nghiệm thu:** Theo chỉ thị bắt buộc, toàn bộ 16 ca của mode này được ghi nhận chính xác là `DEGRADED`, không được tính là LLM 7B đã pass.

### 5. `gpu_finetune` & `gpu_max` — `BLOCKED`
- **Điểm số:** **0.0 / 10** | **Trạng thái:** **BLOCKED (16/16 mỗi mode)** | **Latency TB:** **0.0s**
- **Nhận định kiểm toán:** Worker từ xa trên Kaggle (qua Cloudflare Tunnel `https://grad-populations-logan-walker.trycloudflare.com`) hiện không thể kết nối (HTTP 502 / DNS getaddrinfo failed).
- **Quy tắc nghiệm thu:** Cả 32 lượt của 2 mode này được đánh dấu `BLOCKED`. Không dùng Mock giả mạo để báo PASS.

---

## 3. Ma trận Đánh đổi (Trade-off Matrix)

```
        Tốc độ phản hồi (Speed)
                  ▲
                  │  [cpu_min] (1.22s, 100% Offline)
                  │
                  │         [gemini] (3.14s, 9.47/10 điểm)
                  │
                  │  [cpu_max-degraded] (3.43s)
                  │
                  │  [hybrid] (6.45s, Zero Gemini Cascade)
                  │
                  └────────────────────────────────────────► Độ phong phú câu trả lời (Quality)
```
