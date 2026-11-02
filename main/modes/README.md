# 🎛️ 6 Tùy Chọn Thực Thi Đa Tác Tử (`main/modes/`)

Thư mục `main/modes/` chứa 6 gói chế độ thực thi độc lập, có thể chuyển đổi toàn cục hoặc phối trộn từng Agent riêng lẻ thông qua [`registry.py`](registry.py).

---

## 🔍 Chi Tiết Từng Chế Độ

1. **[`cpu_min/`](cpu_min/) — Siêu Tốc Cục Bộ (`~0.39s`):**
   - Sử dụng C-Engine SQLite FTS5 + BM25 kết hợp bộ đọc câu nguyên văn từ PDF đang chọn và kiểm tra số học tất định.
   - Là chế độ mặc định khi tải lại trang giao diện Web.
2. **[`cpu_max/`](cpu_max/) — Nghiên Cứu Chuyên Sâu Cục Bộ (`~4.08s`):**
   - Khai thác tối đa phần cứng **AMD Ryzen 7 H 255 (AVX-512) + Radeon 780M Vulkan iGPU**.
   - Kết hợp STORM Multi-Perspective Query + Hybrid Dense (`multilingual-e5-small`) / FTS5 + Cross-Encoder Reranking + MiniCheck NLI + Kiểm toán thống kê (Statistical Auditor).
3. **[`gemini/`](gemini/) — Đám Mây Tối Ưu Hạn Ngạch (2–3 API Calls/Query):**
   - Gộp Generator + Self-Critic vào **1 lần gọi JSON duy nhất** (`summarizer.py`).
   - Thay thế vòng lặp kiểm tra từng câu bằng **1 lần gọi Batch NLI Verifier JSON** (`fact_checker.py`), tiết kiệm 75% hạn ngạch Gemini mỗi ngày.
4. **[`kaggle_fn/`](kaggle_fn/) — Chuyên Gia Tinh Chỉnh Multi-LoRA (`Qwen2.5-7B` + 3 Adapters):**
   - Gọi 3 bộ QLoRA chuyên biệt trên Kaggle GPU: `router` (Retrieval), `generator` + `critic` (vòng lặp tự sửa lỗi trong Summarizer), và Cross-Encoder NLI (Fact-Checker).
   - Tự động chuyển về chế độ tổng hợp trích xuất cục bộ nếu đường hầm Kaggle ngoại tuyến.
5. **[`kaggle_max/`](kaggle_max/) — Đại Mô Hình Suy Luận 32B AWQ (`Qwen3-32B-AWQ` Dual T4 `TP=2`):**
   - Kích hoạt chế độ suy luận sâu `/think` (tách thẻ `<think>...</think>` hiển thị lên giao diện) cho Summarizer và `/no_think` tốc độ cao cho Router & Batch NLI Verifier.
6. **[`hybrid/`](hybrid/) — Tổ Hợp Đối Kháng Đa Mô Hình (`Flagship Mode`):**
   - Tự động thăm dò các nguồn lực đang trực tuyến (`Ollama Vulkan`, `Gemini API`, `Kaggle Tunnel`).
   - Áp dụng **Cross-Model Adversarial Verification**: Bắt buộc Agent Tổng hợp (Generator) và Agent Kiểm chứng (Fact-Checker) sử dụng **2 họ mô hình khác nhau** để triệt tiêu hoàn toàn hiện tượng tự xác nhận ảo giác.
