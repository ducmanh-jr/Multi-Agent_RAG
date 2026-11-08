# 🎛️ Danh Mục Các Tùy Chọn Thực Thi Đa Tác Tử (`main/modes/`)

Thư mục `main/modes/` cung cấp **7 chế độ thực thi độc lập (Execution Modes)**, cho phép chuyển đổi toàn cục hoặc phối ghép linh hoạt từng Agent thông qua bộ điều phối trung tâm [`registry.py`](registry.py).

---

## 🔍 Bảng Tổng Hợp 7 Chế Độ Thực Thi

| # | Chế Độ (Mode) | Môi Trường Vận Hành | Kiến Trúc & Công Nghệ Cốt Lõi | Đặc Tính Kỹ Thuật |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **[`gpu_finetune/`](gpu_finetune/)** | **Kaggle Dual Tesla T4** | **Kiến trúc Tinh chỉnh Toàn diện 7 Mô hình**: BGE-M3 Dense Retriever + CoREB Cross-Encoder Reranker + Router Planner LoRA + LayoutLMv3 Visual Document Parser + Dual-QLoRA Summarizer/Critic + Calibrated BGE NLI Verifier (Macro F1 = 1.0000) + DoRA Academic Writer (Token Acc = 99.6%). | Độ chuẩn xác học thuật tối đa, phục vụ nghiên cứu đa tài liệu chuyên sâu và trích xuất cấu trúc 2D phức tạp. |
| **2** | **[`hybrid/`](hybrid/)** | **Cân Bằng Đa Nền Tảng (Multi-Tier)** | **Kiến trúc Đa Tác tử Cân bằng Tự chủ Cấp độ 4**: Gemini Foundation API (Agent 1 Router) $\rightarrow$ C-Engine PyMuPDF (Agent 2 Reader) $\rightarrow$ Dual-QLoRA Fine-Tuned (Agent 3 Summarizer) $\rightarrow$ Deterministic Numeric Guard & BGE NLI (Agent 4 Fact-Checker) $\rightarrow$ Local Academic Report Builder & Research Gap Radar (Agent 5 Writer). | Tự chủ quyết định, tự phục hồi không gián đoạn (Zero-Downtime) thông qua chuỗi dự phòng thác 3–4 tầng (Cascade Fallback). |
| **3** | **[`gpu_ft/`](gpu_ft/)** | **Kaggle Dual Tesla T4** | **Kiến trúc Tinh chỉnh Tiêu chuẩn 3 Adapter**: Nạp động 3 LoRA Adapters (`router`, `generator`, `critic`) trên nền tảng Qwen2.5-7B (4-bit NF4) kết hợp phân tích văn bản tuyến tính. | Cung cấp cấu hình chuẩn mực cho các nghiên cứu đo lường, đối chứng thực nghiệm (Ablation Baseline). |
| **4** | **[`gpu_max/`](gpu_max/)** | **Kaggle Dual Tesla T4** | **Đại Mô hình Suy luận Chuyên sâu 32B**: `Qwen2.5-32B-Instruct-AWQ` vận hành song song trên 2 GPU qua vLLM ($TP=2$), hỗ trợ phân nhánh chuỗi tư duy mở rộng (`/think` và `/no_think`). | Phục vụ lập luận đa bước phức tạp trên các tài liệu chuyên ngành quy mô lớn (30+ trang). |
| **5** | **[`gemini/`](gemini/)** | **Google Cloud Platform** | **Nền tảng Đám mây Tối ưu Hạn ngạch**: Toàn bộ quy trình vận hành qua Gemini API (`gemini-2.5-flash`), nén tối ưu xuống 2–3 cuộc gọi API/query nhờ kỹ thuật đóng gói đa tác vụ Structured JSON Schema. | Tốc độ phản hồi cao, không đòi hỏi tài nguyên tính toán cục bộ. |
| **6** | **[`cpu_min/`](cpu_min/)** | **100% Cục Bộ (Zero-GPU)** | **Động cơ Trích xuất C-Engine Siêu tốc**: SQLite FTS5 + BM25 + PyMuPDF C-Engine kết hợp Extractive Synthesizer và bộ thẩm định số học/thực thể tất định (`~0.39s`). | Mức chiếm dụng RAM $< 1\text{GB}$, hoàn toàn độc lập với mạng Internet và phần cứng đồ họa. |
| **7** | **[`cpu_max/`](cpu_max/)** | **100% Cục Bộ (Workstation)** | **Trạm Làm việc Cục bộ Nâng cao**: Tối ưu hóa tập lệnh **AVX-512** của AMD Ryzen kết hợp nhân đồ họa tích hợp **Radeon 780M Vulkan** chạy mô hình 7B cục bộ qua Ollama + MiniCheck NLI. | Đảm bảo tính bảo mật và toàn vẹn dữ liệu tuyệt đối theo tiêu chuẩn On-Premise. |

---

## 🏛️ Sơ Đồ Cấu Trúc Mã Nguồn

```text
modes/
├── registry.py        # Dynamic Agent Factory: Phân giải động 7 chế độ x 5 tác tử
├── gpu_finetune/      # Kiến trúc Tinh chỉnh Toàn diện 7 Mô hình (Full 7-Model Fine-Tuned Suite)
├── hybrid/            # Kiến trúc Đa Tác tử Cân bằng Tự chủ Cấp độ 4 (Balanced Multi-Tier Architecture)
├── gpu_ft/            # Kiến trúc Tinh chỉnh Tiêu chuẩn 3 Adapter (Standard 3-Adapter Multi-LoRA Baseline)
├── gpu_max/           # Đại Mô hình Suy luận Chuyên sâu 32B AWQ Tensor Parallelism (TP=2)
├── gemini/            # Nền tảng Đám mây Tối ưu Hạn ngạch (Quota-Optimized Cloud Foundation API)
├── cpu_min/           # Động cơ Trích xuất C-Engine Cục bộ Tinh gọn
└── cpu_max/           # Động cơ Trạm Làm việc Cục bộ Nâng cao (AMD Ryzen AVX-512 + Vulkan iGPU)
```
