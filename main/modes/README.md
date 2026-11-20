# 🎛️ Danh Mục Các Tùy Chọn Thực Thi Đa Tác Tử (`main/modes/`)

Thư mục `main/modes/` cung cấp **6 chế độ thực thi chính thức (Execution Modes)**, cho phép chuyển đổi toàn cục hoặc phối ghép linh hoạt từng Agent thông qua bộ điều phối trung tâm [`registry.py`](registry.py).

---

## 🔍 Bảng Tổng Hợp 6 Chế Độ Thực Thi Chính Thức

| # | Chế Độ (Mode) | Môi Trường Vận Hành | Kiến Trúc & Công Nghệ Cốt Lõi | Đặc Tính Kỹ Thuật |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **[`gpu_finetune/`](gpu_finetune/)** | **Kaggle Dual Tesla T4** | **Kiến trúc Tinh chỉnh Toàn diện 7 Mô hình**: BGE-M3 Dense Retriever + CoREB Cross-Encoder Reranker + Router Planner LoRA + LayoutLMv3 Visual Document Parser + Dual-QLoRA Summarizer/Critic + Calibrated BGE NLI Verifier + DoRA Academic Writer. | Độ chuẩn xác học thuật tối đa, phục vụ nghiên cứu đa tài liệu chuyên sâu và trích xuất cấu trúc 2D phức tạp. |
| **2** | **[`hybrid/`](hybrid/)** | **Cân Bằng Đa Nền Tảng (Multi-Tier)** | **Kiến trúc Đa Tác tử Cân bằng qua `ModelGateway` (Không gọi Gemini)**: Thứ tự ưu tiên `gpu_finetune` $\rightarrow$ `gpu_max` $\rightarrow$ `cpu_max` (Local CPU). | Tự chủ quyết định, tự phục hồi không gián đoạn qua `ModelGateway`, tuân thủ tuyệt đối chính sách cô lập Gemini. |
| **3** | **[`gpu_max/`](gpu_max/)** | **Kaggle Dual Tesla T4** | **Đại Mô hình Suy luận Chuyên sâu 32B**: `Qwen2.5-32B-Instruct-AWQ` vận hành song song trên 2 GPU qua vLLM ($TP=2$), hỗ trợ phân nhánh chuỗi tư duy mở rộng (`/think` và `/no_think`). | Phục vụ lập luận đa bước phức tạp trên các tài liệu chuyên ngành quy mô lớn (30+ trang). |
| **4** | **[`gemini/`](gemini/)** | **Google Cloud Platform** | **Nền tảng Đám mây Tối ưu Hạn ngạch**: Toàn bộ quy trình vận hành qua Gemini API, chế độ duy nhất được phép gọi Gemini API. | Tốc độ phản hồi cao, không đòi hỏi tài nguyên tính toán cục bộ. |
| **5** | **[`cpu_min/`](cpu_min/)** | **100% Cục Bộ (Zero-GPU)** | **Động cơ Trích xuất C-Engine Siêu tốc**: SQLite FTS5 + BM25 + PyMuPDF C-Engine kết hợp Extractive Synthesizer và bộ thẩm định số học/thực thể tất định. | Mức chiếm dụng RAM $< 1\text{GB}$, hoàn toàn độc lập với mạng Internet và phần cứng đồ họa. |
| **6** | **[`cpu_max/`](cpu_max/)** | **100% Cục Bộ (Workstation)** | **Trạm Làm việc Cục bộ Nâng cao**: Tối ưu hóa tập lệnh **AVX-512** của AMD Ryzen kết hợp nhân đồ họa tích hợp **Radeon 780M Vulkan** chạy mô hình 7B cục bộ qua Ollama + MiniCheck NLI. | Đảm bảo tính bảo mật và toàn vẹn dữ liệu tuyệt đối theo tiêu chuẩn On-Premise. |

---

## 🏛️ Sơ Đồ Cấu Trúc Mã Nguồn

```text
modes/
├── registry.py        # Dynamic Agent Factory: Phân giải động 6 chế độ chính thức x 5 tác tử
├── gpu_finetune/      # Kiến trúc Tinh chỉnh Toàn diện 7 Mô hình (Tên mode chính thức)
├── hybrid/            # Kiến trúc Đa Tác tử Cân bằng (GPU Finetune -> GPU Max -> Local CPU, không Gemini)
├── gpu_max/           # Đại Mô hình Suy luận Chuyên sâu 32B AWQ Tensor Parallelism (TP=2)
├── gemini/            # Nền tảng Đám mây Gemini API (chế độ duy nhất được phép gọi Gemini)
├── cpu_min/           # Động cơ Trích xuất C-Engine Cục bộ Tinh gọn
└── cpu_max/           # Động cơ Trạm Làm việc Cục bộ Nâng cao (AMD Ryzen AVX-512 + Vulkan iGPU)
```
