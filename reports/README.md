# 📚 BỘ BÁO CÁO KIẾN TRÚC HỆ THỐNG MULTI-AGENT RESEARCH RAG (TECHNICAL MONOGRAPHS)

Chào mừng bạn đến với trung tâm tài liệu kỹ thuật hoàn chỉnh của dự án **Multi-Agent Research RAG**. Hệ thống bao gồm 5 Tác tử Cốt lõi (`Agent 1` đến `Agent 5`), vận hành trên nền tảng **Đồ thị Trạng thái LangGraph** kết hợp **Cơ chế Tự chủ 2 Vòng Lặp (Two-Tier Self-Correction Loop)** và **7 Tùy chọn Thực thi Độc lập (Execution Modes)**.

---

## 🗂️ Mục Lục Điều Hướng Các Tập Báo Cáo Chuyên Sâu

| STT | Báo Cáo Kiến Trúc (Markdown) | Trọng Tâm Kỹ Thuật |
| :---: | :--- | :--- |
| **00** | [**00_TONG_QUAN_KIEN_TRUC_MULTI_AGENT.md**](./00_TONG_QUAN_KIEN_TRUC_MULTI_AGENT.md) | **Kiến Trúc Tổng Thể Toàn Hệ Thống**: Đồ thị trạng thái LangGraph, Cơ chế Tự chủ 2 Vòng lặp (Tier 1 Reflection & Tier 2 Consensus), ma trận 5 Agent (`Agent 1` $\to$ `Agent 5`) và tách biệt tiền xử lý Agent 0. |
| **01** | [**01_TUY_CHON_1_CPU_MIN.md**](./01_TUY_CHON_1_CPU_MIN.md) | **Tùy chọn 1 (`cpu_min` — Ultra-Lightweight)**: Vận hành không mạng, RAM $< 1\text{GB}$, kết hợp BM25 Okapi + TF-IDF SVD/LSA, đồ thị trọng tâm câu và bộ lọc số học/thực thể ($15.6\text{ ms}$). |
| **02** | [**02_TUY_CHON_2_CPU_MAX.md**](./02_TUY_CHON_2_CPU_MAX.md) | **Tùy chọn 2 (`cpu_max` — Heavy Local Workstation)**: Nhúng đa ngữ `multilingual-e5-small` + BM25 RRF ($k=60, \alpha=0.65$) + Reranker MiniLM + Động cơ suy luận Ollama cục bộ. |
| **03** | [**03_TUY_CHON_3_GEMINI_CLOUD.md**](./03_TUY_CHON_3_GEMINI_CLOUD.md) | **Tùy chọn 3 (`gemini` — Cloud Foundation API)**: Phân rã truy vấn đa chiều (Sub-Queries), ép kiểu JSON Schema nghiêm ngặt, vòng lặp phản biện Generator-Critic và Giám khảo Đa nhân cách. |
| **04** | [**04_TUY_CHON_4_GPU_FT_MULTILORA.md**](./04_TUY_CHON_4_KAGGLE_FN_MULTILORA.md) | **Tùy chọn 4 (`gpu_ft` — Standard Multi-LoRA Serving)**: Hoán đổi nóng 3 Adapter LoRA (`router`, `generator`, `critic`) trên nền Qwen2.5-7B (Dual T4) phục vụ đo lường đối chứng chuẩn (Ablation Baseline). |
| **05** | [**05_TUY_CHON_5_GPU_FINETUNE_SOTA.md**](./05_TUY_CHON_5_KAGGLE_MAX_ENSEMBLE.md) | **Tùy chọn 5 (`gpu_finetune` — Full 7-Model Specialized Suite)**: Trọn bộ 7 mô hình chuyên sâu (BGE-M3 MoAI, CoREB Reranker, Router LoRA, LayoutLMv3 2D OCR, Dual-QLoRA Summarizer, BGE NLI Verifier F1=1.0, DoRA Writer 99.6%). |
| **06** | [**06_TUY_CHON_6_GPU_MAX_32B.md**](./05_TUY_CHON_5_KAGGLE_MAX_ENSEMBLE.md) | **Tùy chọn 6 (`gpu_max` — Heavyweight 32B AWQ Reasoning)**: Đại mô hình Qwen3-32B-AWQ chạy song song trên 2 GPU qua vLLM ($TP=2$), hỗ trợ phân nhánh chuỗi tư duy `<think>...</think>`. |
| **07** | [**07_TUY_CHON_7_HYBRID_BALANCED.md**](./06_TUY_CHON_6_HYBRID_KET_HOP.md) | **Tùy chọn 7 (`hybrid` — Balanced Autonomous Multi-Tier Multi-Agent)**: Phân bổ tối ưu đa nền tảng (Gemini Router $\to$ PyMuPDF Reader $\to$ Dual-QLoRA Summarizer $\to$ BGE NLI Guard $\to$ Local Gap Writer) với cơ chế thác dự phòng 3–4 tầng chống sập 100%. |
| **08** | [**08_PHONG_THI_NGHIEM_TRAINING_LAB.md**](./07_PHONG_THI_NGHIEM_TRAINING_LAB.md) | **Kiến Trúc Huấn Luyện 5 Agent (`training_lab/`)**: Cấu trúc gói `src/`, các thuật toán SOTA (`DoRA + rsLoRA + SDPA + RPO/DPO + Focal Loss + R-Drop`) và đồng bộ Google Drive tự động. |
