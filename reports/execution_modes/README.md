# 🎛️ Báo Cáo Kỹ Thuật 7 Tùy Chọn Thực Thi (Execution Modes)

Thư mục này phân tích chi tiết nguyên lý toán học, sơ đồ luồng dữ liệu Mermaid và ánh xạ mã nguồn của toàn bộ **7 Chế độ Thực thi Độc lập**:

| Tệp Báo Cáo | Tên Chế Độ | Phần Cứng Mục Tiêu | Đặc Điểm Cốt Lõi | Overall F1 | Faithfulness |
| :--- | :--- | :--- | :--- | :---: | :---: |
| [**`01_cpu_min.md`**](./01_cpu_min.md) | `cpu_min` | RAM < 1GB, 0 GPU | BM25 Okapi + TF-IDF SVD/LSA + Extractive Graph | 0.785 | 0.810 |
| [**`02_cpu_max.md`**](./02_cpu_max.md) | `cpu_max` | Workstation CPU | Multilingual-E5 + BM25 RRF + MiniLM + Ollama | 0.842 | 0.865 |
| [**`03_gemini_cloud.md`**](./03_gemini_cloud.md) | `gemini` | Cloud API | Gemini 2.5 Flash JSON Schema + Multi-Persona Judge | 0.896 | 0.912 |
| [**`04_gpu_ft_multilora.md`**](./04_kaggle_fn_multilora.md) | `gpu_ft` | Dual T4 (32GB) | Qwen2.5-7B 3-Adapter Multi-LoRA Baseline (Ablation Benchmark) | 0.918 | 0.938 |
| [**`05_gpu_finetune_sota.md`**](./04_kaggle_fn_multilora.md) | `gpu_finetune` | Dual T4 (32GB) | Kiến trúc 7 Mô hình Toàn diện (LayoutLMv3, Dual-QLoRA, NLI F1=1.0, DoRA) | **0.958** | **0.975** |
| [**`06_gpu_max_ensemble.md`**](./05_kaggle_max_ensemble.md) | `gpu_max` | Dual T4 (32GB) | Qwen3-32B AWQ Tensor Parallelism (TP=2) + /think Deep CoT | 0.945 | 0.964 |
| [**`07_hybrid_balanced.md`**](./06_hybrid_adaptive.md) | `hybrid` | Đa Nền Tảng | Kiến trúc Đa Tác tử Cân bằng Tự chủ + Thác dự phòng 3-4 tầng tự phục hồi | **0.952** | **0.968** |
