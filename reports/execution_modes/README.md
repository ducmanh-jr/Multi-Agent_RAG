# 🎛️ Báo Cáo Chi Tiết 6 Tùy Chọn Thực Thi (Execution Modes)

Thư mục này phân tích chi tiết nguyên lý toán học, sơ đồ luồng dữ liệu Mermaid và ánh xạ mã nguồn của cả **6 Chế độ Thực thi**:

| Tệp Báo Cáo | Tên Chế Độ | Phần Cứng Mục Tiêu | Đặc Điểm Cốt Lõi | Overall F1 | Faithfulness |
| :--- | :--- | :--- | :--- | :---: | :---: |
| [**`01_cpu_min.md`**](./01_cpu_min.md) | `cpu_min` | RAM < 1GB, 0 GPU | BM25 Okapi + TF-IDF SVD/LSA + Extractive Graph | 0.785 | 0.810 |
| [**`02_cpu_max.md`**](./02_cpu_max.md) | `cpu_max` | Workstation CPU | Multilingual-E5 + BM25 RRF + MiniLM + Ollama | 0.842 | 0.865 |
| [**`03_gemini_cloud.md`**](./03_gemini_cloud.md) | `gemini` | Cloud API | Gemini 3 Flash JSON Schema + Multi-Persona Judge | 0.896 | 0.912 |
| [**`04_kaggle_fn_multilora.md`**](./04_kaggle_fn_multilora.md) | `kaggle_fn` | Dual T4 (32GB) | Qwen2.5-7B 4-Adapter Multi-LoRA (DoRA r=64) | 0.918 | 0.938 |
| [**`05_kaggle_max_ensemble.md`**](./05_kaggle_max_ensemble.md) | `kaggle_max` | Dual T4 (32GB) | Async Gather + Hội đồng 3 Giám khảo Đồng thuận 2/3 | 0.945 | 0.964 |
| [**`06_hybrid_adaptive.md`**](./06_hybrid_adaptive.md) | `hybrid` | Hỗn hợp Đa tầng | Per-Agent Routing + Cầu dao tự động 3 trạng thái | **0.952** | **0.968** |
