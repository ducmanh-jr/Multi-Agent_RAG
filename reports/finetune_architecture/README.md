# 🧪 Trung Tâm Kiến Trúc Huấn Luyện 5 Agent (Fine-Tuning Architecture)

Thư mục này dành riêng cho việc phân tích chuyên sâu kiến trúc fine-tuning, thuật toán toán học SOTA và cấu trúc mô-đun `src/` của **từng Agent độc lập** trên phần cứng **Kaggle Dual Tesla T4 (2 x 16GB = 32GB VRAM)**:

| Tệp Báo Cáo Agent | Tác Tử Mục Tiêu | Mô Hình Nền & Kỹ Thuật PEFT | Kích Thước Trọng Số FP16 |
| :--- | :--- | :--- | :---: |
| [**`01_finetune_agent1_router.md`**](./01_finetune_agent1_router.md) | **Agent 1: Retrieval Router** | Qwen2.5-7B + DoRA + rsLoRA ($r=64, \alpha=128$) + SDPA | 77.05 MB |
| [**`02_finetune_agent2_reader.md`**](./02_finetune_agent2_reader.md) | **Agent 2: Document Reader** | LayoutLMv3-base (11 nhãn) + Multi-Class Focal Loss ($\gamma=2.0$) | ~480 MB |
| [**`03_finetune_agent3_summarizer.md`**](./03_finetune_agent3_summarizer.md) | **Agent 3: Summarizer** | Qwen2.5-7B Two-Stage SFT + RPO/DPO (Gen) & Hard-Negatives (Critic) | 77.05 MB + 19.28 MB |
| [**`04_finetune_agent4_factchecker.md`**](./04_finetune_agent4_factchecker.md) | **Agent 4: Fact-Checker** | BGE-Reranker-v2-M3 + Weighted Focal + R-Drop KL + ECE Calibration | ~1.1 GB |
| [**`05_finetune_agent5_writer.md`**](./05_finetune_agent5_writer.md) | **Agent 5: Academic Writer** | Qwen2.5-7B Two-Stage SFT + RPO Citation Alignment ($r=32$) | 19.28 MB |

---

## 🚀 Huấn Luyện Đồng Bộ Tự Động
Mọi kịch bản huấn luyện đều được điều phối tập trung qua [`training_lab/Train_Master_T4x2.ipynb`](../../training_lab/Train_Master_T4x2.ipynb) và tự động đồng bộ kết quả về Google Drive Folder: [`1upmXzTIlxpcCVGU8Vp9ZGV3xNa1yu-Q6`](https://drive.google.com/drive/folders/1upmXzTIlxpcCVGU8Vp9ZGV3xNa1yu-Q6?usp=sharing).
