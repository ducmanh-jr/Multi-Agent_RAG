# 🧭 Agent 1: Retrieval Router & Supervisor Planner (`training_lab/agent1_retrieval/`)

Thư mục huấn luyện độc lập cho **Agent 1 (Retrieval Router)** tối ưu hóa trên phần cứng **Kaggle Dual Tesla T4 (`2 × 16GB = 32GB VRAM`)**.

## 📂 Cấu Trúc Thư Mục
- [`Train_Agent1_Router_T4x2.ipynb`](Train_Agent1_Router_T4x2.ipynb): Notebook chạy huấn luyện tự động Agent 1 trên Kaggle T4x2 / Colab.
- [`train_agent1_t4x2.py`](train_agent1_t4x2.py): Script huấn luyện QLoRA 4-bit NF4 + DoRA (`r=64, lora_alpha=128`, `NEFTune alpha=5.0`).
- [`dataset/router_sft.jsonl`](dataset/router_sft.jsonl): `381` mẫu phân tích ý định, độ phức tạp (`LOW`/`MEDIUM`/`HIGH`), `required_entities` và `sub_queries`.
- [`weights/`](weights/): Lưu trữ trọng số `adapter_model.safetensors` chuẩn `float16` (`77 MB`).
