# ✍️ Agent 3: Summarizer (`Generator` SFT+DPO & `Critic` Hard-Negatives) (`training_lab/agent3_summarizer/`)

Thư mục huấn luyện độc lập cho **Agent 3 (Summarizer)** gồm 2 mô hình chuyên gia phối hợp trong vòng lặp tự phản biện nội bộ (Inner Generator-Critic Loop), tối ưu hóa cho **Kaggle Dual Tesla T4 (`2 × 16GB = 32GB VRAM`)**.

## 📂 Cấu Trúc Thư Mục
- [`Train_Agent3_Summarizer_T4x2.ipynb`](Train_Agent3_Summarizer_T4x2.ipynb): Notebook chạy huấn luyện tự động Agent 3 trên Kaggle T4x2 / Colab.
- [`train_agent3_t4x2.py`](train_agent3_t4x2.py): Script huấn luyện 2 chặng (`Generator SFT + DPO` & `Critic Hard-Negatives` với `r=64, max_seq_length=3072`).
- [`dataset/`](dataset/):
  - `generator_sft.jsonl` (`381` mẫu tổng hợp gắn trích dẫn `[1]`, `[2]`).
  - `generator_dpo.jsonl` (`381` cặp ưu tiên `chosen` vs `rejected`).
  - `critic_sft.jsonl` (`762` mẫu tự phản biện có gài bẫy số liệu).
- [`weights/`](weights/):
  - `generator_qlora/`: Trọng số QLoRA của Generator (`77 MB` chuẩn `float16`).
  - `critic_qlora/`: Trọng số QLoRA của Critic (`19 MB` chuẩn `float16`).
