# Bộ Công Cụ Fine-Tuning Thế Hệ 2 (`Kaggle FN v2` — SFT + DPO Multi-LoRA + NLI)

Thư mục này cung cấp toàn bộ pipeline chuẩn nghiên cứu để bạn huấn luyện (Fine-tune) bộ trọng số chuyên biệt mạnh nhất cho chế độ **`Kaggle FN`** trên **Kaggle Dual Tesla T4 (`2 × 16 GB VRAM`)**.

---

## 1. Kiến Trúc 5 Chuyên Gia (`Multi-LoRA + NLI Specialist Suite`)

| Tên Adapter Chuẩn | Tên Tương Thích Ngược | Base Model | Phương Pháp Huấn Luyện | Nhiệm Vụ Trong Hệ Thống Agentic RAG |
| :--- | :--- | :--- | :--- | :--- |
| **`router`** | `agent1_router`, `fn_router_planner` | `Qwen/Qwen2.5-7B-Instruct` | **QLoRA SFT** (`r=32, alpha=64`) | Phân tích Intent, Độ phức tạp (`LOW/MEDIUM/HIGH`), tách `sub_queries` đa bước và chọn `alpha`, `top_k`. |
| **`generator`** | `generator_qlora`, `fn_generator` | `Qwen/Qwen2.5-7B-Instruct` | **QLoRA SFT + DPO** (`beta=0.1`) | Tổng hợp câu trả lời học thuật bám sát Context, gắn trích dẫn `[1], [2]` và trích xuất `evidence_spans` nguyên văn. |
| **`critic`** | `critic_qlora`, `fn_critic` | `Qwen/Qwen2.5-7B-Instruct` | **QLoRA SFT với Hard Negatives** | Phát hiện mệnh đề bịa đặt (`Unsupported_claims`), ý bị sót (`Missing_key_points`) và ra quyết định `approve`, `rewrite`, hoặc `request_more_context`. |
| **`agent4_nli`** | `fn_nli_verifier` | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | **Full Fine-Tuning (Cross-Entropy + Focal Loss)** | Thẩm định nhanh từng cặp `(Evidence, Claim)` song ngữ Việt - Anh với tốc độ > 100 claims/giây. |

---

## 2. Quy Trình 3 Bước Để Bạn Chạy Fine-Tune Trên Kaggle

1. **Bước 1 — Sinh & Kiểm định bộ dữ liệu SFT + DPO (`01_generate_sft_dpo_dataset.py`):**
   ```bash
   python training_lab/kaggle_fn_v2/01_generate_sft_dpo_dataset.py --output_dir training_lab/kaggle_fn_v2/datasets
   ```
   Script này tự động trích xuất các đoạn văn từ kho tài liệu PDF (`main/data/corpus.jsonl`), sinh mẫu chuẩn ShareGPT/ChatML cho `router`, `generator`, `critic` và tạo các cặp `(chosen, rejected)` chứa **Hard Negatives** (sửa lệch số liệu 5–10%, đảo năm, thêm suy diễn ngoài văn bản) cho giai đoạn **DPO** và **Critic**.

2. **Bước 2 — Chạy Huấn Luyện Trên Kaggle GPU T4 × 2 (`Kaggle_MultiLoRA_SFT_DPO_Master.ipynb` hoặc `02_train_multilora_sft_dpo.py`):**
   - Tải file [Kaggle_MultiLoRA_SFT_DPO_Master.ipynb](file:///c:/Users/Admin/.ng/.m/hoc-may-nang-cao/.Multi-Agent_RAG/training_lab/kaggle_fn_v2/Kaggle_MultiLoRA_SFT_DPO_Master.ipynb) lên Kaggle.
   - Chọn **Accelerator: GPU T4 x2**.
   - Nhấn **Run All** để huấn luyện lần lượt `router`, `generator` (SFT + DPO), `critic`, và `agent4_nli`, sau đó tự động đóng gói thành `kaggle_fn_v2_weights.zip`.

3. **Bước 3 — Triển Khai Lên Worker (`kaggle/kaggle_worker.ipynb`):**
   - Gắn Dataset chứa `kaggle_fn_v2_weights.zip` vào Notebook Worker [kaggle_worker.ipynb](file:///c:/Users/Admin/.ng/.m/hoc-may-nang-cao/.Multi-Agent_RAG/kaggle/kaggle_worker.ipynb), hệ thống sẽ tự động nạp cả 3 QLoRA Adapters (`router`, `generator`, `critic`) + `agent4_nli` lên 2 GPU T4.
