# 🛡️ Agent 4: Fact-Checker NLI Verifier (`training_lab/agent4_fact_checker/`)

Thư mục huấn luyện độc lập cho **Agent 4 (Fact-Checker)** sử dụng mô hình **Cross-Encoder NLI 3 nhãn** (`SUPPORT`, `CONTRADICT`, `NOT_ENOUGH_INFO`).

## 📂 Cấu Trúc Thư Mục
- [`Train_Agent4_FactChecker_T4x2.ipynb`](Train_Agent4_FactChecker_T4x2.ipynb): Notebook chạy huấn luyện tự động Agent 4 trên Kaggle T4x2 / Colab (~10 phút).
- [`train_agent4_t4x2.py`](train_agent4_t4x2.py): Script huấn luyện Cross-Encoder Sequence Classification với Cosine LR.
- [`dataset/nli_verifier.jsonl`](dataset/nli_verifier.jsonl): `762` cặp `(premise, hypothesis, label)` bao gồm các mẫu gài bẫy số liệu/năm tháng.
- [`weights/`](weights/): Nơi lưu trữ `model.safetensors` chuẩn `float16` sau khi huấn luyện.
