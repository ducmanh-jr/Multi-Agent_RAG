# 🧬 Phòng Thí Nghiệm Huấn Luyện Multi-LoRA & NLI (`training_lab/`)

`training_lab/` là thư mục chuyên trách việc **sinh dữ liệu huấn luyện (SFT + DPO + Hard Negatives)** và **tinh chỉnh các bộ trọng số QLoRA / Cross-Encoder** cho hệ thống Multi-Agent RAG.

Đặc biệt, thư mục này đã được tích hợp sẵn **Notebook tự động hóa hoàn toàn (`GitHub Clone + Google Drive Sync 2 chiều`)** để bạn không bao giờ phải tải hay upload dữ liệu/trọng số thủ công.

---

## 🗂️ Cấu Trúc Thư Mục `training_lab/`

| Thư mục / Tệp | Chức năng chi tiết |
| :--- | :--- |
| **[`Train_Master_GitHub_Drive.ipynb`](Train_Master_GitHub_Drive.ipynb)** | **Notebook Huấn Luyện Chính (1-Click)**: Tự động `git clone` dự án từ GitHub, kết nối Google Drive (`drive.mount` hoặc `gdown`), chạy huấn luyện SFT + DPO cho cả 3 LoRA Adapters + NLI Verifier và tự động nén + sao lưu `weights_bundle_v2.zip` lên Google Drive. |
| **[`kaggle_fn_v2/`](kaggle_fn_v2/)** | **Pipeline Huấn Luyện Thế Hệ Mới (v2)**:<br/>• `01_generate_sft_dpo_dataset.py`: Trích xuất 2,667 mẫu huấn luyện từ kho PDF.<br/>• `02_train_multilora_sft_dpo.py`: Huấn luyện QLoRA 4-bit NF4 (`r=32, alpha=64, DoRA`) với SFT + DPO ($\beta=0.1$).<br/>• `datasets/`: Chứa 5 tập dữ liệu `.jsonl` đã tạo sẵn. |
| **[`weights_drive_bundle/`](weights_drive_bundle/)** | **Bộ Trọng Số QLoRA Chuẩn `float16` (Sẵn Sàng Sử Dụng)**:<br/>• `agent1_router/` (`77 MB`): Adapter phân tích ý định & tách truy vấn.<br/>• `generator_qlora/final/` (`77 MB`): Adapter tổng hợp câu trả lời có trích dẫn.<br/>• `critic_qlora/final/` (`19 MB`): Adapter tự phản biện phát hiện ảo giác.<br/>• `layoutlmv3_doclaynet_model/`: Cấu hình nhận diện bố cục PDF. |
| `agent1_retrieval/` .. `agent4_factchecker/` | Lưu trữ các kịch bản huấn luyện riêng lẻ của từng Agent ở phiên bản v1. |

---

## 🚀 Quy Trình Huấn Luyện Tự Động (Không Cần Tải Thủ Công)

1. Mở trực tiếp file **[`Train_Master_GitHub_Drive.ipynb`](Train_Master_GitHub_Drive.ipynb)** trên **Google Colab** (khuyên dùng nếu muốn tự động lưu thẳng vào Google Drive) hoặc **Kaggle GPU (T4 x2 / P100)**.
2. Bấm **Run All (Chạy tất cả các ô)**:
   - **Ô 1**: Tự động `git clone https://github.com/ducmanh-jr/Multi-Agent_RAG.git` để lấy toàn bộ code và 2,667 mẫu dữ liệu trong `training_lab/kaggle_fn_v2/datasets/`.
   - **Ô 2**: Tự động mount Google Drive (`/content/drive/MyDrive/MultiAgent_RAG_Weights`) hoặc tải checkpoint qua `gdown`.
   - **Ô 3 & 4**: Huấn luyện tuần tự `router` (SFT) $ightarrow$ `generator` (SFT + DPO $\beta=0.1$) $ightarrow$ `critic` (Hard-Negative SFT) $ightarrow$ `fact_checker_nli`.
   - **Ô 5**: Tự động đóng gói `weights_bundle_v2.zip` và chép thẳng vào Google Drive của bạn.
