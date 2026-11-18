# 🛡️ Kiến Trúc Huấn Luyện Agent 4: Fact-Checker NLI Verifier

> **Thư Mục Mã Nguồn**: `training_lab/agent4_fact_checker/`  
> **Mô Hình Nền**: `BAAI/bge-reranker-v2-m3` / `allenai/scibert_scivocab_uncased`  
> **Thuật Toán Tổn Thất**: Class-Weighted Focal Loss ($\gamma=2.0$, CONTRADICT trọng số $1.35\times$) + R-Drop KL Consistency Regularization  
> **Hiệu Chuẩn Hậu Huấn Luyện**: Temperature Scaling Calibration (ECE Minimization)  
> **Tập Dữ Liệu**: `dataset/nli_verifier.jsonl` (3,000 cặp SciFact + VitaminC)  

---

## 1. Sơ Đồ Quy Trình Huấn Luyện (Mermaid Flowchart)

```mermaid
flowchart TD
    D["Tập Dữ Liệu Thẩm Định Khoa Học<br/>dataset/nli_verifier.jsonl (3,000 cặp claim/evidence)"]
    ENC["Mã Hóa Cặp Chuỗi Cross-Encoder<br/>Tokenizer(claim, evidence, max_length=512)"]
    SP["Chia 85/15 Train/Val Split"]

    subgraph TRAIN_FLOW ["WeightedCrossEncoderTrainer (Kaggle Dual T4)"]
        FOCAL["Class-Weighted Focal Loss (gamma=2.0)<br/>Phạt nặng gấp 1.35x nếu bỏ sót nhãn CONTRADICT"]
        RDROP["R-Drop Regularization<br/>Ép phân phối KL-Divergence giữa 2 lần forward pass"]
        FOCAL --> RDROP
    end

    ECE["Hiệu Chuẩn Nhiệt Độ Xác Suất (Temperature Scaling ECE)<br/>Tối ưu hóa nhiệt độ T trong [0.5, 2.5] giảm thiểu ECE"]
    OUT["Xuất Trọng Số NLI Verifier & eval_metrics.json<br/>Macro-F1 >= 0.918 | Calibrated Temp T=1.15"]
    DRIVE["Đồng Bộ Trực Tiếp Lên Google Drive"]

    D --> ENC
    ENC --> SP
    SP --> TRAIN_FLOW
    TRAIN_FLOW --> ECE
    ECE --> OUT
    OUT --> DRIVE

    classDef purple fill:#f5f3ff,stroke:#a78bfa,stroke-width:1.5px,color:#2e1065;
    classDef input fill:#ede9fe,stroke:#8b5cf6,stroke-width:2px,color:#3b0764;
    classDef green fill:#f0fdf4,stroke:#4ade80,stroke-width:2px,color:#14532d;

    class D input;
    class ENC,SP,FOCAL,RDROP,ECE purple;
    class OUT,DRIVE green;
```

---

## 2. Cấu Trúc Gói Mã Nguồn `src/` Trong `training_lab/agent4_fact_checker/`

| Tệp Mã Nguồn | Vai Trò Kỹ Thuật | Điểm Nổi Bật SOTA |
| :--- | :--- | :--- |
| `src/config.py` | Cấu hình NLI 3-Class | Ánh xạ nhãn `SUPPORT: 0`, `CONTRADICT: 1`, `NOT_ENOUGH_INFO: 2`. |
| `src/data_loader.py` | Tokenize cặp câu | Mã hóa cặp `(claim, evidence)` theo chuẩn Cross-Encoder Sequence Classification. |
| `src/model_builder.py` | Khởi tạo mô hình Cross-Encoder | Nạp `bge-reranker-v2-m3` cấu hình đầu ra phân loại 3 lớp. |
| `src/trainer.py` | Huấn luyện Focal + R-Drop | Cài đặt `WeightedCrossEncoderTrainer` kết hợp mất mát tiêu điểm và khoảng cách đối xứng KL. |
| `src/evaluator.py` | Tính toán chỉ số & ECE | Đánh giá Macro-F1, F1 từng lớp và tìm nhiệt độ tối ưu qua hàm `calibrate_temperature_ece()`. |
