# 📝 Kiến Trúc Huấn Luyện Agent 5: Academic Report Writer

> **Thư Mục Mã Nguồn**: `training_lab/agent5_writer/`  
> **Mô Hình Nền**: `Qwen/Qwen2.5-7B-Instruct`  
> **Kỹ Thuật PEFT**: DoRA + rsLoRA ($r=32, \alpha=64$, `attn_implementation="sdpa"`)  
> **Huấn Luyện 2 Giai Đoạn**: Stage 1 SFT (1,500 mẫu) $\rightarrow$ Stage 2 RPO/DPO Citation Alignment (1,000 cặp)  
> **Đầu Ra Trọng Số**: FP16 `adapter_model.safetensors` (19.28 MB — tích hợp làm Adapter thứ 4 trên GPU 0)  

---

## 1. Sơ Đồ Quy Trình Huấn Luyện (Mermaid Flowchart)

```mermaid
flowchart TD
    D1["dataset/writer_sft.jsonl (1,500 mẫu)<br/>Cấu trúc IEEE: Executive Summary, Math LaTeX, Tables, [DocX]"]
    D2["dataset/writer_dpo.jsonl (1,000 cặp)<br/>Chosen (trích dẫn chuẩn) vs Rejected (thiếu bảng/trích dẫn)"]

    subgraph TRAIN_STAGES ["Quy Trình Huấn Luyện 2 Giai Đoạn"]
        S1["Stage 1: SFT (3 Epochs)<br/>DoRA + rsLoRA r=32 | NEFTune alpha=5.0"]
        S2["Stage 2: RPO/DPO (2 Epochs)<br/>beta=0.1 | rpo_alpha=0.1 (ngăn quên cú pháp trích dẫn)"]
        S1 --> S2
    end

    EXP["Xuất Trọng Số FP16 adapter_model.safetensors (19.28 MB)<br/>Hoán đổi nóng trực tiếp làm Adapter thứ 4 trên Qwen2.5-7B"]
    DRIVE["Đồng Bộ Trực Tiếp Lên Google Drive & GitHub"]

    D1 --> S1
    D2 --> S2
    S2 --> EXP
    EXP --> DRIVE

    classDef purple fill:#f5f3ff,stroke:#a78bfa,stroke-width:1.5px,color:#2e1065;
    classDef input fill:#ede9fe,stroke:#8b5cf6,stroke-width:2px,color:#3b0764;
    classDef green fill:#f0fdf4,stroke:#4ade80,stroke-width:2px,color:#14532d;

    class D1,D2 input;
    class S1,S2 purple;
    class EXP,DRIVE green;
```

---

## 2. Cấu Trúc Gói Mã Nguồn `src/` Trong `training_lab/agent5_writer/`

| Tệp Mã Nguồn | Vai Trò Kỹ Thuật | Điểm Nổi Bật SOTA |
| :--- | :--- | :--- |
| `src/config.py` | Cấu hình siêu tham số Writer | Thiết lập DoRA + rsLoRA $r=32$, SDPA Attention và $rpo\_alpha=0.1$. |
| `src/data_loader.py` | Nạp dữ liệu báo cáo học thuật | Nạp mẫu SFT cấu trúc và cặp đối chiếu DPO ưu tiên trích dẫn nội văn. |
| `src/model_builder.py` | Khởi tạo mô hình Qwen2.5-7B | Lượng tử hóa 4-bit NF4 cân bằng tải trên cụm Dual T4. |
| `src/trainer.py` | Huấn luyện SFT và RPO/DPO | Điều phối chuỗi 2 giai đoạn và xuất file trọng số FP16 siêu gọn 19.28 MB. |
| `src/evaluator.py` | Thẩm định cấu trúc báo cáo | Kiểm tra tự động 4 yếu tố: tiêu đề `###`, bảng Markdown `|`, công thức LaTeX `$$` và thẻ `[DocX]`. |
