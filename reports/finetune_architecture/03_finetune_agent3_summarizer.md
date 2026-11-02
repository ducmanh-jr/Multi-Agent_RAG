# ✍️ Kiến Trúc Huấn Luyện Agent 3: Academic Summarizer (Generator & Critic)

> **Thư Mục Mã Nguồn**: `training_lab/agent3_summarizer/`  
> **Mô Hình Nền**: `Qwen/Qwen2.5-7B-Instruct`  
> **Generator PEFT**: Two-Stage SFT + RPO/DPO ($r=64, \alpha=128$, `rpo_alpha=0.1`)  
> **Critic PEFT**: Adversarial Hard-Negatives SFT ($r=32, \alpha=64$)  
> **Đầu Ra Trọng Số**: `generator_qlora` (77.05 MB) & `critic_qlora` (19.28 MB)  

---

## 1. Sơ Đồ Quy Trình Huấn Luyện 2 Giai Đoạn (Mermaid Flowchart)

```mermaid
flowchart TD
    subgraph STAGE1 ["Giai Đoạn 1: Huấn Luyện SFT Cơ Sở (SFTTrainer)"]
        SFT_D["dataset/generator_sft.jsonl (1,500 mẫu)<br/>Học sinh bản thảo học thuật kèm trích dẫn"]
        SFT_T["SFT Training (3 Epochs)<br/>DoRA + rsLoRA r=64 | NEFTune alpha=5.0"]
        SFT_D --> SFT_T
    end

    subgraph STAGE2 ["Giai Đoạn 2: Căn Chỉnh Ưu Tiên RPO/DPO (DPOTrainer)"]
        DPO_D["dataset/generator_dpo.jsonl (1,000 cặp)<br/>Chosen (trích dẫn chuẩn) vs Rejected (bịa đặt)"]
        DPO_T["RPO/DPO Training (2 Epochs)<br/>beta=0.1 | rpo_alpha=0.1 (chính quy hóa NLL)"]
        DPO_D --> DPO_T
    end

    subgraph CRITIC_FLOW ["Huấn Luyện Tác Tử Phản Biện Độc Lập (Critic SFT)"]
        C_D["dataset/critic_sft.jsonl (1,500 mẫu)<br/>Hard-Negatives bắt lỗi suy diễn ngoài ngữ cảnh"]
        C_T["Critic SFT Training (3 Epochs)<br/>DoRA r=32, alpha=64"]
        C_D --> C_T
    end

    EXP["Xuất 2 Adapter FP16 Riêng Biệt Vào weights/<br/>generator_qlora (77.05 MB) + critic_qlora (19.28 MB)"]
    SYNC["Tự Động Đẩy Lên GitHub & Google Drive"]

    STAGE1 --> STAGE2
    STAGE2 --> EXP
    CRITIC_FLOW --> EXP
    EXP --> SYNC

    classDef purple fill:#f5f3ff,stroke:#a78bfa,stroke-width:1.5px,color:#2e1065;
    classDef input fill:#ede9fe,stroke:#8b5cf6,stroke-width:2px,color:#3b0764;
    classDef green fill:#f0fdf4,stroke:#4ade80,stroke-width:2px,color:#14532d;

    class SFT_D,DPO_D,C_D input;
    class SFT_T,DPO_T,C_T purple;
    class EXP,SYNC green;
```

---

## 2. Cấu Trúc Gói Mã Nguồn `src/` Trong `training_lab/agent3_summarizer/`

| Tệp Mã Nguồn | Vai Trò Kỹ Thuật | Điểm Nổi Bật SOTA |
| :--- | :--- | :--- |
| `src/config.py` | Cấu hình Generator và Critic | Cấu hình riêng biệt 2 mức rank ($r=64$ cho Generator, $r=32$ cho Critic). |
| `src/data_loader.py` | Nạp dữ liệu SFT và cặp DPO | Nạp ChatML JSONL và định dạng cặp `(prompt, chosen, rejected)`. |
| `src/model_builder.py` | Khởi tạo mô hình Qwen2.5-7B | Lượng tử hóa 4-bit NF4, SDPA Attention và gắn DoRA. |
| `src/trainer.py` | Điều phối SFT $\rightarrow$ RPO/DPO | Kết hợp DPOTrainer với hệ số phạt NLL `rpo_alpha=0.1` bảo toàn cấu trúc trích dẫn. |
| `src/evaluator.py` | Kiểm tra kích thước adapter | Xác nhận tính toàn vẹn và dung lượng FP16 của 2 adapter. |
