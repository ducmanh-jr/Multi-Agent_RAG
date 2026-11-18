# 🧭 Kiến Trúc Huấn Luyện Agent 1: Retrieval Router & Query Planner

> **Thư Mục Mã Nguồn**: `training_lab/agent1_retrieval/`  
> **Mô Hình Nền**: `Qwen/Qwen2.5-7B-Instruct`  
> **Kỹ Thuật PEFT**: DoRA (`use_dora=True`) + rsLoRA (`use_rslora=True`, $r=64, \alpha=128$)  
> **Tối Ưu Phần Cứng**: `attn_implementation="sdpa"` | Dual T4 `14.5GiB/GPU`  
> **Đầu Ra Trọng Số**: FP16 `adapter_model.safetensors` (77.05 MB — sẵn trên GitHub)  

---

## 1. Sơ Đồ Quy Trình Huấn Luyện (Mermaid Flowchart)

```mermaid
flowchart TD
    D["Tập Dữ Liệu Huấn Luyện<br/>dataset/router_sft.jsonl (1,500 mẫu ChatML)"]
    V["Kiểm Định Schema Nghiêm Ngặt<br/>Bắt buộc có action, rewritten_queries, alpha, reason"]
    SP["Phân Tách Dữ Liệu 90/10<br/>Train: 1,350 mẫu | Val: 150 mẫu"]
    
    subgraph DUAL_T4 ["Phân Bổ VRAM Kaggle Dual T4 (2 x 16GB)"]
        M["Qwen/Qwen2.5-7B-Instruct<br/>4-bit NF4 Quantization + SDPA Attention"]
        PEFT["DoRA + rsLoRA Adapters (r=64, alpha=128)<br/>Target: q, k, v, o, gate, up, down_proj"]
        M --> PEFT
    end

    TR["SFTTrainer (TRL Engine)<br/>NEFTune alpha=5.0 | Paged AdamW 8-bit | Cosine LR 1.5e-4"]
    FP16["Chuyển Đổi Trọng Số Sang FP16 (< 80MB)<br/>adapter_model.safetensors (77.05 MB)"]
    SYNC["Lưu Trữ GitHub (<100MB) & Google Drive Tự Động<br/>Folder ID: 1upmXzTIlxpcCVGU8Vp9ZGV3xNa1yu-Q6"]

    D --> V
    V --> SP
    SP --> TR
    DUAL_T4 --> TR
    TR --> FP16
    FP16 --> SYNC

    classDef purple fill:#f5f3ff,stroke:#a78bfa,stroke-width:1.5px,color:#2e1065;
    classDef input fill:#ede9fe,stroke:#8b5cf6,stroke-width:2px,color:#3b0764;
    classDef green fill:#f0fdf4,stroke:#4ade80,stroke-width:2px,color:#14532d;

    class D input;
    class V,SP,M,PEFT,TR purple;
    class FP16,SYNC green;
```

---

## 2. Cấu Trúc Gói Mã Nguồn `src/` Trong `training_lab/agent1_retrieval/`

| Tệp Mã Nguồn | Vai Trò Kỹ Thuật | Điểm Nổi Bật SOTA |
| :--- | :--- | :--- |
| `src/config.py` | Quản lý siêu tham số huấn luyện | Thiết lập DoRA + rsLoRA $r=64$, SDPA Attention và trần VRAM `14.5GiB/GPU`. |
| `src/data_loader.py` | Nạp và tiền xử lý dữ liệu | Kiểm định JSON Schema đầu ra, mã hóa theo Qwen2.5 ChatML template. |
| `src/model_builder.py` | Khởi tạo mô hình & PEFT | Lượng tử hóa 4-bit NF4 BitsAndBytes, cân bằng tải `device_map="balanced"`. |
| `src/trainer.py` | Vòng lặp huấn luyện SFT | Tích hợp NEFTune noise $\alpha=5.0$, tối ưu hóa Paged AdamW 8-bit và xuất FP16. |
| `src/evaluator.py` | Đánh giá sau huấn luyện | Đo lường tỷ lệ tuân thủ JSON Schema (JSON Compliance Rate) trên tập truy vấn kiểm thử. |
