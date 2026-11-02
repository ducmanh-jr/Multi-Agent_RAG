# 🔬 Báo Cáo Kiến Trúc Phòng Thí Nghiệm Huấn Luyện 5 Agent (TRAINING_LAB/)

> **Thư Mục Phòng Thí Nghiệm**: `training_lab/`  
> **Cấu Hình Mục Tiêu Tối Đa**: Kaggle Dual Tesla T4 ($2 \times 16\text{GB} = 32\text{GB}$ VRAM, Turing Architecture `sm_75`)  
> **Các Kỹ Thuật SOTA 2025/2026**: DoRA + rsLoRA + SDPA + RPO/DPO + Multi-Class Focal Loss + R-Drop KL Regularization  
> **Đồng Bộ Đám Mây**: Google Drive Tự Động (`1upmXzTIlxpcCVGU8Vp9ZGV3xNa1yu-Q6`)  

---

## 1. Sơ Đồ Kiến Trúc Huấn Luyện 5 Agent Độc Lập (Mermaid Flowchart)

```mermaid
flowchart TD
    MASTER["Train_Master_T4x2.ipynb<br/>Notebook Điều Phối Huấn Luyện Tuần Tự Cả 5 Agent (Giải phóng 100% VRAM giữa các Agent)"]

    subgraph A1 ["training_lab/agent1_retrieval/"]
        A1_M["Qwen2.5-7B + DoRA + rsLoRA r=64<br/>router_sft.jsonl (1,500 mẫu)"]
    end

    subgraph A2 ["training_lab/agent2_reader/"]
        A2_M["LayoutLMv3-base (11-Class DocLayNet)<br/>Multi-Class Focal Loss (gamma=2.0)"]
    end

    subgraph A3 ["training_lab/agent3_summarizer/"]
        A3_M["Stage 1: SFT (1,500 mẫu) -> Stage 2: RPO/DPO (1,000 cặp)<br/>+ Critic SFT Hard-Negatives (1,500 mẫu)"]
    end

    subgraph A4 ["training_lab/agent4_fact_checker/"]
        A4_M["BGE-Reranker-v2-M3 Cross-Encoder 3-Class NLI<br/>Weighted Focal Loss + R-Drop KL + Temperature ECE"]
    end

    subgraph A5 ["training_lab/agent5_writer/"]
        A5_M["Qwen2.5-7B Two-Stage SFT + RPO Citation Alignment<br/>writer_sft.jsonl (1,500) + writer_dpo.jsonl (1,000)"]
    end

    SOTA["Chuẩn Xuất Trọng Số FP16 Thuần Nhất (< 80MB)<br/>adapter_model.safetensors lưu trực tiếp lên GitHub không vượt ngưỡng 100MB"]
    DRIVE["ĐỒNG BỘ TỰ ĐỘNG VỀ GOOGLE DRIVE<br/>Folder ID: 1upmXzTIlxpcCVGU8Vp9ZGV3xNa1yu-Q6<br/>MultiAgent_RAG_Weights_All_5_Agents.zip"]

    MASTER --> A1
    MASTER --> A2
    MASTER --> A3
    MASTER --> A4
    MASTER --> A5

    A1 --> SOTA
    A2 --> SOTA
    A3 --> SOTA
    A4 --> SOTA
    A5 --> SOTA

    SOTA --> DRIVE

    classDef purple fill:#f5f3ff,stroke:#a78bfa,stroke-width:1.5px,color:#2e1065;
    classDef master fill:#ede9fe,stroke:#8b5cf6,stroke-width:2px,color:#3b0764;
    classDef sota fill:#fdf4ff,stroke:#d946ef,stroke-width:1.5px,color:#701a75;
    classDef drive fill:#f0fdf4,stroke:#22c55e,stroke-width:2px,color:#14532d;

    class MASTER master;
    class A1_M,A2_M,A3_M,A4_M,A5_M purple;
    class SOTA sota;
    class DRIVE drive;
```

---

## 2. Kiến Trúc Mô-Đun `src/` Chuẩn Mực Trong Từng Thư Mục Agent

Mỗi thư mục Agent được cấu trúc thành một gói mã nguồn hoàn chỉnh:
```text
training_lab/agentX/
├── src/
│   ├── __init__.py          # Khai báo interface của gói
│   ├── config.py            # Dataclass quản lý siêu tham số và phân bổ VRAM Dual T4 (14.5GiB/GPU)
│   ├── data_loader.py       # Nạp JSONL, kiểm định Schema, định dạng ChatML/BBox và chia 90/10 Train/Val
│   ├── model_builder.py     # Lượng tử hóa 4-bit NF4, SDPA Attention và gắn adapter DoRA/rsLoRA
│   ├── trainer.py           # Vòng lặp huấn luyện SOTA (RPO, Focal Loss, R-Drop) & ép kiểu FP16 (<80MB)
│   └── evaluator.py         # Kiểm định JSON compliance, Macro-F1 và hiệu chỉnh nhiệt độ xác suất
├── dataset/                 # Tệp dữ liệu huấn luyện JSONL
├── weights/                 # Trọng số FP16 đã huấn luyện sẵn sàng đẩy lên GitHub
├── train_agentX_t4x2.py     # CLI Entrypoint gọi vào package src.*
└── Train_AgentX_T4x2.ipynb  # Notebook độc lập cho người dùng muốn huấn luyện riêng lẻ từng Agent
```

---

## 3. Các Công Nghệ Huấn Luyện SOTA Khai Thác Kịch Trần Dual T4 (32GB VRAM)

1. **DoRA + rsLoRA**: Tách biên độ/hướng vector trọng số kết hợp hệ số ổn định $rac{lpha}{\sqrt{r}}$ cho phép đẩy hạng $r=64$ ổn định.
2. **SDPA Attention**: Khai thác nhân Scaled Dot-Product Attention của PyTorch 2.x cho Turing `sm_75`, mở rộng độ dài ngữ cảnh lên 3072 tokens.
3. **RPO (Regularized Preference Optimization)**: Bổ sung phạt NLL vào DPO với $lpha_{	ext{rpo}} = 0.1$, loại bỏ hiện tượng quên trích dẫn `[DocX]`.
4. **Multi-Class Focal Loss & R-Drop**: Tăng trọng số các nhãn khó (`Formula`, `Table`, `CONTRADICT`) và ép tính bất biến phân phối xác suất qua KL divergence.
