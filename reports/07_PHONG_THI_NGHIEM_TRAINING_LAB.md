# 🔬 Báo Cáo Kiến Trúc Phòng Thí Nghiệm Huấn Luyện 5 Agent (TRAINING_LAB/)

> **Thư Mục Phòng Thí Nghiệm**: `training_lab/`  
> **Cấu Hình Mục Tiêu Tối Đa**: Kaggle Dual Tesla T4 ($2 \times 16\text{GB} = 32\text{GB}$ VRAM, Turing Architecture `sm_75`)  
> **Các Kỹ Thuật SOTA 2025/2026**: DoRA + rsLoRA + SDPA + RPO/DPO + Multi-Class Focal Loss + R-Drop KL Regularization  
> **Đồng Bộ Đám Mây**: Google Drive Tự Động (`1upmXzTIlxpcCVGU8Vp9ZGV3xNa1yu-Q6`)  

---

## 1. Sơ Đồ Kiến Trúc Huấn Luyện 5 Agent Độc Lập (Mermaid Flowchart)

```mermaid
flowchart TD
    classDef masterNode fill:#ede9fe,stroke:#8b5cf6,stroke-width:2px,color:#3b0764,font-weight:bold,rx:6,ry:6;
    classDef agentBox fill:#f5f3ff,stroke:#a78bfa,stroke-width:1.5px,color:#2e1065,rx:4,ry:4;
    classDef sotaNode fill:#fdf4ff,stroke:#d946ef,stroke-width:1.5px,color:#701a75,rx:4,ry:4;
    classDef driveNode fill:#f0fdf4,stroke:#22c55e,stroke-width:2px,color:#14532d,font-weight:bold,rx:6,ry:6;

    MASTER["Train_Master_T4x2.ipynb<br/>Notebook Điều Phối Huấn Luyện Tuần Tự Cả 5 Agent (Giải phóng 100% VRAM giữa các Agent)"]:::masterNode

    subgraph A1 ["training_lab/agent1_retrieval/"]
        A1_M["Qwen2.5-7B + DoRA + rsLoRA r=64<br/>router_sft.jsonl (1,500 mẫu)"]:::agentBox
    end

    subgraph A2 ["training_lab/agent2_reader/"]
        A2_M["LayoutLMv3-base (11-Class DocLayNet)<br/>Multi-Class Focal Loss (gamma=2.0)"]:::agentBox
    end

    subgraph A3 ["training_lab/agent3_summarizer/"]
        A3_M["Stage 1: SFT (1,500 mẫu) -> Stage 2: RPO/DPO (1,000 cặp)<br/>+ Critic SFT Hard-Negatives (1,500 mẫu)"]:::agentBox
    end

    subgraph A4 ["training_lab/agent4_fact_checker/"]
        A4_M["BGE-Reranker-v2-M3 Cross-Encoder 3-Class NLI<br/>Weighted Focal Loss + R-Drop KL + Temperature ECE"]:::agentBox
    end

    subgraph A5 ["training_lab/agent5_writer/"]
        A5_M["Qwen2.5-7B Two-Stage SFT + RPO Citation Alignment<br/>writer_sft.jsonl (1,500) + writer_dpo.jsonl (1,000)"]:::agentBox
    end

    SOTA["Chuẩn Xuất Trọng Số FP16 Thuần Nhất (< 80MB)<br/>adapter_model.safetensors lưu trực tiếp lên GitHub không vượt ngưỡng 100MB"]:::sotaNode
    DRIVE["ĐỒNG BỘ TỰ ĐỘNG VỀ GOOGLE DRIVE<br/>Folder ID: 1upmXzTIlxpcCVGU8Vp9ZGV3xNa1yu-Q6<br/>MultiAgent_RAG_Weights_All_5_Agents.zip"]:::driveNode

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

### 3.1. DoRA kết hợp rsLoRA (Rank-Stabilized Weight-Decomposed LoRA)
* **DoRA**: Tách ma trận trọng số $W$ thành biên độ $m$ và hướng $V$:
  $$W = m \frac{V + \Delta V}{\|V + \Delta V\|_F}, \quad \Delta V = B A$$
* **rsLoRA**: Thay thế hệ số co dãn $\frac{\alpha}{r}$ bằng $\frac{\alpha}{\sqrt{r}}$. Khi thiết lập $r=64, \alpha=128$, hệ số ổn định giúp các ma trận hạng cao hội tụ sâu hơn mà không làm bão hòa gradient.

### 3.2. Attention Rút Gọn Bộ Nhớ SDPA (Scaled Dot-Product Attention)
Khai thác nhân phần cứng nguyên bản của PyTorch 2.x tối ưu cho kiến trúc Turing `sm_75` của Tesla T4, cho phép tăng độ dài ngữ cảnh tối đa từ 2048 lên 3072 tokens trong cùng mức tiêu thụ VRAM.

### 3.3. Tối Ưu Tỷ Lệ Ưu Tiên RPO (Regularized Preference Optimization) Trong DPO
Tại Agent 3 và Agent 5, hàm mất mát DPO được bổ sung thành phần chính quy hóa NLL trên mẫu được chọn (`chosen`):
$$\mathcal{L}_{\text{RPO}} = \mathcal{L}_{\text{DPO}}(\pi_\theta;\; y_w, y_l) + \alpha_{\text{rpo}} \cdot \mathcal{L}_{\text{NLL}}(\pi_\theta;\; y_w)$$
với $\alpha_{\text{rpo}} = 0.1$, ngăn chặn hiện tượng mô hình quên cú pháp trích dẫn `[DocX]` chuẩn mực.

### 3.4. Hàm Mất Mát Focal Loss kết hợp R-Drop Cho Bộ Phân Loại (Agent 2 & 4)
* **Multi-Class Focal Loss (Agent 2 & 4)**:
  $$\mathcal{L}_{\text{focal}} = - (1 - p_t)^\gamma \log(p_t), \quad \gamma = 2.0$$
  Tự động hạ thấp trọng số của các mẫu văn bản dễ và tập trung vào các nhãn cấu trúc phức tạp (`Formula`, `Table`, `CONTRADICT`).
* **R-Drop KL Consistency Regularization (Agent 4)**: Ép phân phối xác suất giữa hai lần forward pass độc lập qua mạng có Dropout phải tiệm cận nhau qua độ đo đối xứng Kullback-Leibler:
  $$\mathcal{L}_{\text{R-Drop}} = \mathcal{L}_{\text{CE}} + \frac{\beta}{2} \left[ \mathcal{D}_{\text{KL}}(P_1 \parallel P_2) + \mathcal{D}_{\text{KL}}(P_2 \parallel P_1) \right]$$
