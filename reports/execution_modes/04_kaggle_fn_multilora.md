# 🚀 Tùy Chọn Tinh Chỉnh GPU: GPU_FT & GPU_FINETUNE (Specialized Multi-Model Pipelines)

> **Mã Chế Độ Phục Vụ**: `gpu_ft` & `gpu_finetune`  
> **Thư Mục Mã Nguồn**: [`main/modes/gpu_ft/`](../../main/modes/gpu_ft/), [`main/modes/gpu_finetune/`](../../main/modes/gpu_finetune/) & [`kaggle/kaggle_finetune_server.py`](../../kaggle/kaggle_finetune_server.py)  
> **Phần Cứng Mục Tiêu**: Kaggle Dual Tesla T4 (2 x 16GB = 32GB VRAM)  
> **Đột Phá Kỹ Thuật**: Single-Backbone Multi-LoRA Hot-Swapping trong 0.2 ms kết hợp phân luồng 2 bán cầu GPU độc lập  
> **Độ Trễ Trung Bình**: ~5.75s – 5.80s / truy vấn  
> **Chỉ Số Đánh Giá**: Overall F1 = **0.958** | Faithfulness = **0.975** | Context Recall = **0.940**  

---

## 1. Sơ Đồ Khối Thực Thi (Mermaid Architecture)

```mermaid
flowchart TD
    REQ["Yêu Cầu Truy Vấn Đến Máy Chủ Kaggle Dual T4 (/api/...)"]

    subgraph GPU0 ["GPU 0: cuda:0 (Tesla T4 - 16GB VRAM)"]
        BASE["Mô Hình Nền Duy Nhất: Qwen/Qwen2.5-7B-Instruct<br/>(4-bit NF4 Quantization ~ 4.8GB VRAM)"]
        L1["Adapter 1: router (Agent 1)<br/>DoRA + rsLoRA r=64"]
        L2["Adapter 2: generator (Agent 3)<br/>Two-Stage SFT + RPO/DPO"]
        L3["Adapter 3: critic (Agent 3)<br/>Hard-Negatives Adversarial"]
        L4["Adapter 4: writer (Agent 5)<br/>DoRA Academic Writer (Token Acc 99.6%)"]
        
        BASE -. "model.set_adapter('router')" .-> L1
        BASE -. "model.set_adapter('generator')" .-> L2
        BASE -. "model.set_adapter('critic')" .-> L3
        BASE -. "model.set_adapter('writer')" .-> L4
    end

    subgraph GPU1 ["GPU 1: cuda:1 (Tesla T4 - 16GB VRAM)"]
        M2["Agent 2: LayoutLMv3-base<br/>11-Class DocLayNet Parser (~480MB)"]
        M4["Agent 4: Calibrated BGE NLI Verifier<br/>Focal Loss + R-Drop KL (Macro F1 = 1.0000)"]
        M6["Dense Retriever: BGE-M3 MoAI LoRA"]
        M7["Reranker: CoREB BGE-Reranker-Large LoRA"]
    end

    REQ --> GPU0
    REQ --> GPU1

    classDef purple fill:#f5f3ff,stroke:#a78bfa,stroke-width:1.5px,color:#2e1065;
    classDef base fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#ffffff;
    classDef green fill:#f0fdf4,stroke:#22c55e,stroke-width:1.5px,color:#14532d;
    classDef input fill:#ede9fe,stroke:#8b5cf6,stroke-width:2px,color:#3b0764;

    class REQ input;
    class BASE base;
    class L1,L2,L3,L4 purple;
    class M2,M4,M6,M7 green;
```

---

## 2. Nguyên Lý Kỹ Thuật & Công Thức Toán Học

### 2.1. Đột Phá Hoán Đổi Nóng Multi-LoRA (0.2 ms)
Thay vì nạp nhiều mô hình 7B riêng tốn $4 \times 15 = 60\text{GB}$ VRAM, hệ thống chỉ nạp 1 bản 4-bit NF4 (~4.8GB VRAM) trên GPU 0 và hoán đổi con trỏ trọng số DoRA qua hàm `model.set_adapter(name)`:
$$\Delta W_{\text{active}} = \begin{cases} \Delta W_{\text{router}}, & \text{khi định tuyến truy vấn} \\ \Delta W_{\text{generator}}, & \text{khi sinh tóm tắt} \\ \Delta W_{\text{critic}}, & \text{khi tự phản biện} \\ \Delta W_{\text{writer}}, & \text{khi biên tập báo cáo cuối} \end{cases}$$

### 2.2. Phân Bổ Tải Hai Bán Cầu GPU Độc Lập
* **GPU 0 (`cuda:0`)**: Chuyên trách toàn bộ các tác vụ sinh ngôn ngữ tự nhiên của Qwen2.5-7B qua cơ chế Multi-LoRA.
* **GPU 1 (`cuda:1`)**: Phục vụ độc lập mô hình thị giác `LayoutLMv3` (Agent 2), kiểm chứng sự thật `Calibrated BGE NLI Head` (Agent 4), và bộ đôi tìm kiếm nhúng/tái xếp hạng (`BGE-M3` + `CoREB Reranker`), triệt tiêu hoàn toàn nghẽn cổ chai tài nguyên.

---

## 3. Bảng Ánh Xạ Trọng Số Đã Huấn Luyện

| Thư Mục Trọng Số | Thành Phần Kiến Trúc | Tác Tử Tương Ứng | Giao Thức Gọi Trên Máy Chủ Suy Luận |
| :--- | :--- | :--- | :--- |
| `training_lab/final_weights/01_agent1_router/` | Adapter LoRA Router ($r=64$) | Agent 1 (Router) | `POST /api/generate` (`adapter="router"`) |
| `training_lab/final_weights/02_agent2_reader/` | LayoutLMv3 Base DocLayNet | Agent 2 (Reader) | `POST /api/reader/layoutlmv3` |
| `training_lab/final_weights/03_agent3_summarizer/generator_qlora/` | QLoRA Academic Summarizer | Agent 3 (Generator) | `POST /api/summarize/generator` |
| `training_lab/final_weights/03_agent3_summarizer/critic_qlora/` | QLoRA Adversarial Critic | Agent 3 (Critic) | `POST /api/summarize/critic` |
| `training_lab/final_weights/04_agent4_fact_checker/` | Calibrated BGE NLI Verifier (F1=1.0) | Agent 4 (Fact-Checker) | `POST /api/factcheck/nli` |
| `training_lab/final_weights/05_agent5_writer/` | DoRA Academic Report Writer | Agent 5 (Writer) | `POST /api/generate` (`adapter="writer"`) |
| `training_lab/final_weights/06_embedding_bge_m3/` | Dense Retriever LoRA ($r=32$) | Retriever | `POST /api/retrieval/embed` |
| `training_lab/final_weights/07_reranker_bge_large/` | CoREB Cross-Encoder Reranker | Reranker | `POST /api/retrieval/rerank` |
