# 🚀 Báo Cáo Kiến Trúc Tùy Chọn 4: KAGGLE_FN (Fine-Tuned Multi-LoRA Pipeline)

> **Mã Chế Độ**: `kaggle_fn`  
> **Thư Mục Thực Thi**: `main/modes/kaggle_fn/` & `kaggle/kaggle_server.py`  
> **Phần Cứng Mục Tiêu**: Kaggle Dual Tesla T4 ($2 \times 16\text{GB} = 32\text{GB}$ VRAM)  
> **Đột Phá Kỹ Thuật**: Kiến trúc Đơn Mô Hình Nền - Đa Nhánh Tác Tử (Single-Backbone Multi-LoRA Hot-Swapping)  
> **Độ Trễ Trung Bình**: $640.0\text{ ms}$ / truy vấn  

---

## 1. Sơ Đồ Kiến Trúc Hoán Đổi Nóng Multi-LoRA (Mermaid Flowchart)

```mermaid
flowchart TD
    REQ["Yêu Cầu Truy Vấn Đến Máy Chủ Kaggle (/api/...)"]

    subgraph GPU0 ["GPU 0: cuda:0 (Tesla T4 - 16GB VRAM)"]
        BASE["Mô Hình Nền Duy Nhất: Qwen/Qwen2.5-7B-Instruct<br/>(4-bit NF4 Quantization ~ 4.8GB VRAM)"]
        L1["Adapter 1: router (Agent 1)<br/>DoRA + rsLoRA r=64 (77.05 MB)"]
        L2["Adapter 2: generator (Agent 3)<br/>Two-Stage SFT + RPO/DPO (77.05 MB)"]
        L3["Adapter 3: critic (Agent 3)<br/>Hard-Negatives Adversarial (19.28 MB)"]
        L4["Adapter 4: writer (Agent 5)<br/>IEEE/Markdown Synthesis (19.28 MB)"]
        
        BASE -. "model.set_adapter('router')" .-> L1
        BASE -. "model.set_adapter('generator')" .-> L2
        BASE -. "model.set_adapter('critic')" .-> L3
        BASE -. "model.set_adapter('writer')" .-> L4
    end

    subgraph GPU1 ["GPU 1: cuda:1 (Tesla T4 - 16GB VRAM)"]
        M2["Agent 2: LayoutLMv3-base<br/>11-Class DocLayNet Parser (~480MB)"]
        M4["Agent 4: BGE-Reranker-v2-M3 Cross-Encoder NLI<br/>Focal Loss + R-Drop KL (~1.1GB)"]
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
    class M2,M4 green;
```

---

## 2. Cơ Chế Hoạt Động Kỹ Thuật & Đột Phá Bộ Nhớ

### 2.1. Nghịch Lý Bộ Nhớ VRAM & Giải Pháp Multi-LoRA Hot-Swapping
Nếu tải riêng biệt 4 mô hình 7B:
$$4 \times 15\text{GB} = 60\text{GB VRAM} \quad (\gg 32\text{GB của cụm Dual T4})$$
Giải pháp: Chỉ nạp **1 mô hình nền** `Qwen2.5-7B` (4-bit NF4 ~ $4.8\text{GB}$) trên GPU 0, gắn đồng thời 4 bộ trọng số DoRA/rsLoRA (`router`, `generator`, `critic`, `writer`) và chuyển mạch trong $0.2\text{ ms}$ qua lệnh `model.set_adapter(name)`.

### 2.2. Phân Bổ Tải Hai Bán Cầu GPU
* **GPU 0 (`cuda:0`)**: Toàn bộ tác vụ sinh ngôn ngữ tự nhiên (Qwen2.5-7B Multi-LoRA).
* **GPU 1 (`cuda:1`)**: Phân đoạn tài liệu `LayoutLMv3` (Agent 2) và kiểm chứng sự thật `Cross-Encoder NLI` (Agent 4).

---

## 3. Cấu Trúc Mã Nguồn & Ánh Xạ Trọng Số

| Thư Mục Trọng Số | Kích Thước (FP16) | Tác Tử Tương Ứng | Giao Thức Gọi Trên Kaggle Worker |
| :--- | :--- | :--- | :--- |
| `training_lab/agent1_retrieval/weights/` | 77.05 MB | Agent 1 (Router) | `POST /api/generate` (`adapter="router"`) |
| `training_lab/agent2_reader/weights/` | 480 MB | Agent 2 (Reader) | Xử lý nội bộ trên GPU 1 (`LayoutLMv3`) |
| `training_lab/agent3_summarizer/weights/generator_qlora/` | 77.05 MB | Agent 3 (Generator) | `POST /api/summarize/generator` |
| `training_lab/agent3_summarizer/weights/critic_qlora/` | 19.28 MB | Agent 3 (Critic) | `POST /api/summarize/critic` |
| `training_lab/agent4_fact_checker/weights/` | 1.1 GB | Agent 4 (Fact-Checker) | `POST /api/factcheck` (Cross-Encoder NLI) |
| `training_lab/agent5_writer/weights/` | 19.28 MB | Agent 5 (Writer) | `POST /api/generate` (`adapter="writer"`) |
