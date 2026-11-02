# 💻 Báo Cáo Kiến Trúc Tùy Chọn 2: CPU_MAX (Heavy Local Workstation Pipeline)

> **Mã Chế Độ**: `cpu_max`  
> **Thư Mục Thực Thi**: `main/modes/cpu_max/`  
> **Tài Nguyên Yêu Cầu**: CPU Đa Nhân (Core i5/i7/Ultra) \| RAM $4\text{GB} - 8\text{GB}$ \| Mô hình Cục bộ  
> **Độ Trễ Trung Bình**: $412.0\text{ ms}$ / truy vấn  

---

## 1. Sơ Đồ Quy Trình Thực Thi `cpu_max` (Mermaid Flowchart)

```mermaid
flowchart TD
    Q["Câu hỏi người dùng (User Query)"]
    DENSE["Mô hình Nhúng Ngữ Nghĩa Cục bộ<br/>intfloat/multilingual-e5-small (main/models/)"]
    SPARSE["Chỉ mục Từ vựng Tần suất Cao<br/>BM25 Okapi Indexing"]
    RRF["Hợp Nhất Thứ Hạng Reciprocal Rank Fusion (RRF)<br/>k = 60 | alpha = 0.65"]
    RERANK["Mô hình Tái Xếp Hạng Thần Kinh (Neural Reranker)<br/>cross-encoder/ms-marco-MiniLM-L-6-v2"]
    OLLAMA["Động cơ Suy Luận Ngôn Ngữ Cục Bộ (Ollama Engine)<br/>deepseek-r1:14b hoặc qwen2.5"]
    NLI["Mô hình Kiểm Chứng Quan Hệ Kéo Theo (NLI Verifier)<br/>DeBERTa-v3 / MiniLM-NLI (Threshold >= 0.70)"]
    OUT["Báo Cáo Tổng Hợp Hàn Lâm Cục Bộ Hoàn Chỉnh"]

    Q --> DENSE
    Q --> SPARSE
    DENSE --> RRF
    SPARSE --> RRF
    RRF --> RERANK
    RERANK --> OLLAMA
    OLLAMA --> NLI
    NLI --> OUT

    classDef purple fill:#f5f3ff,stroke:#a78bfa,stroke-width:1.5px,color:#2e1065;
    classDef input fill:#ede9fe,stroke:#8b5cf6,stroke-width:2px,color:#3b0764;
    classDef green fill:#f0fdf4,stroke:#4ade80,stroke-width:2px,color:#14532d;
    classDef special fill:#fdf4ff,stroke:#d946ef,stroke-width:1.5px,color:#701a75;

    class Q input;
    class DENSE,SPARSE,RRF,OLLAMA,NLI purple;
    class RERANK special;
    class OUT green;
```

---

## 2. Cơ Chế Hoạt Động Kỹ Thuật & Toán Học

### 2.1. Hợp Nhất Reciprocal Rank Fusion (RRF)
$$\text{RRF\_Score}(d) = \frac{\alpha}{60 + \text{Rank}_{\text{dense}}(d)} + \frac{1 - \alpha}{60 + \text{Rank}_{\text{bm25}}(d)}, \quad \text{với } \alpha = 0.65$$

### 2.2. Neural Cross-Encoder Reranking
Mô hình `ms-marco-MiniLM-L-6-v2` tính toán ma trận Attention toàn phần giữa từng token của câu hỏi $q$ và từng token của đoạn văn $d$:
$$\text{Score}_{\text{rerank}}(q, d) = \text{CrossEncoder}(q \circ d)$$

### 2.3. Suy Luận Cục Bộ Qua Giao Thức Ollama
Kết nối bất đồng bộ tới Ollama qua REST API, truyền system prompt kèm yêu cầu sinh JSON có cấu trúc.

---

## 3. Cấu Trúc Mã Nguồn Trong `main/modes/cpu_max/`

| Tệp Mã Nguồn | Lớp / Hàm Chính | Nhiệm Vụ Kỹ Thuật |
| :--- | :--- | :--- |
| `main/modes/cpu_max/__init__.py` | `CPUMaxPipeline` | Điều phối toàn trình luồng nhúng cục bộ, xếp hạng chéo và sinh LLM. |
| `main/modes/cpu_max/retriever.py` | `DenseRRFankRetriever` | Tải trọng số `multilingual-e5-small` từ `main/models/`, tính RRF và Rerank. |
| `main/modes/cpu_max/local_llm.py` | `OllamaLocalClient` | Quản lý kết nối HTTP persistent tới Ollama daemon, xử lý timeout và fallback. |
| `main/modes/cpu_max/verifier.py` | `LocalNLIVerifier` | Chạy mô hình NLI cục bộ kiểm chứng quan hệ Entailment của câu trả lời. |
