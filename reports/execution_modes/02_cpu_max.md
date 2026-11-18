# 💻 Tùy Chọn 2: CPU_MAX (Heavy Local Workstation Pipeline)

> **Mã Chế Độ**: `cpu_max`  
> **Thư Mục Mã Nguồn**: `main/modes/cpu_max/`  
> **Tài Nguyên Yêu Cầu**: CPU Đa Nhân (Core i5/i7/Ultra) | RAM 4GB - 8GB | Mô hình Cục bộ  
> **Độ Trễ Trung Bình**: 412.0 ms / truy vấn  
> **Chỉ Số Đánh Giá**: Overall F1 = **0.842** | Faithfulness = **0.865** | Context Recall = **0.825**  

---

## 1. Sơ Đồ Khối Thực Thi (Mermaid Architecture)

```mermaid
flowchart TD
    Q["Câu hỏi người dùng (User Query)"]
    DENSE["Mô hình Nhúng Ngữ Nghĩa Cục bộ<br/>intfloat/multilingual-e5-small (main/models/)"]
    SPARSE["Chỉ mục Từ vựng Tần suất Cao<br/>BM25 Okapi Indexing"]
    RRF["Hợp Nhất Thứ Hạng Reciprocal Rank Fusion (RRF)<br/>k = 60 | alpha = 0.65"]
    RERANK["Mô hình Tái Xếp Hạng Thần Kinh (Neural Reranker)<br/>cross-encoder/ms-marco-MiniLM-L-6-v2"]
    OLLAMA["Động cơ Suy Luận Ngôn Ngữ Cục Bộ (Ollama Engine)<br/>deepseek-r1:14b hoặc qwen2.5:7b"]
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

## 2. Nguyên Lý Kỹ Thuật & Công Thức Toán Học

### 2.1. Dung Hợp Thứ Hạng RRF (Reciprocal Rank Fusion)
Dung hợp kết quả từ mô hình nhúng đa ngữ `multilingual-e5-small` và chỉ mục `BM25 Okapi` với hằng số làm mịn $k=60$:
$$\text{RRF\_Score}(d) = \frac{\alpha}{60 + \text{Rank}_{\text{dense}}(d)} + \frac{1 - \alpha}{60 + \text{Rank}_{\text{bm25}}(d)}, \quad \alpha = 0.65$$

### 2.2. Tái Xếp Hạng Thần Kinh (Neural Cross-Encoder Reranking)
Đưa 15 đoạn văn ứng viên qua mạng `ms-marco-MiniLM-L-6-v2` để tính toán Attention toàn phần giữa mọi cặp từ khóa:
$$\text{Score}_{\text{rerank}}(q, d) = \text{CrossEncoder}(q \circ d)$$
Lọc ra Top-5 đoạn văn giàu thông tin học thuật nhất cung cấp cho động cơ sinh.

### 2.3. Tích Hợp Động Cơ Suy Luận Cục Bộ Qua Giao Thức Ollama
Kết nối bất đồng bộ qua REST API tới tiến trình Ollama cục bộ (`http://localhost:11434`), hỗ trợ các mô hình mở `deepseek-r1:14b` hoặc `qwen2.5:7b` với cơ chế bắt lỗi và kiểm soát thời gian chờ (timeout fallback).

---

## 3. Bảng Ánh Xạ Mã Nguồn Trong `main/modes/cpu_max/`

| Tệp Mã Nguồn | Lớp / Hàm Chính | Vai Trò & Thuật Toán |
| :--- | :--- | :--- |
| `main/modes/cpu_max/__init__.py` | `CPUMaxPipeline` | Điều phối luồng xử lý E5-Small + RRF + Cross-Encoder + Ollama. |
| `main/modes/cpu_max/retriever.py` | `DenseRRFankRetriever` | Nạp `main/models/multilingual-e5-small`, tính điểm RRF và chạy Cross-Encoder. |
| `main/modes/cpu_max/local_llm.py` | `OllamaLocalClient` | Quản lý kết nối HTTP persistent tới Ollama daemon, xử lý timeout và fallback. |
| `main/modes/cpu_max/verifier.py` | `LocalNLIVerifier` | Kiểm chứng quan hệ kéo theo (Entailment) bằng mô hình NLI cục bộ. |
