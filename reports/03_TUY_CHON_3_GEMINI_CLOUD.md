# ☁️ Báo Cáo Kiến Trúc Tùy Chọn 3: GEMINI (Cloud Foundation Model Pipeline)

> **Mã Chế Độ**: `gemini`  
> **Thư Mục Thực Thi**: `main/modes/gemini/`  
> **Tài Nguyên Yêu Cầu**: Kết Nối Internet \| API Key Google Gemini (Zero Local GPU)  
> **Mô Hình Khai Thác**: `gemini-3-flash-preview` & `gemini-3.1-flash-lite`  
> **Độ Trễ Trung Bình**: $1,180.0\text{ ms}$ / truy vấn  

---

## 1. Sơ Đồ Quy Trình Thực Thi `gemini` (Mermaid Flowchart)

```mermaid
flowchart TD
    classDef inputNode fill:#ede9fe,stroke:#8b5cf6,stroke-width:2px,color:#3b0764,font-weight:bold,rx:6,ry:6;
    classDef stepNode fill:#f5f3ff,stroke:#a78bfa,stroke-width:1.5px,color:#2e1065,rx:4,ry:4;
    classDef loopNode fill:#fef2f2,stroke:#f87171,stroke-width:1.5px,color:#7f1d1d,rx:4,ry:4;
    classDef judgeNode fill:#fdf4ff,stroke:#c084fc,stroke-width:1.5px,color:#581c87,rx:4,ry:4;
    classDef outNode fill:#f0fdf4,stroke:#4ade80,stroke-width:2px,color:#14532d,font-weight:bold,rx:6,ry:6;

    Q["Câu hỏi nghiên cứu phức tạp (User Query)"]:::inputNode
    DEC["Agent 1 Planner (Gemini 3.1 Flash-Lite)<br/>Phân tích & Tách 3 Truy Vấn Con (Sub-Queries)"]:::stepNode
    HYBRID["Truy Xuất Hỗn Hợp Đa Luồng<br/>Gemini Text-Embedding + BM25 FTS5"]:::stepNode
    GEN["Agent 3 Generator (Gemini 3 Flash)<br/>Sinh bản thảo JSON: answer + evidence_spans"]:::stepNode
    CRIT["Agent 3 Inner Critic (Gemini 3 Flash)<br/>Phản biện độc lập đối chiếu bằng chứng"]:::loopNode
    JUDGE["Agent 4 Multi-Persona Consensus Judge<br/>Giám khảo Chữ nghĩa vs Giám khảo Ngữ cảnh"]:::judgeNode
    OUT["Báo Cáo Nghiên Cứu Chuẩn IEEE/Markdown"]:::outNode

    Q --> DEC
    DEC --> HYBRID
    HYBRID --> GEN
    GEN --> CRIT
    CRIT -- "is_supported == false (Phát hiện ảo giác)" --> GEN
    CRIT -- "is_supported == true (Bằng chứng hợp lệ)" --> JUDGE
    JUDGE -- "Đồng thuận >= 2/3 SUPPORT" --> OUT
    JUDGE -- "Bất đồng / CONTRADICT" --> DEC
```

---

## 2. Cơ Chế Hoạt Động Kỹ Thuật & Toán Học

### 2.1. Phân Rã Truy Vấn Tự Động (Sub-Query Decomposition)
Đối với các câu hỏi so sánh hoặc nghiên cứu đa bước, `GeminiQueryPlanner` phân tách truy vấn gốc $Q$ thành 3 truy vấn con chuyên biệt:
$$Q \longrightarrow \{q_{\text{theoretical}},\; q_{\text{mathematical}},\; q_{\text{empirical}}\}$$
Mỗi truy vấn con được gắn trọng số truy xuất riêng để bao quát trọn vẹn không gian tài liệu.

### 2.2. Vòng Lặp Phản Biện Nội Tại (Generator-Critic Loop) Ép Kiểu JSON
* **Generator**: Trả về cấu trúc JSON bắt buộc:
  ```json
  {"answer": "...", "evidence_spans": ["Trích dẫn 1", "Trích dẫn 2"]}
  ```
* **Inner Critic**: Một phiên gọi độc lập đóng vai trò phản biện, nhận vào `context`, `question`, `answer` và `evidence_spans`. Nếu có bất kỳ mệnh đề nào trong `answer` không tìm thấy căn cứ trong `evidence_spans`, Critic trả về:
  ```json
  {"is_supported": false, "reason": "Mệnh đề X thiếu chứng cứ trong ngữ cảnh", "action": "rewrite"}
  ```
  buộc Generator phải viết lại ngay trong cùng một lượt chạy.

### 2.3. Hội Đồng Giám Khảo Đa Nhân Cách (Multi-Persona Consensus Judge)
Agent 4 khởi tạo 2 giám khảo ảo với nhiệt độ $T=0.0$:
* **Strict Literal Judge**: Chỉ chấp nhận nếu câu chữ tương đương 1-1.
* **Contextual Scientific Judge**: Đánh giá tính hợp lý theo logic khoa học và toán học.
Phán quyết cuối cùng là trung bình trọng số xác suất giữa hai giám khảo.

---

## 3. Cấu Trúc Mã Nguồn Trong `main/modes/gemini/`

| Tệp Mã Nguồn | Lớp / Hàm Chính | Nhiệm Vụ Kỹ Thuật |
| :--- | :--- | :--- |
| `main/modes/gemini/__init__.py` | `GeminiCloudPipeline` | Điều phối toàn bộ vòng đời suy luận trên nền tảng Google GenAI SDK. |
| `main/modes/gemini/planner.py` | `GeminiQueryPlanner` | Tách truy vấn, xác định độ khó và tạo prompt chuyên biệt theo ngữ cảnh. |
| `main/modes/gemini/synthesizer.py` | `GeminiCriticSynthesizer` | Triển khai vòng lặp Generator + Critic ép kiểu `response_mime_type="application/json"`. |
| `main/modes/gemini/judge.py` | `MultiPersonaConsensusJudge` | Hội đồng thẩm định đa nhân cách kiểm định tính chuẩn xác khoa học. |
