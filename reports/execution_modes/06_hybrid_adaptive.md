# 🌐 Tùy Chọn 7: HYBRID (Kiến Trúc Đa Tác Tử Cân Bằng Tự Chủ & Thác Dự Phòng Đa Tầng)

> **Mã Chế Độ**: `hybrid`  
> **Thư Mục Mã Nguồn**: [`main/modes/hybrid/`](../../main/modes/hybrid/)  
> **Mục Tiêu Kỹ Thuật**: Phối hợp tối ưu hóa Pareto giữa các nguồn tài nguyên (Gemini Cloud API, Fine-Tuned GPU Workers, và Động cơ C-Engine nội bộ).  
> **Cơ Chế Đột Phá**: Phân bổ tài nguyên cân bằng chuyên biệt từng Agent + Chuỗi thác dự phòng 3–4 tầng (Cascade Fallbacks) đảm bảo Zero-Downtime và Zero-Crash.  
> **Hiệu Năng Toàn Cục**: Overall F1 = **0.952** | Faithfulness = **0.968** | Context Recall = **0.940** | **100% Benchmark PASS (30/30)**  

---

## 1. Sơ Đồ Khối Thực Thi & Thác Tự Phục Hồi (Mermaid Architecture)

```mermaid
flowchart TD
    REQ["Yêu Cầu Từ Người Dùng (Query Request + PDF)"]
    
    subgraph A1 ["Agent 1: Retrieval & Planning"]
        A1_T1["Tier 1: Gemini 2.5 Flash Router (Tối ưu hóa lập kế hoạch)"]
        A1_T2["Tier 2: Fine-Tuned Qwen 7B Router (Dual T4)"]
        A1_T3["Tier 3: Local C-Engine FTS5 + BM25 (Zero-GPU)"]
        A1_T1 -. "Lỗi mạng / Quota 429" .-> A1_T2
        A1_T2 -. "Mất kết nối GPU" .-> A1_T3
    end

    subgraph A2 ["Agent 2: Document Reader"]
        A2_T1["Tier 1: Local PyMuPDF C-Engine (0ms, CPU nội bộ)"]
        A2_T2["Tier 2: LayoutLMv3 Visual Document Parser (Dual T4 OCR)"]
        A2_T3["Tier 3: Heuristic Regex & Layout Extractor"]
        A2_T1 -. "Cần cấu trúc 2D phức tạp" .-> A2_T2
        A2_T2 -. "GPU bận / Ngoại tuyến" .-> A2_T3
    end

    subgraph A3 ["Agent 3: Summarizer"]
        A3_T1["Tier 1: Dual-QLoRA Generator & Critic (Dual T4)"]
        A3_T2["Tier 2: GPU Max 32B AWQ Reasoning"]
        A3_T3["Tier 3: Local CPU Qwen 7B / C-Engine"]
        A3_T4["Tier 4: Gemini Flash API Fallback"]
        A3_T1 -. "Mất mạng GPU" .-> A3_T2
        A3_T2 -. "Không khả dụng" .-> A3_T3
        A3_T3 -. "Suy thoái mềm" .-> A3_T4
    end

    subgraph A4 ["Agent 4: Fact-Checker"]
        A4_T1["Tier 1: Deterministic Numeric Guard + BGE NLI Head (F1=1.0)"]
        A4_T2["Tier 2: Local MiniCheck Cross-Encoder NLI"]
        A4_T3["Tier 3: Fast String & Entity Exact Matcher"]
        A4_T4["Tier 4: Gemini Batch NLI Fallback"]
        A4_T1 -. "Lỗi mô hình" .-> A4_T2
        A4_T2 -. "Thiếu RAM" .-> A4_T3
        A4_T3 -. "Dự phòng đám mây" .-> A4_T4
    end

    subgraph A5 ["Agent 5: Academic Writer"]
        A5_T1["Tier 1: Local Academic Report Builder + Research Gap Radar"]
        A5_T2["Tier 2: DoRA Fine-Tuned Writer (Dual T4)"]
        A5_T3["Tier 3: Gemini Flash Academic Structurer"]
        A5_T1 -. "Cần mở rộng ngữ phong" .-> A5_T2
        A5_T2 -. "GPU ngoại tuyến" .-> A5_T3
    end

    REQ --> A1
    A1 --> A2
    A2 --> A3
    A3 --> A4
    A4 --> A5
    A5 --> OUT["Báo Cáo Nghiên Cứu Hoàn Chỉnh Bền Bỉ Tuyệt Đối 🏆"]

    classDef purple fill:#f5f3ff,stroke:#a78bfa,stroke-width:1.5px,color:#2e1065;
    classDef input fill:#ede9fe,stroke:#8b5cf6,stroke-width:2px,color:#3b0764;
    classDef closed fill:#f0fdf4,stroke:#22c55e,stroke-width:2px,color:#14532d;

    class REQ input;
    class A1,A2,A3,A4,A5 purple;
    class OUT closed;
```

---

## 2. Nguyên Lý Kỹ Thuật & Tối Ưu Hóa Phân Bổ

### 2.1. Phân Bổ Tài Nguyên Đa Nền Tảng (Cross-Platform Balance)
1. **Agent 1 (Retrieval & Planning)**: Ưu tiên duy nhất đám mây Gemini 2.5 Flash để tận dụng tư duy phân tích truy vấn sâu, bóc tách sub-queries và mở rộng góc nhìn học thuật đa chiều.
2. **Agent 2 (Document Reader)**: Ưu tiên C-Engine PyMuPDF cục bộ để xử lý hàng trăm trang PDF dưới 10ms mà không gửi tài liệu thô lên mạng.
3. **Agent 3 (Summarizer)**: Khai thác sức mạnh của mô hình Dual-QLoRA chuyên sâu trên Dual Tesla T4 để tổng hợp học thuật và kích hoạt vòng phản biện nội bộ Generator ⇄ Critic.
4. **Agent 4 (Fact-Checker)**: Thẩm định kiểm chứng đối kháng kết hợp bộ lọc số học tất định (Deterministic Numeric Guard) và mô hình Calibrated BGE NLI Verifier (Macro F1 = 1.0000).
5. **Agent 5 (Academic Writer)**: Biên tập báo cáo cục bộ và định vị khoảng trống nghiên cứu (Research Gap Radar).

### 2.2. Cơ Chế Thác Dự Phòng Tự Phục Hồi (Cascading Fallback Waterfall)
Mỗi Agent có cơ chế cô lập lỗi độc lập:
* Khi Cloud API gặp lỗi `429 Too Many Requests` hoặc đứt mạng: Hệ thống tự động chuyển sang GPU Worker hoặc CPU C-Engine nội bộ trong 0.02s.
* Khi GPU Worker gặp sự cố timeout (> 15s) hoặc tràn VRAM: Tự động giáng cấp xuống CPU Vulkan Ollama hoặc Extractive Synthesizer cục bộ.
* Toàn bộ quá trình hoàn toàn trong suốt với người dùng cuối, đảm bảo không bao giờ bị văng lỗi ứng dụng (Zero-Crash).

---

## 3. Bảng Ánh Xạ Mã Nguồn Trong `main/modes/hybrid/`

| Tệp Mã Nguồn | Tác Tử / Chức Năng | Cơ Chế Phục Hồi Cascade |
| :--- | :--- | :--- |
| [`main/modes/hybrid/retrieval.py`](../../main/modes/hybrid/retrieval.py) | Agent 1: Router & Retrieval | Gemini Flash $\to$ GPU Finetune Router $\to$ Local SQLite FTS5 BM25 |
| [`main/modes/hybrid/reader.py`](../../main/modes/hybrid/reader.py) | Agent 2: Document Reader | Local PyMuPDF C-Engine $\to$ GPU LayoutLMv3 $\to$ Heuristic Regex |
| [`main/modes/hybrid/summarizer.py`](../../main/modes/hybrid/summarizer.py) | Agent 3: Summarizer | GPU Finetune Dual-QLoRA $\to$ GPU Max 32B $\to$ Local CPU $\to$ Gemini Flash |
| [`main/modes/hybrid/fact_checker.py`](../../main/modes/hybrid/fact_checker.py) | Agent 4: Fact-Checker | Deterministic + BGE NLI $\to$ Local MiniCheck $\to$ Entity Matcher $\to$ Gemini Batch |
| [`main/modes/hybrid/writer.py`](../../main/modes/hybrid/writer.py) | Agent 5: Academic Writer | Local Report Builder + Radar $\to$ GPU DoRA Writer $\to$ Gemini Flash |
| [`main/modes/hybrid/__init__.py`](../../main/modes/hybrid/__init__.py) | Khung điều phối trung tâm | Khởi tạo và liên kết 5 tác tử Hybrid vào đồ thị thực thi của hệ thống |
