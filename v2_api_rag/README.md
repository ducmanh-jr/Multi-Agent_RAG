# Hệ Thống Multi-Agent RAG v2 (API-Driven Architecture)

Hệ thống RAG Đa tác tử (Multi-Agent Research RAG) thế hệ mới được tái cấu trúc hoàn toàn:
- **KHÔNG cần Fine-tuning** (bỏ toàn bộ QLoRA, bitsandbytes, DocLayNet, SciBERT).
- **Chạy nhẹ nhàng trên mọi Laptop cá nhân** (chỉ tốn ~100MB RAM, không yêu cầu GPU).
- **Tận dụng LLM Frontier Sẵn Có** (Google Gemini 2.5 Flash / Groq) thông qua **Structured Output (Pydantic)**.
- **Bóc tách PDF tự nhiên** bằng PyMuPDF (nhanh, chuẩn xác, không cần cài Tesseract binary hay LayoutLMv3).
- **Đầy đủ 5 Agent** với cơ chế tự sửa lỗi đa tầng (LangGraph StateGraph).

---

## 🏗️ Kiến Trúc Hệ Thống

```mermaid
flowchart TD
    START([Bắt đầu]) --> Router["Orchestrator Router<br/>(Xác định Intent)"]
    Router --> Agent1["1. Retrieval Agent<br/>(Dynamic Routing + BM25/Dense)"]
    
    Agent1 -->|"Nếu có PDF"| Agent2["2. Reader Agent<br/>(PyMuPDF Layout Extractor)"]
    Agent1 -->|"Không có PDF"| Agent3["3. Summarizer Agent<br/>(Structured Evidence Extraction)"]
    
    Agent2 --> Agent3
    Agent3 --> Agent4["4. Fact-Checker Agent<br/>(LLM-as-a-Judge / NLI Verifier)"]
    
    Agent4 --> Decision{"fact_check_label?"}
    
    Decision -->|"support"| Agent5["5. Writer Agent<br/>(Soạn Báo Cáo Nghiên Cứu Markdown)"]
    Decision -->|"contradict / not_enough_info<br/>retry < 3"| RetryLoop["Retry Loop (Tăng retry)"]
    Decision -->|"retry >= 3"| Escalation["Escalation Node<br/>(Cảnh báo độ tin cậy thấp)"]
    
    RetryLoop --> Agent1
    Agent5 --> END([Kết thúc])
    Escalation --> END
```

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Cài đặt môi trường cực nhanh (chỉ mất 30 giây)
```bash
pip install -r v2_api_rag/requirements.txt
```

### 2. Thiết lập API Key
Tạo file `.env` trong thư mục `v2_api_rag/` (hoặc ngay tại thư mục gốc dự án) với nội dung:
```env
# Lấy miễn phí tại: https://aistudio.google.com/
GEMINI_API_KEY="AIzaSy..."
```

### 3. Chạy thử nghiệm hệ thống
```bash
python v2_api_rag/main.py
```
