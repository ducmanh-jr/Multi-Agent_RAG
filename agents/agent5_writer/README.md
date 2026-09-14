# TÀI LIỆU ĐẶC TẢ KỸ THUẬT: AGENT 5 - WRITER AGENT
### Vị trí: Tổng hợp Báo cáo Nghiên cứu & Phê bình cấp câu (Sentence-level Grounding)
> **Người thực hiện:** Thành viên 5  
> **Thư mục làm việc độc lập:** `agents/agent5_writer/`  
> **File bàn giao hệ thống:** `agents/agent5_writer/node.py`

---

## 1. TỔNG QUAN NHIỆM VỤ

Writer Agent là "người chốt sổ" cuối cùng trong đồ thị LangGraph trước khi trả kết quả cho người dùng.

Sau khi **Fact-Checker Agent** đã xác nhận câu trả lời đạt tiêu chuẩn (`support`), Writer Agent thực hiện 2 nhiệm vụ chuyên sâu:
1. **Academic Report Synthesis (Tổng hợp báo cáo học thuật):** Soạn thảo báo cáo nghiên cứu hoàn chỉnh, trang trọng theo chuẩn Markdown xuất bản (có Tóm tắt kết quả, Luận điểm đã chứng minh kèm trích dẫn nguyên văn, Danh mục tài liệu tham chiếu).
2. **Sentence-level Grounding Critique (Phê bình từng câu):** Rà soát từng câu văn trong bản thảo báo cáo, bắt buộc mỗi câu khẳng định phải gắn với một `verified_claim` đã kiểm chứng. **Loại bỏ hoặc cảnh báo bất kỳ câu "mồ côi" nào** (câu tự thêm vào mà không có cơ sở dữ liệu).

---

## 2. QUYẾT ĐỊNH THIẾT KẾ: PHÊ BÌNH CẤP CÂU (SENTENCE CRITIQUE)

Một lỗi rất phổ biến trong các hệ thống RAG là: LLM ở bước cuối cùng khi "viết văn cho hay" lại tự động thêm vào các câu nhận xét mang tính chủ quan hoặc kiến thức ngoài lề không có căn cứ.

Để giải quyết triệt để, Writer Agent áp dụng cơ chế:
- **Tách văn bản thành từng câu riêng biệt.**
- Đối chiếu câu đó với danh sách `verified_claims` đã được Fact-Checker đóng dấu.
- Gán nhãn hành động cho từng câu:
  - `action = "keep"`: Câu có liên kết với claim đã kiểm chứng $\rightarrow$ Được giữ lại.
  - `action = "rewrite"`: Câu diễn đạt chưa rõ ràng $\rightarrow$ Yêu cầu viết lại.
  - `action = "delete"`: Câu mồ côi, tự bịa thêm thông tin ngoài lề $\rightarrow$ Xóa bỏ khỏi báo cáo.

---

## 3. GIAO KÈO INTERFACE TRONG `node.py`

File `node.py` là điểm duy nhất mà đồ thị LangGraph gọi vào. Bạn PHẢI tuân thủ:

### 📥 Dữ liệu đầu vào (Đọc từ `AgentState`):
- `state["query"]`: Câu hỏi nghiên cứu ban đầu (`str`).
- `state.get("answer", "")`: Câu trả lời đã qua Summarizer (`str`).
- `state.get("verified_claims", [])`: Danh sách các luận điểm đã được Fact-Checker kiểm chứng (`list[dict]`).
- `state.get("retrieved_docs", [])`: Danh sách tài liệu nguồn để lập mục tham chiếu (`list[dict]`).
- `state.get("document_schema")`: Cấu trúc PDF bóc tách được (nếu có) (`dict`).

### 📤 Dữ liệu đầu ra (Bắt buộc trả về):
```python
return {
    "final_report": """# BÁO CÁO NGHIÊN CỨU: ...
## 1. Tóm tắt kết quả (Executive Summary)
...
## 2. Các luận điểm đã kiểm chứng (Fact-Checked Claims)
- **[C1]** ... — `✅ SUPPORTED`
  > *Trích dẫn nguyên văn:* "..."
## 3. Tài liệu nguồn tham chiếu (Retrieved Sources)
1. **[arxiv_2305_lora]** ...
## 4. Kết luận & Độ tin cậy
...""",
    "reasoning_trace": [{
        "agent": "WriterAgent",
        "status": "report_finalized_and_grounded",
        "total_sentences_checked": 12,
        "grounded_sentences": 10,
        "report_length": 2400
    }]
}
```

---

## 4. CẤU TRÚC BÁO CÁO MỤC TIÊU (MARKDOWN STANDARD)

Báo cáo cuối cùng xuất ra phải tuân thủ chuẩn cấu trúc khoa học:
1. **Header:** Tên đề tài nghiên cứu viết hoa.
2. **Executive Summary:** Câu trả lời cô đọng, khúc chiết.
3. **Fact-Checked Claims:** Từng luận điểm kèm huy hiệu trạng thái (`✅ SUPPORTED`) và khối trích dẫn nguyên văn (`> *Trích dẫn nguyên văn:* ...`).
4. **Retrieved Sources:** Liệt kê ID tài liệu nguồn, điểm số độ liên quan (`RRF / BM25 Score`) và đoạn trích dẫn ngắn.
5. **Document Layout (Nếu có PDF):** Tên file PDF và số trang đã phân tích.
6. **Audit & Reliability Assessment:** Kết luận về độ tin cậy của báo cáo.

---

## 5. THUẬT TOÁN XỬ LÝ TRONG `components/`

- **`components/report_builder.py`:**
  - Lắp ráp các phần của báo cáo từ `answer`, `verified_claims` và `retrieved_docs`.
- **`components/sentence_critic.py`:**
  - Sử dụng regex tách câu: `re.split(r"(?<=[.?!])\s+", text)`.
  - Tính toán độ trùng khớp từ vựng (word overlap) giữa câu văn và tập claims đã duyệt.
  - Loại bỏ các câu không đạt ngưỡng overlap.

---

## 6. TIÊU CHÍ NGHIỆM THU (DEFINITION OF DONE)

- [ ] File `node.py` chạy thành công độc lập.
- [ ] Chạy lệnh `python -m pytest agents/agent5_writer/tests/` đạt 100% PASS.
- [ ] Báo cáo sinh ra có đầy đủ 4-5 phần theo đúng định dạng Markdown chuẩn xuất bản.
- [ ] Mọi luận điểm trong báo cáo đều có trích dẫn nguồn hoặc evidence span kèm theo.
- [ ] Không chứa các câu khẳng định mồ côi (đã qua bộ lọc của `SentenceCritic`).
