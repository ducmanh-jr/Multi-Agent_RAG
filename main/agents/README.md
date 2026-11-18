# 🤖 Hệ Thống 6 Tác Tử Cơ Sở (`main/agents/`)

Thư mục `main/agents/` cung cấp các công cụ (tools) và thành phần xử lý chuyên sâu cho 6 tác tử trong đường ống nghiên cứu khoa học:

| Tác tử (`Agent`) | Thư mục | Vai trò cốt lõi |
| :--- | :--- | :--- |
| **Agent 0: Literature Scout** | [`agent0_scout/`](agent0_scout/) | Tự động tìm kiếm và tải bài báo khoa học từ ArXiv / Semantic Scholar khi người dùng chưa chỉ định file PDF. |
| **Agent 1: Retrieval** | [`agent1_retrieval/`](agent1_retrieval/) | Tìm kiếm lai ghép (SQLite FTS5 + BM25 + Dense Vector `multilingual-e5-small`) và hợp nhất thứ hạng bằng Reciprocal Rank Fusion (RRF) + Cross-Encoder Reranker. |
| **Agent 2: Reader** | [`agent2_reader/`](agent2_reader/) | Phân tích cấu trúc tài liệu PDF bằng PyMuPDF, trích xuất bảng biểu, công thức toán học và đóng gói thành các `SourceUnit` có gắn mã băm `context_hash`. |
| **Agent 3: Summarizer** | [`agent3_summarizer/`](agent3_summarizer/) | Đóng gói ngữ cảnh theo ngân sách token (`pack_context_by_token_budget`), tổng hợp câu trả lời học thuật có trích dẫn `[1]`, `[2]` và tách mệnh đề nguyên tử (`AtomicClaim`). |
| **Agent 4: Fact-Checker** | [`agent4_fact_checker/`](agent4_fact_checker/) | Kiểm chứng từng mệnh đề bằng NLI Cross-Encoder (`minicheck`), đối chiếu trích dẫn nguyên văn (`quote_verifier`), kiểm tra số liệu/ngày tháng và kiểm toán thống kê (`statistical_auditor`). |
| **Agent 5: Writer** | [`agent5_writer/`](agent5_writer/) | Biên tập báo cáo nghiên cứu hoàn chỉnh (`report_builder`) và phân tích khoảng trống nghiên cứu (`gap_analyzer` — Research Gap Radar). |
