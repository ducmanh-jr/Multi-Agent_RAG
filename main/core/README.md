# ⚙️ Core Agentic Engine L3–L4 (`main/core/`)

Thư mục `main/core/` chứa bộ khung điều phối trung tâm giúp nâng cấp toàn bộ 6 tùy chọn thực thi từ đường ống tuần tự (L1) lên **Hệ thống Đa Tác tử Tự chủ (L3–L4)**.

---

## 📄 Các Thành Phần Trong `main/core/`

1. **[`graph.py`](graph.py) — Đồ thị trạng thái LangGraph Two-Tier Self-Correction:**
   - **Vòng lặp tầng 1 (`route_after_summarizer`)**: Nếu Summarizer Critic phát hiện ngữ cảnh thiếu thông tin (`critic_action == "request_more_context"`), đồ thị quay thẳng về `retry_loop_node` $ightarrow$ `retrieval_node` mà không tốn chi phí gọi Fact-Checker.
   - **Vòng lặp tầng 2 (`route_after_fact_check`)**: Nếu Fact-Checker phát hiện mệnh đề mâu thuẫn (`contradict`) hoặc thiếu bằng chứng (`not_enough_info`), đồ thị kích hoạt `retry_loop_node` để sinh câu truy vấn bù lỗ hổng (`gap_requeries`).
2. **[`state.py`](state.py) — Hợp đồng dữ liệu hợp nhất (`AgentState`):**
   - Định nghĩa `SourceUnit`, `EvidenceSpan` (kèm mã băm SHA-256 `context_hash`), `AtomicClaim`, `ClaimCheckResult`, và `EvidenceGap`.
3. **[`schemas.py`](schemas.py) — Chuẩn hóa Schema & Tên LoRA Adapter:**
   - Hợp nhất đầu ra JSON giữa `training_lab/` và `main/modes/` (`RouterPlanOutput`, `SummarizerGeneratorOutput`, `SummarizerCriticOutput`).
   - Ánh xạ tự động tên Adapter (`generator_qlora` $\leftrightarrow$ `generator`, `critic_qlora` $\leftrightarrow$ `critic`, `agent1_router` $\leftrightarrow$ `router`).
4. **[`supervisor.py`](supervisor.py) — Bộ lập kế hoạch `AgenticSupervisorPlanner`:**
   - Phân loại độ phức tạp truy vấn (`LOW` / `MEDIUM` / `HIGH`), trích xuất thực thể bắt buộc (`required_entities`) và sinh truy vấn con (`sub_queries`).
5. **[`tools.py`](tools.py) — `AgentToolRegistry` & Bộ đọc PDF theo phạm vi (`load_or_parse_selected_pdf`):**
   - Giải quyết triệt để lỗi tràn phạm vi tài liệu: tự động phân tách câu có cửa sổ ngữ cảnh (Sentence-Window Chunking) và lưu đệm (cache) trực tiếp trên file PDF người dùng đang chọn.
6. **[`verifier.py`](verifier.py) — Kiểm chứng Tất định & Đa mô hình:**
   - Kiểm tra chính xác số liệu/năm (`check_numerical_and_date_consistency`), đối sánh thuật ngữ xuyên ngôn ngữ Việt–Anh (`deterministic_verify_claims`), và đảm bảo gắn ký hiệu trích dẫn `[1]`, `[2]` (`ensure_inline_citations`).
