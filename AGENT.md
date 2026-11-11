# AGENT.md — Multi-Agent Research RAG System Rules & AI Operating Protocol

> **CRITICAL DIRECTIVE FOR ALL AI AGENTS (Cursor, Copilot, Claude, Windsurf, Antigravity, Aider, etc.):**
>
> You are operating within the **Multi-Agent Research RAG** repository (L3-L4 Autonomous Scientific Assistant).
> You MUST read, understand, and strictly follow every rule defined in this document.
>
> ⛔ **ABSOLUTE RULE NUMBER 1: STRICTLY NO GIT COMMIT & NO GIT PUSH**
> Under NO circumstances shall you execute `git commit`, `git push`, `git merge`, or modify git history.
> All git commits and repository history management belong exclusively to the human USER.
> Violating this rule is considered a critical boundary breach.

---

## 1. Project Context & Architecture Map

Hệ thống RAG Đa tác tử phân tích, trích xuất và thẩm định bài báo khoa học, bao gồm 6 Agents và 7 Chế độ thực thi song song:

### 1.1 Thư mục Lõi (`main/core/`)
- `state.py`: Hợp đồng trạng thái LangGraph (`AgentState`), các dataclass `SourceUnit`, `AtomicClaim`, `ClaimCheckResult`, `EvidenceGap`.
- `graph.py`: StateGraph DAG Orchestrator kết nối toàn bộ 5 nodes chính, vòng lặp tự sửa lỗi (Self-Correction Loop) và Escalation.
- `supervisor.py`: Agentic Supervisor lập kế hoạch truy vấn, phân loại intent, xác định độ phức tạp (LOW/MEDIUM/HIGH) và sinh sub-queries.
- `verifier.py`: Bộ kiểm chứng số học, ngày tháng và đối soát trích dẫn nguyên văn với ranh giới từ (word boundaries).
- `db.py`: Động cơ SQLite (`main/data/workspace.db`) quản lý tài liệu, lịch sử hội thoại, cache LLM và log API.
- `config.py`: Quản lý cấu hình toàn cục (`settings`).
- `tools.py`: Đăng ký công cụ trích xuất PDF và xử lý toán học an toàn có chặn giới hạn tràn bộ nhớ.

### 1.2 6 Tác tử Chuyên môn hóa (`main/agents/`)
- `agent0_scout`: Trinh sát học thuật ArXiv / CrossRef / OpenAlex tự động tải và lập chỉ mục bài báo khi chưa có PDF.
- `agent1_retrieval`: Truy hồi lai (BM25 + C FTS5 + Dense RRF) và phân tích góc nhìn Stanford STORM.
- `agent2_reader`: Phân tích bố cục tài liệu đa cột (DocLayNet-style) và băm SHA-256 provenance đảm bảo nguồn trích.
- `agent3_summarizer`: Tác tử tóm tắt tích hợp bộ phê bình nội bộ (Inner Critic) phát hiện thiếu sót ngữ cảnh.
- `agent4_fact_checker`: Thẩm phán NLI SciFact, đối soát số liệu, ngày tháng và kiểm chứng mã nguồn (Zero Hallucination).
- `agent5_writer`: Tác tử biên soạn báo cáo nghiên cứu học thuật hai nửa (Executive Summary + Audit Ledger) và radar khoảng trống tri thức.

### 1.3 7 Chế độ Vận hành Độc lập (`main/modes/`)
- `gemini`: Đám mây tốc độ cao qua Gemini 2.5/1.5 Flash API với cơ chế tự động xoay vòng model.
- `cpu_min`: Chế độ siêu nhẹ chạy offline 100% trên CPU bất kỳ (FTS5 + BM25 + Extractive Guard, latency < 100ms).
- `cpu_max`: Chế độ CPU đa luồng kết hợp Vulkan Ollama cục bộ và động cơ C-Engine FTS5.
- `gpu_ft`: Chế độ GPU tinh chỉnh với các adapter LoRA chuyên biệt.
- `gpu_finetune`: Chế độ GPU 7-Model SOTA với BGE-Reranker NLI Head đạt F1 = 1.0.
- `gpu_max`: Chế độ suy luận sâu chuỗi tư duy (CoT) 32B trên hạ tầng GPU lớn.
- `hybrid`: Chế độ lai cờ đầu tự động dò tìm năng lực phần cứng và định tuyến thông minh giữa GPU, CPU và Cloud.

### 1.4 API Backend & Web UI
- `main/api/app.py`: FastAPI server hỗ trợ SSE streaming sự kiện thời gian thực, upload file giới hạn 100MB, và endpoint quản trị.
- `main/web_demo2/`: Giao diện Web Workspace hiện đại bằng HTML/CSS/JS thuần, polling có exponential backoff và hiển thị trace suy luận.

---

## 2. Các nguyên tắc kỹ thuật bất biến (Coding Invariants)

1. **Bảo toàn Hợp đồng StateGraph:** Tuyệt đối không thay đổi kiểu dữ liệu hoặc cấu trúc các trường trong `AgentState` (`main/core/state.py`), đặc biệt là trường reducer `reasoning_trace: Annotated[list[dict[str, Any]], operator.add]`.
2. **Kỷ luật Zero-Hallucination & Chuẩn hóa 3 Kịch bản Phản hồi:** Hệ thống đặt tính trung thực học thuật lên hàng đầu:
   - *Kịch bản 1 (Có trong tài liệu):* Trả lời trực diện, nguyên tắc cốt lõi **"ĐÚNG VÀ ĐỦ"**, kèm mỏ neo trích dẫn `[1]` *(Trang X)* và bảo toàn số liệu, công thức, khối cấu hình/code.
   - *Kịch bản 2 (Không có thông tin):* BẮT BUỘC trả về `NOT_FOUND_IN_DOCUMENT` kèm thông điệp từ chối trung thực: *"Dựa trên các bài báo/tài liệu hiện có trong hệ thống, thông tin về \"{query}\" **hoàn toàn không được đề cập**. Vui lòng hỏi các câu hỏi liên quan đến nội dung tài liệu được cung cấp hoặc tải thêm tài liệu liên quan lên hệ thống để phân tích."*
   - *Kịch bản 3 (Có đề cập nhưng mơ hồ / phân nhánh):* Khử mơ hồ chủ động (*Active Disambiguation*): Khẳng định có thông tin, liệt kê 2–4 khía cạnh/bảng kết quả có trong bài và yêu cầu làm rõ: *"Vui lòng cung cấp thêm chi tiết để tôi hỗ trợ chính xác hơn dựa trên tài liệu."*
3. **Tính tương thích cả 7 Chế độ:** Bất kỳ sửa đổi nào ở tầng `core` hoặc `agents` không được làm gãy bất kỳ chế độ nào trong 7 chế độ vận hành.
4. **An toàn Môi trường Windows & PowerShell:**
   - Luôn sử dụng encoding `utf-8` khi đọc/ghi file hoặc in ra màn hình console.
   - Thao tác đường dẫn phải dùng `os.path` hoặc `pathlib` để tương thích Windows (`\\` vs `/`).
   - Tuyệt đối không nhúng chuỗi ký tự `$` trong lệnh PowerShell inline mà không escape để tránh bị parse nhầm biến shell.
   - Không dùng lệnh shell `sleep` làm bộ đếm thời gian.
5. **Clickable Markdown File Links:** Luôn tạo đường link markdown `file://` có thể click mở file trực tiếp tới từng class/hàm/dòng code được đề cập.

---

## 3. Quy trình Kiểm soát Thực thi (Workflow Gate)

Mỗi khi nhận nhiệm vụ, AI Agent phải thực hiện theo trình tự:
1. **Xác định phạm vi (Task Scope):** Chỉ can thiệp vào các file liên quan trực tiếp đến yêu cầu; không tự ý tái cấu trúc (refactor) các file ngoài phạm vi.
2. **Nguyên tắc can thiệp chính xác (Surgical Changes):** Giữ nguyên định dạng và comment của các đoạn mã không liên quan.
3. **Cấm cài đặt tùy tiện:** Tuyệt đối không chạy lệnh `pip install` thêm gói bên thứ ba mới nếu chưa có sự đồng ý của người dùng.

---

## 4. Điểm dừng Khẩn cấp (Stop Conditions)

AI Agent phải **lập tức dừng lại và hỏi ý kiến người dùng** khi:
- Yêu cầu dẫn đến việc thay đổi cấu trúc bảng cơ sở dữ liệu SQLite (`workspace.db`).
- Sửa đổi hợp đồng dữ liệu giữa các agent trong StateGraph.
- Cần xóa file tài liệu gốc trong `uploads/` hoặc các mô hình đã tải về trong `models/`.
- Phát hiện xung đột kiến trúc hoặc nguy cơ làm hỏng các bài kiểm thử hồi quy hiện có.

---

## 5. Tiêu chuẩn Nghiệm thu (Definition of Done - DoD)

Một nhiệm vụ chỉ được xem là hoàn thành khi thỏa mãn đầy đủ 4 tiêu chí:
1. Yêu cầu của người dùng được giải quyết triệt để và hoạt động ổn định.
2. Không sinh ra file rác, dead code, hoặc import mồ côi.
3. **100% các bộ kiểm thử tự động vượt qua (PASS):**
   - Unit tests: `python -m pytest main/tests/root_suite/unit -o "pythonpath=main"`
   - Integration tests: `python -m pytest main/tests/root_suite/integration -o "pythonpath=main"`
   - Architecture tests: `python -m pytest main/tests/test_modes_architecture.py -o "pythonpath=main"`
   - System Sanity Check: `python main/tests/system_sanity_check.py`
4. **Tuyệt đối không có lệnh `git commit` nào được thực thi.**

---

## 6. Mẫu Báo cáo Kết thúc Nhiệm vụ

Sau khi hoàn thành, Agent phải xuất báo cáo tổng kết theo mẫu sau:

```markdown
### 📋 Báo cáo Hoàn thành Nhiệm vụ

#### 1. Nội dung đã thực hiện
- [Tóm tắt tính năng hoặc bản sửa lỗi đã áp dụng]

#### 2. Danh sách file thay đổi
- `[file://đường/dẫn/file]` (Tạo mới / Chỉnh sửa / Xóa bỏ)

#### 3. Kết quả Kiểm thử Nghiệm thu
- **Unit Tests:** [37/37 PASSED]
- **Integration Tests:** [4/4 PASSED]
- **Architecture Tests:** [15/15 PASSED]
- **System Sanity Check:** [35/35 PASSED]

#### 4. Cam kết An toàn
- Xác nhận 100% không thực hiện bất kỳ lệnh `git commit` hay `git push` nào.
```

---

## 7. Tiêu chuẩn Phản hồi: Nguyên tắc "ĐÚNG VÀ ĐỦ" (Accuracy & Sufficiency Protocol)

Mọi Tác tử (Agents) tham gia sinh câu trả lời (Summarizer, Writer, Extractive Engines, LLMs) **BẮT BUỘC** phải tuân thủ nguyên tắc cốt lõi: **"ĐÚNG VÀ ĐỦ — KHÔNG THỪA THÃI, KHÔNG THIẾU SÓT"**.

### 7.1 Định nghĩa 4 Trụ Cột

| Trụ cột | Yêu cầu bắt buộc | Dấu hiệu vi phạm (Phải loại bỏ) |
| :--- | :--- | :--- |
| **1. ĐÚNG** *(Accurate)* | Trả lời chính xác bản chất câu hỏi, bám sát 100% ngữ cảnh tài liệu nguồn, trung thực học thuật (Zero Hallucination). | Trả lời lạc đề, suy diễn ngoài tài liệu, nhầm lẫn khái niệm. |
| **2. ĐỦ** *(Sufficient)* | Trình bày trọn vẹn câu trả lời: có định nghĩa bản chất + khối cú pháp/mã lệnh (nếu là kỹ thuật) + 1–2 cơ chế vận hành mật thiết. | Trả lời cụt ngủn (1–2 dòng cộc lốc), chỉ nêu tên mà không giải thích cách dùng. |
| **3. KHÔNG THỪA** *(Non-redundant)* | Cắt bỏ hoàn toàn thông tin phụ trợ, không lặp lại ý, không kéo theo các mục ngoài lề không liên quan trực tiếp đến câu hỏi. | Kéo theo phân công công việc (Task, slide), link tham khảo (Bài viết, video), hay các cấu hình không liên quan. |
| **4. KHÔNG THIẾU** *(Non-omissive)* | Đầy đủ các thành phần kỹ thuật cần thiết để người đọc hiểu và áp dụng được ngay mà không phải hỏi lại. | Giải thích lý thuyết nhưng bỏ quên cú pháp cấu hình hoặc quên nêu thuật toán cốt lõi. |

### 7.2 Quy tắc Phản hồi theo Dạng Câu hỏi

1. **Dạng Khái niệm / Định nghĩa Kỹ thuật (`X là gì?`, `Cú pháp X`):**
   - **Bản chất & Mục đích:** Định nghĩa trực diện $X$ là gì và giải quyết vấn đề gì trong thực tế.
   - **Cú pháp / Mã lệnh mẫu:** Nếu $X$ là cấu hình hoặc lệnh, **bắt buộc** định dạng bằng khối code chuẩn (ví dụ ````nginx ... ````, ````bash ... ````).
   - **Cơ chế cốt lõi gắn liền:** Chỉ bổ sung tối đa 1–2 điểm kỹ thuật mật thiết nhất (ví dụ: cách trỏ request tới $X$, thuật toán của $X$).
   - ⛔ **Cấm:** Không kéo theo thông tin giáo trình/slide (*Task, người thực hiện, thời gian thuyết trình*), danh mục tham khảo (*Bài viết, video, certbot*), hoặc các thông số không liên quan nằm cùng trang (*client_max_body_size, SSL tự ký*).
2. **Dạng Liệt kê / Quy trình (`Gồm những gì?`, `Các bước thực hiện`):**
   - Xác định rõ số lượng thành phần. Trình bày dạng danh sách đánh số `1.`, `2.`, `3.` với tiêu đề in đậm rõ ràng.
3. **Dạng Thời gian / Số liệu / Sự kiện (`Khi nào?`, `Bao nhiêu?`):**
   - Trả lời trực tiếp mốc thời gian / con số cụ thể ngay ở câu mở đầu + 1 câu bối cảnh trực tiếp. Dừng lại, không lan man.
4. **Dạng Tóm tắt Toàn diện (`/tom-tat`, `Tóm tắt nội dung`):**
   - Phải cung cấp bản tóm tắt đa chiều bao quát đầy đủ các phần: *Mục tiêu $\rightarrow$ Kiến trúc nền tảng $\rightarrow$ Kỹ thuật triển khai $\rightarrow$ Lưu ý vận hành*.
5. **Dạng Ngoài tài liệu (Out-of-Domain Zero Hallucination):**
   - Trả lời minh bạch: *"Dựa trên các bài báo/tài liệu hiện có trong hệ thống, thông tin về '{query}' hoàn toàn không được đề cập."* Tuyệt đối không suy diễn.

### 7.3 Checklist Kiểm Thử Nhanh (3 KHÔNG — 3 PHẢI)

- ❌ **KHÔNG** chứa rác hành chính, giáo trình, phân công nhiệm vụ (*Task, slide, thuyết trình*).
- ❌ **KHÔNG** chứa danh mục liên kết đọc thêm (*Bài viết: ..., Video: ...*).
- ❌ **KHÔNG** kéo theo các chỉ thị cấu hình không liên quan trực tiếp đến thực thể được hỏi.
- ✅ **PHẢI** trả lời trúng trọng tâm câu hỏi ngay trong 2 dòng đầu.
- ✅ **PHẢI** có khối code block chuẩn định dạng nếu hỏi về cú pháp/kỹ thuật.
- ✅ **PHẢI** có trích dẫn số trang nguồn `[1] (Trang X)` đối soát minh bạch.
