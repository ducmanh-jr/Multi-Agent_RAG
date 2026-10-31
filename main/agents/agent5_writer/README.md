# TÀI LIỆU ĐẶC TẢ KỸ THUẬT: AGENT 5 - WRITER AGENT
### Vị trí: Tổng hợp Báo cáo Nghiên cứu & Phê bình cấp câu (Sentence-level Grounding Critique)
> **Người thực hiện:** Thành viên 5  
> **Thư mục làm việc độc lập:** `agents/agent5_writer/`  
> **File bàn giao hệ thống:** `agents/agent5_writer/node.py`  
> **Môi trường chạy mục tiêu:** Kaggle Notebook (GPU T4 x2) hoặc Môi trường thực thi cục bộ

---

## 1. TỔNG QUAN NHIỆM VỤ & VỊ TRÍ KIẾN TRÚC

Writer Agent là "người chốt sổ" cuối cùng trong đồ thị LangGraph trước khi trả kết quả cho người dùng. 

Sau khi **Fact-Checker Agent** đã xác nhận câu trả lời đạt tiêu chuẩn (`support`), Writer Agent thực hiện 2 nhiệm vụ chuyên biệt:
1. **Academic Report Synthesis (Tổng hợp báo cáo học thuật):** Soạn thảo báo cáo nghiên cứu hoàn chỉnh, trang trọng theo chuẩn Markdown xuất bản (có Tóm tắt kết quả, Luận điểm đã chứng minh kèm trích dẫn nguyên văn, Danh mục tài liệu tham chiếu).
2. **Sentence-level Grounding Critique (Phê bình từng câu văn):** Rà soát từng câu văn trong bản thảo báo cáo, bắt buộc mỗi câu khẳng định phải gắn với một `verified_claim` đã kiểm chứng từ Agent 4. **Loại bỏ hoặc cảnh báo bất kỳ câu "mồ côi" nào** (câu tự thêm vào mà không có cơ sở dữ liệu).

---

## 2. HIỆN TRẠNG TỪ CODE CŨ & ĐỊNH HƯỚNG TRIỂN KHAI

- **Hiện trạng:** Trong các nhánh cũ (`branches/`), **Agent 5 hoàn toàn chưa có code**.
- **Cạm bẫy của việc viết báo cáo bằng LLM thông thường:**
  Khi yêu cầu LLM viết một báo cáo dài, LLM thường có thói quen tự thêm thắt các nhận xét chủ quan, từ ngữ văn vẻ hoặc kiến thức tổng quát ngoài lề không nằm trong tài liệu nguồn. Đây chính là nguyên nhân gây ra ảo giác ở khâu cuối cùng!
- **Giải pháp kỹ thuật của Writer Agent:**
  Kết hợp **Mô hình sinh báo cáo** cùng **Bộ lọc phê bình cấp câu (Sentence Critic)**:
  - Tách văn bản báo cáo thành danh sách các câu đơn lẻ bằng regex: `re.split(r"(?<=[.?!])\s+", text)`.
  - Dùng mô hình so khớp tương đồng ngữ nghĩa cực nhẹ: `sentence-transformers/all-MiniLM-L6-v2` (~80MB VRAM, chạy trong 5ms trên CPU/GPU).
  - So sánh từng câu văn trong báo cáo với danh sách các luận điểm đã được kiểm chứng (`verified_claims`):
    - Nếu độ tương đồng cosine $\ge 0.65$: Gán hành động `keep` (giữ lại câu).
    - Nếu câu mang tính liên kết ngữ pháp thuần túy (header, chuyển đoạn): Giữ lại.
    - Nếu là câu khẳng định sự kiện nhưng độ tương đồng $< 0.5$ (câu mồ côi): Gán hành động `delete` (loại bỏ khỏi báo cáo) hoặc cảnh báo!

---

## 3. THIẾT LẬP TRÊN KAGGLE / MÔI TRƯỜNG CỤC BỘ

### Cài đặt thư viện:
```bash
!pip install -q -U "sentence-transformers" "scikit-learn" "pydantic"
```

Mô hình embedding `all-MiniLM-L6-v2` chỉ nặng ~80MB, hoàn toàn có thể chạy chung trên GPU 0 hoặc GPU 1 của Kaggle T4 mà không hề gây ảnh hưởng đến bộ nhớ VRAM của các agent khác.

---

## 4. CẤU TRÚC BÁO CÁO CHUẨN KHOA HỌC (MARKDOWN SPECIFICATION)

Báo cáo nghiên cứu do Agent 5 xuất ra (`final_report`) phải tuân thủ nghiêm ngặt cấu trúc Markdown sau:

```markdown
# BÁO CÁO NGHIÊN CỨU: <TIÊU ĐỀ ĐỀ TÀI>

## 1. Tóm tắt kết quả (Executive Summary)
<Đoạn văn ngắn gọn, súc tích 3-5 câu trả lời thẳng vào câu hỏi>

## 2. Các luận điểm khoa học đã kiểm chứng (Fact-Checked Claims)
- **[Luận điểm 1]:** <Nội dung khẳng định> — `[✅ XÁC THỰC]`
  > *Trích dẫn chứng cứ nguyên văn:* "<Đoạn text trích nguyên văn từ tài liệu nguồn>"
- **[Luận điểm 2]:** ...

## 3. Danh mục tài liệu tham chiếu (Retrieved Sources)
1. **[doc_id_1]** (Độ liên quan RRF: 0.032)
   - Trích đoạn: "..."
2. **[doc_id_2]** ...

## 4. Phân tích cấu trúc tài liệu PDF (Nếu có PDF)
- Tên tài liệu: `Attention_Is_All_You_Need.pdf`
- Tổng số trang phân tích: 12 trang (Bóc tách bởi Reader Agent).

## 5. Kết luận & Đánh giá độ tin cậy
- Trạng thái kiểm chứng: **Đạt tiêu chuẩn Factual Grounding**
- Tỉ lệ câu có căn cứ: 100% (Đã lọc qua Sentence-level Critic).
```

---

## 5. ĐẶC TẢ GIAO DIỆN `node.py`

File `agents/agent5_writer/node.py` phải tuân thủ nghiêm ngặt giao diện:

```python
from typing import Any
from core.base_agent import BaseAgent
from core.state import AgentState

class WriterAgent(BaseAgent):
    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="WriterAgent", max_internal_retry=max_internal_retry)
        # Nạp ReportBuilder và SentenceCritic (MiniLM-L6-v2)
        
    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        answer = state.get("answer", "")
        verified_claims = state.get("verified_claims", [])
        retrieved_docs = state.get("retrieved_docs", [])
        document_schema = state.get("document_schema")
        
        # 1. Soạn thảo bản thảo báo cáo ban đầu
        # 2. Rà soát phê bình từng câu (Sentence-level grounding critique)
        # 3. Loại bỏ câu khẳng định mồ côi
        
        final_markdown_report = "# BÁO CÁO NGHIÊN CỨU: ..."
        
        return {
            "final_report": final_markdown_report,
            "reasoning_trace": [
                {
                    "agent": self.name,
                    "status": "report_finalized_and_grounded",
                    "total_sentences_checked": 8,
                    "grounded_sentences": 8,
                    "report_length": len(final_markdown_report)
                }
            ]
        }
```

---

## 6. TIÊU CHÍ NGHIỆM THU (DEFINITION OF DONE)

- [ ] File `node.py` chạy độc lập thành công.
- [ ] Báo cáo đầu ra có cấu trúc rõ ràng, định dạng Markdown chuẩn xuất bản với đầy đủ 4-5 phần.
- [ ] Toàn bộ các câu khẳng định trong báo cáo đều có nguồn gốc từ `verified_claims` hoặc ngữ cảnh nguồn.
- [ ] Bộ lọc `SentenceCritic` nhận diện và loại bỏ được các câu bịa đặt ngẫu nhiên được chèn vào.
- [ ] Toàn bộ unit tests trong `agents/agent5_writer/tests/` đều vượt qua (100% PASS).
