# QUY TẮC PHỐI HỢP PHÁT TRIỂN DỰ ÁN (TEAMWORK GUIDELINES)
### Dự án: Multi-Agent Research RAG System
> **Dành cho tất cả thành viên trong nhóm phát triển.**  
> Bắt buộc đọc kỹ và tuân thủ 100% để tránh xung đột Git (Merge Conflict), lệch định dạng dữ liệu và vỡ đồ thị LangGraph khi tích hợp.

---

## 1. NGUYÊN TẮC CỐT LÕI (CORE PHILOSOPHY)

> **"LỎNG Ở RUỘT, CHẶT Ở GIAO DIỆN" (Loose Coupling, Strict Contract)**

- **Ruột (Bên trong folder của bạn):** Bạn toàn quyền sáng tạo, thử nghiệm công cụ, chuẩn bị dữ liệu và viết script train trên Kaggle/Colab.
- **Giao diện (Điểm nối với hệ thống):** Bạn phải tuân thủ nghiêm ngặt chuẩn đầu vào/đầu ra qua file `node.py` để đồ thị LangGraph có thể gọi được mà không bị lỗi.

---

## 2. PHÂN CHIA THƯ MỤC & PHẠM VI QUYỀN HẠN

```
DM_Multi-Agent_RAG/
├── core/                           ⛔ [VÙNG CẤM] Chỉ Lead được sửa (State, Graph, BaseAgent)
├── data/corpus.jsonl               ⛔ [VÙNG CẤM] Kho tri thức dùng chung
├── main.py                         ⛔ [VÙNG CẤM] File chạy CLI chính
│
├── agents/                         🟢 [VÙNG HOẠT ĐỘNG CỦA THÀNH VIÊN]
│   ├── agent1_retrieval/           👉 Thành viên 1 toàn quyền quản lý
│   ├── agent2_reader/              👉 Thành viên 2 toàn quyền quản lý
│   ├── agent3_summarizer/          👉 Thành viên 3 toàn quyền quản lý
│   ├── agent4_fact_checker/        👉 Thành viên 4 toàn quyền quản lý
│   └── agent5_writer/              👉 Thành viên 5 toàn quyền quản lý
│
└── tests/                          🧪 Bộ kiểm thử tích hợp E2E
```

### ⚠️ Quy định phân quyền:
1. **Tuyệt đối KHÔNG tự ý chỉnh sửa** code trong thư mục `core/`, file `main.py`, hoặc các thư mục của thành viên khác.
2. Nếu cần bổ sung trường mới vào `AgentState`, phải **trao đổi trước với Lead** để thống nhất cập nhật tập trung tại `core/state.py`.
3. Mọi công việc của bạn phải nằm gọn trong thư mục `agents/agentX_.../` được giao.

---

## 3. BẢN GIAO KÈO INTERFACE BẮT BUỘC (`node.py`)

Trong thư mục của mỗi thành viên, bắt buộc phải có file `node.py`. File này chứa class kế thừa từ `core.base_agent.BaseAgent` với mẫu chuẩn như sau:

```python
# File: agents/agentX_your_agent/node.py
from typing import Any
from core.base_agent import BaseAgent
from core.state import AgentState

class YourAgent(BaseAgent):
    def __init__(self, max_internal_retry: int = 2):
        # Đặt tên định danh duy nhất cho Agent của bạn
        super().__init__(name="YourAgentName", max_internal_retry=max_internal_retry)
        # Nạp model / load công cụ phụ trợ ở đây

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Hàm thực thi chính mà LangGraph sẽ gọi.
        """
        # 1. ĐỌC DỮ LIỆU TỪ STATE
        query = state.get("query", "")
        
        # 2. XỬ LÝ NỘI BỘ (Model, Tools, CoT, v.v.)
        # ... logic của bạn ...
        
        # 3. TRẢ VỀ DICT CẬP NHẬT VÀO AGENTSTATE (Phải đúng tên trường quy định)
        return {
            "your_output_field": "giá trị",
            "reasoning_trace": [{
                "agent": self.name,
                "status": "success",
                "details": "Mô tả ngắn gọn bước làm"
            }]
        }
```

---

## 4. BẢNG ĐẶC TẢ INPUT / OUTPUT CHO TỪNG AGENT

Mỗi thành viên kiểm tra đúng Agent mình phụ trách:

| Agent | Thành viên | Đọc trường gì từ `state`? | Bắt buộc trả về trường gì? |
|---|---|---|---|
| **Agent 1** (Retrieval) | Thành viên 1 | `query`, `retry_count` | `retrieved_docs` (list dict), `sub_queries` (list str), `reasoning_trace` |
| **Agent 2** (Reader) | Thành viên 2 | `pdf_path` | `document_schema` (dict chứa pages, sections, bbox), `reasoning_trace` |
| **Agent 3** (Summarizer) | Thành viên 3 | `query`, `retrieved_docs`, `document_schema` | `answer` (str), `evidence_spans` (list str nguyên văn), `confidence` (str), `reasoning_trace` |
| **Agent 4** (Fact-Checker) | Thành viên 4 | `answer`, `evidence_spans`, `retrieved_docs` | `claims` (list str), `verified_claims` (list dict), `fact_check_label` (`support` / `contradict` / `not_enough_info`), `reasoning_trace` |
| **Agent 5** (Writer) | Thành viên 5 | `query`, `answer`, `verified_claims`, `retrieved_docs`, `document_schema` | `final_report` (str Markdown hoàn chỉnh), `reasoning_trace` |

---

## 5. QUY TẮC HUẤN LUYỆN & QUẢN LÝ TRỌNG SỐ (MLOps)

Thành viên có nhiệm vụ fine-tune mô hình (Agent 1, 2, 3, 4) phải tuân theo các quy tắc sau:

1. **Vị trí script train:** Toàn bộ code sinh dữ liệu, tiền xử lý và huấn luyện trên Kaggle/Colab/GPU phải đặt trong thư mục con `training/` của agent mình (ví dụ: `agents/agent1_retrieval/training/`).
2. **Quy tắc tường lửa:** File `node.py` (chạy inference phục vụ người dùng) **KHÔNG ĐƯỢC import chéo** vào thư mục `training/`.
3. **⛔ TUYỆT ĐỐI KHÔNG COMMIT FILE MODEL LÊN GIT:**
   - Các file nặng: `*.safetensors`, `*.bin`, `*.pt`, `checkpoints/`... không bao giờ được push lên Git repo (đã được cấu hình chặn trong `.gitignore`).
4. **Cách lưu trữ và nạp model sau khi train xong:**
   - **Cách khuyến nghị (Ưu tiên số 1):** Đẩy LoRA adapter nhẹ (~50MB-100MB) lên **Hugging Face Hub** cá nhân. Trong `node.py` chỉ cần gọi `PeftModel.from_pretrained(base_model, "username/repo-name")`.
   - **Hỗ trợ Dual Mode trong `node.py`:** Khuyến khích code có 2 chế độ: nạp model local/Hugging Face nếu máy có GPU, và fallback sang gọi API (Gemini/OpenAI) nếu chạy trên máy yếu.

---

## 6. QUY TRÌNH NỘP CODE VÀ MERGE (GIT WORKFLOW)

Để không bao giờ bị đè code của nhau:

1. **Không làm việc trực tiếp trên nhánh `main`:**
   - Mỗi người tạo một nhánh riêng từ `main`:
     ```bash
     git checkout -b feature/agent1-retrieval
     ```
2. **Trước khi tạo Pull Request (PR) về `main`:**
   - Chạy test riêng của Agent mình:
     ```bash
     python -m pytest agents/agentX_your_agent/tests/
     ```
   - Chạy bộ test tích hợp toàn hệ thống:
     ```bash
     python -m pytest tests/
     ```
   - Chạy thử E2E:
     ```bash
     python main.py --query "What is LoRA?"
     ```
   - **Chỉ khi 100% test đều PASS** mới được tạo Pull Request gửi Lead review.

---

## 7. CHECKLIST TRƯỚC KHI BÀN GIAO (DEFINITION OF DONE)

Một Agent được coi là **hoàn thành** khi đạt đủ 5 tiêu chí:
- [ ] Thư mục nằm đúng cấu trúc tại `agents/agentX_.../`.
- [ ] File `node.py` chạy độc lập thành công, không ném exception làm sập đồ thị.
- [ ] Trả về đúng 100% các trường dữ liệu được phân công trong bảng đặc tả.
- [ ] Có ít nhất 1 file test đơn vị trong thư mục `tests/` của Agent đó và pass.
- [ ] File `README.md` trong thư mục Agent được cập nhật hướng dẫn sử dụng rõ ràng.
