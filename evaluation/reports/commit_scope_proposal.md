# Đề Xuất Phân Rã & Phạm Vi Commit (Commit Scope Proposal)

**Thời điểm:** 2026-11-20  
**Nhánh Git:** `refactor/assistant-core-reconstruction`  
**Base Branch:** `develop`  
**Trạng thái PR:** **DRAFT** (Tuyệt đối không merge hoặc tạo commit khi chưa có phê duyệt từ người dùng).

---

## 1. Kiểm Toán Hiện Trạng Toàn Bộ Working Tree (`git status -s`)

| File Đường Dẫn | Trạng Thái Git | Phân Loại Nội Dung | Khuyến Nghị Xử Lý |
| :--- | :---: | :--- | :--- |
| `main/utils/llm_client.py` | `M` | **Sửa lỗi DEF-01** (Phân vùng cache key đa chiều). | **Stage vào Commit Sửa Lỗi** |
| `main/utils/json_repair.py` | `??` | **Sửa lỗi DEF-02** (Bộ quét giải mã JSON LaTeX chuẩn RFC). | **Stage vào Commit Sửa Lỗi** |
| `main/modes/gemini/summarizer.py` | `M` | **Sửa lỗi DEF-02** (Tích hợp `extract_and_parse_json`). | **Stage vào Commit Sửa Lỗi** |
| `main/modes/gemini/fact_checker.py` | `M` | **Sửa lỗi DEF-02** (Tích hợp `extract_and_parse_json`). | **Stage vào Commit Sửa Lỗi** |
| `main/core/supervisor.py` | `M` | **Sửa lỗi DEF-03 & DEF-04** (Phân định dịch thuật & lệnh OS). | **Stage vào Commit Sửa Lỗi** |
| `main/core/graph.py` | `M` | Sửa unwrap JSON của Direct Assistant. | **Stage vào Commit Sửa Lỗi** |
| `main/api/app.py` | `M` | **Sửa lỗi DEF-05** (Cho phép run không kèm tài liệu). | **Stage vào Commit Sửa Lỗi** |
| `main/tests/test_evaluation_regression.py` | `??` | **Bộ kiểm thử hồi quy 5 lỗi** (5 bài test, 100% pass). | **Stage vào Commit Sửa Lỗi** |
| `evaluation/` | `??` | **Hệ sinh thái Đánh giá Toàn diện Đợt 7** (Dataset, Runners, Ground Truth, Results, Reports, Schemas). | **Stage vào Commit Đánh Giá (hoặc chung)** |
| `.gitignore` | `M` | Baseline đã sửa (bảo lưu local, không đưa vào commit mới). | **KHÔNG STAGE (Giữ nguyên local)** |
| `AGENT.md` | `M` | Baseline đã sửa (bảo lưu local, không đưa vào commit mới). | **KHÔNG STAGE (Giữ nguyên local)** |
| `training_lab/` | `??` | Baseline mã nguồn huấn luyện (bảo lưu local). | **KHÔNG STAGE (Giữ nguyên local)** |
| `start.cmd` | `??` | Script khởi động tiện ích cục bộ. | **KHÔNG STAGE (Giữ nguyên local)** |
| `main/uploads/*.pdf` | `??` | File PDF tài liệu người dùng và file test. | **KHÔNG STAGE (Giữ nguyên local)** |

---

## 2. Đề Xuất Phương Án Đóng Gói Commit

Để đảm bảo tính độc lập, minh bạch và dễ dàng rà soát mã nguồn (code review), chúng tôi đề xuất **Phương án 2 Commit Nguyên Tử (Two Atomic Commits)**:

### Phương Án A (Khuyến Nghị Cao Nhất): Tách 2 Commit Độc Lập

#### **Commit 5: Core Defect Fixes & Comprehensive Regression Suite**
- **Mục tiêu:** Đóng gói toàn bộ 5 bản sửa lỗi kỹ thuật phát hiện trong quá trình đánh giá và file test hồi quy bảo vệ.
- **Danh sách file cụ thể:**
  ```text
  main/api/app.py
  main/core/graph.py
  main/core/supervisor.py
  main/modes/gemini/fact_checker.py
  main/modes/gemini/summarizer.py
  main/utils/json_repair.py
  main/utils/llm_client.py
  main/tests/test_evaluation_regression.py
  ```
- **Thông điệp đề xuất:**
  ```text
  fix(core): resolve evaluation defects DEF-01 to DEF-05 and add regression suite

  - DEF-01: partition semantic LLM cache by mode, agent, and generation params
  - DEF-02: robust JSON LaTeX parsing preserving escapes and repairing unescaped commands
  - DEF-03: disambiguate translation queries referencing 'tai lieu' from Document QA
  - DEF-04: guard TASK_EXECUTION against general explanation of commands and software
  - DEF-05: permit API runs without documents for general assistant supervisor routing
  - Add 5 comprehensive regression tests covering all edge cases (108/108 tests passing)
  ```

#### **Commit 6: Independent AI System Evaluation Subsystem**
- **Mục tiêu:** Bổ sung toàn bộ khung đánh giá độc lập, 16 Golden Cases, bộ chạy tự động và các báo cáo nghiệm thu kỹ thuật.
- **Danh sách thư mục cụ thể:**
  ```text
  evaluation/
  ```
- **Thông điệp đề xuất:**
  ```text
  test(evaluation): introduce standardized AI System Evaluation Plan and reports

  - Add 16 Golden Test Cases across 5 functional evaluation groups
  - Add ground truth facts, expected routes, and 10-point scoring rubric
  - Add automated evaluation runners for live matrix, API, and reliability
  - Add comprehensive audit reports (pre-commit, defect, integrity, API/FE)
  ```

---

### Phương Án B: Gom 1 Commit Duy Nhất (Single Consolidated Commit)
- Đóng gói toàn bộ các file trên và thư mục `evaluation/` vào một commit duy nhất mang tên:
  `test(evaluation): add Đợt 7 evaluation subsystem and fix defects DEF-01..05`.

---

## 3. Lệnh Git Cụ Thể (Sau Khi Được Người Dùng Phê Duyệt)

> [!CAUTION]
> **Dừng lại trước khi thực thi!** Tuyệt đối không dùng `git add .` hoặc `git add -A`.
> Chỉ thực hiện lệnh khi có sự đồng ý tường minh từ người dùng.

```powershell
# BƯỚC 1: Stage chính xác các file mã nguồn và test hồi quy
git add main/api/app.py
git add main/core/graph.py
git add main/core/supervisor.py
git add main/modes/gemini/fact_checker.py
git add main/modes/gemini/summarizer.py
git add main/utils/json_repair.py
git add main/utils/llm_client.py
git add main/tests/test_evaluation_regression.py

# BƯỚC 2: Tạo Commit 5
git commit -m "fix(core): resolve evaluation defects DEF-01 to DEF-05 and add regression suite"

# BƯỚC 3: Stage thư mục evaluation
git add evaluation/

# BƯỚC 4: Tạo Commit 6
git commit -m "test(evaluation): introduce standardized AI System Evaluation Plan and reports"

# BƯỚC 5: Kiểm tra trạng thái Git đảm bảo các file baseline không bị ảnh hưởng
git status -s
```
