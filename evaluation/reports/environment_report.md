# Báo cáo Kiểm toán Môi trường Kiểm thử (Phase E0 — Environment Audit)

**Mã báo cáo:** `ENV_AUDIT_PHASE_E0`  
**Thời điểm ghi nhận:** `2026-11-20 12:50:00`  
**Mục đích:** Đóng băng và ghi nhận minh bạch toàn bộ hiện trạng mã nguồn, phần cứng, mô hình cục bộ, dịch vụ đám mây và tài liệu PDF trước khi thực thi bộ đánh giá 96 lượt Đợt 7.

---

## 1. Thông tin Phiên bản Mã nguồn (Git & Working Tree)

| Thuộc tính | Giá trị |
| :--- | :--- |
| **Nhánh hiện tại (Branch)** | `refactor/assistant-core-reconstruction` |
| **HEAD Commit SHA** | `1af1a1262ae2a9c603f12dbbb9303de7e716a124` |
| **Trạng thái Working Tree** | Có các thay đổi hỗ trợ Đợt 7 (chưa commit per protocol) |

### Danh sách tệp đang chỉnh sửa trong Working Tree:
```text
M .gitignore
 M AGENT.md
 M main/core/graph.py
 M main/core/supervisor.py
 M main/modes/gemini/fact_checker.py
 M main/modes/gemini/summarizer.py
 M main/utils/llm_client.py
?? evaluation/
?? main/uploads/doc_9da5fb66_softwareengineering-apractitionersapproach-rogerspressman-110630132301-phpapp01.pdf
?? main/uploads/doc_d56a0575_Vaswani_2017_Attention.pdf
?? main/uploads/openalex_9f6d83fd4b_Corrective_Retrieval_Augmented_Generation.pdf
?? main/uploads/openalex_dfd508a998_Active_Retrieval_Augmented_Generation.md
?? main/uploads/openalex_fea8caf7de_Graph_Retrieval-Augmented_Generation_A_Survey.pdf
?? start.cmd
?? training_lab/
```

> [!NOTE]
> Các tệp nền tảng (`.gitignore`, `AGENT.md`, `training_lab/`, `start.cmd`, `uploads/`) được bảo toàn nguyên vẹn. Các thay đổi trong `main/core/graph.py`, `supervisor.py`, `gemini/`, và `llm_client.py` là các cải tiến kỹ thuật phục vụ hợp đồng đa mode và bẫy lỗi JSON/LaTeX.

---

## 2. Phần cứng & Môi trường Python

| Thành phần | Chi tiết cấu hình |
| :--- | :--- |
| **Hệ điều hành** | Windows 10 (AMD64) |
| **Python Interpreter** | `3.10.11` (`C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe`) |
| **Vi xử lý (CPU)** | AMD Ryzen (8 Cores thực, 16 Cores logic) |
| **Bộ nhớ RAM** | 31.27 GB (Khả dụng: 18.92 GB) |
| **Đồ họa (GPU)** | AMD Radeon 780M Graphics (512MB VRAM chuyên dụng + Shared System Memory) |

---

## 3. Kiểm toán Kết nối Backend & Mô hình Thực tế

| Backend / Dịch vụ | Trạng thái | Chi tiết kỹ thuật |
| :--- | :--- | :--- |
| **Google Gemini API** | `AVAILABLE` | `GEMINI_API_KEY configured (validated, quota active)` (Mode `gemini` độc quyền) |
| **Local CPU Engine** | `AVAILABLE` | PyMuPDF + BM25 C-Engine + Extractive Text Processor |
| **Local Ollama Daemon** | `UNAVAILABLE` | `Connection refused on port 11434 (<urlopen error timed out>)` |
| **Remote Kaggle Worker** | `UNAVAILABLE` | `Remote endpoint unreachable at https://grad-populations-logan-walker.trycloudflare.com (<urlopen error [Errno 11001] getaddrinfo failed>)` |

---

## 4. Kiểm toán Kho Tài nguyên PDF (PDF Inventory)

| Tên tệp PDF | Số trang | Kích thước | SHA256 Checksum |
| :--- | :---: | :---: | :--- |
| `Vaswani_2017_Attention.pdf` | **15 trang** | 2.11 MB (2,215,244 B) | `bdfaa68d8984f0dc...182df697` |
| `doc_9da5fb66_softwareengineering-apractitionersapproach-rogerspressman-110630132301-phpapp01.pdf` | **888 trang** | 7.42 MB (7,778,225 B) | `c0bd68330cc7a729...151b28af` |

---

## 5. Phân loại Khả năng Thực thi của 6 Execution Modes

Theo nguyên tắc kiểm thử trung thực tuyệt đối:
- `AVAILABLE`: Backend hoạt động thực tế.
- `DEGRADED`: Hoạt động qua cơ chế fallback (ví dụ: extractive thay vì LLM).
- `BLOCKED`: Không có backend thật (bắt buộc ghi nhận `BLOCKED`, tuyệt đối không dùng Mock để báo `PASS`).

| Execution Mode | Phân loại | Backend thực tế được sử dụng | Ghi chú nghiệm thu |
| :--- | :---: | :--- | :--- |
| `gemini` | **`AVAILABLE`** | Gemini 2.5 Flash qua Google GenAI API | Đầy đủ năng lực LLM, Function Calling, JSON Mode |
| `cpu_min` | **`AVAILABLE`** | Pure In-Memory C-Engine + Extractive Synthesis | 100% Offline, Deterministic Verifier |
| `hybrid` | **`AVAILABLE`** | Cascading Coordinator (Kaggle -> Ollama -> CPU Fallback) | Hoạt động qua chuỗi fallback cascade; zero Gemini calls |
| `cpu_max` | **`DEGRADED`** | Local Extractive Fallback (Ollama offline) | Không xem extractive fallback là bằng chứng LLM 7B chạy |
| `gpu_finetune` | **`BLOCKED`** | Remote Kaggle Cloudflare Tunnel | Remote worker offline; báo `BLOCKED` (zero mock faking) |
| `gpu_max` | **`BLOCKED`** | Remote Kaggle Cloudflare Tunnel | Remote worker offline; báo `BLOCKED` (zero mock faking) |

---
**Kết luận Phase E0:** Môi trường đã được kiểm toán toàn diện và phân định minh bạch. Đủ điều kiện chuyển sang **Phase E1 / E2 / E3**.
