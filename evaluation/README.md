# Multi-Agent RAG — Hệ thống Đánh giá Toàn diện (Evaluation Subsystem)

Thư mục này chứa toàn bộ bộ dữ liệu chuẩn, kịch bản kiểm thử, bộ chạy tự động và báo cáo nghiệm thu chất lượng hệ thống cho dự án **Multi-Agent RAG**.

---

## 1. Cấu trúc Thư mục

```text
evaluation/
├── README.md                      # Hướng dẫn tổng quan và cách chạy
│
├── dataset/                       # 16 Golden Test Cases chuẩn hóa
│   ├── manifest.json              # Danh mục tài liệu tham chiếu & checksums
│   ├── general_qa.json            # Nhóm A: General Assistant (GQA-01..04)
│   ├── document_qa.json           # Nhóm B: Document Intelligence (DOC-01..05)
│   ├── web_research.json          # Nhóm C: Web Research OpenAlex (WEB-01..02)
│   ├── mixed_qa.json              # Nhóm D: Mixed QA PDF + Web (MIX-01..02)
│   └── edge_cases.json            # Nhóm E: Edge Cases & Abstention (EDGE-01..03)
│
├── ground_truth/                  # Đáp án tham chiếu từ PDF & Thang điểm
│   ├── document_facts.json        # Dữ kiện kiểm chứng từ Vaswani (2017) & Pressman (2001)
│   ├── expected_citations.json    # Số trang và đoạn trích bắt buộc
│   ├── expected_routes.json       # Phân loại intent và node đồ thị kỳ vọng
│   └── scoring_rubric.json        # Thang điểm 10 chuẩn hóa (Correctness, Completeness, Groundedness...)
│
├── runners/                       # Các bộ chạy kiểm thử độc lập
│   ├── generate_env_report.py     # Phase E0: Kiểm toán môi trường phần cứng, API, PDF
│   ├── run_eval.py                # Phase E3: Chạy ma trận 96 lượt (16 cases x 6 modes)
│   ├── run_api_eval.py            # Phase E5: Kiểm thử 10 kịch bản HTTP API (API-01..10)
│   ├── run_reliability.py         # Phase E4: Kiểm thử 10 kịch bản chịu lỗi (REL-01..10)
│   ├── run_frontend_eval.py       # Phase E5: Kiểm thử 8 luồng người dùng frontend (FE-01..08)
│   └── generate_reports.py        # Phase E6: Tổng hợp 4 báo cáo Markdown chuẩn hóa
│
├── results/                       # Kết quả JSON máy đọc được
│   ├── summary_metrics.json       # Bảng điểm và trạng thái 6 modes
│   ├── gemini.json                # Chi tiết 16 runs mode gemini
│   ├── cpu_min.json               # Chi tiết 16 runs mode cpu_min
│   ├── hybrid.json                # Chi tiết 16 runs mode hybrid
│   ├── cpu_max.json               # Chi tiết 16 runs mode cpu_max (DEGRADED)
│   ├── gpu_finetune.json          # Chi tiết 16 runs mode gpu_finetune (BLOCKED)
│   ├── gpu_max.json               # Chi tiết 16 runs mode gpu_max (BLOCKED)
│   ├── api_evaluation.json        # Chi tiết 10 API tests
│   ├── reliability_evaluation.json# Chi tiết 10 Reliability tests
│   └── frontend_evaluation.json   # Chi tiết 8 Frontend tests
│
├── reports/                       # Báo cáo đánh giá Markdown
│   ├── environment_report.md      # Báo cáo kiểm toán môi trường (Phase E0)
│   ├── functional_evaluation_report.md # Báo cáo kết quả 96 lượt (Phase E3)
│   ├── cross_mode_comparison.md   # So sánh đối chiếu 6 execution modes
│   └── system_acceptance_report.md# Báo cáo nghiệm thu & khuyến nghị release
│
└── schemas/                       # JSON Schemas xác thực dữ liệu
    ├── test_case.schema.json      # Schema cho Golden Test Cases
    ├── evaluation_result.schema.json # Schema cho kết quả kiểm thử
    └── validate_dataset.py        # Script kiểm tra tính hợp lệ của dataset
```

---

## 2. Hướng dẫn Tái hiện Kết quả (Reproducible Execution)

### Bước 1: Kiểm toán Môi trường (Phase E0)
```powershell
python evaluation/runners/generate_env_report.py
```

### Bước 2: Xác thực Tính Hợp lệ của Dataset (Phase E1)
```powershell
python evaluation/schemas/validate_dataset.py
```

### Bước 3: Chạy Toàn bộ Ma trận 96 Lượt Đánh giá (Phase E3)
```powershell
python evaluation/runners/run_eval.py
```

### Bước 4: Chạy Kiểm thử Chịu lỗi & Độ ổn định (Phase E4)
```powershell
python evaluation/runners/run_reliability.py
```

### Bước 5: Chạy Kiểm thử API & Frontend Journeys (Phase E5)
```powershell
python evaluation/runners/run_api_eval.py
python evaluation/runners/run_frontend_eval.py
```

### Bước 6: Tổng hợp Tất cả Báo cáo (Phase E6)
```powershell
python evaluation/runners/generate_reports.py
```
