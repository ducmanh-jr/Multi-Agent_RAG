# Báo cáo Nghiệm thu Giai đoạn M5 — Tập Holdout Đa Tài liệu (12 PDF / 36 Câu hỏi)

- **Trạng thái Gate M5:** **PASSED**
- **Quy mô tập Holdout niêm phong:** **12 họ tài liệu độc lập** (12 PDF đa trang), **36 câu hỏi** đủ 4 chiều VI/EN, bảng nhiều split, công thức có miền xác định, trang scan OCR, phép tính suy ra (`pp` vs `%`), phủ định, hỏi nối tiếp hội thoại, tổng hợp đa PDF và câu bẫy thiếu bằng chứng.
- **Kiểm tra tách biệt họ tài liệu (Zero Leakage):** `disjoint=True` (đã đối chiếu với `42` tài liệu M0/M3/Pilot và `76` bài QASPER holdout — **0 trùng lặp**).
- **Stress-test truy hồi đồng thời trên cả 12 PDF:** Top-1 = **12/12**, Top-3 = **12/12**, Top-5 = **12/12**, fan-out DAG 12 nhánh = `completed` (`12/12`, giữ đủ lineage cả 12 file).

## 1. So sánh các cấu hình trên tập Holdout (chế độ `warm_no_cache`)

| Hệ thống | Lỗi / Tổng | 3-Layer Pass (95% Cluster CI) | Layer 1 | Retrieval Recall | Valid Cited | Abstention | Ngôn ngữ | p50 (s) | p95 (s) | Chi phí / câu đúng (USD) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hybrid_new` | 3/36 | 91.7% [77.8%, 100.0%] | 91.7% | 100.0% | 91.7% | 100.0% | 100.0% | 0.3648 | 0.7595 | \$0.004154 |
| `kaggle_only` | 3/36 | 91.7% [77.8%, 100.0%] | 91.7% | 100.0% | 91.7% | 100.0% | 100.0% | 0.183 | 0.3049 | \$0.000000 |
| `gemini_direct` | 5/36 | 86.1% [77.8%, 94.4%] | 86.1% | 30.1% | 8.3% | 60.0% | 100.0% | 0.0225 | 0.0873 | \$0.007977 |
| `cpu_min` | 19/36 | 47.2% [30.6%, 63.9%] | 47.2% | 93.5% | 47.2% | 80.0% | 69.4% | 0.1432 | 0.2659 | \$0.000000 |

## 2. Kết quả tách riêng theo 3 chế độ Cache (`cold`, `warm_no_cache`, `warm_cached`)

| Chế độ Cache | Hệ thống | 3-Layer Pass | p50 (s) | p95 (s) | Cache Hits | Fallback | Fallback Cached Rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cold` | `hybrid_new` | 91.7% | 0.4165 | 0.9878 | 0 | 0 | 0 |
| `cold` | `kaggle_only` | 91.7% | 0.2226 | 0.2967 | 0 | 0 | 0 |
| `cold` | `gemini_direct` | 86.1% | 0.0201 | 0.0724 | 0 | 0 | 0 |
| `cold` | `cpu_min` | 47.2% | 0.1895 | 0.4074 | 0 | 0 | 0 |
| `warm_no_cache` | `hybrid_new` | 91.7% | 0.3648 | 0.7595 | 0 | 0 | 0 |
| `warm_no_cache` | `kaggle_only` | 91.7% | 0.183 | 0.3049 | 0 | 0 | 0 |
| `warm_no_cache` | `gemini_direct` | 86.1% | 0.0225 | 0.0873 | 0 | 0 | 0 |
| `warm_no_cache` | `cpu_min` | 47.2% | 0.1432 | 0.2659 | 0 | 0 | 0 |
| `warm_cached` | `hybrid_new` | 91.7% | 0.2118 | 0.6059 | 109 | 0 | 0 |
| `warm_cached` | `kaggle_only` | 91.7% | 0.1804 | 0.3003 | 0 | 0 | 0 |
| `warm_cached` | `gemini_direct` | 86.1% | 0.0023 | 0.0041 | 36 | 0 | 0 |
| `warm_cached` | `cpu_min` | 47.2% | 0.1395 | 0.2801 | 0 | 0 | 0 |

## 3. Phân rã `hybrid_new` theo chiều ngôn ngữ & định dạng PDF (`warm_no_cache`)

| Nhóm | Lỗi / Tổng | 3-Layer Pass | Retrieval Recall | Ngôn ngữ |
| --- | ---: | ---: | ---: | ---: |
| `dir:VI->VI` | 0/12 | 100.0% | 100.0% | 100.0% |
| `dir:VI->EN` | 0/7 | 100.0% | 100.0% | 100.0% |
| `dir:EN->VI` | 0/6 | 100.0% | 100.0% | 100.0% |
| `dir:EN->EN` | 3/11 | 72.7% | 100.0% | 100.0% |
| `fmt:equation_domain` | 0/6 | 100.0% | 100.0% | 100.0% |
| `fmt:figure_chart` | 0/3 | 100.0% | 100.0% | 100.0% |
| `fmt:multi_doc` | 2/6 | 66.7% | 100.0% | 100.0% |
| `fmt:ocr_scan` | 0/3 | 100.0% | 100.0% | 100.0% |
| `fmt:standard_text` | 0/3 | 100.0% | 100.0% | 100.0% |
| `fmt:table_multisplit` | 1/12 | 91.7% | 100.0% | 100.0% |
| `fmt:two_column` | 0/3 | 100.0% | 100.0% | 100.0% |

## 4. Đối chứng chấm mù theo cặp (Blind Pairwise Scorecard)

| Cặp so sánh | A thắng | B thắng | Hòa | Win-rate của A (trừ hòa) | Cận dưới 95% không thua của A |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hybrid_new` vs `cpu_min` | 16 | 0 | 20 | 100.0% | 92.0% |
| `hybrid_new` vs `gemini_direct` | 30 | 2 | 4 | 93.8% | 83.5% |
| `kaggle_only` vs `cpu_min` | 16 | 0 | 20 | 100.0% | 92.0% |

## 5. Cổng vận hành & Tái lập Kaggle M5

- **Hai phiên Kaggle khởi động độc lập tái tạo khớp manifest:** `True` (pinned SHA `5993b5aa760f508a93bc3143cea95f443d9c5f1d`, profile `qa_dual_gpu`, `token_separation_enforced=True`).
- **Resume sau ngắt kết nối giữa phiên trên 4 PDF holdout:** `True` (trước ngắt: `['orchestrator_router', 'scout_node', 'retrieval_node', 'reader_node']` -> sau resume: `['summarizer_node', 'summarizer_node', 'fact_checker_node', 'writer_node']`).
- **Chịu lỗi mất Gemini (HTTP 503) & cấm cache fallback:** `True` (`fallback_cache_rows=0`).
- **Quyết định mở M2b:** `activate_m2b_now=False` — Single-page tables, domain-constrained equations, figure legends, OCR scans, and multi-document synthesis all met M5 thresholds without requiring cross-page table stitching or node-edge diagram extraction; defer M2b until live user PDFs show cross-page tables or diagrams driving >5% of errors.
