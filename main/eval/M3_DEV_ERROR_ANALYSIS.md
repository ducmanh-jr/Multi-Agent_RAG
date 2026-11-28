# Báo cáo Đánh giá Dev & Phân tích Lỗi Gốc — Giai đoạn M3

- Tập dev độc lập: **8 PDF song ngữ (4 EN + 4 VI)** · **24 câu hỏi** phủ đủ 4 chiều (`VI->VI`, `VI->EN`, `EN->VI`, `EN->EN`), tiếng Việt không dấu, công thức xếp tầng/miền xác định, bảng nhiều cột, trang scan OCR, câu hỏi tiếp nối hội thoại và câu bẫy thiếu bằng chứng.
- Cổng Gate M3: **`PASSED`**

## 1. So sánh `sparse → hybrid → hybrid_rerank` trên cùng split dev

| Cấu hình | Page Recall@5 | Page MRR@10 | Page nDCG@10 | Region Recall@5 | Region MRR@10 | Region nDCG@10 | p95 (ms) | Số ca trượt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sparse` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9565 | 0.9679 | 1779.60 | 0 |
| `hybrid` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9420 | 0.9565 | 63.15 | 0 |
| `hybrid_rerank` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9565 | 0.9679 | 109.81 | 0 |

## 2. Đối chiếu 4 tổ hợp ngôn ngữ hỏi / ngôn ngữ nguồn (`hybrid_rerank`)

| Chiều ngôn ngữ | Số câu | Page Recall@5 | Region Recall@5 | Region MRR@10 | Đúng ngôn ngữ trả lời |
|---|---:|---:|---:|---:|---:|
| `VI->VI` | 6 | 1.0000 | 1.0000 | 1.0000 | 100.0% |
| `VI->EN` | 6 | 1.0000 | 1.0000 | 0.9167 | 100.0% |
| `EN->VI` | 6 | 1.0000 | 1.0000 | 1.0000 | 100.0% |
| `EN->EN` | 6 | 1.0000 | 1.0000 | 0.9000 | 100.0% |

## 3. Phân tích lỗi gốc trên tập dev (Root-Cause Error Analysis)

- Độ đúng ngôn ngữ trả lời toàn tập dev: **100.0%**
- Độ đúng số liệu (`cpu_min` offline extractive baseline): **79.0%** (19 ca có đáp án số)
- Độ đúng từ chối câu bẫy thiếu bằng chứng: **100.0%**

### Ca trượt truy hồi ở `sparse` (0 ca)
- Không có ca trượt truy hồi trang/vùng ở ngưỡng `Recall@5`.

### Ca trượt truy hồi ở `hybrid` (0 ca)
- Không có ca trượt truy hồi trang/vùng ở ngưỡng `Recall@5`.

### Ca trượt truy hồi ở `hybrid_rerank` (0 ca)
- Không có ca trượt truy hồi trang/vùng ở ngưỡng `Recall@5`.

### Ca lệch tổng hợp QA cục bộ (`cpu_min` không LLM) (4 ca)
- `m3-vivi-05` (`VI->VI`): extractive_numeric_synthesis_miss
- `m3-vien-04` (`VI->EN`): extractive_numeric_synthesis_miss
- `m3-envi-03` (`EN->VI`): extractive_numeric_synthesis_miss
- `m3-enen-03` (`EN->EN`): extractive_numeric_synthesis_miss

## 4. Khóa cấu hình truy hồi sau M3

- Bật mặc định mở rộng song ngữ VI↔EN (kể cả tiếng Việt không dấu) và truy tìm cấu trúc phương trình/bảng/hình cho mọi chế độ truy hồi.
- Giữ `page_diverse=True`, `dense_weight=0.67`, `dense_rrf_k=2`, `rerank_keep_prefix=1` (bảo vệ hạng đầu từ Hybrid RRF) và fallback an toàn có ghi vết về `sparse` khi worker không khả dụng.
- Luôn giải tham chiếu hội thoại tại `orchestrator_router_node` và lưu `chat_history`, `resolved_query`, `inherited_entities`, `document_scope` vào SQLite checkpoint.
