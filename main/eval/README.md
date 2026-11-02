# 📏 Bộ Kiểm Thử Đánh Giá 120 Câu Hỏi Song Ngữ (`main/eval/`)

Thư mục `main/eval/` chứa bộ khung đánh giá định lượng toàn diện cho cả 6 tùy chọn thực thi của hệ thống.

---

## 📦 Cấu Trúc Thư Mục `main/eval/`

- **[`benchmark_120.jsonl`](benchmark_120.jsonl)**: Bộ 120 câu hỏi kiểm thử song ngữ (**84 câu Tiếng Việt — 70% / 36 câu Tiếng Anh — 30%**) được trích xuất trực tiếp từ các tài liệu PDF thực tế trong `main/uploads/`, chia đều thành 4 danh mục:
  1. `single_hop` (30 câu): Truy xuất sự kiện, thông số kỹ thuật và công thức đơn bước.
  2. `multi_hop` (30 câu): Suy luận tổng hợp kết nối nhiều trang/mục khác nhau trong tài liệu.
  3. `table_comparison` (30 câu): So sánh bảng biểu, kiến trúc và kết quả thực nghiệm.
  4. `adversarial_trap` (30 câu): Câu hỏi bẫy chứa thực thể/con số giả định không tồn tại trong tài liệu để kiểm tra khả năng chống ảo giác (Abstention).
- **[`generate_benchmark.py`](generate_benchmark.py)**: Script tự động tái tạo bộ câu hỏi `benchmark_120.jsonl` từ kho PDF.
- **[`metrics.py`](metrics.py)**: Định nghĩa 6 thước đo chuẩn:
  1. **Groundedness (Faithfulness)**: Tỷ lệ mệnh đề được chứng minh bởi bằng chứng gốc.
  2. **Citation F1**: Độ chính xác và độ phủ của ký hiệu trích dẫn `[1]`, `[2]`.
  3. **Numeric/Date Exactness**: Độ khớp tuyệt đối của các con số, phần trăm, năm tháng.
  4. **Abstention Accuracy**: Độ chính xác khi từ chối/bác bỏ thông tin giả định trong câu hỏi bẫy.
  5. **Latency (`p50`, `p95`)**: Thời gian phản hồi trung vị và phân vị 95%.
- **[`run_benchmark.py`](run_benchmark.py)**: Trình chạy đánh giá tự động trên cả 6 chế độ và xuất báo cáo [`benchmark_report.md`](benchmark_report.md).

---

## ▶️ Cách Chạy Đánh Giá

```bash
# Chạy kiểm thử nhanh (4 câu đại diện cho 4 nhóm trên cả 6 chế độ)
py -3.10 -X utf8 main/eval/run_benchmark.py --limit-per-cat 1

# Chạy toàn bộ 120 câu hỏi cho một hoặc nhiều chế độ cụ thể
py -3.10 -X utf8 main/eval/run_benchmark.py --modes cpu_min cpu_max hybrid --limit-per-cat 0
```
