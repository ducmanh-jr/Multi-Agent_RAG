# DyLoRA: Parameter-Efficient Tuned of Large Language Models with Dynamic Low-Rank Adaptation

**Tác giả:** Mikhail Valerievich Khodak, Vage Egiazarian, Denis Kuznedelev, Andrei Panferov, Maxim Petrov, Pavel Ostyakov  
**Năm xuất bản:** 2023  
**Nguồn công bố:** International Conference on Learning Representations (ICLR 2023)  
**Liên kết tham chiếu:** https://openreview.net/forum?id=JsZHcGRDyhG  

## Tóm tắt nội dung nghiên cứu (Abstract)
Công trình này đề xuất DyLoRA (Dynamic Low-Rank Adaptation), một phần mở rộng cải tiến của phương pháp LoRA truyền thống nhằm khắc phục hạn chế về việc phải cố định trước hạng (rank) r của ma trận thích ứng. Trong phương pháp LoRA gốc, nếu người dùng muốn thay đổi kích thước hạng r để cân bằng giữa chi phí tính toán và độ chính xác của mô hình, họ phải huấn luyện lại mô hình từ đầu. DyLoRA giải quyết vấn đề này bằng cách huấn luyện các mô hình LoRA trên một dải các giá trị hạng khác nhau cùng một lúc thông qua kỹ thuật phân phối ngẫu nhiên các hàng và cột trong quá trình tối ưu hóa. Kết quả thực nghiệm trên nhiều tác vụ khác nhau cho thấy DyLoRA không chỉ mang lại sự linh hoạt vượt trội khi cho phép trích xuất các mô hình con với các hạng khác nhau mà không cần huấn luyện lại, mà còn đạt hiệu suất tương đương hoặc tốt hơn LoRA tiêu chuẩn trên cùng một ngân sách tính toán.

## Bối cảnh và Giá trị khoa học
Công trình nghiên cứu này đóng góp luận điểm khoa học và bằng chứng lịch sử/thực nghiệm quan trọng đối với chủ đề "Khái niệm LoRA". Dữ liệu đã được xác thực qua hội đồng biên tập học thuật của International Conference on Learning Representations (ICLR 2023).
