# QLoRA: Efficient Finetuning of Quantized LLMs

**Tác giả:** Tim Dettmers, Artidoro Pagnoni, Holger Holtzman, Luke Zettlemoyer  
**Năm xuất bản:** 2023  
**Nguồn công bố:** Conference on Neural Information Processing Systems (NeurIPS 2023)  
**Liên kết tham chiếu:** https://arxiv.org/abs/2305.14314  

## Tóm tắt nội dung nghiên cứu (Abstract)
Nghiên cứu này đề xuất QLoRA, một phương pháp tinh chỉnh hiệu quả giúp giảm đáng kể lượng bộ nhớ sử dụng để huấn luyện các mô hình ngôn ngữ lớn, cho phép tinh chỉnh mô hình 65 tỷ tham số trên một GPU duy nhất loại 48GB mà vẫn giữ nguyên độ chính xác. QLoRA giới thiệu các đổi mới kỹ thuật cốt lõi bao gồm: định lượng 4-bit bình thường (NormalFloat4 - NF4), việc sử dụng double quantization để tiết kiệm bộ nhớ phụ trợ, và các trang tính trang thái tối ưu hóa (Paged Optimizers) để quản lý hiện tượng tràn bộ nhớ khi xử lý các chuỗi dữ liệu dài. Bài báo cung cấp đánh giá thực nghiệm sâu rộng trên các chuẩn mực hội thoại và suy luận, chứng minh rằng mô hình Guanaco được tinh chỉnh bằng QLoRA đạt hiệu suất tương đương 99.3% so với chuẩn ChatGPT trên các tập dữ liệu benchmark tiêu chuẩn, đồng thời giải quyết bài toán rào cản tài nguyên phần cứng trong nghiên cứu trí tuệ nhân tạo.

## Bối cảnh và Giá trị khoa học
Công trình nghiên cứu này đóng góp luận điểm khoa học và bằng chứng lịch sử/thực nghiệm quan trọng đối với chủ đề "Khái niệm LoRA". Dữ liệu đã được xác thực qua hội đồng biên tập học thuật của Conference on Neural Information Processing Systems (NeurIPS 2023).
