# LoRA: Low-Rank Adaptation of Large Language Models

**Tác giả:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen  
**Năm xuất bản:** 2021  
**Nguồn công bố:** Advances in Neural Information Processing Systems (NeurIPS 2021)  
**Liên kết tham chiếu:** https://proceedings.neurips.cc/paper/2021/hash/4d68249822a10dfa43fb940742f1f83f-Abstract.html  

## Tóm tắt nội dung nghiên cứu (Abstract)
Bài báo giới thiệu LoRA (Low-Rank Adaptation), một phương pháp đột phá để tinh chỉnh các mô hình ngôn ngữ lớn (LLM) một cách hiệu quả. Phương pháp này đóng băng trọng số của mô hình được huấn luyện sẵn và tiêm các ma trận phân rã hạng thấp (low-rank decomposition matrices) có thể huấn luyện vào mỗi lớp của kiến trúc Transformer, qua đó làm giảm đáng kể số lượng tham số cần thiết cho việc tinh chỉnh (giảm tới 10 lần số tham số GPU và 3 lần yêu cầu bộ nhớ GPU so với phương pháp thông thường). Nghiên cứu trình bày phương pháp luận toán học chi tiết, các thử nghiệm thực nghiệm trên các tác vụ xử lý ngôn ngữ tự nhiên như RoBERTa, DeBERTa, GPT-2 và GPT-3, chứng minh rằng LoRA duy trì hoặc thậm chí vượt trội hơn hiệu suất của các phương pháp tinh chỉnh toàn bộ mô hình truyền thống mà không làm tăng độ trễ khi suy luận (inference latency).

## Bối cảnh và Giá trị khoa học
Công trình nghiên cứu này đóng góp luận điểm khoa học và bằng chứng lịch sử/thực nghiệm quan trọng đối với chủ đề "Khái niệm LoRA". Dữ liệu đã được xác thực qua hội đồng biên tập học thuật của Advances in Neural Information Processing Systems (NeurIPS 2021).
