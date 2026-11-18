# BÁO CÁO SO SÁNH & ĐÁNH GIÁ 6 TÙY CHỌN MÔ HÌNH MULTI-AGENT RAG
- **Thời gian khảo nghiệm**: 2026-11-03 02:44:04
- **Tài liệu kiểm định**: Các bài báo trong thư mục `bao/` (*Báo Hà Nội mới*, *Chất vấn Môi trường HĐND*)

## 1. Bảng Tổng Hợp Hiệu Năng & Độ Chính Xác Giữa 6 Tùy Chọn

| Tùy chọn Mô hình | Độ khớp Thực thể TB | Thời gian TB | Trích dẫn TB | Đánh giá thế mạnh & Ứng dụng |
| :--- | :---: | :---: | :---: | :--- |
| **CPU Min** | **68.8%** | 1.1s | 4.0 | Siêu tốc, 100% offline, chuẩn xác số liệu và tên người, tiết kiệm tài nguyên tuyệt đối. |
| **CPU Max** | **43.8%** | 14.7s | 2.8 | Suy luận sâu (DeepSeek-R1 14B), kiểm định thực thể bằng MiniCheck cục bộ. |
| **Kaggle FN** | **0.0%** | 62.3s | 0.0 | Cụm GPU Dual T4 Qwen2.5-7B, trích xuất cấu trúc văn bản học thuật chuyên sâu. |
| **Kaggle Max** | **0.0%** | 62.0s | 0.0 | Mô hình quy mô lớn trên đám mây, phù hợp báo cáo đa bài báo tổng hợp dài. |
| **Gemini API** | **43.8%** | 13.8s | 2.8 | Google Cloud, văn phong tự nhiên, tóm lược mượt mà như ChatGPT / Gemini chính thức. |
| **Hybrid** | **43.8%** | 20.2s | 2.8 | Tối ưu nhất: Kết hợp tốc độ của C-Engine với sự sâu sắc của AI đám mây. |

## 2. Chi Tiết Câu Trả Lời Của Từng Mô Hình Theo Từng Câu Hỏi

| Mô hình | Câu hỏi | Trạng thái | Thời gian | Trích dẫn | Độ khớp | Trích đoạn câu trả lời |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `CPU Min` | Báo Thủ đô - tiền thân của báo Hà N... | `completed` | 1.1s | 4 | **100%** | Tóm Tắt Tổng Quan (Executive Summary) Dựa trên tài liệu nghiên cứu (**Báo Hà Nội mới ra số đầu tiên (1)**): ⚠️ **Tài liệ... |
| `CPU Min` | Mục tiêu thành lập của Liên Hợp Quố... | `completed` | 1.0s | 4 | **100%** | Tóm Tắt Tổng Quan (Executive Summary) Dựa trên tài liệu nghiên cứu (**Báo Hà Nội mới ra số đầu tiên (1)**): - Liên Hợp Q... |
| `CPU Min` | Nội dung phiên chất vấn tập trung v... | `completed` | 1.1s | 4 | **0%** | Tóm Tắt Tổng Quan (Executive Summary) Dựa trên tài liệu nghiên cứu (**Thường trực Hội đồng nhân dân thành phố Hà Nội tổ ... |
| `CPU Min` | Chủ tịch HĐND thành phố Hà Nội Phùn... | `completed` | 1.1s | 4 | **75%** | Tóm Tắt Tổng Quan (Executive Summary) Dựa trên tài liệu nghiên cứu (**Thường trực Hội đồng nhân dân thành phố Hà Nội tổ ... |
| `CPU Max` | Báo Thủ đô - tiền thân của báo Hà N... | `completed` | 15.9s | 4 | **100%** | Tóm tắt Tổng quan (Executive Summary) Dựa trên tài liệu đã chọn (**Báo Hà Nội mới ra số đầu tiên (1)**): • Ngày 24.10.19... |
| `CPU Max` | Mục tiêu thành lập của Liên Hợp Quố... | `completed` | 15.8s | 1 | **25%** | Tóm tắt Tổng quan (Executive Summary) Dựa trên tài liệu đã chọn (**Báo Hà Nội mới ra số đầu tiên (1)**): • Mỹ và nhiều q... |
| `CPU Max` | Nội dung phiên chất vấn tập trung v... | `completed` | 15.5s | 3 | **0%** | Tóm tắt Tổng quan (Executive Summary) Dưới đây là tóm tắt các phát hiện và nội dung khoa học trọng tâm từ (**Thường trực... |
| `CPU Max` | Chủ tịch HĐND thành phố Hà Nội Phùn... | `completed` | 11.4s | 3 | **50%** | Tóm tắt Tổng quan (Executive Summary) Dưới đây là tóm tắt các phát hiện và nội dung khoa học trọng tâm từ (**Thường trực... |
| `Kaggle FN` | Báo Thủ đô - tiền thân của báo Hà N... | `processing` | 62.0s | 0 | **0%** | Chưa có phản hồi... |
| `Kaggle FN` | Mục tiêu thành lập của Liên Hợp Quố... | `processing` | 62.6s | 0 | **0%** | Chưa có phản hồi... |
| `Kaggle FN` | Nội dung phiên chất vấn tập trung v... | `processing` | 62.4s | 0 | **0%** | Chưa có phản hồi... |
| `Kaggle FN` | Chủ tịch HĐND thành phố Hà Nội Phùn... | `processing` | 62.1s | 0 | **0%** | Chưa có phản hồi... |
| `Kaggle Max` | Báo Thủ đô - tiền thân của báo Hà N... | `processing` | 61.8s | 0 | **0%** | Chưa có phản hồi... |
| `Kaggle Max` | Mục tiêu thành lập của Liên Hợp Quố... | `processing` | 62.1s | 0 | **0%** | Chưa có phản hồi... |
| `Kaggle Max` | Nội dung phiên chất vấn tập trung v... | `processing` | 61.9s | 0 | **0%** | Chưa có phản hồi... |
| `Kaggle Max` | Chủ tịch HĐND thành phố Hà Nội Phùn... | `processing` | 62.1s | 0 | **0%** | Chưa có phản hồi... |
| `Gemini API` | Báo Thủ đô - tiền thân của báo Hà N... | `completed` | 39.4s | 4 | **100%** | Tóm tắt Tổng quan (Executive Summary) Dựa trên tài liệu đã chọn (**Báo Hà Nội mới ra số đầu tiên (1)**): • Ngày 24.10.19... |
| `Gemini API` | Mục tiêu thành lập của Liên Hợp Quố... | `completed` | 5.2s | 1 | **25%** | Tóm tắt Tổng quan (Executive Summary) Dựa trên tài liệu đã chọn (**Báo Hà Nội mới ra số đầu tiên (1)**): • Mỹ và nhiều q... |
| `Gemini API` | Nội dung phiên chất vấn tập trung v... | `completed` | 4.1s | 3 | **0%** | Tóm tắt Tổng quan (Executive Summary) Dưới đây là tóm tắt các phát hiện và nội dung khoa học trọng tâm từ (**Thường trực... |
| `Gemini API` | Chủ tịch HĐND thành phố Hà Nội Phùn... | `completed` | 6.3s | 3 | **50%** | Tóm tắt Tổng quan (Executive Summary) Dưới đây là tóm tắt các phát hiện và nội dung khoa học trọng tâm từ (**Thường trực... |
| `Hybrid` | Báo Thủ đô - tiền thân của báo Hà N... | `completed` | 12.4s | 4 | **100%** | Tóm tắt Tổng quan (Executive Summary) Dựa trên tài liệu đã chọn (**Báo Hà Nội mới ra số đầu tiên (1)**): • Ngày 24.10.19... |
| `Hybrid` | Mục tiêu thành lập của Liên Hợp Quố... | `completed` | 20.8s | 1 | **25%** | Tóm tắt Tổng quan (Executive Summary) Dựa trên tài liệu đã chọn (**Báo Hà Nội mới ra số đầu tiên (1)**): • Mỹ và nhiều q... |
| `Hybrid` | Nội dung phiên chất vấn tập trung v... | `completed` | 12.5s | 3 | **0%** | Tóm tắt Tổng quan (Executive Summary) Dưới đây là tóm tắt các phát hiện và nội dung khoa học trọng tâm từ (**Thường trực... |
| `Hybrid` | Chủ tịch HĐND thành phố Hà Nội Phùn... | `completed` | 35.2s | 3 | **50%** | Tóm tắt Tổng quan (Executive Summary) Dưới đây là tóm tắt các phát hiện và nội dung khoa học trọng tâm từ (**Thường trực... |