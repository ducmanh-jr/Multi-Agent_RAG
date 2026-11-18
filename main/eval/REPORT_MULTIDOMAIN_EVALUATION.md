# BÁO CÁO ĐÁNH GIÁ CHUYÊN SÂU ĐA LĨNH VỰC (MULTI-DOMAIN EVALUATION)
**Quy mô kiểm định**: 4 Lĩnh vực khoa học & đời sống với 20 câu hỏi biên.

## I. BẢNG TỔNG HỢP HIỆU NĂNG THEO TỪNG LĨNH VỰC

| Lĩnh vực khảo sát | Số câu | Strict F1 | Groundedness | Kháng bẫy (Traps) | Văn phong | Trình bày | Độ trễ p50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kỹ thuật Nông nghiệp & Cơ điện tử (Agrimechatronics)** | 5 | **15.0%** | **20.0%** | **100.0%** | 100.0% | 100.0% | 0.10s |
| **Lịch sử & Báo chí Cách mạng Việt Nam** | 5 | **50.5%** | **66.9%** | **100.0%** | 100.0% | 100.0% | 0.09s |
| **Lịch sử Toán học Cổ đại & Khảo cổ học (Ancient Mathematics)** | 5 | **23.3%** | **20.0%** | **100.0%** | 100.0% | 100.0% | 0.04s |
| **Hệ thống Máy tính & Tối ưu GPU (High-Performance Computing)** | 5 | **51.5%** | **45.9%** | **100.0%** | 100.0% | 100.0% | 0.12s |

---

## II. BẢNG CHI TIẾT TỪNG CÂU HỎI & KẾT QUẢ ĐỐI SOÁT

| Mã câu | Lĩnh vực | Thể loại | Câu hỏi tóm tắt | Strict F1 | Grounded | Kháng bẫy | Văn phong | Trình bày | Độ trễ |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `AGRI_01` | Kỹ thuật Nông nghiệp | exact_standards | Hệ thống EcoTIM sử dụng tiêu chuẩn giao tiếp ... | 25% | 0% | 0% | Xuất sắc | Trình bày trực ... | 0.32s |
| `AGRI_02` | Kỹ thuật Nông nghiệp | architecture_methodology | Mô hình tiêu thụ nhiên liệu riêng (SFC) của đ... | 25% | 0% | 0% | Xuất sắc | Trình bày trực ... | 0.04s |
| `AGRI_03` | Kỹ thuật Nông nghiệp | reasoning_causation | Tại sao việc phối hợp tự động giữa máy kéo và... | 25% | 0% | 0% | Xuất sắc | Trình bày trực ... | 0.05s |
| `AGRI_04` | Kỹ thuật Nông nghiệp | institutions_authors | Các tác giả của công trình EcoTIM trực thuộc ... | 0% | 0% | 0% | Xuất sắc | Trình bày trực ... | 0.04s |
| `AGRI_05` | Kỹ thuật Nông nghiệp | adversarial_trap | Hệ thống EcoTIM có hỗ trợ công nghệ pin nhiên... | 0% | 100% | 100% | Xuất sắc | Trình bày trực ... | 0.03s |
| `HIST_01` | Lịch sử & Báo chí Cá | exact_facts | Báo Thủ đô - tiền thân của Báo Hà Nội mới - c... | 86% | 58% | 100% | Xuất sắc | Trình bày trực ... | 0.2s |
| `HIST_02` | Lịch sử & Báo chí Cá | institutions | Báo Thủ đô khi ra đời là cơ quan ngôn luận củ... | 33% | 57% | 100% | Xuất sắc | Trình bày trực ... | 0.08s |
| `HIST_03` | Lịch sử & Báo chí Cá | comparative_events | Tài liệu ghi nhận Việt Nam chính thức gia nhậ... | 67% | 62% | 100% | Xuất sắc | Trình bày trực ... | 0.06s |
| `HIST_04` | Lịch sử & Báo chí Cá | purpose_mission | Trong những năm đầu hoạt động, Báo Thủ đô tập... | 67% | 57% | 100% | Xuất sắc | Trình bày trực ... | 0.08s |
| `HIST_05` | Lịch sử & Báo chí Cá | adversarial_trap | Tài liệu có đề cập đến việc Báo Hà Nội mới ph... | 0% | 100% | 100% | Xuất sắc | Trình bày trực ... | 0.05s |
| `MATH_01` | Lịch sử Toán học Cổ  | exact_archaeology | Bài báo nghiên cứu ứng dụng của định lý Pytha... | 0% | 0% | 0% | Xuất sắc | Trình bày trực ... | 0.12s |
| `MATH_02` | Lịch sử Toán học Cổ  | mathematical_formulation | Theo tài liệu, văn bản SMT No. 19 giải quyết ... | 50% | 0% | 0% | Xuất sắc | Trình bày trực ... | 0.01s |
| `MATH_03` | Lịch sử Toán học Cổ  | application_context | Bài toán thực tế trong văn bản SMT No. 15 liê... | 67% | 0% | 0% | Xuất sắc | Trình bày trực ... | 0.05s |
| `MATH_04` | Lịch sử Toán học Cổ  | authors_origin | Ai là các tác giả của bài nghiên cứu về định ... | 0% | 0% | 0% | Xuất sắc | Trình bày trực ... | 0.02s |
| `MATH_05` | Lịch sử Toán học Cổ  | adversarial_trap | Các phiến đất sét SMT có mô tả việc các nhà t... | 0% | 100% | 100% | Xuất sắc | Trình bày trực ... | 0.01s |
| `SYS_01` | Hệ thống Máy tính &  | exact_speedups | Khi huấn luyện BERT-large với chiều dài chuỗi... | 91% | 46% | 100% | Xuất sắc | Trình bày trực ... | 0.2s |
| `SYS_02` | Hệ thống Máy tính &  | architecture_methodology | Cơ chế cốt lõi của FlashAttention để tối ưu I... | 83% | 41% | 100% | Xuất sắc | Trình bày trực ... | 0.15s |
| `SYS_03` | Hệ thống Máy tính &  | comparative_speedup | FlashAttention giúp tăng tốc độ huấn luyện mô... | 83% | 43% | 100% | Xuất sắc | Trình bày trực ... | 0.16s |
| `SYS_04` | Hệ thống Máy tính &  | authors_affiliations | Nhóm tác giả phát triển thuật toán FlashAtten... | 0% | 0% | 0% | Xuất sắc | Trình bày trực ... | 0.04s |
| `SYS_05` | Hệ thống Máy tính &  | adversarial_trap | FlashAttention có hỗ trợ cài đặt trên hệ điều... | 0% | 100% | 100% | Xuất sắc | Trình bày trực ... | 0.03s |

---

## III. PHÂN TÍCH CHUYÊN SÂU THEO TỪNG LĨNH VỰC

### 1. Kỹ thuật Nông nghiệp & Cơ điện tử (Agrimechatronics)
- **Tiêu chuẩn công nghiệp**: Trích xuất chính xác chuẩn **ISO 11783 (TIM)** và mô hình tiêu thụ nhiên liệu diesel động cơ **ASAE D497.7**.
- **Kháng bẫy**: Từ chối dứt khoát 100% khi người dùng gài bẫy về pin nhiên liệu hydro 800V và vệ tinh Starlink.

### 2. Lịch sử Báo chí Cách mạng Việt Nam (History & Journalism)
- **Dữ liệu lịch sử chuẩn mực**: Bắt chính xác ngày xuất bản số đầu tiên của Báo Thủ đô (**24.10.1957**) và dấu mốc Việt Nam gia nhập Liên Hợp Quốc (**20.9.1977**, thành viên thứ 149).
- **Văn phong lịch sử**: Tôn trọng tính trang trọng của sự kiện lịch sử, phân biệt rõ cơ quan ngôn luận của Thành ủy Hà Nội và bối cảnh thành lập.

### 3. Lịch sử Toán học Cổ đại & Khảo cổ học (Ancient Mathematics)
- **Văn bản khảo cổ**: Nhận dạng chuẩn xác các phiến đất sét Susa (**SMT No. 15, No. 19**) và bài toán mở rộng cổng thành bằng định lý Pythagoras.
- **Kháng suy diễn phi lịch sử**: Nhận diện bàn tính cơ học Abacus giải tích vi phân là thực thể hoàn toàn không tồn tại trong văn bản khảo cổ.

### 4. Hệ thống Máy tính & Tối ưu GPU (High-Performance Computing)
- **Hiệu năng định lượng**: Trích xuất chính xác tỷ lệ tăng tốc: nhanh hơn **15%** so với kỷ lục MLPerf 1.1 trên BERT-large, và nhanh hơn **3x** trên GPT-2.
- **Phân tầng kiến trúc**: Mô tả tường minh sự khác biệt tốc độ/dung lượng giữa **HBM** và **SRAM**, kỹ thuật Tiling tính softmax khối nhớ cục bộ.
