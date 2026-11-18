# ☁️ Triển Khai Đám Mây & GPU Worker (`kaggle/`)

Thư mục `kaggle/` cung cấp giải pháp thực thi suy luận phân tán trên cụm phần cứng **Kaggle Dual Tesla T4 (2x16GB VRAM = 32GB VRAM)** hoặc **Google Colab**.

---

## 🌟 Tệp Notebook Duy Nhất: 1 Cell Bấm Chạy Lấy Link (`kaggle_master_worker.ipynb`)

Toàn bộ các tác vụ cấu hình phức tạp đã được gói gọn hoàn toàn:

| Tệp Notebook | Trọng Tâm Kỹ Thuật | Trải Nghiệm Người Dùng |
| :--- | :--- | :--- |
| **[`kaggle_master_worker.ipynb`](kaggle_master_worker.ipynb)** 🏆 | **All-in-One 1-Click Master Runner**: Tự động tải mã nguồn, cài đặt môi trường, kéo trọng số 7 mô hình SOTA, kích hoạt Supervisor Gateway và mở Cloudflare Tunnel. | **Chỉ đúng 1 ô code ngắn nhất.** Bấm `Run` -> Nhận ngay link `REMOTE_INFERENCE_URL`. |

---

## 🛠️ Danh Sách Thành Phần Trong Thư Mục `kaggle/`

```text
kaggle/
├── kaggle_master_worker.ipynb   # 🌟 Notebook 1-cell duy nhất khởi chạy toàn bộ GPU
├── run_kaggle.py                # Script điều phối toàn diện: nạp VRAM, Supervisor, Cloudflare
├── kaggle_supervisor_gateway.py # Gateway chuyển đổi nóng (Hot-Swap) giữa 7 Mô Hình SOTA <-> 32B AWQ
├── kaggle_finetune_server.py    # Máy chủ FastAPI điều phối 7 mô hình SOTA (Dual T4)
├── kaggle_server.py             # Máy chủ FastAPI điều phối 3 LoRA adapter baseline
└── README.md                    # Tài liệu hướng dẫn vận hành
```

---

## 🔗 Quy Trình Khởi Chạy (3 Bước Chuẩn)

1. Mở tệp **`kaggle_master_worker.ipynb`** trên **Kaggle**:
   - Phần cứng: **Accelerator: GPU T4 x 2**
   - Mạng: **Internet: On**
2. Bấm nút **Run** ở ô code duy nhất. Màn hình sẽ tự động chạy và in ra hộp thoại:
   ```text
   ============================================================================
   🚀 ĐƯỜNG DẪN KẾT NỐI CHO MULTI-AGENT RAG (SAO CHÉP DÒNG DƯỚI):

       REMOTE_INFERENCE_URL=https://xxxx-xxxx-xxxx.trycloudflare.com

   ============================================================================
   ```
3. Sao chép link `https://xxxx.trycloudflare.com` dán vào tệp `.env` hoặc dán trực tiếp trên giao diện **Web Demo2**.

---

## 🔄 Chuyển Đổi Mô Hình Nóng (Hot-Swap Không Cần Chạy Lại Kaggle)
- Khi khởi động: Mặc định nạp **Cụm 7 Mô Hình SOTA** trên Dual T4.
- Khi muốn chuyển sang **GPU Max 32B AWQ**: Bấm chọn `[GPU Max]` trực tiếp trên giao diện **Web Demo2**. Hệ thống tự động thu hồi VRAM và nạp Qwen-32B qua vLLM. Các nút GPU còn lại trên Web tự động mờ đi và khóa lại để chống tràn bộ nhớ.
- Khi muốn quay về: Bấm `[Đổi về 7 Mô Hình (SOTA)]` ngay trên thanh thông báo ở Web Demo2!
