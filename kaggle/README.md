# ☁️ Triển Khai Đám Mây & GPU Worker (`kaggle/`)

Thư mục `kaggle/` chứa **3 Notebook `.ipynb` tự động hóa 100%** (tự động `git clone` từ GitHub và nạp trọng số từ kho lưu trữ / Google Drive) giúp bạn chạy toàn bộ dự án hoặc chạy máy chủ suy luận GPU từ xa trên **Kaggle (Dual Tesla T4 32GB VRAM)** hoặc **Google Colab**.

---

## 📓 Danh Sách 3 Notebook Trong `kaggle/`

| Tệp Notebook / Script | Mục đích sử dụng | Khi nào nên chạy? |
| :--- | :--- | :--- |
| **[`01_Run_Full_Project_Cloud.ipynb`](01_Run_Full_Project_Cloud.ipynb)** | **Chạy TRỌN BỘ cả dự án trên đám mây**: Khởi chạy đồng thời cả GPU Worker (cổng `8001`) lẫn Web Workspace chính (`main/main.py --serve` cổng `8000`) và xuất đường link công khai `https://xxxx.trycloudflare.com/demo2/`. | Khi bạn muốn sử dụng toàn bộ Web UI + 6 chế độ trên trình duyệt mà **không cần bật máy cá nhân**. |
| **[`kaggle_worker.ipynb`](kaggle_worker.ipynb)** | **GPU Worker cho chế độ `Kaggle FN` (Multi-LoRA 7B)**: Tự động `git clone` dự án, nạp 3 bộ QLoRA Adapter (`router`, `generator`, `critic`) từ `training_lab/weights_drive_bundle/` (hoặc Google Drive) và xuất link Cloudflare Tunnel. | Khi bạn chạy giao diện Web trên máy cá nhân (`localhost:8000/demo2/`) và muốn kết nối chế độ **`Kaggle FN`** hoặc **`Hybrid`**. |
| **[`kaggle_max_32b_vllm.ipynb`](kaggle_max_32b_vllm.ipynb)** | **GPU Worker cho chế độ `Kaggle Max` (32B AWQ Reasoning)**: Chạy mô hình suy luận **`Qwen/Qwen3-32B-AWQ`** (hoặc `QwQ-32B-AWQ`) với Tensor Parallelism `TP=2` trên 2 GPU Tesla T4 (`32GB VRAM`) bằng `vLLM`. | Khi bạn muốn sử dụng sức mạnh suy luận chuỗi tư duy `<think>...</think>` của đại mô hình 32B trong chế độ **`Kaggle Max`** hoặc **`Hybrid`**. |
| **[`kaggle_server.py`](kaggle_server.py)** | Mã nguồn FastAPI Server quản lý nạp mô hình Qwen2.5-7B 4-bit, chuyển đổi linh hoạt giữa các LoRA Adapters (`set_adapter`) và cung cấp các endpoint `/api/router`, `/api/summarize/generator`, `/api/summarize/critic`, `/api/nli`. | Được gọi tự động bởi các Notebook trên. |

---

## 🔗 Hướng Dẫn Kết Nối Với Máy Cá Nhân Trong 3 Bước

1. Mở 1 trong 3 file `.ipynb` ở trên trong **Kaggle Notebook** (chọn Accelerator: **GPU T4 x2** và bật **Internet: On**).
2. Bấm **Run All** và đợi khoảng 2–3 phút đến khi màn hình in ra đường dẫn:
   ```text
   ✅ KAGGLE WORKER ONLINE: https://xxxx-yyyy-zzzz.trycloudflare.com
   ```
3. Mở giao diện Web tại `http://localhost:8000/demo2/` (nếu dùng `02` hoặc `03`), dán đường dẫn trên vào mục **Kaggle GPU Tunnel URL** (hoặc cập nhật biến `REMOTE_INFERENCE_URL` trong `.env`).
