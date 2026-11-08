# ☁️ Triển Khai Đám Mây & GPU Worker (`kaggle/`)

Thư mục `kaggle/` cung cấp các Notebook `.ipynb` tự động hóa và máy chủ FastAPI phục vụ việc thực thi suy luận phân tán trên cụm phần cứng **Kaggle Dual Tesla T4 (2x16GB VRAM)** hoặc **Google Colab**.

---

## 📓 Danh Sách Thành Phần Vận Hành Trong `kaggle/`

| Tệp Notebook / Script | Trọng Tâm Kỹ Thuật | Chế Độ Phục Vụ |
| :--- | :--- | :--- |
| **[`kaggle_finetune_worker.ipynb`](kaggle_finetune_worker.ipynb)** | **GPU Worker Kiến trúc Tinh chỉnh Toàn diện (Full 7-Model Suite)**: Tự động tải gói trọng số `final_weights_bundle.zip` (2.85 GB) từ Google Drive, phân bổ song song 7 mô hình lên 2x GPU T4 và thiết lập đường hầm bảo mật Cloudflare Tunnel. | **`gpu_finetune`** & **`hybrid`** |
| **[`kaggle_finetune_server.py`](kaggle_finetune_server.py)** | Máy chủ FastAPI chuyên dụng điều phối song song: **GPU 0** đảm nhiệm Qwen2.5-7B (4-bit NF4) nạp động 4 LoRA (`router`, `generator`, `critic`, `writer`); **GPU 1** đảm nhiệm BGE NLI Verifier (Macro F1 = 1.0000), LayoutLMv3 Base, BGE-M3 MoAI LoRA và CoREB Reranker LoRA. | Khởi chạy tự động bởi `kaggle_finetune_worker.ipynb` |
| **[`kaggle_worker.ipynb`](kaggle_worker.ipynb)** | **GPU Worker Kiến trúc Tinh chỉnh Tiêu chuẩn (Standard 3-Adapter Baseline)**: Nạp 3 bộ QLoRA (`router`, `generator`, `critic`) phục vụ mục đích kiểm thử đối chứng và đo lường thực nghiệm (Ablation Baseline). | **`gpu_ft`** |
| **[`kaggle_server.py`](kaggle_server.py)** | Máy chủ FastAPI cho kiến trúc tinh chỉnh tiêu chuẩn 3 adapter. | Khởi chạy tự động bởi `kaggle_worker.ipynb` |
| **[`kaggle_max_32b_vllm.ipynb`](kaggle_max_32b_vllm.ipynb)** | **GPU Worker Đại Mô hình Suy luận 32B**: Chạy `Qwen2.5-32B-Instruct-AWQ` qua engine suy luận vLLM với cấu hình Tensor Parallelism ($TP=2$), hỗ trợ phân nhánh chuỗi tư duy `<think>...</think>`. | **`gpu_max`** |

---

## 📦 Phân Định Các Gói Trọng Số Lưu Trữ Trên Google Drive

Hệ thống phân tách rành mạch hai gói trọng số độc lập phục vụ các kiến trúc suy luận tương ứng:

| Phân Loại Kiến Trúc | Google Drive Folder & Định Danh | Định Dạng Lưu Trữ | Danh Mục Mô Hình Thành Phần |
| :--- | :--- | :--- | :--- |
| **Kiến trúc Tinh chỉnh Toàn diện 7 Mô hình (Full 7-Model Suite)** | [`1UPF13BNjxFFhd8YFv5PpuyfOR4G-VNte`](https://drive.google.com/drive/folders/1UPF13BNjxFFhd8YFv5PpuyfOR4G-VNte?usp=sharing) | `final_weights_bundle.zip` (2.85 GB) | **Trọn bộ 7 mô hình chuyên sâu**: 01 Router LoRA, 02 LayoutLMv3 Base, 03 Dual-QLoRA Summarizer/Critic, 04 Calibrated BGE NLI Head (F1 = 1.0000), 05 DoRA Academic Writer (99.6%), 06 BGE-M3 Dense Retriever LoRA, 07 CoREB Cross-Encoder Reranker LoRA. |
| **Kiến trúc Tinh chỉnh Tiêu chuẩn 3 Adapter (Standard 3-Adapter Baseline)** | [`1upmXzTIlxpcCVGU8Vp9ZGV3xNa1yu-Q6`](https://drive.google.com/drive/folders/1upmXzTIlxpcCVGU8Vp9ZGV3xNa1yu-Q6?usp=sharing) | `weights_bundle.zip` | 3 LoRA adapters tiêu chuẩn phục vụ thử nghiệm đối chứng (`router`, `generator`, `critic`). |

---

## 🔗 Quy Trình Kích Hoạt Vận Hành Phân Tán (3 Bước Chuẩn)

1. Mở tệp **`kaggle_finetune_worker.ipynb`** trên môi trường **Kaggle Notebook**:
   - Thiết lập phần cứng: **Accelerator: GPU T4 x 2**
   - Thiết lập mạng: **Internet: On**
2. Thực thi toàn bộ (**Run All**). Hệ thống tự động đồng bộ hóa trọng số, cấu hình bộ đệm GPU và xuất bản địa chỉ Cloudflare Tunnel:
   ```text
   🚀 REMOTE_INFERENCE_URL CHO MULTI-AGENT RAG:
       REMOTE_INFERENCE_URL=https://xxxx-xxxx-xxxx.trycloudflare.com
   ```
3. Cập nhật biến môi trường tại tệp [`.env`](../.env) ở máy trạm cục bộ:
   ```env
   REMOTE_INFERENCE_URL=https://xxxx-xxxx-xxxx.trycloudflare.com
   ```
   Hệ thống sẵn sàng kích hoạt các chế độ `gpu_finetune`, `hybrid` hoặc `gpu_ft`.
