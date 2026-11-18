# 📦 Local Models Directory (`main/models`)

Thư mục này lưu trữ các mô hình trọng số cục bộ (offline) phục vụ trực tiếp cho các tác vụ của hệ thống **Multi-Agent Research RAG**.

---

## 📌 Danh sách mô hình hiện có

### 1. `multilingual-e5-small`
- **Loại mô hình**: SentenceTransformer / Dense Text Embedding (Đa ngôn ngữ: Anh, Việt, v.v.).
- **Dung lượng**: ~493 MB.
- **Agent sử dụng**: **Agent 0 (Scout)** tại [`main/agents/agent0_scout/paper_ranker.py`](../agents/agent0_scout/paper_ranker.py).
- **Mục đích & Chức năng**:
  - Dùng để tính toán độ tương đồng ngữ nghĩa (*Cosine Similarity*) giữa câu truy vấn của người dùng và các bài báo khoa học thu thập được từ ArXiv.
  - Hoạt động như một bộ lọc thông minh (*Semantic Relevance Gatekeeper*), loại bỏ triệt để các bài báo trùng từ khóa nhưng sai chủ đề (tránh xung đột nghĩa).
  - Giúp **Agent 0** chạy **100% Offline** mượt mà, không cần tải lại mô hình `all-MiniLM-L6-v2` từ Hugging Face qua internet mỗi khi khởi động.

---

## ⚠️ Lưu ý quan trọng
- **KHÔNG XÓA** thư mục này để tránh việc Agent 0 phải tải lại mô hình qua mạng.
- **Phân biệt với RAG Retrieval chính**: 
  - Mô hình tại đây (`multilingual-e5-small`) chỉ dùng riêng cho **Agent 0 (Scout Paper Ranking)**.
  - Khâu tìm kiếm nội dung tài liệu RAG chính (Agent 1) sử dụng **BGE-M3** và **BGE-Reranker** (hoặc các adapter LoRA chuyên biệt tại `training_lab/final_weights/`).
