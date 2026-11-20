# Báo Cáo Thẩm Định Kiểm Thử API & Frontend (API & Frontend Validation Report)

**Thời điểm:** 2026-11-20  
**Tài liệu tham chiếu:** `evaluation/runners/run_api_eval.py`, `evaluation/runners/run_frontend_eval.py`  
**Mục tiêu:** Phân định rõ ràng bản chất kỹ thuật của các tầng kiểm thử API và Frontend; minh bạch ranh giới giữa kiểm thử tích hợp in-process và kiểm thử trình duyệt thực tế.

---

## 1. Phân Tích Kỹ Thuật Tầng Kiểm Thử API (Level 3)

### A. Phương Pháp Thực Thi
Bộ kiểm thử API gồm 10 kịch bản (`API-01` đến `API-10`) được thực thi bằng thư viện `httpx.AsyncClient` kết hợp trực tiếp với `ASGITransport(app=app)`.

### B. Ranh Giới Kỹ Thuật (ASGI Integration vs Live HTTP Server)
| Tiêu Chí | In-Process ASGI Integration (Phương pháp đang dùng) | Live HTTP Server (TCP Socket) |
| :--- | :--- | :--- |
| **Giao thức truyền thông** | Gọi trực tiếp qua interface chuẩn ASGI của Python (in-memory async callable). | Mở cổng mạng TCP (ví dụ `127.0.0.1:8000`), truyền nhận qua socket HTTP/1.1. |
| **Phạm vi kiểm tra** | Toàn bộ Route handlers, Request validation (Pydantic), Authentication/Authorization middleware, Lifespan events, Dependency injection, SQLite DB persistence, LangGraph asynchronous task execution. | Như bên trái + OS socket binding, network timeouts, keep-alive, proxy reverse... |
| **Mục đích phù hợp** | Tối ưu tuyệt đối cho CI/CD, chạy nhanh, cô lập tài nguyên, không xung đột port mạng. | Kiểm thử tải thực tế, kiểm tra cấu hình NGINX reverse proxy hoặc tường lửa mạng. |
| **Trạng thái thực tế** | **10 / 10 PASS (100%)** | Chưa chạy (không mở daemon nền bên ngoài trong vòng test tự động). |

### C. Chi Tiết 10 Kịch Bản API Đã Được Thẩm Định
1. `API-01`: Upload tài liệu đơn lẻ (`POST /api/upload`) $\rightarrow$ Mã băm SHA-256, lưu file an toàn, trả về `document_id`.
2. `API-02`: Upload tài liệu thứ hai (Multi-document support).
3. `API-03`: Tải lên file PDF bị hỏng cấu trúc (Corrupted PDF) $\rightarrow$ Trả về mã lỗi 400 rõ ràng, không làm sập server.
4. `API-04`: Truy vấn danh mục tài liệu (`GET /api/documents`) $\rightarrow$ Kiểm tra danh sách, trạng thái indexed và kích thước file.
5. `API-05`: Tạo Run mới (`POST /api/runs`) với câu hỏi kiến thức chung không kèm tài liệu $\rightarrow$ Phân luồng thành công.
6. `API-06`: Polling trạng thái Run (`GET /api/runs/{run_id}`) $\rightarrow$ Trạng thái chuyển mượt mà từ `processing` sang `completed`.
7. `API-07`: Lấy chi tiết kết quả Run đã hoàn thành $\rightarrow$ Đầy đủ `answer`, `claims`, `evidence_spans`, `reasoning_trace`.
8. `API-08`: Truy xuất luồng sự kiện Server-Sent Events (`GET /api/runs/{run_id}/stream`) $\rightarrow$ Nhận các gói tin `data: ...\n\n`.
9. `API-09`: Hủy tác vụ đang chạy (`POST /api/runs/{run_id}/cancel`) $\rightarrow$ Hủy an toàn async task, cập nhật trạng thái `cancelled`.
10. `API-10`: Kiểm tra cấu hình và trạng thái hệ thống (`GET /api/health` & `/api/config`).

---

## 2. Phân Tích Kỹ Thuật Tầng Kiểm Thử Frontend (Level 4)

### A. Phương Pháp Thực Thi
Bộ kiểm thử giao diện người dùng gồm 8 kịch bản (`FE-01` đến `FE-08`) tại [run_frontend_eval.py](file:///c:/Users/Admin/.ng\.m\.Multi-Agent_RAG/evaluation/runners/run_frontend_eval.py).

### B. Ranh Giới Kỹ Thuật (Contract Validation vs Real Browser Automation)
| Tiêu Chí | UI Contract & State Validation (Phương pháp đang dùng) | Real Browser E2E Automation |
| :--- | :--- | :--- |
| **Công cụ thực thi** | Python JSON / DOM contract parser & SSE event stream consumer. | Headless Chromium / Firefox qua Playwright, Puppeteer, Selenium. |
| **Phạm vi kiểm tra** | Xác minh cấu trúc gói tin UI, định dạng Citation Drawer, định dạng nhãn Badge, trạng thái nút hủy, cấu trúc dropdown tài liệu. | Click chuột thật trên giao diện web, chụp ảnh màn hình (screenshot), render CSS thực tế, tính toán layout pixels. |
| **Trạng thái thực tế** | **8 / 8 PASS (100%)** | **BLOCKED (Thiếu Headless Browser Driver)** |

### C. Báo Cáo Minh Bạch Về Real Browser E2E
> [!IMPORTANT]
> **Ghi nhận minh bạch:** Trong môi trường hiện tại, **chưa cài đặt các công cụ headless browser automation** (như `playwright install chromium` hoặc `selenium-webdriver`).
> Do đó, chúng tôi **tuyệt đối không tuyên bố 8 bài kiểm thử Frontend là kiểm thử click chuột trên trình duyệt thật**.
> - Bộ kiểm thử `FE-01..08` là bộ kiểm tra **tính tương thích hợp đồng dữ liệu giao diện (UI Contract & Stream Compatibility)**.
> - Kiểm thử **Real Browser E2E với chụp ảnh màn hình hiện ở trạng thái `BLOCKED`** do thiếu driver trình duyệt trong môi trường CI/CD.

### D. Chi Tiết 8 Hợp Đồng Giao Diện Đã Thẩm Định
1. `FE-01`: SSE Event Stream Binding $\rightarrow$ Kiểm tra luồng sự kiện UI realtime không bị mất gói tin.
2. `FE-02`: Multi-Document Selector Payload $\rightarrow$ Dropdown tài liệu truyền nhận danh sách mảng ID chuẩn xác.
3. `FE-03`: Citation Drawer Payload $\rightarrow$ Đầy đủ số trang (`page_no`), đoạn trích (`quote`), tên file (`document_name`).
4. `FE-04`: Verification Badge Contract $\rightarrow$ Chuẩn hóa 3 trạng thái màu sắc (`VERIFIED` [Xanh], `NOT_ENOUGH_INFO` [Vàng], `CONTRADICTED` [Đỏ]).
5. `FE-05`: Direct Assistant UI Route $\rightarrow$ Giao diện nhận diện đúng câu trả lời trực tiếp không hiển thị Citation Drawer rỗng.
6. `FE-06`: Auto-Scout Banner Contract $\rightarrow$ Hiển thị thanh tiến trình khi kích hoạt tìm kiếm bài báo khoa học ngoài.
7. `FE-07`: User Cancellation Action $\rightarrow$ Nút dừng tác vụ gửi tín hiệu hủy và giao diện chuyển trạng thái ngay lập tức.
8. `FE-08`: Error State Alert Payload $\rightarrow$ Hộp thoại lỗi thân thiện khi tài liệu bị lỗi hoặc không có tài liệu được chọn.
