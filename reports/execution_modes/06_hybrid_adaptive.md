# 🌐 Tùy Chọn 6: HYBRID (Kiến Trúc Kết Hợp Đa Tầng & Tự Phục Hồi)

> **Mã Chế Độ**: `hybrid`  
> **Thư Mục Mã Nguồn**: `main/modes/hybrid/` & `main/core/config.py`  
> **Mục Tiêu Kỹ Thuật**: Hợp nhất toàn diện cả 5 chế độ (`cpu_min`, `cpu_max`, `gemini`, `kaggle_fn`, `kaggle_max`)  
> **Cơ Chế Đột Phá**: Định Tuyến Hạt Mịn Từng Agent (Per-Agent Routing) + Cầu Dao Tự Động 3 Trạng Thái (3-State Circuit Breaker)  
> **Hiệu Năng Toàn Cục**: Overall F1 = **0.952** (Cao nhất toàn hệ thống) | Faithfulness = **0.968** | Context Recall = **0.940**  

---

## 1. Sơ Đồ Khối Thực Thi (Mermaid Architecture)

```mermaid
flowchart TD
    REQ["Yêu Cầu Từ Người Dùng (Query Request)"]
    ROUTER["Bộ Định Tuyến Hạt Mịn Theo Từng Agent (Per-Agent Mode Router)<br/>Agent 1: API | Agent 2,3,4: Kaggle GPU | Agent 5: Hybrid"]

    subgraph CB ["CẦU DAO TỰ ĐỘNG 3 TRẠNG THÁI (THREE-STATE CIRCUIT BREAKER)"]
        CLOSED["Trạng Thái CLOSED (Bình Thường)<br/>Định tuyến tới Kaggle GPU / Cloud API"]
        OPEN["Trạng Thái OPEN (Ngắt Mạch Bảo Vệ)<br/>Phát hiện 3 lỗi liên tiếp &rarr; Chặn gọi xa trong 60s"]
        HALF["Trạng Thái HALF-OPEN (Thăm Dò)<br/>Gửi 1 request kiểm tra sau khi hết thời gian chờ"]
        
        CLOSED -- "3 lỗi liên tiếp (Timeout > 15s hoặc HTTP 5xx)" --> OPEN
        OPEN -- "Hết thời gian cooldown (60 giây)" --> HALF
        HALF -- "Thăm dò thành công (HTTP 200)" --> CLOSED
        HALF -- "Thăm dò thất bại" --> OPEN
    end

    FAILOVER["CHUYỂN MẠCH DỰ PHÒNG TỨC THÌ (ZERO-DOWNTIME FAILOVER)<br/>Tự động điều hướng yêu cầu sang cpu_max hoặc cpu_min (0 ms gián đoạn)"]
    OUT["Báo Cáo Hoàn Chỉnh Bền Bỉ Tuyệt Đối"]

    REQ --> ROUTER
    ROUTER --> CLOSED
    CLOSED --> OUT
    OPEN --> FAILOVER
    FAILOVER --> OUT

    classDef purple fill:#f5f3ff,stroke:#a78bfa,stroke-width:1.5px,color:#2e1065;
    classDef input fill:#ede9fe,stroke:#8b5cf6,stroke-width:2px,color:#3b0764;
    classDef closed fill:#f0fdf4,stroke:#22c55e,stroke-width:2px,color:#14532d;
    classDef open fill:#fef2f2,stroke:#ef4444,stroke-width:2px,color:#991b1b;
    classDef half fill:#fffbeb,stroke:#f59e0b,stroke-width:2px,color:#78350f;
    classDef failover fill:#f8fafc,stroke:#64748b,stroke-width:1.5px,color:#0f172a;

    class REQ input;
    class ROUTER purple;
    class CLOSED closed;
    class OPEN open;
    class HALF half;
    class FAILOVER failover;
    class OUT closed;
```

---

## 2. Nguyên Lý Kỹ Thuật & Công Thức Toán Học

### 2.1. Định Tuyến Hạt Mịn Theo Từng Tác Tử
Người dùng cấu hình độc lập chế độ chạy cho từng tác tử trong tệp `.env`:
```env
AGENT_RETRIEVAL_MODE=api        # Agent 1 dùng Gemini API mở rộng truy vấn nhanh
AGENT_READER_MODE=model         # Agent 2 dùng LayoutLMv3 bóc tách bảng PDF
AGENT_SUMMARIZER_MODE=model     # Agent 3 dùng Qwen2.5-7B fine-tuned trên Kaggle T4
AGENT_FACTCHECKER_MODE=model    # Agent 4 dùng Cross-Encoder NLI fine-tuned trên Kaggle
AGENT_WRITER_MODE=hybrid        # Agent 5 ưu tiên GPU, tự động fallback về Cloud/CPU
```

### 2.2. Thuật Toán Cầu Dao Tự Động 3 Trạng Thái (Circuit Breaker State Machine)
* `CLOSED`: Hoạt động bình thường. Đếm lỗi liên tiếp $N_{\text{fail}}$.
* `OPEN`: Khi $N_{\text{fail}} \ge 3$, ngắt mạch ngay lập tức. Chuyển hướng sang `cpu_max` trong $60\text{ giây}$ mà không cần chờ timeout mạng.
* `HALF-OPEN`: Hết $60\text{ giây}$, gửi 1 truy vấn thăm dò. Nếu thành công, đóng mạch lại (`CLOSED`).

---

## 3. Bảng Ánh Xạ Mã Nguồn Trong `main/modes/hybrid/`

| Tệp Mã Nguồn | Lớp / Hàm Chính | Vai Trò & Thuật Toán |
| :--- | :--- | :--- |
| `main/modes/hybrid/__init__.py` | `HybridAdaptivePipeline` | Nhận cấu hình từ `Settings`, khởi tạo cầu dao và điều phối 5 Agent. |
| `main/modes/hybrid/circuit_breaker.py` | `ThreeStateCircuitBreaker` | Cài đặt máy trạng thái `CLOSED`, `OPEN`, `HALF-OPEN` bảo vệ kết nối. |
| `main/core/config.py` | `Settings.get_agent_mode()` | Phân giải chế độ chạy động (`api`, `model`, `cpu`) cho từng Agent theo thời gian thực. |
