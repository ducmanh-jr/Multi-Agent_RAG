# 🔬 Phòng Nghiên cứu & Huấn luyện Tác tử (Training & Research Lab)

Thư mục `training_lab/` là không gian độc lập phục vụ toàn bộ chu trình **Nghiên cứu & Phát triển (R&D), Huấn luyện tham số (Fine-tuning), Sinh dữ liệu tổng hợp (Synthetic Data) và Đánh giá chất lượng (Evaluation)** cho từng tác tử trong hệ thống Multi-Agent Research RAG.

Kiến trúc này tuân thủ chuẩn **MLOps Lifecycle Separation**, tách biệt hoàn toàn giữa:
1. **Phòng thí nghiệm (Training Lab - R&D)**: Nơi kỹ sư AI và nhà nghiên cứu làm việc với GPU, notebook, QLoRA, loss curves và synthetic data.
2. **Hệ thống chạy thực tế (Production Runtime)**: Nơi ứng dụng web, FastAPI server, SQLite C-Speed FTS5 và LangGraph phục vụ người dùng cuối với độ trễ thấp và độ ổn định 100%.

---

## 📁 Cấu trúc thư mục

```text
training_lab/
├── README.md                           # Bản đặc tả phòng nghiên cứu
├── agent1_retrieval/                   # R&D cho Tác tử 1: Router & Retrieval
│   ├── eval_router_accuracy.py         # Đo đạc độ chính xác phân loại công cụ (target >= 80%)
│   ├── retrieval_eval_manual.jsonl     # Tập test bench 30 kịch bản truy vấn thực tế
│   ├── router_policy.py                # Thuật toán phân luồng heuristic & guardrails
│   ├── schemas.py                      # Pydantic schema chuẩn cho Router Decision
│   └── retrieval_router_critique_qlora/# Checkpoint adapter LoRA sau fine-tune
├── agent2_reader/                      # R&D cho Tác tử 2: Layout Parser & Reader
│   ├── reader_agent.py                 # Thử nghiệm bóc tách cấu trúc tài liệu đa cột
│   ├── reader_policy.py                # Chính sách phân loại độ phức tạp trang (simple vs complex)
│   ├── tool_layoutlmv3.py              # Thử nghiệm phân đoạn thị giác LayoutLMv3
│   ├── tool_ocr_fallback.py            # Cơ chế phát hiện trang PDF scan ảnh & OCR
│   ├── tool_alt_segmentation.py        # Thuật toán phân đoạn thay thế
│   └── test_tool_layoutlmv3.py         # Unit test độc lập cho parser
└── agent3_summarizer/                  # R&D cho Tác tử 3: Generator & Critic LoRA
    ├── Code/                           # Script huấn luyện SFT kép (Generator + Critic)
    ├── Script test/                    # Kịch bản kiểm thử độ hội tụ và giảm thiểu ảo giác
    └── Weights/                        # Trọng số QLoRA Generator & Critic
```

---

## 🚀 Quy trình chuyển giao từ Lab sang Production

Khi một tác tử hoàn thành huấn luyện trong `training_lab/`:
1. Trọng số LoRA tối ưu được kiểm thử đạt chuẩn sẽ được xuất sang thư mục [`weights/`](../weights/) (ví dụ: `weights/agent1_router/`).
2. Logic phân loại và heuristic guardrails được tích hợp vào tác tử thực thi tương ứng trong [`agents/`](../agents/).
3. Hệ thống chạy chính [`main.py`](../main.py) và FastAPI [`api/`](../api/) chỉ nạp các artifact cuối cùng mà không cần phụ thuộc vào môi trường huấn luyện cồng kềnh.
