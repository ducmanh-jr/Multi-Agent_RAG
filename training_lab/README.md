# 🧪 `training_lab/` — Phòng Thí Nghiệm Huấn Luyện Độc Lập Từng Agent (Chuẩn Kaggle Dual T4 `2 × 16GB = 32GB VRAM`)

Toàn bộ thư mục `training_lab/` được thiết kế theo kiến trúc phần mềm chuẩn mực, **tách biệt độc lập từng Agent** kèm gói mã nguồn mô-đun `src/` riêng biệt cho mỗi Agent:

```text
training_lab/
├── agent1_retrieval/                          # 🧭 AGENT 1: Retrieval Router & Supervisor Query Planner
│   ├── src/                                   # Kiến trúc mã nguồn Fine-Tuning mô-đun hóa
│   │   ├── __init__.py                        # Package exports
│   │   ├── config.py                          # Cấu hình siêu tham số DoRA/QLoRA & phân bổ VRAM Dual T4
│   │   ├── data_loader.py                     # Nạp dữ liệu ChatML, kiểm định JSON schema, chia Train/Val
│   │   ├── model_builder.py                   # Khởi tạo Qwen2.5-7B 4-bit NF4 + Dual-GPU device_map="balanced"
│   │   ├── trainer.py                         # SFTTrainer (NEFTune=5.0, Paged AdamW 8-bit) + Xuất FP16 (<80MB)
│   │   └── evaluator.py                       # Kiểm định JSON Schema Compliance Rate sau huấn luyện
│   ├── dataset/router_sft.jsonl               # 1,500 mẫu SFT chiến lược điều phối & viết lại truy vấn
│   ├── weights/                               # Trọng số QLoRA FP16 đã huấn luyện (77.05 MB - sẵn trên GitHub)
│   ├── train_agent1_t4x2.py                   # CLI Entrypoint gọi vào package `src.*`
│   ├── Train_Agent1_Router_T4x2.ipynb         # Notebook huấn luyện riêng Agent 1 trên Kaggle T4x2
│   └── README.md                              # Tài liệu kỹ thuật chi tiết Agent 1
│
├── agent2_reader/                             # 📄 AGENT 2: Document Layout & Formula/Table Parser
│   ├── src/                                   # Kiến trúc mã nguồn Fine-Tuning LayoutLMv3
│   │   ├── __init__.py
│   │   ├── config.py                          # Cấu hình 11 nhãn DocLayNet & siêu tham số huấn luyện
│   │   ├── data_loader.py                     # Bộ nạp tọa độ Bounding-Box 2D & Token Classification
│   │   ├── model_builder.py                   # Khởi tạo `microsoft/layoutlmv3-base` (11 classes)
│   │   ├── trainer.py                         # Huấn luyện & xuất `model.safetensors` + `label_map.json`
│   │   └── evaluator.py                       # Kiểm định độ chính xác phân đoạn cấu trúc trang PDF
│   ├── dataset/layout_doclaynet_sample.jsonl  # Tập mẫu tọa độ không gian 2D & nhãn cấu trúc DocLayNet
│   ├── weights/                               # Trọng số LayoutLMv3 đã tinh chỉnh
│   ├── train_agent2_t4x2.py                   # CLI Entrypoint gọi vào package `src.*`
│   ├── Train_Agent2_Reader_T4x2.ipynb         # Notebook huấn luyện riêng Agent 2 trên Kaggle T4x2
│   └── README.md                              # Tài liệu kỹ thuật chi tiết Agent 2
│
├── agent3_summarizer/                         # ✍️ AGENT 3: Academic Summarizer (Generator + Critic)
│   ├── src/                                   # Kiến trúc mã nguồn Fine-Tuning 2 Giai đoạn (SFT + DPO)
│   │   ├── __init__.py
│   │   ├── config.py                          # Cấu hình DoRA r=64 (Generator) & r=32 (Critic) trên Dual T4
│   │   ├── data_loader.py                     # Bộ nạp SFT ChatML & Cặp đối chiếu DPO (Chosen vs Rejected)
│   │   ├── model_builder.py                   # Khởi tạo Qwen2.5-7B 4-bit NF4 cân bằng tải 2 GPU T4
│   │   ├── trainer.py                         # Pipeline Stage 1 SFT -> Stage 2 DPO + Critic Hard-Negatives
│   │   └── evaluator.py                       # Kiểm định kích thước & độ trung thực trích dẫn của 2 Adapter
│   ├── dataset/
│   │   ├── generator_sft.jsonl                # 1,500 mẫu SFT tóm tắt học thuật có trích dẫn chuẩn
│   │   ├── generator_dpo.jsonl                # 1,000 cặp DPO (`chosen` trung thực vs `rejected` bịa đặt)
│   │   └── critic_sft.jsonl                   # 1,500 mẫu SFT phản biện phát hiện lỗi trích dẫn
│   ├── weights/
│   │   ├── generator_qlora/                   # Trọng số Generator QLoRA FP16 (77.05 MB - sẵn trên GitHub)
│   │   └── critic_qlora/                      # Trọng số Critic QLoRA FP16 (19.28 MB - sẵn trên GitHub)
│   ├── train_agent3_t4x2.py                   # CLI Entrypoint gọi vào package `src.*`
│   ├── Train_Agent3_Summarizer_T4x2.ipynb     # Notebook huấn luyện riêng Agent 3 trên Kaggle T4x2
│   └── README.md                              # Tài liệu kỹ thuật chi tiết Agent 3
│
├── agent4_fact_checker/                       # 🛡️ AGENT 4: Atomic Fact-Checker & NLI Consensus Verifier
│   ├── src/                                   # Kiến trúc mã nguồn Fine-Tuning Cross-Encoder NLI
│   │   ├── __init__.py
│   │   ├── config.py                          # Cấu hình 3 nhãn NLI (`SUPPORT`, `CONTRADICT`, `NOT_ENOUGH_INFO`)
│   │   ├── data_loader.py                     # Bộ nạp cặp `(claim, evidence)` từ SciFact + VitaminC
│   │   ├── model_builder.py                   # Khởi tạo `BAAI/bge-reranker-v2-m3` / `SciBERT` 3-class
│   │   ├── trainer.py                         # `WeightedCrossEncoderTrainer` (phạt nặng lỗi bỏ sót CONTRADICT)
│   │   └── evaluator.py                       # Tính Macro-F1 & Hiệu chỉnh nhiệt độ xác suất (Temperature ECE)
│   ├── dataset/nli_verifier.jsonl             # 3,000 cặp thẩm định sự kiện khoa học
│   ├── weights/                               # Thư mục lưu trọng số NLI Verifier + `eval_metrics.json`
│   ├── train_agent4_t4x2.py                   # CLI Entrypoint gọi vào package `src.*`
│   ├── Train_Agent4_FactChecker_T4x2.ipynb    # Notebook huấn luyện riêng Agent 4 trên Kaggle T4x2
│   └── README.md                              # Tài liệu kỹ thuật chi tiết Agent 4
│
├── Train_Master_T4x2.ipynb                    # 🚀 Notebook Tổng Hợp: Chạy huấn luyện cả 4 Agent liên hoàn
└── README.md                                  # Tài liệu kiến trúc tổng thể `training_lab/`
```
