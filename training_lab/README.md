# 🧬 Phòng Thí Nghiệm Fine-Tune Phân Tách Theo Từng Agent (`training_lab/`)

Thư mục `training_lab/` đã được dọn sạch toàn bộ các file rác và chuẩn hóa theo cấu trúc **Mỗi Agent Một Thư Mục Độc Lập**, tối ưu hóa cực đại cho phần cứng **Kaggle Dual Tesla T4 (`2 × 16GB = 32GB VRAM`)**.

---

## 📂 Cấu Trúc Phân Tách Từng Agent Trong `training_lab/`

```text
training_lab/
├── README.md                                      # Tài liệu tổng quan kiến trúc Fine-Tune T4x2
├── Train_Master_T4x2.ipynb                        # Notebook tổng hợp (chạy cả 3 Agent trong 1 lần bấm)
│
├── agent1_retrieval/                              # [AGENT 1] RETRIEVAL ROUTER & SUPERVISOR PLANNER
│   ├── README.md
│   ├── Train_Agent1_Router_T4x2.ipynb             # Notebook riêng để train Agent 1 trên Kaggle T4x2
│   ├── train_agent1_t4x2.py                       # Script QLoRA + DoRA (r=64, alpha=128, NEFTune)
│   ├── dataset/router_sft.jsonl                   # 381 mẫu SFT phân tích độ phức tạp & tách truy vấn
│   └── weights/                                   # Trọng số Agent 1 (adapter_model.safetensors - 77 MB)
│
├── agent2_reader/                                 # [AGENT 2] DOCUMENT LAYOUT & STRUCTURE READER
│   ├── README.md
│   └── weights/                                   # Cấu hình nhận diện bố cục PDF (LayoutLMv3 DocLayNet)
│
├── agent3_summarizer/                             # [AGENT 3] SUMMARIZER (GENERATOR SFT+DPO & CRITIC)
│   ├── README.md
│   ├── Train_Agent3_Summarizer_T4x2.ipynb         # Notebook riêng để train Agent 3 trên Kaggle T4x2
│   ├── train_agent3_t4x2.py                       # Script 2 chặng: Generator (SFT + DPO) & Critic (Hard-Negatives)
│   ├── dataset/
│   │   ├── generator_sft.jsonl                    # 381 mẫu tổng hợp gắn trích dẫn [1], [2]
│   │   ├── generator_dpo.jsonl                    # 381 cặp ưu tiên DPO (Chosen vs Rejected)
│   │   └── critic_sft.jsonl                       # 762 mẫu tự phản biện có gài bẫy số liệu
│   └── weights/
│       ├── generator_qlora/                       # Trọng số Generator QLoRA (77 MB chuẩn FP16)
│       └── critic_qlora/                          # Trọng số Critic QLoRA (19 MB chuẩn FP16)
│
└── agent4_fact_checker/                           # [AGENT 4] FACT-CHECKER (3-CLASS NLI VERIFIER)
    ├── README.md
    ├── Train_Agent4_FactChecker_T4x2.ipynb        # Notebook riêng để train Agent 4 trên Kaggle T4x2
    ├── train_agent4_t4x2.py                       # Script huấn luyện Cross-Encoder NLI 3 nhãn
    ├── dataset/nli_verifier.jsonl                 # 762 mẫu kiểm chứng mệnh đề (SUPPORT/CONTRADICT/NEI)
    └── weights/                                   # Nơi lưu trọng số NLI Verifier sau khi huấn luyện
```

---

## 🎯 Cách Chọn Notebook Để Đi Huấn Luyện

- **Muốn huấn luyện riêng từng Agent:** Vào thẳng thư mục của Agent đó và mở file `.ipynb` tương ứng:
  - Agent 1: [`agent1_retrieval/Train_Agent1_Router_T4x2.ipynb`](agent1_retrieval/Train_Agent1_Router_T4x2.ipynb)
  - Agent 3: [`agent3_summarizer/Train_Agent3_Summarizer_T4x2.ipynb`](agent3_summarizer/Train_Agent3_Summarizer_T4x2.ipynb)
  - Agent 4: [`agent4_fact_checker/Train_Agent4_FactChecker_T4x2.ipynb`](agent4_fact_checker/Train_Agent4_FactChecker_T4x2.ipynb)
- **Muốn chạy 1 mạch cả 3 Agent:** Mở [`Train_Master_T4x2.ipynb`](Train_Master_T4x2.ipynb) ở ngoài cùng.
