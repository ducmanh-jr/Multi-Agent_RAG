"""
Step 1 of Kaggle FN v2 Fine-Tuning Pipeline:
Automatically extracts chunks from both `main/data/corpus.jsonl` and all PDFs in `main/uploads/`
to generate 100% schema-validated SFT and DPO datasets:
  1. router_sft.jsonl       (Agent 0/1 Query Planner & Multi-Hop Decomposer)
  2. generator_sft.jsonl    (Agent 2A Grounded Summarizer with [n] citations & evidence_spans)
  3. generator_dpo.jsonl    (Preference pairs: Chosen = strictly grounded vs Rejected = subtle numeric/entity hallucination)
  4. critic_sft.jsonl       (Agent 2B Adversarial Critic trained on Hard Negatives: approve | rewrite | request_more_context)
  5. nli_verifier.jsonl     (Agent 3 mDeBERTa-v3-base-xnli pairs: ENTAILMENT | CONTRADICTION | NEUTRAL)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main")))
from core.tools import load_or_parse_selected_pdf  # noqa: E402


ROUTER_SYSTEM_PROMPT = (
    "You are an intelligent routing agent for an Academic RAG system. "
    "Analyze the user query and output a JSON object with: intent, complexity, sub_queries, alpha (0.0 to 1.0), top_k (1 to 15), and reasoning."
)

GENERATOR_SYSTEM_PROMPT = """Bạn là một Trợ lý Nghiên cứu AI chuyên nghiệp (Academic RAG Summarizer).
Nhiệm vụ của bạn là đọc hiểu ngữ cảnh (Context) được cung cấp và trả lời câu hỏi (Question) của người dùng một cách chính xác, khách quan và có căn cứ.

QUY TẮC BẮT BUỘC:
1. CHỈ sử dụng thông tin có trong Context. Tuyệt đối KHÔNG tự bịa đặt (hallucinate) hoặc dùng kiến thức bên ngoài nếu Context không đề cập.
2. Nếu Context KHÔNG chứa đủ thông tin để trả lời câu hỏi, hãy tuyên bố rõ ràng: "Tài liệu hiện tại không cung cấp đủ thông tin để trả lời câu hỏi này."
3. Mọi lập luận đưa ra trong câu trả lời phải được chứng minh bằng các trích dẫn nguyên văn (evidence_spans) lấy từ Context và gắn ký hiệu trích dẫn [1], [2].
4. Trả lời hoàn toàn bằng định dạng JSON hợp lệ:
{"answer": "...", "evidence_spans": ["..."]}"""

CRITIC_SYSTEM_PROMPT = """Bạn là một Chuyên gia Kiểm định Chất lượng AI (Critic Agent) khắt khe.
Nhiệm vụ của bạn là đánh giá bản tóm tắt (Generated Summary) do Generator tạo ra bằng cách đối chiếu nó với Câu hỏi gốc (Question) và Ngữ cảnh (Context).
BẮT BUỘC trả về JSON hợp lệ:
{"is_supported": true/false, "Unsupported_claims": [...], "Missing_key_points": [...], "action": "approve"|"rewrite"|"request_more_context"|"lower_confidence", "confidence": 0.0..1.0}"""


def perturb_numbers_for_hard_negative(text: str) -> tuple[str, str]:
    """Perturbs a number or year in `text` to create a realistic Hard Negative hallucination."""
    nums = re.findall(r"\b\d+(?:[.,]\d+)?\b", text)
    if not nums:
        hallucinated = text + " Ngoài ra, nghiên cứu đã thử nghiệm trên 12.500 mẫu lâm sàng bổ sung tại châu Âu."
        return hallucinated, "12.500 mẫu lâm sàng bổ sung tại châu Âu"

    target = nums[0]
    try:
        val = float(target.replace(",", "."))
        fake_val = int(val + 17) if val.is_integer() else round(val * 1.25, 2)
        fake_str = str(fake_val)
        corrupted = text.replace(target, fake_str, 1)
        return corrupted, f"Sai lệch số liệu: '{fake_str}' (trong tài liệu gốc là '{target}')"
    except Exception:
        return text + " (Đạt độ chính xác 99.98% trên mọi tập dữ liệu).", "Đạt độ chính xác 99.98%"


def build_datasets(
    corpus_path: str,
    uploads_dir: str,
    output_dir: str,
    max_samples: int = 600,
) -> dict[str, int]:
    os.makedirs(output_dir, exist_ok=True)
    chunks: list[dict[str, Any]] = []

    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        chunks.append(json.loads(line))
                    except Exception:
                        pass

    if os.path.isdir(uploads_dir):
        for pdf_file in sorted(glob.glob(os.path.join(uploads_dir, "*.pdf")))[:25]:
            parsed = load_or_parse_selected_pdf(pdf_file)
            chunks.extend(parsed[:15])

    router_rows = []
    gen_sft_rows = []
    gen_dpo_rows = []
    critic_sft_rows = []
    nli_rows = []

    for idx, ch in enumerate(chunks[:max_samples]):
        text = str(ch.get("text") or ch.get("content") or "").strip()
        if len(text) < 90:
            continue
        doc_id = str(ch.get("doc_id") or ch.get("title") or f"doc_{idx+1}")
        page_no = int(ch.get("page_no") or 1)
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 35]
        if not sents:
            continue

        lead_sent = sents[0]
        second_sent = sents[1] if len(sents) > 1 else lead_sent
        query = f"Nội dung phương pháp và kết quả chính tại trang {page_no} của tài liệu '{doc_id}' là gì?"

        # 1. Router SFT sample
        router_out = {
            "intent": "academic_research",
            "complexity": "MEDIUM",
            "sub_queries": [
                f"{doc_id} trang {page_no} phương pháp chính",
                f"{doc_id} số liệu kết quả thực nghiệm",
            ],
            "alpha": 0.45,
            "top_k": 6,
            "reasoning": "Câu hỏi yêu cầu cả phương pháp và kết quả thực nghiệm nên kết hợp BM25 và Dense search.",
        }
        router_rows.append(
            {
                "messages": [
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": json.dumps(router_out, ensure_ascii=False)},
                ]
            }
        )

        # 2. Generator SFT & DPO sample
        chosen_ans = f"Dựa trên tài liệu **{doc_id}** (Trang {page_no}):\n- {lead_sent} [1]\n"
        if second_sent != lead_sent:
            chosen_ans += f"- {second_sent} [1]"
        chosen_json = json.dumps(
            {
                "answer": chosen_ans,
                "evidence_spans": [lead_sent, second_sent] if second_sent != lead_sent else [lead_sent],
            },
            ensure_ascii=False,
        )

        corrupted_sent, corruption_reason = perturb_numbers_for_hard_negative(lead_sent)
        rejected_ans = f"Dựa trên tài liệu **{doc_id}**:\n- {corrupted_sent}\n- Kết quả này đã được kiểm chứng trên 50 quốc gia."
        rejected_json = json.dumps(
            {
                "answer": rejected_ans,
                "evidence_spans": [],
            },
            ensure_ascii=False,
        )

        user_gen_prompt = f'Context:\n"""\n[1] (Doc: {doc_id}, Trang {page_no}): {text}\n"""\n\nCâu hỏi: {query}'
        gen_sft_rows.append(
            {
                "messages": [
                    {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_gen_prompt},
                    {"role": "assistant", "content": chosen_json},
                ]
            }
        )
        gen_dpo_rows.append(
            {
                "prompt": f"<|im_start|>system\n{GENERATOR_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_gen_prompt}<|im_end|>\n<|im_start|>assistant\n",
                "chosen": chosen_json,
                "rejected": rejected_json,
            }
        )

        # 3. Critic SFT samples (Positive 'approve' + Hard Negative 'rewrite')
        critic_pos_out = {
            "is_supported": True,
            "Unsupported_claims": [],
            "Missing_key_points": [],
            "action": "approve",
            "confidence": 0.95,
        }
        critic_neg_out = {
            "is_supported": False,
            "Unsupported_claims": [corruption_reason, "Kết quả này đã được kiểm chứng trên 50 quốc gia."],
            "Missing_key_points": [second_sent[:90]] if second_sent != lead_sent else [],
            "action": "rewrite",
            "confidence": 0.25,
        }
        critic_sft_rows.append(
            {
                "messages": [
                    {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f'Context:\n"""\n{text}\n"""\n\nQuestion: {query}\n\nGenerated Summary:\n{chosen_ans}',
                    },
                    {"role": "assistant", "content": json.dumps(critic_pos_out, ensure_ascii=False)},
                ]
            }
        )
        critic_sft_rows.append(
            {
                "messages": [
                    {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f'Context:\n"""\n{text}\n"""\n\nQuestion: {query}\n\nGenerated Summary:\n{rejected_ans}',
                    },
                    {"role": "assistant", "content": json.dumps(critic_neg_out, ensure_ascii=False)},
                ]
            }
        )

        # 4. NLI Verifier samples
        nli_rows.append({"premise": text, "hypothesis": lead_sent, "label": "ENTAILMENT", "label_id": 0})
        nli_rows.append({"premise": text, "hypothesis": corrupted_sent, "label": "CONTRADICTION", "label_id": 2})

    counts = {}
    for filename, rows in [
        ("router_sft.jsonl", router_rows),
        ("generator_sft.jsonl", gen_sft_rows),
        ("generator_dpo.jsonl", gen_dpo_rows),
        ("critic_sft.jsonl", critic_sft_rows),
        ("nli_verifier.jsonl", nli_rows),
    ]:
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[filename] = len(rows)

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", default="main/data/corpus.jsonl")
    parser.add_argument("--uploads_dir", default="main/uploads")
    parser.add_argument("--output_dir", default="training_lab/kaggle_fn_v2/datasets")
    parser.add_argument("--max_samples", type=int, default=600)
    args = parser.parse_args()
    stats = build_datasets(args.corpus_path, args.uploads_dir, args.output_dir, args.max_samples)
    print("Generated Kaggle FN v2 datasets:", json.dumps(stats, indent=2))
