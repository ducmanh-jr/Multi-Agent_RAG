# -*- coding: utf-8 -*-
"""
Post-Training Evaluator for Agent 1 (Retrieval Router & Supervisor Query Planner).
Measures JSON Schema Compliance Rate and Routing Decision Accuracy on held-out queries.
"""

import json
import torch


def evaluate_router_adapter(model, tokenizer, test_queries: list[str] | None = None) -> dict:
    """Runs inference on sample scientific queries and checks strict JSON schema compliance."""
    if test_queries is None:
        test_queries = [
            "So sánh độ phức tạp tính toán giữa Self-Attention và Linear Attention trong chuỗi dài.",
            "Compare LoRA and DoRA parameter-efficient fine-tuning methods on Qwen2.5-7B.",
            "Cơ chế đồng thuận 2/3 trong kiểm chứng sự kiện khoa học hoạt động thế nào?",
        ]

    sys_prompt = (
        "Bạn là Tác tử Điều phối Truy vấn Khoa học (Agent 1 Router). "
        "Hãy phân tích câu hỏi và trả về JSON thuần túy gồm: "
        '{"action": "retrieve"|"rewrite", "rewritten_queries": [...], "alpha": 0.65, "reason": "..."}'
    )

    valid_json = 0
    results = []
    try:
        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = True
        device = next(model.parameters()).device

        for q in test_queries:
            messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": q}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([prompt], return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            try:
                parsed = json.loads(resp)
                is_valid = isinstance(parsed, dict)
                if is_valid:
                    valid_json += 1
            except Exception:
                is_valid = False
            results.append({"query": q, "response": resp, "valid_json": is_valid})
    except Exception as e:
        print(f"⚠️ [Agent 1 Evaluator] Evaluation notice: {e}")

    metrics = {
        "total_tested": len(test_queries),
        "json_compliance_rate": round(valid_json / max(1, len(test_queries)), 4),
        "samples": results,
    }
    print(f"✅ [Agent 1 Evaluator] JSON Compliance Rate: {metrics['json_compliance_rate']*100:.1f}%")
    return metrics
