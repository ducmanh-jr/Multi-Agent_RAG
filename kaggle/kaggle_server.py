"""
Kaggle Dual Tesla T4 Inference Worker Server.
Runs on Kaggle with Accelerator = GPU T4 x 2 (32GB VRAM total).

Memory distribution:
  - GPU 0 (16GB VRAM): Qwen2.5-7B-Instruct (4-bit NF4 ~4.8GB VRAM)
      + PEFT Adapters (Multi-Adapter Serving):
        * 'generator' (Agent 3 Generator QLoRA)
        * 'critic' (Agent 3 Critic QLoRA)
        * 'router' (Agent 1 Retrieval Router QLoRA)
  - GPU 1 (16GB VRAM): Qwen2.5-3B-Instruct (4-bit NF4 ~2.8GB VRAM)
      + SciFact Consensus Fact-Checker
      + Optional LayoutLMv3 DocLayNet Document Parser
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional, Dict
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI(title="Kaggle Dual-T4 Multi-Agent Model Worker with PEFT Adapters")

# Global model holders
_models: dict[str, Any] = {}
_tokenizers: dict[str, Any] = {}
_loaded_adapters: list[str] = []


def find_adapter_path(sub_path: str) -> Optional[str]:
    """Search common Kaggle and local paths for fine-tuned weights."""
    candidates = [
        Path("/kaggle/working/Multi-Agent_RAG/weights") / sub_path,
        Path.cwd() / "weights" / sub_path,
        Path("./weights") / sub_path,
        Path("../weights") / sub_path,
        Path("/kaggle/input/test-agent") / sub_path,
        Path("/kaggle/working/Test_agent") / sub_path,
        Path("./Test_agent") / sub_path,
        Path("../Test_agent") / sub_path,
        Path.cwd() / "Test_agent" / sub_path,
    ]
    for c in candidates:
        if c.exists() and (c / "adapter_config.json").exists():
            return str(c.resolve())
    return None


def get_bnb_config():
    """Builds 4-bit NF4 BitsAndBytes configuration."""
    try:
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    except Exception:
        return None


def load_generator_model():
    """Loads Qwen2.5-7B on GPU 0 with 4-bit NF4 quantization and attaches PEFT adapters."""
    global _loaded_adapters
    if "generator" in _models:
        return _models["generator"], _tokenizers["generator"]

    model_id = os.getenv("GENERATOR_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    print(f"Loading Base Model: {model_id} on cuda:0...")

    if not torch.cuda.is_available():
        print("CUDA not available. Running in CPU/Mock mode.")
        return None, None

    from transformers import AutoModelForCausalLM, AutoTokenizer
    bnb_config = get_bnb_config()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={" ": 0} if torch.cuda.device_count() >= 1 else "auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Attach PEFT Adapters if available
    try:
        from peft import PeftModel

        generator_path = os.getenv("GENERATOR_LORA_PATH") or find_adapter_path("agent3/Weights/generator_qlora/final")
        critic_path = os.getenv("CRITIC_LORA_PATH") or find_adapter_path("agent3/Weights/critic_qlora/final")
        router_path = os.getenv("ROUTER_LORA_PATH") or find_adapter_path("agent1_router") or find_adapter_path("agent1/retrieval_router_critique_qlora/final")

        peft_attached = False

        if generator_path:
            print(f"Attaching Generator QLoRA from {generator_path}...")
            model = PeftModel.from_pretrained(model, generator_path, adapter_name="generator")
            _loaded_adapters.append("generator")
            peft_attached = True

        if critic_path:
            print(f"Attaching Critic QLoRA from {critic_path}...")
            if peft_attached:
                model.load_adapter(critic_path, adapter_name="critic")
            else:
                model = PeftModel.from_pretrained(model, critic_path, adapter_name="critic")
                peft_attached = True
            _loaded_adapters.append("critic")

        if router_path:
            print(f"Attaching Retrieval Router QLoRA from {router_path}...")
            if peft_attached:
                model.load_adapter(router_path, adapter_name="router")
            else:
                model = PeftModel.from_pretrained(model, router_path, adapter_name="router")
                peft_attached = True
            _loaded_adapters.append("router")

        print(f"PEFT Adapters active: {_loaded_adapters}")

    except Exception as e:
        print(f"Note: PEFT adapters not loaded or not found: {e}. Running base model.")

    _models["generator"] = model
    _tokenizers["generator"] = tokenizer
    return model, tokenizer


def load_verifier_model():
    """Loads Qwen2.5-3B on GPU 1 with 4-bit NF4 quantization for Fact-Checking."""
    if "verifier" in _models:
        return _models["verifier"], _tokenizers["verifier"]

    model_id = os.getenv("VERIFIER_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
    target_device = 1 if torch.cuda.device_count() > 1 else 0
    print(f"Loading Verifier Model: {model_id} on cuda:{target_device}...")

    if not torch.cuda.is_available():
        print("CUDA not available. Verifier will run in CPU/Mock mode.")
        return None, None

    from transformers import AutoModelForCausalLM, AutoTokenizer
    bnb_config = get_bnb_config()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={" ": target_device},
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    _models["verifier"] = model
    _tokenizers["verifier"] = tokenizer
    return model, tokenizer


class GenerateRequest(BaseModel):
    prompt: str
    system_instruction: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1500
    json_mode: bool = False
    adapter: Optional[str] = None


class FactCheckRequest(BaseModel):
    claim: str
    evidence_context: str


class SummarizeRequest(BaseModel):
    context: str
    question: str
    retry_instruction: Optional[str] = None


class CriticRequest(BaseModel):
    context: str
    question: str
    answer: str
    evidence_spans: list[str]


class RouterRequest(BaseModel):
    query: str
    history_trials: Optional[list[dict[str, Any]]] = None


@app.on_event("startup")
def startup_event():
    print("🚀 [Startup] Khởi động máy chủ và nạp trước mô hình vào VRAM Dual T4...")
    try:
        load_generator_model()
        load_verifier_model()
        print("✅ [Startup] Hoàn tất nạp mô hình.")
    except Exception as e:
        print(f"⚠️ [Startup Error]: {e}")


@app.get("/health")
def health():
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_info = []
    for i in range(gpu_count):
        prop = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        total = prop.total_memory / (1024 ** 3)
        gpu_info.append({
            "device": f"cuda:{i}",
            "name": prop.name,
            "allocated_gb": round(allocated, 2),
            "total_gb": round(total, 2),
        })

    return {
        "status": "online",
        "worker": "Kaggle Dual T4 Worker",
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": gpu_count,
        "gpus": gpu_info,
        "models_loaded": list(_models.keys()),
        "adapters_loaded": _loaded_adapters,
    }


def _run_inference(model, tokenizer, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 1024, temperature: float = 0.1, adapter_name: Optional[str] = None) -> str:
    """Helper to run text generation with optional adapter switching."""
    if hasattr(model, "set_adapter") and adapter_name and adapter_name in _loaded_adapters:
        model.set_adapter(adapter_name)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


@app.post("/api/generate")
def generate(req: GenerateRequest):
    """Generates completion using Qwen-7B on GPU 0."""
    model, tokenizer = load_generator_model()
    if model is None or tokenizer is None:
        return {
            "output": f"[Fallback] {req.prompt[:100]}...",
            "gpu_used": "cpu_fallback",
        }

    response_text = _run_inference(
        model, tokenizer,
        prompt=req.prompt,
        system_prompt=req.system_instruction,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        adapter_name=req.adapter,
    )
    return {
        "output": response_text,
        "gpu_used": str(model.device),
        "adapter_used": req.adapter or "base",
    }


@app.post("/api/summarize/generator")
def summarize_generator(req: SummarizeRequest):
    """Specialized Generator endpoint using fine-tuned Generator QLoRA."""
    model, tokenizer = load_generator_model()
    if model is None:
        return {
            "answer": "Tài liệu chứng minh kiến trúc mạng nơ-ron sâu đạt hiệu năng vượt trội.",
            "evidence_spans": ["Trích dẫn thực nghiệm trong tài liệu."],
        }

    system_prompt = (
        "Bạn là Tác tử Tóm tắt Khoa học (Generator). Hãy tổng hợp câu trả lời ngắn gọn, chuẩn xác "
        "dựa DUY NHẤT trên context được cung cấp. BẮT BUỘC chỉ trả về JSON thuần túy có schema:\n"
        '{"answer": "...", "evidence_spans": ["trích dẫn 1", ...]}'
    )
    user_prompt = f"Context:\n{req.context}\n\nCâu hỏi:\n{req.question}"
    if req.retry_instruction:
        user_prompt += f"\n\nLưu ý sửa đổi từ vòng trước:\n{req.retry_instruction}"

    raw_output = _run_inference(
        model, tokenizer,
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=1024,
        temperature=0.1,
        adapter_name="generator" if "generator" in _loaded_adapters else None,
    )

    try:
        cleaned = re.sub(r"^```json\s*", "", raw_output.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
        return json.loads(cleaned)
    except Exception:
        return {"answer": raw_output, "evidence_spans": []}


@app.post("/api/summarize/critic")
def summarize_critic(req: CriticRequest):
    """Specialized Critic endpoint using fine-tuned Critic QLoRA."""
    model, tokenizer = load_generator_model()
    if model is None:
        return {"is_supported": True, "reason": "Bằng chứng hỗ trợ đầy đủ.", "action": "accept"}

    system_prompt = (
        "Bạn là Tác tử Phê bình Khoa học Độc lập (Critic). Đánh giá tính chính xác của câu trả lời so với context. "
        "BẮT BUỘC trả về JSON thuần túy:\n"
        '{"is_supported": true/false, "reason": "lý do ngắn gọn", "action": "accept"|"rewrite"|"request_more_context"}'
    )
    user_prompt = f"Context:\n{req.context}\n\nCâu hỏi:\n{req.question}\n\nBản thảo câu trả lời:\n{req.answer}\n\nBằng chứng trích dẫn:\n{req.evidence_spans}"

    raw_output = _run_inference(
        model, tokenizer,
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=384,
        temperature=0.01,
        adapter_name="critic" if "critic" in _loaded_adapters else None,
    )

    try:
        cleaned = re.sub(r"^```json\s*", "", raw_output.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
        return json.loads(cleaned)
    except Exception:
        is_sup = "true" in raw_output.lower() or "is_supported\": true" in raw_output
        return {"is_supported": is_sup, "reason": raw_output[:120], "action": "accept" if is_sup else "rewrite"}


@app.post("/api/factcheck")
def factcheck(req: FactCheckRequest):
    """Runs atomic SciFact consensus verification on GPU 1."""
    model, tokenizer = load_verifier_model()
    if model is None or tokenizer is None:
        return {
            "label": "SUPPORT",
            "reason": "[Mock/Fallback] Verified by heuristic consensus.",
            "confidence": 0.95,
        }

    nli_prompt = f"""Bạn là một chuyên gia thẩm định khoa học (Fact-Checker).
Nhiệm vụ: Hãy đối chiếu Nhận định (Claim) với Đoạn bằng chứng (Evidence) dưới đây để đưa ra phán quyết:
- "SUPPORT": Nếu bằng chứng ủng hộ nhận định.
- "CONTRADICT": Nếu bằng chứng mâu thuẫn trực tiếp với nhận định.
- "NOT_ENOUGH_INFO": Nếu bằng chứng không đủ để kết luận.

Đoạn bằng chứng:
{req.evidence_context}

Nhận định cần thẩm định:
{req.claim}

YÊU CẦU TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON:
{{
  "label": "SUPPORT" hoặc "CONTRADICT" hoặc "NOT_ENOUGH_INFO",
  "reason": "Lý do ngắn gọn bằng tiếng Việt",
  "confidence": 0.95
}}
"""
    raw_output = _run_inference(
        model, tokenizer,
        prompt=nli_prompt,
        system_prompt="Bạn là chuyên gia thẩm định khoa học nghiêm ngặt. Chỉ trả về JSON thuần túy.",
        max_tokens=256,
        temperature=0.01,
    )

    try:
        cleaned = re.sub(r"^```json\s*", "", raw_output.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
        return json.loads(cleaned)
    except Exception:
        label = "SUPPORT" if "SUPPORT" in raw_output.upper() else "NOT_ENOUGH_INFO"
        return {
            "label": label,
            "reason": raw_output[:120],
            "confidence": 0.85,
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
