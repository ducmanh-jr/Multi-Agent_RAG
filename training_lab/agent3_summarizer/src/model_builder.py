# -*- coding: utf-8 -*-
"""
Model Builder for Agent 3 (Academic Summarizer & Critic).
Loads `Qwen/Qwen2.5-7B-Instruct` in 4-bit NF4 (~4.8GB) on `cuda:0` for zero-latency, single-device execution.
Ensures all trainable adapter parameters are strictly `torch.float32` master weights for PyTorch AMP GradScaler.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def get_dual_t4_device_map(max_mem: str = "14.5GiB"):
    """Pins 4-bit model to current CUDA device (~5GB fits in 16GB VRAM) to avoid cross-device tensor mismatch in Trainer."""
    if torch.cuda.is_available():
        target = torch.cuda.current_device()
        print(f"⚡ [Agent 3 ModelBuilder] Pinning 4-bit Qwen2.5-7B to cuda:{target} (~5.2GB / 16GB VRAM)...")
        return {"": target}, None
    return "auto", None


def build_summarizer_model_and_tokenizer(
    base_model_id: str,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    use_dora: bool = True,
    max_mem: str = "14.5GiB",
):
    """Loads 4-bit NF4 Qwen2.5-7B-Instruct on current CUDA device and attaches DoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    dev_map, mem_map = get_dual_t4_device_map(max_mem)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map=dev_map,
        max_memory=mem_map,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.torch_dtype = torch.float16
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        use_rslora=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Chuẩn AMP của PyTorch trên Tesla T4: Trọng số LoRA trainable phải ở float32 để GradScaler unscale:
    for param in model.parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    model.print_trainable_parameters()
    return model, tokenizer, peft_config
