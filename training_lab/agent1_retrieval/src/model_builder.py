# -*- coding: utf-8 -*-
"""
Model & PEFT QLoRA/DoRA Builder for Agent 1 (Retrieval Router) on Kaggle Dual T4.
Loads `Qwen/Qwen2.5-7B-Instruct` in 4-bit NF4 (~4.8GB) on `cuda:0` for zero-latency, single-device execution.
Ensures all trainable adapter parameters are strictly `torch.float16` for native Tesla T4 FP16 GradScaler support.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .config import Agent1TrainingConfig


def get_dual_t4_device_map(max_mem: str = "14.5GiB"):
    """Pins 4-bit model to current CUDA device (~5GB fits in 16GB VRAM) to avoid cross-device tensor mismatch in Trainer."""
    if torch.cuda.is_available():
        target = torch.cuda.current_device()
        print(f"⚡ [Agent 1 ModelBuilder] Pinning 4-bit Qwen2.5-7B to cuda:{target} (~5.2GB / 16GB VRAM)...")
        return {"": target}, None
    return "auto", None


def build_router_model_and_tokenizer(cfg: Agent1TrainingConfig):
    """Initializes 4-bit NF4 Qwen2.5-7B-Instruct and attaches DoRA/QLoRA adapters in pure FP16."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    dev_map, max_mem = get_dual_t4_device_map(cfg.max_memory_per_gpu)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        quantization_config=bnb_config,
        device_map=dev_map,
        max_memory=max_mem,
        torch_dtype=torch.float16,
        attn_implementation=cfg.attn_implementation,
        trust_remote_code=True,
    )
    model.config.torch_dtype = torch.float16
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        use_dora=cfg.use_dora,
        use_rslora=cfg.use_rslora,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Đảm bảo 100% tham số trainable LoRA/DoRA ở định dạng FP16 (tránh lỗi GradScaler BFloat16 trên Tesla T4):
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float16)

    model.print_trainable_parameters()
    return model, tokenizer, peft_config
