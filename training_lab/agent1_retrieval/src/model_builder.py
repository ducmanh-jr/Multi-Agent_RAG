# -*- coding: utf-8 -*-
"""
Model & PEFT QLoRA/DoRA Builder for Agent 1 (Retrieval Router) on Kaggle Dual T4.
Distributes `Qwen/Qwen2.5-7B-Instruct` across `cuda:0` and `cuda:1` via `device_map="balanced"`.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .config import Agent1TrainingConfig


def get_dual_t4_device_map(max_mem: str = "14.5GiB"):
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus >= 2:
        print(f"⚡ [Agent 1 ModelBuilder] Detected {num_gpus} GPUs (Kaggle T4x2). Balancing across cuda:0 & cuda:1 ({max_mem}/GPU)...")
        return "balanced", {0: max_mem, 1: max_mem}
    elif num_gpus == 1:
        return {"": 0}, None
    return "auto", None


def build_router_model_and_tokenizer(cfg: Agent1TrainingConfig):
    """Initializes 4-bit NF4 Qwen2.5-7B-Instruct and attaches DoRA/QLoRA adapters."""
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
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        use_dora=cfg.use_dora,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer, peft_config
