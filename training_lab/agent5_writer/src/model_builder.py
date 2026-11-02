# -*- coding: utf-8 -*-
"""
SOTA Dual-T4 Model Builder for Agent 5 (Academic Writer) with 4-bit NF4, SDPA, DoRA & rsLoRA.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .config import Agent5TrainingConfig


def get_dual_t4_device_map(max_mem: str = "14.5GiB"):
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus >= 2:
        return "balanced", {0: max_mem, 1: max_mem}
    elif num_gpus == 1:
        return {"": 0}, None
    return "auto", None


def build_writer_model_and_tokenizer(cfg: Agent5TrainingConfig):
    """Initializes Qwen2.5-7B-Instruct with SDPA Attention and attaches DoRA + rsLoRA adapter."""
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
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        use_dora=cfg.use_dora,
        use_rslora=cfg.use_rslora,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer, peft_config
