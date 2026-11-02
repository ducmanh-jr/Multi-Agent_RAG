# -*- coding: utf-8 -*-
"""
Dual-T4 Model & PEFT DoRA/QLoRA Builder for Agent 3 (Generator & Critic).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def get_dual_t4_device_map(max_mem: str = "14.5GiB"):
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus >= 2:
        return "balanced", {0: max_mem, 1: max_mem}
    elif num_gpus == 1:
        return {"": 0}, None
    return "auto", None


def build_summarizer_model_and_tokenizer(
    base_model_id: str,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    use_dora: bool = True,
    max_mem: str = "14.5GiB",
):
    """Loads 4-bit NF4 Qwen2.5-7B-Instruct balanced across Dual T4 and attaches DoRA adapter."""
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
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer, peft_config
