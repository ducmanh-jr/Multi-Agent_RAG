"""
Step 2 of Kaggle FN v2 Fine-Tuning Pipeline:
Trains the 3 QLoRA Specialists ('router', 'generator' SFT+DPO, 'critic' SFT) on Qwen/Qwen2.5-7B-Instruct
and the 'agent4_nli' Verifier on MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
using Kaggle Dual Tesla T4 (2 x 16 GB VRAM, float16 + NF4 4-bit).

Usage on Kaggle:
  python 02_train_multilora_sft_dpo.py --adapter all --data_dir ./datasets --output_dir ./kaggle_fn_v2_weights
"""

from __future__ import annotations

import argparse
import os


def train_qlora_sft(
    base_model: str,
    train_jsonl: str,
    output_dir: str,
    lora_r: int = 32,
    lora_alpha: int = 64,
    epochs: int = 3,
    lr: float = 2e-4,
):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # T4 (sm_75) uses float16
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)

    dataset = load_dataset("json", data_files=train_jsonl, split="train")

    def _format_chat(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = dataset.map(_format_chat)

    sft_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=3072,
        neftune_noise_alpha=5.0,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_args,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[OK] Saved QLoRA SFT adapter to: {output_dir}")


def train_generator_dpo(
    base_model: str,
    sft_adapter_dir: str,
    dpo_jsonl: str,
    output_dir: str,
    beta: float = 0.1,
    lr: float = 5e-6,
):
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, sft_adapter_dir, is_trainable=True)
    dataset = load_dataset("json", data_files=dpo_jsonl, split="train")

    dpo_args = DPOConfig(
        output_dir=output_dir,
        beta=beta,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=2560,
        max_prompt_length=1792,
        report_to="none",
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    print(f"[OK] Saved DPO-aligned Generator adapter to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", choices=["router", "generator", "critic", "all"], default="all")
    parser.add_argument("--data_dir", default="training_lab/kaggle_fn_v2/datasets")
    parser.add_argument("--output_dir", default="training_lab/kaggle_fn_v2/weights")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.adapter in ("router", "all"):
        train_qlora_sft(
            args.base_model,
            os.path.join(args.data_dir, "router_sft.jsonl"),
            os.path.join(args.output_dir, "agent1_router"),
        )
    if args.adapter in ("generator", "all"):
        sft_dir = os.path.join(args.output_dir, "generator_qlora_sft")
        final_gen_dir = os.path.join(args.output_dir, "generator_qlora")
        train_qlora_sft(
            args.base_model,
            os.path.join(args.data_dir, "generator_sft.jsonl"),
            sft_dir,
        )
        train_generator_dpo(
            args.base_model,
            sft_dir,
            os.path.join(args.data_dir, "generator_dpo.jsonl"),
            final_gen_dir,
        )
    if args.adapter in ("critic", "all"):
        train_qlora_sft(
            args.base_model,
            os.path.join(args.data_dir, "critic_sft.jsonl"),
            os.path.join(args.output_dir, "critic_qlora"),
        )
