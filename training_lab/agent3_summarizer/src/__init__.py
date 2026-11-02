"""Agent 3 (Academic Summarizer: Generator SFT+DPO & Critic Hard-Negatives) Package (Kaggle Dual T4)."""
from .config import Agent3TrainingConfig
from .data_loader import load_sft_jsonl_dataset, load_dpo_jsonl_dataset
from .model_builder import build_summarizer_model_and_tokenizer
from .trainer import train_generator_sft_and_dpo, train_critic_sft
from .evaluator import evaluate_generator_and_critic
