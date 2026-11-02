"""Agent 5 (Academic Report Writer & Citation Synthesizer) Fine-Tuning Package (Kaggle Dual T4)."""
from .config import Agent5TrainingConfig
from .data_loader import load_writer_sft_dataset, load_writer_dpo_dataset
from .model_builder import build_writer_model_and_tokenizer
from .trainer import run_writer_sft_and_rpo_training
from .evaluator import evaluate_academic_writer
