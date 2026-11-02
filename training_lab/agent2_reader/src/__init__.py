"""Agent 2 (Document Reader — LayoutLMv3 DocLayNet) Fine-Tuning Package (Kaggle Dual T4)."""
from .config import Agent2TrainingConfig, DOCLAYNET_LABELS
from .data_loader import load_doclaynet_dataset
from .model_builder import build_layoutlmv3_model
from .trainer import run_reader_training
from .evaluator import evaluate_layout_parser
