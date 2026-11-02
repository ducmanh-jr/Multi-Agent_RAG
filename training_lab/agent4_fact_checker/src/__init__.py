"""Agent 4 (Fact-Checker & NLI Consensus Verifier) Fine-Tuning Package (Kaggle Dual T4)."""
from .config import Agent4TrainingConfig, LABEL2ID, ID2LABEL
from .data_loader import load_nli_dataset
from .model_builder import build_nli_cross_encoder
from .trainer import WeightedCrossEncoderTrainer, run_factchecker_training
from .evaluator import compute_nli_metrics, calibrate_temperature_ece
