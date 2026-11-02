"""
Hybrid Mode (6th Canonical Mode - Flagship L4 Adaptive Supervisor & Cross-Model Verification).
Combines:
  - Local ThinkBook G8+ (AMD Ryzen 7 H 255 + Radeon 780M Vulkan) for zero-latency PDF reading & Hybrid Retrieval.
  - Dynamic Capability Routing across Kaggle 32B / Multi-LoRA, Gemini 2.5 Flash, and Local 7B Vulkan.
  - Cross-Model Adversarial Verification (Generator and Verifier strictly use different model families).
"""

from modes.hybrid.retrieval import HybridRetrievalAgent
from modes.hybrid.reader import HybridReaderAgent
from modes.hybrid.summarizer import HybridSummarizerAgent
from modes.hybrid.fact_checker import HybridFactCheckerAgent
from modes.hybrid.writer import HybridWriterAgent

__all__ = [
    "HybridRetrievalAgent",
    "HybridReaderAgent",
    "HybridSummarizerAgent",
    "HybridFactCheckerAgent",
    "HybridWriterAgent",
]
