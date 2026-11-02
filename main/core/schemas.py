"""
Unified Schemas & Contracts (Single Source of Truth) across:
  - training_lab/agent1_retrieval/schemas.py
  - training_lab/agent3_summarizer/Code/prompts_generator.py & prompts_critic.py
  - All 6 Execution Modes: cpu_min, cpu_max, gemini, kaggle_fn, kaggle_max, hybrid
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


IntentType = Literal[
    "summarize",
    "compare",
    "methodology",
    "results",
    "limitation",
    "code_request",
    "general_question",
    "academic_research",
    "numeric_fact",
]

CriticActionType = Literal[
    "approve",
    "rewrite",
    "request_more_context",
    "lower_confidence",
]

ComplexityLevel = Literal["LOW", "MEDIUM", "HIGH"]


# Canonical Adapter Name Map between main/modes/kaggle_fn and kaggle/kaggle_worker.ipynb
ADAPTER_ALIASES: dict[str, str] = {
    "router": "router",
    "agent1_router": "router",
    "retrieval": "router",
    "fn_router_planner": "router",
    "generator": "generator",
    "generator_qlora": "generator",
    "summarizer": "generator",
    "fn_generator": "generator",
    "critic": "critic",
    "critic_qlora": "critic",
    "factchecker": "critic",
    "fn_critic": "critic",
    "writer": "generator",
    "fn_writer": "generator",
}


def canonical_adapter_name(name: Optional[str]) -> str:
    """Normalizes any adapter alias ('generator_qlora', 'agent1_router', etc.) to canonical worker adapter name."""
    if not name:
        return "generator"
    clean = name.strip().lower()
    return ADAPTER_ALIASES.get(clean, clean)


@dataclass
class RouterPlanOutput:
    """Output contract of Agent 0 / Agent 1 Router & Supervisor Planner (matches training_lab/agent1_retrieval/schemas.py)."""
    intent: str = "academic_research"
    complexity: ComplexityLevel = "MEDIUM"
    sub_queries: list[str] = field(default_factory=list)
    required_entities: list[str] = field(default_factory=list)
    is_numeric_query: bool = False
    is_comparative_query: bool = False
    alpha: float = 0.5
    top_k: int = 6
    reasoning: str = ""
    recommended_generator: str = "auto"
    recommended_verifier: str = "auto"

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_query: str = "") -> "RouterPlanOutput":
        if not isinstance(data, dict):
            data = {}
        subs = data.get("sub_queries") or ([default_query] if default_query else [])
        if isinstance(subs, str):
            subs = [subs]
        subs = [str(s).strip() for s in subs if str(s).strip()][:4]
        if not subs and default_query:
            subs = [default_query]
        try:
            alpha = float(data.get("alpha", 0.5))
            alpha = max(0.0, min(1.0, alpha))
        except Exception:
            alpha = 0.5
        try:
            top_k = int(data.get("top_k", 6))
            top_k = max(2, min(15, top_k))
        except Exception:
            top_k = 6

        complexity = str(data.get("complexity", "MEDIUM")).upper()
        if complexity not in ("LOW", "MEDIUM", "HIGH"):
            complexity = "MEDIUM"

        return cls(
            intent=str(data.get("intent", "academic_research")),
            complexity=complexity,  # type: ignore[arg-type]
            sub_queries=subs,
            required_entities=[str(x) for x in (data.get("required_entities") or [])],
            is_numeric_query=bool(data.get("is_numeric_query", False)),
            is_comparative_query=bool(data.get("is_comparative_query", False)),
            alpha=alpha,
            top_k=top_k,
            reasoning=str(data.get("reasoning", "")),
            recommended_generator=str(data.get("recommended_generator", "auto")),
            recommended_verifier=str(data.get("recommended_verifier", "auto")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SummarizerGeneratorOutput:
    """Output contract of Summarizer Generator (matches training_lab/agent3_summarizer/Code/prompts_generator.py)."""
    answer: str = ""
    evidence_spans: list[Any] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    thinking: str = ""
    model_family: str = "unknown"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SummarizerGeneratorOutput":
        if not isinstance(data, dict):
            return cls(answer=str(data))
        ans = str(data.get("answer") or data.get("summary") or data.get("raw_text") or "").strip()
        # Extract <think>...</think> if embedded in answer
        thinking = str(data.get("thinking") or "").strip()
        think_match = re.search(r"<think>(.*?)</think>", ans, flags=re.DOTALL)
        if think_match:
            if not thinking:
                thinking = think_match.group(1).strip()
            ans = re.sub(r"<think>.*?</think>", "", ans, flags=re.DOTALL).strip()

        ev_spans = data.get("evidence_spans") or []
        if not isinstance(ev_spans, list):
            ev_spans = []
        claims = data.get("claims") or []
        if not isinstance(claims, list):
            claims = []
        return cls(
            answer=ans,
            evidence_spans=ev_spans,
            claims=claims,
            thinking=thinking,
            model_family=str(data.get("model_family", "unknown")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SummarizerCriticOutput:
    """Output contract of Summarizer Critic (matches training_lab/agent3_summarizer/Code/prompts_critic.py)."""
    is_supported: bool = True
    unsupported_claims: list[str] = field(default_factory=list)
    missing_key_points: list[str] = field(default_factory=list)
    action: CriticActionType = "approve"
    confidence: float = 0.9
    critique_rationale: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SummarizerCriticOutput":
        if not isinstance(data, dict):
            return cls()
        is_sup = bool(data.get("is_supported", True))
        unsup = (
            data.get("unsupported_claims")
            or data.get("Unsupported_claims")
            or data.get("rejected_claims")
            or []
        )
        if not isinstance(unsup, list):
            unsup = [str(unsup)]
        missing = (
            data.get("missing_key_points")
            or data.get("Missing_key_points")
            or data.get("evidence_gaps")
            or []
        )
        if not isinstance(missing, list):
            missing = [str(missing)]

        raw_action = str(data.get("action") or ("approve" if is_sup else "rewrite")).lower().strip()
        if raw_action not in ("approve", "rewrite", "request_more_context", "lower_confidence"):
            raw_action = "approve" if is_sup else "rewrite"

        try:
            conf = float(data.get("confidence", 0.9 if is_sup else 0.45))
        except Exception:
            conf = 0.9 if is_sup else 0.45

        return cls(
            is_supported=is_sup,
            unsupported_claims=[str(u) for u in unsup if str(u).strip()],
            missing_key_points=[str(m) for m in missing if str(m).strip()],
            action=raw_action,  # type: ignore[arg-type]
            confidence=conf,
            critique_rationale=str(data.get("critique_rationale") or data.get("reason") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_supported": self.is_supported,
            "unsupported_claims": self.unsupported_claims,
            "Unsupported_claims": self.unsupported_claims,
            "missing_key_points": self.missing_key_points,
            "Missing_key_points": self.missing_key_points,
            "action": self.action,
            "confidence": self.confidence,
            "critique_rationale": self.critique_rationale,
        }
