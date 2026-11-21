"""
Abstract Base Agent Protocol for the 5 Specialized Agents.
Every Agent class must implement:
  - name: Identification string
  - max_internal_retry: Limit for internal repair loops
  - run(state: AgentState) -> dict[str, Any]
"""

from abc import ABC, abstractmethod
from typing import Any

from core.config import agent_inference_mode, get_run_settings
from core.state import AgentState
from utils.logger import logger


class BaseAgent(ABC):
    """
    Unified abstract base class for all 5 Agent nodes in the LangGraph StateGraph.
    """

    def __init__(self, name: str, max_internal_retry: int | None = None):
        self.name = name
        self.max_internal_retry = (
            self.settings.DEFAULT_MAX_INTERNAL_RETRY if max_internal_retry is None else max_internal_retry
        )
        self.logger = logger

    @property
    def settings(self):
        return get_run_settings()

    @abstractmethod
    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Execute the agent's responsibilities and return a dictionary of state updates.
        """
        pass

    def run_in_mode(self, state: AgentState, mode: str) -> dict[str, Any]:
        """Run a delegated specialist with its own inference settings."""
        with agent_inference_mode(mode):
            return self.run(state)

    def log_action(self, action: str, details: Any = None) -> dict[str, Any]:
        """Helper to format a structured reasoning trace record."""
        import time
        record = {
            "agent": self.name,
            "action": action,
            "details": details,
            "timestamp": time.time(),
        }
        self.logger.info(f"[{self.name}] {action}")
        return record
