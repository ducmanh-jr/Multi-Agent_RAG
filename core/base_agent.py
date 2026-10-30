"""
Abstract Base Agent Protocol for the 5 Specialized Agents.
Every Agent class must implement:
  - name: Identification string
  - max_internal_retry: Limit for internal repair loops
  - run(state: AgentState) -> dict[str, Any]
"""

from abc import ABC, abstractmethod
from typing import Any

from core.config import settings
from core.state import AgentState
from utils.logger import logger


class BaseAgent(ABC):
    """
    Unified abstract base class for all 5 Agent nodes in the LangGraph StateGraph.
    """

    def __init__(self, name: str, max_internal_retry: int = settings.DEFAULT_MAX_INTERNAL_RETRY):
        self.name = name
        self.max_internal_retry = max_internal_retry
        self.logger = logger
        self.settings = settings

    @abstractmethod
    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Execute the agent's responsibilities and return a dictionary of state updates.
        """
        pass

    def log_action(self, action: str, details: Any = None) -> dict[str, Any]:
        """Helper to format a structured reasoning trace record."""
        record = {
            "agent": self.name,
            "action": action,
            "details": details,
        }
        self.logger.info(f"[{self.name}] {action}")
        return record
