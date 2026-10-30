from abc import ABC, abstractmethod
from typing import Any
from core.state import AgentState
from core.config import settings
from utils.logger import logger

class BaseAgent(ABC):
    """
    Abstract Base Class chuẩn cho toàn bộ các Agent trong v2.
    Mỗi Agent chịu trách nhiệm nhận AgentState, suy luận và trả về dict update.
    """
    def __init__(self, name: str, max_internal_retry: int = settings.DEFAULT_MAX_INTERNAL_RETRY):
        self.name = name
        self.max_internal_retry = max_internal_retry
        self.logger = logger

    @abstractmethod
    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Thực thi nhiệm vụ và trả về dictionary cập nhật cho AgentState.
        """
        pass
