"""
Abstract Base Agent Protocol for the 5 Specialized Agents.
Every Agent class must implement:
  - name: Identification string
  - max_internal_retry: Limit for internal repair loops
  - run(state: AgentState) -> dict[str, Any]
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

from core.config import agent_inference_mode, get_run_settings
from core.state import AgentState
from utils.logger import logger


MAX_SPECIALIST_TOOL_STEPS = 4


class BaseAgent(ABC):
    """
    Unified abstract base class for all Specialist Agent nodes in the LangGraph StateGraph.
    Supports typed `AgentTask` invocation (`run_task`) and bounded local tool loops
    (`run_bounded_tool_loop`) without adding extra LLM calls in deterministic CPU modes.
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

    def run_task(self, state: AgentState, task: Any = None) -> dict[str, Any]:
        """
        Execute a typed `AgentTask` (or dict representation) by scoping handoff inputs when present
        and delegating to `self.run(...)`, preserving full backward compatibility.
        """
        if task is None:
            return self.run(state)
        request = None
        if hasattr(task, "to_handoff_request") and callable(task.to_handoff_request):
            request = task.to_handoff_request()
        elif isinstance(task, Mapping):
            from core.agent_protocol import AgentTask

            request = AgentTask.from_dict(task).to_handoff_request()
        if request:
            from core.handoff_inputs import prepare_handoff_input

            scoped = prepare_handoff_input(state, request)
            return self.run(scoped)
        return self.run(state)

    def run_bounded_tool_loop(
        self,
        tools: Mapping[str, Callable[..., Any]],
        planner_fn: Callable[[int, list[dict[str, Any]]], tuple[str, dict[str, Any]] | None],
        *,
        max_steps: int = 3,
    ) -> dict[str, Any]:
        """
        Execute a bounded specialist-local tool loop (max 1..4 steps).
        `planner_fn(step_index, history)` selects `(tool_name, kwargs)` or returns `None` to stop.
        Records structured step outcomes and `tools_used` without crashing on individual tool errors.
        """
        cap = max(1, min(int(max_steps or 3), MAX_SPECIALIST_TOOL_STEPS))
        history: list[dict[str, Any]] = []
        tools_used: list[str] = []
        stopped_reason = "max_steps_reached"

        for step_idx in range(cap):
            proposal = planner_fn(step_idx, history)
            if proposal is None:
                stopped_reason = "completed"
                break
            tool_name, tool_kwargs = proposal
            clean_name = str(tool_name or "").strip()
            tool_fn = tools.get(clean_name)
            if tool_fn is None:
                history.append({
                    "step": step_idx,
                    "tool": clean_name,
                    "ok": False,
                    "error": f"unknown_tool:{clean_name}",
                })
                stopped_reason = "unknown_tool"
                break
            if clean_name not in tools_used:
                tools_used.append(clean_name)
            try:
                output = tool_fn(**(dict(tool_kwargs) if isinstance(tool_kwargs, Mapping) else {}))
                history.append({
                    "step": step_idx,
                    "tool": clean_name,
                    "ok": True,
                    "output": output,
                })
            except Exception as exc:
                self.logger.warning("[%s] Bounded tool '%s' failed on step %d: %s", self.name, clean_name, step_idx, exc)
                history.append({
                    "step": step_idx,
                    "tool": clean_name,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        return {
            "steps": history,
            "tools_used": tools_used,
            "step_count": len(history),
            "max_steps": cap,
            "stopped_reason": stopped_reason,
        }

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
