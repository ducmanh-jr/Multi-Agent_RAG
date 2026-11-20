"""
Tool Registry & Permission Guard for TASK_EXECUTION (AGENT.md Sections 4, 7, 18).

Provides the execution boundary for real tool invocations:
  - Explicit tool registration with input schema validation.
  - PermissionGuard enforcing L0-L4 authorization rules (L3/L4 require explicit approval).
  - Clear error separation: ToolNotFoundError vs ToolPermissionDeniedError vs ToolExecutionError.
  - Never fabricates or simulates tool execution when a tool is missing or denied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from utils.logger import logger

PermissionLevel = Literal["L0", "L1", "L2", "L3", "L4"]

_LEVEL_RANK: dict[str, int] = {
    "L0": 0,
    "L1": 1,
    "L2": 2,
    "L3": 3,
    "L4": 4,
}


class ToolError(RuntimeError):
    """Base error for all ToolRegistry failures."""
    error_code: str = "TOOL_ERROR"


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not registered or is disabled."""
    error_code: str = "TOOL_NOT_FOUND"


class ToolPermissionDeniedError(ToolError):
    """Raised when the caller lacks permission or explicit approval to execute a tool."""
    error_code: str = "PERMISSION_DENIED"


class ToolExecutionError(ToolError):
    """Raised when input validation fails or the tool handler raises at runtime."""
    error_code: str = "TOOL_EXECUTION_FAILED"


@dataclass
class ToolSpec:
    """Declarative specification for a registered executable tool."""
    name: str
    description: str
    handler: Callable[[dict[str, Any]], Any]
    input_schema: dict[str, Any] = field(default_factory=dict)
    permission_level: PermissionLevel = "L0"
    enabled: bool = True

    def __post_init__(self) -> None:
        self.name = (self.name or "").strip().lower()
        if not self.name:
            raise ValueError("ToolSpec requires a non-empty 'name'.")
        if self.permission_level not in _LEVEL_RANK:
            raise ValueError(
                f"Invalid permission_level {self.permission_level!r}. Expected one of {sorted(_LEVEL_RANK)}."
            )
        if not callable(self.handler):
            raise ValueError(f"ToolSpec '{self.name}' handler must be callable.")


class PermissionGuard:
    """Enforces AGENT.md Section 4 (L0 -> L4) authorization boundaries before tool execution."""

    def check(
        self,
        tool: ToolSpec,
        granted_level: PermissionLevel = "L0",
        approved_tools: Optional[set[str]] = None,
    ) -> None:
        if granted_level not in _LEVEL_RANK:
            raise ToolPermissionDeniedError(
                f"Invalid granted permission level '{granted_level}' for tool '{tool.name}'."
            )

        approved = {t.strip().lower() for t in (approved_tools or set())}

        # L3 and L4 are strictly approval-gated per AGENT.md Section 4
        if tool.permission_level in ("L3", "L4") and tool.name not in approved:
            raise ToolPermissionDeniedError(
                f"Tool '{tool.name}' requires explicit user approval (level {tool.permission_level})."
            )

        if _LEVEL_RANK[granted_level] < _LEVEL_RANK[tool.permission_level]:
            raise ToolPermissionDeniedError(
                f"Tool '{tool.name}' requires permission '{tool.permission_level}', "
                f"but caller only has '{granted_level}'."
            )


class ToolRegistry:
    """Central registry controlling discovery, schema validation, permission checks, and execution."""

    def __init__(self, permission_guard: Optional[PermissionGuard] = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._guard = permission_guard or PermissionGuard()

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def unregister(self, name: str) -> None:
        self._tools.pop((name or "").strip().lower(), None)

    def has_tool(self, name: str) -> bool:
        key = (name or "").strip().lower()
        spec = self._tools.get(key)
        return bool(spec and spec.enabled)

    def get_tool(self, name: str) -> ToolSpec:
        key = (name or "").strip().lower()
        spec = self._tools.get(key)
        if not spec or not spec.enabled:
            raise ToolNotFoundError(
                f"Tool '{key or 'unspecified'}' is not registered or is currently disabled."
            )
        return spec

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "permission_level": spec.permission_level,
                "input_schema": spec.input_schema,
                "enabled": spec.enabled,
            }
            for spec in self._tools.values()
        ]

    def validate_inputs(self, spec: ToolSpec, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ToolExecutionError(
                f"Tool '{spec.name}' expects a dict payload, got {type(payload).__name__}."
            )
        schema = spec.input_schema or {}
        required = schema.get("required") or []
        missing = [k for k in required if k not in payload or payload[k] is None]
        if missing:
            raise ToolExecutionError(
                f"Tool '{spec.name}' missing required input fields: {missing}."
            )
        properties = schema.get("properties") or {}
        type_map: dict[str, type | tuple[type, ...]] = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        for key, prop_spec in properties.items():
            if key not in payload or payload[key] is None:
                continue
            expected_name = prop_spec.get("type") if isinstance(prop_spec, dict) else None
            if expected_name and expected_name in type_map:
                val = payload[key]
                expected_type = type_map[expected_name]
                if expected_name in ("integer", "number") and isinstance(val, bool):
                    raise ToolExecutionError(
                        f"Tool '{spec.name}' field '{key}' expected {expected_name}, got bool."
                    )
                if not isinstance(val, expected_type):
                    raise ToolExecutionError(
                        f"Tool '{spec.name}' field '{key}' expected {expected_name}, got {type(val).__name__}."
                    )

    def execute(
        self,
        name: str,
        payload: Optional[dict[str, Any]] = None,
        granted_level: PermissionLevel = "L0",
        approved_tools: Optional[set[str]] = None,
    ) -> dict[str, Any]:
        spec = self.get_tool(name)
        self._guard.check(spec, granted_level=granted_level, approved_tools=approved_tools)
        safe_payload = payload if payload is not None else {}
        self.validate_inputs(spec, safe_payload)

        try:
            output = spec.handler(safe_payload)
        except ToolError:
            raise
        except Exception as exc:
            logger.error(f"[ToolRegistry] Tool '{spec.name}' execution failed: {exc}")
            raise ToolExecutionError(f"Tool '{spec.name}' failed during execution: {exc}") from exc

        return {
            "status": "SUCCESS",
            "tool": spec.name,
            "permission_level": spec.permission_level,
            "output": output,
        }


# Default global registry (intentionally empty until tools are explicitly registered)
tool_registry = ToolRegistry()
