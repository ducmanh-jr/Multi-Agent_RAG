"""Task state and dependency rules for research plans."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from graphlib import CycleError, TopologicalSorter
from threading import Lock
from types import MappingProxyType


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"

    @property
    def is_terminal(self) -> bool:
        return self in {self.COMPLETED, self.FAILED, self.CANCELLED, self.SKIPPED}


_TRANSITIONS = {
    TaskStatus.PENDING: {TaskStatus.RUNNING, TaskStatus.CANCELLED, TaskStatus.SKIPPED},
    TaskStatus.RUNNING: {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED},
}


class DependencyStatus(StrEnum):
    READY = "READY"
    WAITING = "WAITING"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class DependencyCheck:
    status: DependencyStatus
    waiting_for: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResearchTask:
    id: str
    depends_on_success: tuple[str, ...] = ()
    depends_on_terminal: tuple[str, ...] = ()
    status: TaskStatus = TaskStatus.PENDING

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Task ID must be a nonempty string.")
        for field in ("depends_on_success", "depends_on_terminal"):
            dependencies = getattr(self, field)
            if isinstance(dependencies, str) or not isinstance(dependencies, Iterable):
                raise ValueError(f"{field} must contain nonempty task IDs.")
            dependencies = tuple(dependencies)
            if any(not isinstance(task_id, str) or not task_id.strip() for task_id in dependencies):
                raise ValueError(f"{field} must contain nonempty task IDs.")
            object.__setattr__(self, field, tuple(dict.fromkeys(dependencies)))
        object.__setattr__(self, "status", TaskStatus(self.status))

    @property
    def dependencies(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.depends_on_success, *self.depends_on_terminal)))


class ResearchPlan:
    """A validated DAG with atomic task status transitions."""

    def __init__(self, tasks: Iterable[ResearchTask]):
        self._tasks: dict[str, ResearchTask] = {}
        for task in tasks:
            if task.id in self._tasks:
                raise ValueError(f"Duplicate task ID {task.id!r}.")
            self._tasks[task.id] = task
        for task in self._tasks.values():
            for dependency in task.dependencies:
                if dependency not in self._tasks:
                    raise ValueError(f"Task {task.id!r} depends on unknown task {dependency!r}.")
        try:
            TopologicalSorter({task.id: task.dependencies for task in self._tasks.values()}).prepare()
        except CycleError as exc:
            cycle = " -> ".join(exc.args[1])
            raise ValueError(f"Task dependencies contain a cycle: {cycle}.") from exc
        for task in self._tasks.values():
            if task.status in {TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.FAILED}:
                if self._check_dependencies(task).status != DependencyStatus.READY:
                    raise ValueError(f"Started task {task.id!r} has unsatisfied dependencies.")
        self._lock = Lock()

    @property
    def tasks(self) -> Mapping[str, ResearchTask]:
        with self._lock:
            return MappingProxyType(self._tasks.copy())

    def _check_dependencies(self, task: ResearchTask) -> DependencyCheck:
        blocked_by = tuple(
            dependency for dependency in task.depends_on_success
            if self._tasks[dependency].status.is_terminal
            and self._tasks[dependency].status != TaskStatus.COMPLETED
        )
        waiting_for = tuple(
            dependency for dependency in task.dependencies if not self._tasks[dependency].status.is_terminal
        )
        if blocked_by:
            return DependencyCheck(DependencyStatus.BLOCKED, waiting_for, blocked_by)
        if waiting_for:
            return DependencyCheck(DependencyStatus.WAITING, waiting_for)
        return DependencyCheck(DependencyStatus.READY)

    def check_dependencies(self, task_id: str) -> DependencyCheck:
        """Return unmet dependencies even when another dependency already blocks the task."""
        with self._lock:
            return self._check_dependencies(self._tasks[task_id])

    def transition(self, task_id: str, status: TaskStatus) -> ResearchTask:
        """Apply a lifecycle transition, checking dependencies before starting a task."""
        target = TaskStatus(status)
        with self._lock:
            task = self._tasks[task_id]
            if target not in _TRANSITIONS.get(task.status, ()):
                raise ValueError(f"Task {task_id!r} cannot transition from {task.status} to {target}.")
            if target == TaskStatus.RUNNING:
                check = self._check_dependencies(task)
                if check.status != DependencyStatus.READY:
                    raise ValueError(f"Task {task_id!r} cannot start with {check.status} dependencies.")
            updated = replace(task, status=target)
            self._tasks[task_id] = updated
            return updated
