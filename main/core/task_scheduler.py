"""Bounded asynchronous execution of a validated research DAG."""

import asyncio
import math
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from typing import Any

from core.research_tasks import DependencyStatus, ResearchPlan, ResearchTask, TaskStatus


class TaskTimeoutError(TimeoutError):
    """One executor attempt exceeded its time limit."""


class ScheduleTimeoutError(TimeoutError):
    """The schedule exhausted its overall time limit."""


def _seconds(value: float, name: str, *, allow_zero: bool = False) -> float:
    try:
        seconds = float(value) if type(value) in {int, float} else math.nan
    except OverflowError:
        seconds = math.inf
    if not math.isfinite(seconds) or seconds < 0 or (seconds == 0 and not allow_zero):
        raise ValueError(f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} number.")
    return seconds


@dataclass(frozen=True)
class TaskPolicy:
    """Limit each attempt and retry only configured exceptions after it has finished."""

    timeout_seconds: float | None = None
    max_retries: int = 0
    retry_delay_seconds: float = 0
    retry_on: tuple[type[Exception], ...] = (TimeoutError,)

    def __post_init__(self):
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer.")
        if self.timeout_seconds is not None:
            timeout = _seconds(self.timeout_seconds, "timeout_seconds")
            object.__setattr__(self, "timeout_seconds", timeout)
        delay = _seconds(self.retry_delay_seconds, "retry_delay_seconds", allow_zero=True)
        object.__setattr__(self, "retry_delay_seconds", delay)
        if not isinstance(self.retry_on, Iterable):
            raise ValueError("retry_on must contain Exception classes.")
        retry_on = tuple(self.retry_on)
        if any(not isinstance(error, type) or not issubclass(error, Exception) for error in retry_on):
            raise ValueError("retry_on must contain Exception classes.")
        object.__setattr__(self, "retry_on", retry_on)


@dataclass(frozen=True)
class TaskOutcome:
    status: TaskStatus
    output: Any = None
    error: Exception | None = None
    blocked_by: tuple[str, ...] = ()
    attempts: int = 0


TaskExecutor = Callable[[ResearchTask, Mapping[str, TaskOutcome]], Awaitable[Any]]


async def _execute(
    executor: TaskExecutor, task: ResearchTask, dependencies: Mapping[str, TaskOutcome],
    policy: TaskPolicy, attempt_counts: dict[str, int],
) -> TaskOutcome:
    execution = asyncio.current_task()
    clock = asyncio.get_running_loop().time
    attempt = 0
    while True:
        attempt += 1
        attempt_counts[task.id] = attempt
        deadline = asyncio.timeout(policy.timeout_seconds)
        error = None
        output = None
        try:
            async with deadline:
                output = await executor(task, dependencies)
        except Exception as exc:
            error = exc
        if execution is not None and execution.cancelling():
            raise asyncio.CancelledError
        expires_at = deadline.when()
        if deadline.expired() or (expires_at is not None and clock() >= expires_at):
            error = TaskTimeoutError(f"Task {task.id!r} attempt {attempt} timed out.")
        if error is None:
            return TaskOutcome(TaskStatus.COMPLETED, output=output, attempts=attempt)
        if attempt > policy.max_retries or not isinstance(error, policy.retry_on):
            return TaskOutcome(TaskStatus.FAILED, error=error, attempts=attempt)
        await asyncio.sleep(policy.retry_delay_seconds)


class TaskScheduler:
    """Own the task state for one plan and pass dependency outcomes to its executor."""

    def __init__(
        self, tasks: Iterable[ResearchTask], *, max_concurrency: int,
        task_policies: Mapping[str, TaskPolicy] | None = None,
        run_timeout_seconds: float | None = None,
    ):
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer.")
        self._plan = ResearchPlan(tasks)
        self._max_concurrency = max_concurrency
        self._default_policy = TaskPolicy()
        self._task_policies = dict(task_policies or {})
        self._run_timeout = (
            None if run_timeout_seconds is None else _seconds(run_timeout_seconds, "run_timeout_seconds")
        )
        self._run_lock = Lock()
        self._outcomes: dict[str, TaskOutcome] = {}
        snapshot = self._plan.tasks
        for task_id, policy in self._task_policies.items():
            if task_id not in snapshot:
                raise ValueError(f"Execution policy names unknown task {task_id!r}.")
            if not isinstance(policy, TaskPolicy):
                raise ValueError(f"Execution policy for task {task_id!r} must be a TaskPolicy.")
        for task in snapshot.values():
            if task.status == TaskStatus.RUNNING:
                raise ValueError(f"Task {task.id!r} is already running.")
            if task.status.is_terminal:
                self._outcomes[task.id] = TaskOutcome(task.status)

    @property
    def tasks(self) -> Mapping[str, ResearchTask]:
        return self._plan.tasks

    def _record(self, task_id: str, outcome: TaskOutcome) -> None:
        self._plan.transition(task_id, outcome.status)
        self._outcomes[task_id] = outcome

    async def run(self, executor: TaskExecutor) -> dict[str, TaskOutcome]:
        """Run pending work and return outcomes in plan order, retaining prior results."""
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("The scheduler is already running.")
        attempt_counts: dict[str, int] = {}
        deadline = asyncio.timeout(self._run_timeout)
        try:
            pending = {task.id: task for task in self.tasks.values() if task.status == TaskStatus.PENDING}
            active: dict[asyncio.Task[TaskOutcome], str] = {}
            async with deadline, asyncio.TaskGroup() as group:
                while pending or active:
                    skipped = False
                    for task_id, task in tuple(pending.items()):
                        check = self._plan.check_dependencies(task_id)
                        if check.status == DependencyStatus.BLOCKED:
                            self._record(task_id, TaskOutcome(TaskStatus.SKIPPED, blocked_by=check.blocked_by))
                            del pending[task_id]
                            skipped = True
                        elif check.status == DependencyStatus.READY and len(active) < self._max_concurrency:
                            dependencies = MappingProxyType({key: self._outcomes[key] for key in task.dependencies})
                            running = self._plan.transition(task_id, TaskStatus.RUNNING)
                            policy = self._task_policies.get(task_id, self._default_policy)
                            future = group.create_task(_execute(executor, running, dependencies, policy, attempt_counts))
                            active[future] = task_id
                            del pending[task_id]

                    # Skips can unlock cleanup or block an earlier task in plan order.
                    if skipped:
                        continue
                    if not active:
                        if pending:
                            raise RuntimeError("The research plan has pending tasks with no way to progress.")
                        break
                    done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                    for future in done:
                        task_id = active.pop(future)
                        outcome = (
                            TaskOutcome(TaskStatus.CANCELLED, attempts=attempt_counts.get(task_id, 0))
                            if future.cancelled() else future.result()
                        )
                        self._record(task_id, outcome)
            return {task_id: self._outcomes[task_id] for task_id in self.tasks}
        except TimeoutError as exc:
            if deadline.expired():
                raise ScheduleTimeoutError("Research schedule timed out.") from exc
            raise
        finally:
            # TaskGroup waits for active executors to finish cancellation before we publish it.
            try:
                for task in self.tasks.values():
                    if not task.status.is_terminal:
                        self._record(
                            task.id, TaskOutcome(TaskStatus.CANCELLED, attempts=attempt_counts.get(task.id, 0)),
                        )
            finally:
                self._run_lock.release()
