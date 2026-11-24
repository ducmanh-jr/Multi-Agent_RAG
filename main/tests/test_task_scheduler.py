import asyncio

import pytest

from core.research_tasks import ResearchTask, TaskStatus
from core.task_scheduler import ScheduleTimeoutError, TaskOutcome, TaskPolicy, TaskScheduler, TaskTimeoutError


def test_scheduler_runs_ready_tasks_up_to_capacity_and_passes_dependency_outputs():
    async def exercise():
        scheduler = TaskScheduler([
            ResearchTask("write", depends_on_success=("left", "right")),
            ResearchTask("left", depends_on_success=("retrieve",)),
            ResearchTask("right", depends_on_success=("retrieve",)),
            ResearchTask("extra", depends_on_success=("retrieve",)),
            ResearchTask("retrieve"),
        ], max_concurrency=2)
        both_started = asyncio.Event()
        release = asyncio.Event()
        working = set()
        peak = 0

        async def execute(task, dependencies):
            nonlocal peak
            assert task.status == TaskStatus.RUNNING
            working.add(task.id)
            peak = max(peak, len(working))
            try:
                if task.id == "retrieve":
                    return 5
                if task.id in {"left", "right"}:
                    if working == {"left", "right"}:
                        both_started.set()
                    await release.wait()
                    return dependencies["retrieve"].output + (1 if task.id == "left" else 2)
                if task.id == "write":
                    return dependencies["left"].output + dependencies["right"].output
                return dependencies["retrieve"].output
            finally:
                working.remove(task.id)

        scheduled = asyncio.create_task(scheduler.run(execute))
        await asyncio.wait_for(both_started.wait(), timeout=5)
        assert scheduler.tasks["extra"].status == TaskStatus.PENDING
        assert scheduler.tasks["write"].status == TaskStatus.PENDING
        release.set()
        results = await scheduled
        assert peak == 2
        assert results["write"] == TaskOutcome(TaskStatus.COMPLETED, output=13, attempts=1)
        assert results["extra"].output == 5
        assert list(results) == list(scheduler.tasks)

    asyncio.run(asyncio.wait_for(exercise(), timeout=10))


def test_scheduler_propagates_skips_and_runs_terminal_consumers_and_other_branches():
    async def exercise():
        scheduler = TaskScheduler([
            ResearchTask("report", depends_on_terminal=("validate",)),
            ResearchTask("validate", depends_on_success=("extract",)),
            ResearchTask("extract", depends_on_success=("retrieve",)),
            ResearchTask("retrieve"),
            ResearchTask("independent"),
        ], max_concurrency=2)
        failure = ValueError("fixture failed")
        called = []

        async def execute(task, dependencies):
            called.append(task.id)
            if task.id == "retrieve":
                raise failure
            if task.id == "report":
                return dependencies["validate"].blocked_by
            return "done"

        results = await scheduler.run(execute)
        assert set(called) == {"retrieve", "report", "independent"}
        assert results["retrieve"] == TaskOutcome(TaskStatus.FAILED, error=failure, attempts=1)
        assert results["extract"] == TaskOutcome(TaskStatus.SKIPPED, blocked_by=("retrieve",))
        assert results["validate"] == TaskOutcome(TaskStatus.SKIPPED, blocked_by=("extract",))
        assert results["report"] == TaskOutcome(TaskStatus.COMPLETED, output=("extract",), attempts=1)
        assert results["independent"].status == TaskStatus.COMPLETED
        assert all(task.status.is_terminal for task in scheduler.tasks.values())

    asyncio.run(asyncio.wait_for(exercise(), timeout=5))


def test_cancelled_executor_keeps_terminal_followup_available():
    async def exercise():
        scheduler = TaskScheduler([
            ResearchTask("retrieve"),
            ResearchTask("followup", depends_on_terminal=("retrieve",)),
            ResearchTask("read", depends_on_success=("retrieve",)),
        ], max_concurrency=1)

        async def execute(task, dependencies):
            if task.id == "retrieve":
                raise asyncio.CancelledError
            assert task.id == "followup"
            return dependencies["retrieve"].status

        results = await scheduler.run(execute)
        assert results["retrieve"].status == TaskStatus.CANCELLED
        assert results["read"].status == TaskStatus.SKIPPED
        assert results["followup"].output == TaskStatus.CANCELLED

    asyncio.run(asyncio.wait_for(exercise(), timeout=5))


def test_cancelling_a_schedule_drains_active_executors_and_never_starts_pending_work():
    async def exercise():
        scheduler = TaskScheduler([
            ResearchTask("retrieve"),
            ResearchTask("slow", depends_on_success=("retrieve",)),
            ResearchTask("queued", depends_on_success=("retrieve",)),
            ResearchTask("cleanup", depends_on_terminal=("slow",)),
        ], max_concurrency=1)
        started, drained = asyncio.Event(), asyncio.Event()
        called = []

        async def execute(task, dependencies):
            called.append(task.id)
            if task.id == "retrieve":
                return "cached evidence"
            assert task.id == "slow"
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                drained.set()

        scheduled = asyncio.create_task(scheduler.run(execute))
        await asyncio.wait_for(started.wait(), timeout=5)
        with pytest.raises(RuntimeError, match="already running"):
            await scheduler.run(execute)
        scheduled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await scheduled
        assert drained.is_set()
        results = await scheduler.run(execute)
        assert called == ["retrieve", "slow"]
        assert results["retrieve"].output == "cached evidence"
        assert all(results[key].status == TaskStatus.CANCELLED for key in ("slow", "queued", "cleanup"))

    asyncio.run(asyncio.wait_for(exercise(), timeout=10))


def test_scheduler_preserves_terminal_snapshots_and_reuses_completed_results():
    async def exercise():
        scheduler = TaskScheduler([
            ResearchTask("prior", status=TaskStatus.COMPLETED),
            ResearchTask("next", depends_on_success=("prior",)),
        ], max_concurrency=1)
        called = []

        async def execute(task, dependencies):
            called.append(task.id)
            assert dependencies["prior"].status == TaskStatus.COMPLETED
            return "new evidence"

        first = await scheduler.run(execute)
        second = await scheduler.run(execute)
        assert called == ["next"]
        assert first == second
        assert first["next"].output == "new evidence"

    asyncio.run(asyncio.wait_for(exercise(), timeout=5))


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_scheduler_rejects_invalid_concurrency(limit):
    with pytest.raises(ValueError, match="positive integer"):
        TaskScheduler([ResearchTask("retrieve")], max_concurrency=limit)


def test_scheduler_rejects_running_tasks_without_an_executor():
    with pytest.raises(ValueError, match="already running"):
        TaskScheduler([ResearchTask("retrieve", status=TaskStatus.RUNNING)], max_concurrency=1)


def test_retry_holds_the_slot_and_exposes_only_final_output_to_dependents():
    async def exercise():
        scheduler = TaskScheduler([
            ResearchTask("retrieve"),
            ResearchTask("write", depends_on_success=("retrieve",)),
            ResearchTask("independent"),
        ], max_concurrency=1, task_policies={
            "retrieve": TaskPolicy(max_retries=2, retry_on=(ConnectionError,)),
        })
        called = []

        async def execute(task, dependencies):
            called.append(task.id)
            if task.id == "retrieve":
                if called.count("retrieve") < 3:
                    raise ConnectionError("transient failure")
                return "verified evidence"
            if task.id == "write":
                evidence = dependencies["retrieve"]
                assert evidence == TaskOutcome(TaskStatus.COMPLETED, output="verified evidence", attempts=3)
                return evidence.output

        results = await scheduler.run(execute)
        assert called == ["retrieve", "retrieve", "retrieve", "write", "independent"]
        assert results["write"].output == "verified evidence"

    asyncio.run(asyncio.wait_for(exercise(), timeout=5))


@pytest.mark.parametrize(("error", "attempts"), [(ConnectionError("retry"), 3), (ValueError("permanent"), 1)])
def test_exhausted_and_nonretryable_failures_release_terminal_dependents(error, attempts):
    async def exercise():
        scheduler = TaskScheduler([
            ResearchTask("retrieve"),
            ResearchTask("write", depends_on_success=("retrieve",)),
            ResearchTask("audit", depends_on_terminal=("retrieve",)),
        ], max_concurrency=2, task_policies={
            "retrieve": TaskPolicy(max_retries=2, retry_on=(ConnectionError,)),
        })
        called = []

        async def execute(task, dependencies):
            called.append(task.id)
            if task.id == "retrieve":
                raise error
            assert task.id == "audit"
            return dependencies["retrieve"].attempts

        results = await scheduler.run(execute)
        assert called == ["retrieve"] * attempts + ["audit"]
        assert results["retrieve"] == TaskOutcome(TaskStatus.FAILED, error=error, attempts=attempts)
        assert results["write"].status == TaskStatus.SKIPPED
        assert results["audit"].output == attempts

    asyncio.run(asyncio.wait_for(exercise(), timeout=5))


def test_timeout_retries_after_cleanup_and_never_accepts_a_late_result():
    async def exercise():
        scheduler = TaskScheduler([ResearchTask("read")], max_concurrency=1, task_policies={
            "read": TaskPolicy(timeout_seconds=0.02, max_retries=1),
        })
        cleaned = []

        async def execute(task, dependencies):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return "late result"
            finally:
                cleaned.append(task.id)

        result = (await scheduler.run(execute))["read"]
        assert result.status == TaskStatus.FAILED
        assert isinstance(result.error, TaskTimeoutError)
        assert result.attempts == 2 and result.output is None
        assert cleaned == ["read", "read"]

    asyncio.run(asyncio.wait_for(exercise(), timeout=5))


def test_cancelling_during_retry_delay_stops_further_attempts():
    async def exercise():
        scheduler = TaskScheduler([ResearchTask("read")], max_concurrency=1, task_policies={
            "read": TaskPolicy(max_retries=2, retry_delay_seconds=30, retry_on=(ConnectionError,)),
        })
        failed = asyncio.Event()
        called = []

        async def execute(task, dependencies):
            called.append(task.id)
            failed.set()
            raise ConnectionError("transient failure")

        running = asyncio.create_task(scheduler.run(execute))
        await asyncio.wait_for(failed.wait(), timeout=5)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        result = (await scheduler.run(execute))["read"]
        assert result == TaskOutcome(TaskStatus.CANCELLED, attempts=1)
        assert called == ["read"]

    asyncio.run(asyncio.wait_for(exercise(), timeout=10))


def test_schedule_deadline_preserves_finished_work_and_cancels_remaining_tasks():
    async def exercise():
        scheduler = TaskScheduler([
            ResearchTask("retrieve"),
            ResearchTask("read", depends_on_success=("retrieve",)),
            ResearchTask("write", depends_on_terminal=("read",)),
        ], max_concurrency=1, run_timeout_seconds=0.2, task_policies={
            "read": TaskPolicy(max_retries=2, retry_on=(ConnectionError,)),
        })
        called = []
        cleaned = asyncio.Event()

        async def execute(task, dependencies):
            called.append(task.id)
            if task.id == "retrieve":
                return "evidence"
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise ConnectionError("executor returned an error during cancellation")
            finally:
                cleaned.set()

        with pytest.raises(ScheduleTimeoutError):
            await scheduler.run(execute)
        assert cleaned.is_set()
        results = await scheduler.run(execute)
        assert called == ["retrieve", "read"]
        assert results["retrieve"] == TaskOutcome(TaskStatus.COMPLETED, output="evidence", attempts=1)
        assert results["read"] == TaskOutcome(TaskStatus.CANCELLED, attempts=1)
        assert results["write"] == TaskOutcome(TaskStatus.CANCELLED)

    asyncio.run(asyncio.wait_for(exercise(), timeout=5))


@pytest.mark.parametrize(("options", "field"), [
    ({"timeout_seconds": 0}, "timeout_seconds"),
    ({"timeout_seconds": float("nan")}, "timeout_seconds"),
    ({"max_retries": -1}, "max_retries"),
    ({"max_retries": True}, "max_retries"),
    ({"retry_delay_seconds": float("inf")}, "retry_delay_seconds"),
    ({"retry_on": (asyncio.CancelledError,)}, "retry_on"),
])
def test_invalid_task_policy_is_rejected(options, field):
    with pytest.raises(ValueError, match=field):
        TaskPolicy(**options)


def test_scheduler_validates_run_deadline_and_task_policy_keys():
    with pytest.raises(ValueError, match="run_timeout_seconds"):
        TaskScheduler([], max_concurrency=1, run_timeout_seconds=-1)
    with pytest.raises(ValueError, match="unknown task"):
        TaskScheduler([], max_concurrency=1, task_policies={"read": TaskPolicy()})
