from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from core.research_tasks import DependencyCheck, DependencyStatus, ResearchPlan, ResearchTask, TaskStatus


def test_success_opens_a_chain_and_multiple_prerequisites_must_finish():
    plan = ResearchPlan([
        ResearchTask("write", depends_on_success=("read",), depends_on_terminal=("retrieve",)),
        ResearchTask("read", depends_on_success=("retrieve",)),
        ResearchTask("retrieve"),
    ])
    assert plan.check_dependencies("retrieve").status == DependencyStatus.READY
    assert plan.check_dependencies("write") == DependencyCheck(DependencyStatus.WAITING, ("read", "retrieve"))
    with pytest.raises(ValueError, match="WAITING"):
        plan.transition("read", TaskStatus.RUNNING)
    plan.transition("retrieve", TaskStatus.RUNNING)
    assert plan.check_dependencies("read").status == DependencyStatus.WAITING
    plan.transition("retrieve", TaskStatus.COMPLETED)
    assert plan.check_dependencies("read").status == DependencyStatus.READY
    assert plan.check_dependencies("write").waiting_for == ("read",)
    plan.transition("read", TaskStatus.RUNNING)
    plan.transition("read", TaskStatus.COMPLETED)
    assert plan.transition("write", TaskStatus.RUNNING).status == TaskStatus.RUNNING


@pytest.mark.parametrize("outcome", [TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.SKIPPED])
def test_failed_cancelled_or_skipped_only_block_success_dependencies(outcome):
    plan = ResearchPlan([
        ResearchTask("retrieve"),
        ResearchTask("read", depends_on_success=("retrieve",)),
        ResearchTask("inspect", depends_on_terminal=("retrieve",)),
        ResearchTask("cleanup", depends_on_terminal=("read",)),
    ])
    assert plan.check_dependencies("inspect").status == DependencyStatus.WAITING
    if outcome == TaskStatus.FAILED:
        plan.transition("retrieve", TaskStatus.RUNNING)
        assert plan.check_dependencies("inspect").status == DependencyStatus.WAITING
    plan.transition("retrieve", outcome)
    assert plan.check_dependencies("read") == DependencyCheck(DependencyStatus.BLOCKED, blocked_by=("retrieve",))
    with pytest.raises(ValueError, match="BLOCKED"):
        plan.transition("read", TaskStatus.RUNNING)
    assert plan.transition("inspect", TaskStatus.RUNNING).status == TaskStatus.RUNNING
    assert plan.check_dependencies("cleanup").status == DependencyStatus.WAITING
    plan.transition("read", TaskStatus.SKIPPED)
    assert plan.transition("cleanup", TaskStatus.RUNNING).status == TaskStatus.RUNNING


def test_mixed_prerequisites_report_the_terminal_blocker_and_unfinished_work():
    plan = ResearchPlan([
        ResearchTask("retrieve", status=TaskStatus.FAILED),
        ResearchTask("other"),
        ResearchTask("write", depends_on_success=("retrieve",), depends_on_terminal=("retrieve", "other")),
    ])
    assert plan.check_dependencies("write") == DependencyCheck(
        DependencyStatus.BLOCKED, waiting_for=("other",), blocked_by=("retrieve",),
    )


@pytest.mark.parametrize(("status", "target"), [
    (TaskStatus.PENDING, TaskStatus.COMPLETED),
    (TaskStatus.PENDING, TaskStatus.FAILED),
    (TaskStatus.RUNNING, TaskStatus.PENDING),
    (TaskStatus.RUNNING, TaskStatus.SKIPPED),
    (TaskStatus.COMPLETED, TaskStatus.RUNNING),
    (TaskStatus.FAILED, TaskStatus.RUNNING),
    (TaskStatus.CANCELLED, TaskStatus.COMPLETED),
    (TaskStatus.SKIPPED, TaskStatus.RUNNING),
])
def test_invalid_transition_preserves_task_snapshot(status, target):
    task = ResearchTask("task", status=status)
    plan = ResearchPlan([task])
    with pytest.raises(ValueError, match="cannot transition"):
        plan.transition(task.id, target)
    assert plan.tasks[task.id] is task


@pytest.mark.parametrize(("tasks", "message"), [
    ([ResearchTask("retrieve"), ResearchTask("retrieve")], "Duplicate task ID"),
    ([ResearchTask("read", depends_on_success=("missing",))], "unknown task"),
    ([ResearchTask("read", depends_on_terminal=("missing",))], "unknown task"),
    ([ResearchTask("read", depends_on_terminal=("read",))], "cycle"),
    ([
        ResearchTask("retrieve", depends_on_terminal=("write",)),
        ResearchTask("read", depends_on_success=("retrieve",)),
        ResearchTask("write", depends_on_success=("read",)),
    ], "cycle"),
    ([
        ResearchTask("retrieve"),
        ResearchTask("read", depends_on_success=("retrieve",), status=TaskStatus.RUNNING),
    ], "unsatisfied dependencies"),
])
def test_invalid_dag_cannot_be_admitted(tasks, message):
    with pytest.raises(ValueError, match=message):
        ResearchPlan(tasks)


def test_snapshots_and_dependencies_are_independent_of_caller_mutation():
    dependencies = ["retrieve", "retrieve"]
    plan = ResearchPlan([ResearchTask("retrieve"), ResearchTask("read", depends_on_success=dependencies)])
    snapshot = plan.tasks
    dependencies.append("unrelated")
    assert snapshot["read"].depends_on_success == ("retrieve",)
    with pytest.raises(TypeError):
        snapshot["other"] = ResearchTask("other")
    plan.transition("retrieve", TaskStatus.RUNNING)
    assert snapshot["retrieve"].status == TaskStatus.PENDING
    assert plan.tasks["retrieve"].status == TaskStatus.RUNNING


def test_competing_start_requests_can_only_start_one_task_once():
    plan = ResearchPlan([ResearchTask("retrieve")])
    barrier = Barrier(2)

    def start():
        barrier.wait(timeout=5)
        try:
            plan.transition("retrieve", TaskStatus.RUNNING)
        except ValueError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.submit(start), executor.submit(start)
        assert sorted([first.result(timeout=5), second.result(timeout=5)]) == [False, True]
    assert plan.tasks["retrieve"].status == TaskStatus.RUNNING
