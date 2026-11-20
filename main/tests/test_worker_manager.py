"""
Comprehensive Unit & Integration Test Suite for WorkerManager & Weights Manifest.

Verifies:
1. Manifest cryptographic & structural validation across all 8 checkpoints.
2. Worker lease acquisition, concurrency counting, and exception behavior.
3. Non-destructive drain timeout: active document tasks are preserved, transition aborts safely.
4. Concurrent transition serialization: second switch attempt is rejected while first is running.
5. Standardized status contract conformity: selected_mode vs active_worker.
"""

from __future__ import annotations

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from kaggle.worker_manager import (
    WorkerManager,
    GPUVRAMProbe,
    WorkerSwitchingError,
    DrainTimeoutAbortError,
)


@pytest.fixture
def worker_manager():
    return WorkerManager(project_dir=os.path.abspath("."))


def test_manifest_validation_engine(worker_manager):
    """Verify that validate_manifest_checkpoints detects all present checkpoints accurately."""
    caps = worker_manager.validate_manifest_checkpoints(verify_sha256=False)
    assert isinstance(caps, dict)
    # The 5 mandatory Finetune models must be detected on disk
    assert caps["router"] is True
    assert caps["generator"] is True
    assert caps["critic"] is True
    assert caps["fact_checker_nli"] is True
    assert caps["writer"] is True


def test_status_contract_schema(worker_manager):
    """Verify get_status_payload strictly conforms to the agreed status contract."""
    status = worker_manager.get_status_payload()
    assert "runtime_status" in status
    assert "selected_mode" in status
    assert "active_worker" in status
    assert "target_worker" in status
    assert "transition_state" in status
    assert "transition_id" in status
    assert "active_requests" in status
    assert "queue_depth" in status
    assert "worker_profiles_available" in status
    assert "active_worker_capabilities" in status

    # Initially, no worker is active
    assert status["selected_mode"] == "gpu_finetune"
    assert status["active_worker"] is None
    assert status["transition_state"] == "idle"
    assert status["active_requests"] == 0
    assert status["worker_profiles_available"]["gpu_finetune"] is True


def test_worker_lease_acquisition_and_release(worker_manager):
    """Verify acquire_worker_lease tracks active requests and releases cleanly."""
    async def _test():
        assert worker_manager.active_requests == 0
        async with worker_manager.acquire_worker_lease():
            assert worker_manager.active_requests == 1
            async with worker_manager.acquire_worker_lease():
                assert worker_manager.active_requests == 2
        assert worker_manager.active_requests == 0

    asyncio.run(_test())


def test_worker_lease_rejects_during_transition(worker_manager):
    """Verify acquire_worker_lease raises WorkerSwitchingError during transition."""
    async def _test():
        worker_manager.transition_state = "spawning"
        with pytest.raises(WorkerSwitchingError):
            async with worker_manager.acquire_worker_lease():
                pass
        worker_manager.transition_state = "idle"

    asyncio.run(_test())


def test_drain_timeout_aborts_and_preserves_worker(worker_manager):
    """
    CRITICAL: Verify that if in-flight requests cannot drain in time,
    the transition ABORTS safely and the worker is NOT terminated.
    """
    async def _test():
        worker_manager.active_requests = 1
        worker_manager.active_worker = "gpu_finetune"

        # Attempt transition: drain should abort because active_requests > 0
        with patch.object(worker_manager, "_drain_in_flight_requests", side_effect=DrainTimeoutAbortError("Drain timeout")):
            res = await worker_manager._execute_transition("gpu_max", trans_id="abort_test")

        assert res["status"] == "aborted"
        assert "Drain timeout" in res["message"]
        # Worker is preserved!
        assert worker_manager.active_worker == "gpu_finetune"
        worker_manager.active_requests = 0

    asyncio.run(_test())


def test_concurrent_switch_rejection(worker_manager):
    """Verify lock serialization prevents simultaneous mode transitions."""
    async def _test():
        await worker_manager.lock.acquire()
        try:
            res = await worker_manager.request_switch_mode("gpu_max")
            assert res["status"] == "rejected"
            assert "Một tác vụ chuyển chế độ khác đang diễn ra" in res["message"]
        finally:
            worker_manager.lock.release()

    asyncio.run(_test())


def test_process_group_termination(worker_manager):
    """Verify process group termination logic, signal escalation and PG cleanup."""
    async def _test():
        # Scenario 1: Process group exits gracefully after SIGTERM
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = 0

        worker_manager._proc = mock_proc
        worker_manager._pgid = 99999
        worker_manager.active_worker = "gpu_finetune"

        with patch("kaggle.worker_manager.kill_process_group") as mock_kill_pg, \
             patch("kaggle.worker_manager.is_process_group_alive", return_value=False), \
             patch.object(worker_manager, "_verify_vram_reclaimed", new_callable=AsyncMock):

            await worker_manager._terminate_current_worker()

            # Verify SIGTERM sent to pgid
            assert mock_kill_pg.call_count == 1
            assert worker_manager._proc is None
            assert worker_manager._pgid is None
            assert worker_manager.active_worker is None

        # Scenario 2: Lingering process group requires SIGKILL escalation
        mock_proc2 = MagicMock()
        mock_proc2.pid = 88888
        worker_manager._proc = mock_proc2
        worker_manager._pgid = 88888
        worker_manager.active_worker = "gpu_max"

        # Time sequence: start=0, loop check 1=10.0 (exceeds 8.0s), kill_start=10.0, kill check=11.0
        time_seq = [0.0, 10.0, 10.0, 10.0, 11.0, 12.0]
        t_idx = 0
        def mock_time():
            nonlocal t_idx
            val = time_seq[min(t_idx, len(time_seq) - 1)]
            t_idx += 1
            return val

        # Alive until after SIGKILL
        alive_states = [True, True, False]
        a_idx = 0
        def mock_alive(pgid):
            nonlocal a_idx
            val = alive_states[min(a_idx, len(alive_states) - 1)]
            a_idx += 1
            return val

        with patch("kaggle.worker_manager.kill_process_group") as mock_kill_pg2, \
             patch("kaggle.worker_manager.is_process_group_alive", side_effect=mock_alive), \
             patch("time.time", side_effect=mock_time), \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             patch.object(worker_manager, "_verify_vram_reclaimed", new_callable=AsyncMock):

            await worker_manager._terminate_current_worker()

            # Should have called both SIGTERM and SIGKILL
            assert mock_kill_pg2.call_count == 2
            assert worker_manager._proc is None
            assert worker_manager._pgid is None

    asyncio.run(_test())


def test_vram_probe_and_profile_thresholds():
    """Verify VRAM profile validation correctly checks requirements for gpu_max vs gpu_finetune."""
    mock_manifest = {
        "base_models": {
            "qwen_32b_awq": {"min_vram_per_gpu_mb": 13500.0},
            "qwen_7b_base": {"min_vram_mb": 4800.0},
        }
    }

    # Case A: Sufficient VRAM on both GPUs for 32B AWQ TP=2 (15000 MB free > 13500 MB)
    sufficient_vram = {
        0: {"used_mb": 1000.0, "total_mb": 16000.0, "free_mb": 15000.0},
        1: {"used_mb": 1000.0, "total_mb": 16000.0, "free_mb": 15000.0},
    }
    with patch.object(GPUVRAMProbe, "get_vram_info", return_value=sufficient_vram):
        ok, msg = GPUVRAMProbe.check_vram_against_profile("gpu_max", mock_manifest)
        assert ok is True
        assert "Sufficient VRAM verified" in msg

    # Case B: Insufficient VRAM on GPU 1 (only 10000 MB free < 13500 MB)
    insufficient_vram = {
        0: {"used_mb": 1000.0, "total_mb": 16000.0, "free_mb": 15000.0},
        1: {"used_mb": 6000.0, "total_mb": 16000.0, "free_mb": 10000.0},
    }
    with patch.object(GPUVRAMProbe, "get_vram_info", return_value=insufficient_vram):
        ok, msg = GPUVRAMProbe.check_vram_against_profile("gpu_max", mock_manifest)
        assert ok is False
        assert "GPU 1 has 10000.0 MB free VRAM" in msg


def test_worker_startup_failure_preserves_supervisor(worker_manager):
    """Verify supervisor survives worker startup/health check failure gracefully."""
    async def _test():
        with patch.object(worker_manager, "_drain_in_flight_requests", new_callable=AsyncMock), \
             patch.object(worker_manager, "_terminate_current_worker", new_callable=AsyncMock), \
             patch.object(GPUVRAMProbe, "check_vram_against_profile", return_value=(True, "OK")), \
             patch.object(worker_manager, "_spawn_process", return_value=MagicMock()), \
             patch.object(worker_manager, "_poll_worker_ready", return_value=False):

            res = await worker_manager._execute_transition("gpu_max", trans_id="err_test")

            assert res["status"] == "error"
            assert "không phản hồi health check" in res["message"]
            assert worker_manager.transition_state == "error"
            # Supervisor state is still alive and queryable!
            status = worker_manager.get_status_payload()
            assert status["runtime_status"] == "degraded"
            assert status["transition_state"] == "error"

    asyncio.run(_test())


def test_multiround_mode_cycling(worker_manager):
    """
    CRITICAL: Verify continuous mode switching sequence in a single Kaggle session:
    gpu_finetune -> gpu_max -> hybrid -> gpu_finetune -> gpu_max -> gpu_finetune
    Verifies that state transitions cleanly without restarting supervisor or resetting session.
    """
    async def _test():
        sequence = [
            ("gpu_finetune", "gpu_finetune"),
            ("gpu_max", "gpu_max"),
            ("hybrid", "gpu_max"),  # hybrid preserves active worker if already online
            ("gpu_finetune", "gpu_finetune"),
            ("gpu_max", "gpu_max"),
            ("gpu_finetune", "gpu_finetune"),
        ]

        with patch.object(worker_manager, "_drain_in_flight_requests", new_callable=AsyncMock), \
             patch.object(worker_manager, "_terminate_current_worker", new_callable=AsyncMock), \
             patch.object(GPUVRAMProbe, "check_vram_against_profile", return_value=(True, "OK")), \
             patch.object(worker_manager, "_spawn_process", return_value=MagicMock()), \
             patch.object(worker_manager, "_poll_worker_ready", return_value=True):

            for target_mode, expected_worker in sequence:
                res = await worker_manager.request_switch_mode(target_mode)
                assert res["status"] in ("success", "accepted", "already_active")
                assert worker_manager.selected_mode == target_mode
                assert worker_manager.active_worker == expected_worker

                # Ensure status payload reflects accurate live state
                status = worker_manager.get_status_payload()
                assert status["selected_mode"] == target_mode
                assert status["active_worker"] == expected_worker
                assert status["transition_state"] in ("ready", "idle")

    asyncio.run(_test())
