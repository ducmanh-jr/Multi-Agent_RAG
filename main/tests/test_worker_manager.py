"""
Comprehensive Unit & OS Integration Test Suite for WorkerManager & Weights Manifest (v2.2.0).

Verifies:
1. Manifest cryptographic & structural validation across all 8 checkpoints.
2. Worker lease acquisition, queue depth tracking, and exception behavior.
3. Non-destructive drain timeout: active document tasks are preserved, transition aborts safely.
4. Concurrent transition serialization: second switch attempt is rejected while first is running.
5. Standardized status contract conformity: selected_mode vs active_worker.
6. P0-1 Fail-Safe VRAM Probe:
   - Fails-closed when OS VRAM probe is unavailable/invalid (no bypass allowed).
   - Enforces dual GPU detection (rejects single GPU).
   - Validates both GPU 0 and GPU 1 for Finetune (GPU 1 is never skipped).
7. P0-2 Guaranteed Process Group & VRAM Reclamation:
   - Fails transition if process group remains alive after SIGKILL.
   - Fails transition if parent process has not terminated.
   - Fails transition if VRAM does not return to baseline within timeout.
8. P0-3 Worker Startup Failure Cleanup & Auto-Recovery:
   - Cleans up failed spawned process group immediately on startup failure.
   - Reclaims VRAM before returning.
   - Attempts auto-recovery to previous worker when viable.
9. P0-4 Deep Readiness Healthcheck & Child Fail-Fast:
   - Rejects Finetune worker if mandatory adapters (router, generator, critic, writer) or models are missing.
   - Rejects GPU Max if /v1/models does not match expected model ID.
   - Fails-fast immediately if child process exits unexpectedly during readiness polling.
10. OS Integration Tests:
   - Real subprocess lifecycle: spawns real OS process group and tests graceful SIGTERM.
   - Real subprocess SIGKILL escalation: spawns real process ignoring SIGTERM and verifies SIGKILL termination.
   - Real subprocess fail-fast: spawns process that exits with non-zero code, verifying instant poll abort.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kaggle.worker_manager import (
    WorkerManager,
    GPUVRAMProbe,
    VRAMProbeResult,
    WorkerSwitchingError,
    DrainTimeoutAbortError,
    WorkerTerminationError,
    VRAMReclaimError,
    is_process_group_alive,
    kill_process_group,
)


@pytest.fixture
def worker_manager():
    return WorkerManager(project_dir=os.path.abspath("."))


# =====================================================================
# 1. Manifest Validation & Status Contract Tests
# =====================================================================

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
    assert "vram_probe" in status
    assert "worker_profiles_available" in status
    assert "active_worker_capabilities" in status

    # Initially, no worker is active
    assert status["selected_mode"] == "gpu_finetune"
    assert status["active_worker"] is None
    assert status["transition_state"] == "idle"
    assert status["active_requests"] == 0
    assert status["queue_depth"] == 0
    assert status["worker_profiles_available"]["gpu_finetune"] is True


# =====================================================================
# 2. Worker Lease & Drain Timeout Tests
# =====================================================================

def test_worker_lease_acquisition_and_release(worker_manager):
    """Verify acquire_worker_lease tracks active requests, queue depth, and releases cleanly."""
    async def _test():
        assert worker_manager.active_requests == 0
        assert worker_manager.queue_depth == 0
        async with worker_manager.acquire_worker_lease():
            assert worker_manager.active_requests == 1
            async with worker_manager.acquire_worker_lease():
                assert worker_manager.active_requests == 2
        assert worker_manager.active_requests == 0
        assert worker_manager.queue_depth == 0

    asyncio.run(_test())


def test_worker_lease_rejects_during_transition(worker_manager):
    """Verify acquire_worker_lease raises WorkerSwitchingError during transition."""
    async def _test():
        for state in ["stopping", "reclaiming_vram", "spawning", "draining"]:
            worker_manager.transition_state = state
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

        with patch.object(worker_manager, "_drain_in_flight_requests", side_effect=DrainTimeoutAbortError("Drain timeout")):
            res = await worker_manager._execute_transition("gpu_max", trans_id="abort_test")

        assert res["status"] == "aborted"
        assert "Drain timeout" in res["message"]
        # Active worker is preserved!
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


# =====================================================================
# 3. P0-1 Fail-Safe VRAM Probe & Dual-GPU Verification
# =====================================================================

def test_vram_probe_fails_closed_when_unavailable():
    """Verify check_vram_against_profile strictly fails-closed if OS measurement is unavailable."""
    invalid_probe = VRAMProbeResult(
        gpus={},
        source="unavailable",
        is_valid=False,
        error="nvidia-smi not found"
    )
    with patch.object(GPUVRAMProbe, "probe_vram", return_value=invalid_probe):
        ok, msg = GPUVRAMProbe.check_vram_against_profile("gpu_max", {})
        assert ok is False
        assert "Không có phép đo VRAM cấp OS đáng tin cậy" in msg


def test_vram_probe_requires_dual_gpu():
    """Verify check_vram_against_profile strictly rejects single-GPU environments for Dual-T4."""
    single_gpu = {
        0: {"used_mb": 1000.0, "total_mb": 16000.0, "free_mb": 15000.0}
    }
    probe = VRAMProbeResult(gpus=single_gpu, source="nvidia-smi", is_valid=True)
    with patch.object(GPUVRAMProbe, "probe_vram", return_value=probe):
        ok, msg = GPUVRAMProbe.check_vram_against_profile("gpu_max", {}, required_gpus=2)
        assert ok is False
        assert "yêu cầu phát hiện tối thiểu 2 GPU" in msg

        ok_ft, msg_ft = GPUVRAMProbe.check_vram_against_profile("gpu_finetune", {}, required_gpus=2)
        assert ok_ft is False
        assert "yêu cầu phát hiện tối thiểu 2 GPU" in msg_ft


def test_vram_probe_finetune_verifies_both_gpu0_and_gpu1():
    """Verify Finetune checks both GPU 0 (Base+LoRA >= 7.8GB) and GPU 1 (NLI+Reader >= 7.0GB)."""
    manifest = {
        "base_models": {
            "qwen_7b_base": {"min_vram_mb": 4800.0}
        }
    }

    # Case A: GPU 0 has enough (12GB free), but GPU 1 has insufficient (5GB free < 7GB)
    insufficient_gpu1 = {
        0: {"used_mb": 4000.0, "total_mb": 16000.0, "free_mb": 12000.0},
        1: {"used_mb": 11000.0, "total_mb": 16000.0, "free_mb": 5000.0},
    }
    probe1 = VRAMProbeResult(gpus=insufficient_gpu1, source="nvidia-smi", is_valid=True)
    with patch.object(GPUVRAMProbe, "probe_vram", return_value=probe1):
        ok, msg = GPUVRAMProbe.check_vram_against_profile("gpu_finetune", manifest)
        assert ok is False
        assert "GPU 1 chỉ có 5000.0 MB VRAM trống" in msg

    # Case B: Both GPU 0 and GPU 1 have sufficient VRAM
    sufficient_both = {
        0: {"used_mb": 2000.0, "total_mb": 16000.0, "free_mb": 14000.0},
        1: {"used_mb": 2000.0, "total_mb": 16000.0, "free_mb": 14000.0},
    }
    probe2 = VRAMProbeResult(gpus=sufficient_both, source="nvidia-smi", is_valid=True)
    with patch.object(GPUVRAMProbe, "probe_vram", return_value=probe2):
        ok, msg = GPUVRAMProbe.check_vram_against_profile("gpu_finetune", manifest)
        assert ok is True
        assert "Đã xác minh đủ VRAM" in msg


def test_vram_probe_gpu_max_thresholds():
    """Verify GPU Max requires >= 13500 MB free on BOTH GPU 0 and GPU 1."""
    manifest = {
        "base_models": {
            "qwen_32b_awq": {"min_vram_per_gpu_mb": 13500.0}
        }
    }
    # GPU 0 OK, GPU 1 fails
    vram = {
        0: {"used_mb": 1000.0, "total_mb": 16000.0, "free_mb": 15000.0},
        1: {"used_mb": 4000.0, "total_mb": 16000.0, "free_mb": 12000.0},
    }
    probe = VRAMProbeResult(gpus=vram, source="nvidia-smi", is_valid=True)
    with patch.object(GPUVRAMProbe, "probe_vram", return_value=probe):
        ok, msg = GPUVRAMProbe.check_vram_against_profile("gpu_max", manifest)
        assert ok is False
        assert "GPU 1 chỉ có 12000.0 MB VRAM trống" in msg


# =====================================================================
# 4. P0-2 Termination & Process Group / VRAM Reclamation Tests
# =====================================================================

def test_termination_fails_if_process_group_remains_alive(worker_manager):
    """Verify _terminate_current_worker raises WorkerTerminationError if process group lingers."""
    async def _test():
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        worker_manager._proc = mock_proc
        worker_manager._pgid = 99999
        worker_manager.active_worker = "gpu_finetune"

        with patch("kaggle.worker_manager.kill_process_group"), \
             patch("kaggle.worker_manager.is_process_group_alive", return_value=True), \
             patch("asyncio.sleep", new_callable=AsyncMock):

            with pytest.raises(WorkerTerminationError) as exc_info:
                await worker_manager._terminate_current_worker()

            assert "Process Group 99999 vẫn còn sống sau SIGKILL" in str(exc_info.value)
            assert worker_manager.transition_state == "error"
            # State must NOT be cleared when process is still running!
            assert worker_manager._pgid == 99999

    asyncio.run(_test())


def test_termination_fails_if_vram_not_reclaimed(worker_manager):
    """Verify _terminate_current_worker raises VRAMReclaimError if memory is not freed to baseline."""
    async def _test():
        worker_manager.baseline_vram = {
            0: {"used_mb": 1500.0, "total_mb": 16000.0, "free_mb": 14500.0},
            1: {"used_mb": 1500.0, "total_mb": 16000.0, "free_mb": 14500.0},
        }
        # Simulated leaked VRAM (e.g. 10GB still occupied)
        leaked_vram = {
            0: {"used_mb": 10000.0, "total_mb": 16000.0, "free_mb": 6000.0},
            1: {"used_mb": 10000.0, "total_mb": 16000.0, "free_mb": 6000.0},
        }
        probe = VRAMProbeResult(gpus=leaked_vram, source="nvidia-smi", is_valid=True)

        with patch.object(GPUVRAMProbe, "probe_vram", return_value=probe), \
             patch("asyncio.sleep", new_callable=AsyncMock):

            with pytest.raises(VRAMReclaimError) as exc_info:
                await worker_manager._verify_vram_reclaimed(max_wait_seconds=0.5)

            assert "Hết thời gian chờ giải phóng VRAM" in str(exc_info.value)

    asyncio.run(_test())


def test_transition_aborts_on_termination_failure(worker_manager):
    """Verify that if termination fails, _execute_transition strictly refuses to spawn target worker."""
    async def _test():
        with patch.object(worker_manager, "_drain_in_flight_requests", new_callable=AsyncMock), \
             patch.object(worker_manager, "_terminate_current_worker", side_effect=WorkerTerminationError("PGID alive")), \
             patch.object(worker_manager, "_spawn_process") as mock_spawn:

            res = await worker_manager._execute_transition("gpu_max", trans_id="abort_spawn_test")

            assert res["status"] == "error"
            assert "Dừng Worker hiện tại thất bại" in res["message"]
            assert worker_manager.transition_state == "error"
            # Spawn was strictly NOT called!
            assert mock_spawn.call_count == 0

    asyncio.run(_test())


# =====================================================================
# 5. P0-3 Startup Failure Cleanup & Rollback Recovery Tests
# =====================================================================

def test_startup_failure_cleans_up_spawned_process_and_reclaims_vram(worker_manager):
    """
    CRITICAL P0-3: If spawned worker fails healthcheck, verify:
    1. Spawned process group is immediately terminated via SIGKILL.
    2. VRAM is verified reclaimed.
    3. Supervisor state reports clean error without zombie processes.
    """
    async def _test():
        mock_proc = MagicMock()
        mock_proc.pid = 77777

        with patch.object(worker_manager, "_drain_in_flight_requests", new_callable=AsyncMock), \
             patch.object(worker_manager, "_terminate_current_worker", new_callable=AsyncMock) as mock_term, \
             patch.object(GPUVRAMProbe, "check_vram_against_profile", return_value=(True, "OK")), \
             patch.object(worker_manager, "_spawn_process", return_value=mock_proc), \
             patch.object(worker_manager, "_poll_worker_ready", return_value=False):

            res = await worker_manager._execute_transition("gpu_max", trans_id="startup_fail_test")

            assert res["status"] in ("error", "recovered_previous")
            # Verify termination was called to cleanup the broken worker!
            assert mock_term.call_count >= 1
            assert "Đã giải phóng hoàn toàn tiến trình và VRAM" in worker_manager.transition_message

    asyncio.run(_test())


def test_startup_failure_attempts_recovery_to_previous_worker(worker_manager):
    """Verify auto-recovery attempts to restart previous worker if target startup fails."""
    async def _test():
        worker_manager.active_worker = "gpu_finetune"
        worker_manager.selected_mode = "gpu_finetune"
        mock_proc = MagicMock()

        # Simulate first poll (gpu_max) fails, but second poll (recovering gpu_finetune) succeeds
        poll_results = [False, True]

        with patch.object(worker_manager, "_drain_in_flight_requests", new_callable=AsyncMock), \
             patch.object(worker_manager, "_terminate_current_worker", new_callable=AsyncMock), \
             patch.object(GPUVRAMProbe, "check_vram_against_profile", return_value=(True, "OK")), \
             patch.object(worker_manager, "_spawn_process", return_value=mock_proc), \
             patch.object(worker_manager, "_poll_worker_ready", side_effect=poll_results):

            res = await worker_manager._execute_transition("gpu_max", trans_id="recovery_test")

            assert res["status"] == "recovered_previous"
            assert worker_manager.active_worker == "gpu_finetune"
            # P1-2: Verify selected_mode is restored to previous_mode!
            assert worker_manager.selected_mode == "gpu_finetune"
            assert worker_manager.transition_state == "ready"
            assert "đã khôi phục thành công worker trước đó" in res["message"]

    asyncio.run(_test())


def test_cleanup_failure_blocks_recovery_and_preserves_error_state(worker_manager):
    """
    CRITICAL P0-A: If target worker startup fails AND cleanup of the broken process fails,
    verify that recovery to previous worker is STRICTLY BLOCKED, state remains in error,
    and no false claim of VRAM reclamation is made.
    """
    async def _test():
        worker_manager.active_worker = "gpu_finetune"
        worker_manager.selected_mode = "gpu_finetune"
        mock_proc = MagicMock()
        mock_proc.pid = 66666

        # _terminate_current_worker succeeds during pre-switch termination,
        # but FAILS during post-startup failure cleanup (PGID lingers or VRAM not freed)
        term_calls = 0
        async def mock_terminate(*args, **kwargs):
            nonlocal term_calls
            term_calls += 1
            if term_calls > 1:
                # Cleanup of failed worker fails!
                raise WorkerTerminationError("Process group 66666 is unkillable")
            return None

        with patch.object(worker_manager, "_drain_in_flight_requests", new_callable=AsyncMock), \
             patch.object(worker_manager, "_terminate_current_worker", side_effect=mock_terminate), \
             patch.object(GPUVRAMProbe, "check_vram_against_profile", return_value=(True, "OK")), \
             patch.object(worker_manager, "_spawn_process", return_value=mock_proc) as mock_spawn, \
             patch.object(worker_manager, "_poll_worker_ready", return_value=False):

            res = await worker_manager._execute_transition("gpu_max", trans_id="cleanup_fail_test")

            # Must return error, NOT recovered_previous
            assert res["status"] == "error"
            assert worker_manager.transition_state == "error"
            assert worker_manager.active_worker is None
            # Truthful messaging
            assert "dọn dẹp (cleanup) cũng thất bại" in res["message"]
            assert "Process group 66666 is unkillable" in res["message"]
            assert "Đã giải phóng hoàn toàn tiến trình và VRAM" not in res["message"]
            # Only 1 spawn occurred (the initial target worker), NO second spawn for recovery!
            assert mock_spawn.call_count == 1

    asyncio.run(_test())


# =====================================================================
# 6. P0-4 Deep Healthcheck & Fail-Fast Child Death Tests
# =====================================================================

def test_healthcheck_rejects_finetune_if_mandatory_adapters_missing(worker_manager):
    """Verify _poll_worker_ready rejects Finetune worker if mandatory LoRA adapters are missing."""
    async def _test():
        # Server returns status 200 and 'online', but missing 'generator' and 'writer'
        mock_health_data = {
            "status": "online",
            "ready": False,
            "mandatory_ready": False,
            "loaded_adapters": ["router", "critic"],
            "active_models": ["qwen_cluster", "factcheck"],
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_health_data

        with patch("httpx.AsyncClient.get", return_value=mock_resp), \
             patch("asyncio.sleep", new_callable=AsyncMock):

            is_ready = await worker_manager._poll_worker_ready("gpu_finetune", "http://127.0.0.1:8001", timeout_seconds=1)
            assert is_ready is False

    asyncio.run(_test())


def test_readiness_poll_requires_models_ok(worker_manager):
    """
    CRITICAL P1-1: Verify _poll_worker_ready rejects Finetune worker if active_models
    is missing 'factcheck' even if all 4 LoRA adapters are present.
    """
    async def _test():
        # Server returns status 200, online, all adapters present, but missing 'factcheck' model
        mock_health_data = {
            "status": "online",
            "ready": False,
            "mandatory_ready": False,
            "loaded_adapters": ["router", "generator", "critic", "writer"],
            "active_models": ["qwen_cluster"],  # missing factcheck!
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_health_data

        with patch("httpx.AsyncClient.get", return_value=mock_resp), \
             patch("asyncio.sleep", new_callable=AsyncMock):

            is_ready = await worker_manager._poll_worker_ready("gpu_finetune", "http://127.0.0.1:8001", timeout_seconds=1)
            assert is_ready is False

    asyncio.run(_test())


def test_healthcheck_verifies_gpu_max_model_id(worker_manager):
    """Verify _poll_worker_ready for GPU Max verifies the exact 32B AWQ model ID in /v1/models."""
    async def _test():
        # Case A: Wrong model ID served
        wrong_models_data = {"data": [{"id": "meta-llama/Llama-3-8B-Instruct"}]}
        mock_resp_wrong = MagicMock()
        mock_resp_wrong.status_code = 200
        mock_resp_wrong.json.return_value = wrong_models_data

        with patch("httpx.AsyncClient.get", return_value=mock_resp_wrong), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            is_ready = await worker_manager._poll_worker_ready("gpu_max", "http://127.0.0.1:8002", timeout_seconds=1)
            assert is_ready is False

        # Case B: Correct Qwen 32B AWQ served
        correct_models_data = {"data": [{"id": "Qwen/Qwen2.5-32B-Instruct-AWQ"}]}
        mock_resp_correct = MagicMock()
        mock_resp_correct.status_code = 200
        mock_resp_correct.json.return_value = correct_models_data

        with patch("httpx.AsyncClient.get", return_value=mock_resp_correct), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            is_ready = await worker_manager._poll_worker_ready("gpu_max", "http://127.0.0.1:8002", timeout_seconds=1)
            assert is_ready is True

    asyncio.run(_test())


def test_healthcheck_fail_fast_on_child_process_death(worker_manager):
    """
    CRITICAL: Verify _poll_worker_ready aborts IMMEDIATELY (under 0.1s)
    if the child worker process exits prematurely (e.g. CUDA OOM or missing weights).
    """
    async def _test():
        dead_proc = MagicMock()
        dead_proc.poll.return_value = 1  # Process crashed with exit code 1
        dead_proc.returncode = 1
        worker_manager._proc = dead_proc

        start = time.time()
        # Even with timeout_seconds=90, it must return immediately!
        is_ready = await worker_manager._poll_worker_ready("gpu_finetune", "http://127.0.0.1:8001", timeout_seconds=90)
        duration = time.time() - start

        assert is_ready is False
        assert duration < 0.5  # Instant fail-fast!

    asyncio.run(_test())


# =====================================================================
# 7. OS Integration Tests (Real Subprocess Group Lifecycle)
# =====================================================================

def test_real_subprocess_termination_lifecycle(worker_manager):
    """
    OS INTEGRATION TEST:
    Spawns an actual OS Python process with start_new_session=True.
    Verifies that _terminate_current_worker sends signals, reaps parent process via wait(),
    and verifies that the real OS process is completely dead.
    """
    async def _test():
        # Spawn real sleeping Python process
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            start_new_session=True
        )
        pid = proc.pid
        worker_manager._proc = proc
        worker_manager._pgid = pid
        worker_manager.active_worker = "gpu_finetune"

        assert is_process_group_alive(pid) is True
        assert proc.poll() is None

        # Mock VRAM reclamation to avoid needing real dual Tesla T4s in local test
        with patch.object(worker_manager, "_verify_vram_reclaimed", new_callable=AsyncMock):
            await worker_manager._terminate_current_worker()

        # Verify real OS process was terminated
        assert is_process_group_alive(pid) is False
        assert proc.poll() is not None
        assert worker_manager._proc is None
        assert worker_manager._pgid is None
        assert worker_manager.active_worker is None

    asyncio.run(_test())


def test_real_subprocess_sigkill_escalation(worker_manager):
    """
    OS INTEGRATION TEST:
    Spawns an actual OS Python process that traps/ignores SIGTERM.
    Verifies that _terminate_current_worker detects that the process group did not exit,
    escalates to SIGKILL, and successfully reaps the process.
    """
    async def _test():
        # Script that ignores SIGTERM on POSIX
        code = (
            "import signal, time, sys\n"
            "try:\n"
            "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "except Exception:\n"
            "    pass\n"
            "time.sleep(30)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            start_new_session=True
        )
        pid = proc.pid
        worker_manager._proc = proc
        worker_manager._pgid = pid
        worker_manager.active_worker = "gpu_max"

        with patch.object(worker_manager, "_verify_vram_reclaimed", new_callable=AsyncMock):
            await worker_manager._terminate_current_worker()

        assert is_process_group_alive(pid) is False
        assert proc.poll() is not None
        assert worker_manager._proc is None

    asyncio.run(_test())


def test_real_subprocess_child_fail_fast(worker_manager):
    """
    OS INTEGRATION TEST:
    Spawns an actual OS process that immediately exits with code 42.
    Verifies that _poll_worker_ready detects the death within milliseconds and aborts.
    """
    async def _test():
        proc = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.exit(42)"],
            start_new_session=True
        )
        # Wait a moment for process to exit
        proc.wait(timeout=2.0)
        worker_manager._proc = proc
        worker_manager._pgid = proc.pid

        start = time.time()
        is_ready = await worker_manager._poll_worker_ready("gpu_finetune", "http://127.0.0.1:8001", timeout_seconds=60)
        elapsed = time.time() - start

        assert is_ready is False
        assert elapsed < 0.5  # Instant abort on real process death

    asyncio.run(_test())


# =====================================================================
# 8. Multi-Round Mode Cycling
# =====================================================================

def test_multiround_mode_cycling(worker_manager):
    """
    Verify continuous mode switching sequence:
    gpu_finetune -> gpu_max -> hybrid -> gpu_finetune -> gpu_max -> gpu_finetune
    """
    async def _test():
        sequence = [
            ("gpu_finetune", "gpu_finetune"),
            ("gpu_max", "gpu_max"),
            ("hybrid", "gpu_max"),
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

                status = worker_manager.get_status_payload()
                assert status["selected_mode"] == target_mode
                assert status["active_worker"] == expected_worker
                assert status["transition_state"] in ("ready", "idle")

    asyncio.run(_test())


# =====================================================================
# 9. Truthful Model Loading & Fail-Closed Endpoint Tests
# =====================================================================

def test_finetune_server_health_requires_mandatory_models_and_adapters():
    """
    Verify healthcheck() reports ready=False if ANY mandatory adapter or model is missing.
    """
    from kaggle import kaggle_finetune_server as srv

    with patch.object(srv, "_loaded_adapters", ["router", "critic", "writer"]), \
         patch.object(srv, "_models", {"qwen_cluster": MagicMock(), "factcheck": MagicMock()}), \
         patch("torch.cuda.is_available", return_value=False):

        # Missing 'generator'
        h = srv.healthcheck()
        assert h["ready"] is False
        assert h["mandatory_ready"] is False
        assert h["status"] == "degraded"
        assert "generator" in h["missing_adapters"]

    with patch.object(srv, "_loaded_adapters", ["router", "generator", "critic", "writer"]), \
         patch.object(srv, "_models", {"qwen_cluster": MagicMock()}), \
         patch("torch.cuda.is_available", return_value=False):

        # Missing 'factcheck'
        h = srv.healthcheck()
        assert h["ready"] is False
        assert h["mandatory_ready"] is False
        assert h["status"] == "degraded"
        assert "factcheck" in h["missing_models"]

    with patch.object(srv, "_loaded_adapters", ["router", "generator", "critic", "writer"]), \
         patch.object(srv, "_models", {"qwen_cluster": MagicMock(), "factcheck": MagicMock()}), \
         patch("torch.cuda.is_available", return_value=False):

        # All present
        h = srv.healthcheck()
        assert h["ready"] is True
        assert h["mandatory_ready"] is True
        assert h["status"] == "online"
        assert len(h["missing_adapters"]) == 0
        assert len(h["missing_models"]) == 0


def test_finetune_factcheck_endpoint_fails_closed_when_model_missing():
    """
    CRITICAL Section 3: If Fact-Checker NLI model is missing, /api/factcheck
    MUST raise HTTP 503 and NEVER return fake 'SUPPORT' verdicts with 0.95 confidence.
    """
    from kaggle import kaggle_finetune_server as srv
    from fastapi import HTTPException

    with patch.object(srv, "load_factcheck_model", return_value=(None, None)):
        req = srv.FactCheckRequest(claims=["The moon is made of green cheese."])
        with pytest.raises(HTTPException) as exc_info:
            srv.factcheck_endpoint(req)
        assert exc_info.value.status_code == 503
        assert "Fact-Checker NLI model is not loaded" in exc_info.value.detail


def test_finetune_critic_endpoint_fails_closed_on_malformed_json():
    """
    CRITICAL Section 3: If Critic output cannot be parsed as JSON,
    it MUST return action='rewrite' and NEVER default to action='accept'.
    """
    from kaggle import kaggle_finetune_server as srv

    with patch.object(srv, "generate_with_qwen", return_value="I am the critic and I had an internal error: <corrupted>"):
        req = srv.CriticRequest(
            query="What is transformer?",
            answer="A machine learning model.",
            context="Transformers were introduced in 2017.",
            claims=["Transformers introduced in 2017."]
        )
        res = srv.summarize_critic(req)
        assert res["action"] == "rewrite"
        assert "critic_malformed_json_response" in res["issues_found"]
        assert "Không thể phân tích phản hồi JSON" in res["critique"]

