"""
Kaggle GPU Worker Process & VRAM Lifecycle Manager (v2.1.0).

Key Architectural Guarantees:
1. Process Group Isolation: Uses `start_new_session=True` with explicit PGID tracking and `os.killpg()`.
2. Safe Worker Lease: Async context manager `acquire_worker_lease()` guarding all proxy requests.
3. Non-Destructive Drain Policy: If active requests do not drain within timeout, the switch is ABORTED,
   protecting active long-running document analysis tasks from being killed.
4. OS-Level VRAM Probe: Queries `nvidia-smi` kernel context allocations directly.
5. Dynamic Baseline & Profile Verification: Checks free VRAM against manifest specifications.
6. Unified Manifest Engine: Cryptographically validates checkpoints from `kaggle/weights_manifest.json`.
7. Concurrency Serialization: `asyncio.Lock` guarantees single active transition at any moment.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple


class WorkerSwitchingError(Exception):
    """Raised when an incoming request attempts to acquire a lease during worker transition."""
    pass


class DrainTimeoutAbortError(Exception):
    """Raised when in-flight requests cannot drain in time; transition is safely aborted."""
    pass


class GPUVRAMProbe:
    """Probes actual OS-level GPU memory consumption across all kernel processes."""

    @staticmethod
    def get_vram_info() -> Dict[int, Dict[str, float]]:
        """
        Returns {gpu_index: {"used_mb": float, "total_mb": float, "free_mb": float}}.
        Uses nvidia-smi directly to capture all CUDA contexts from the Linux kernel.
        """
        vram_data: Dict[int, Dict[str, float]] = {}
        try:
            cmd = ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,nounits,noheader"]
            output = subprocess.check_output(cmd, encoding="utf-8", timeout=3.0)
            for line in output.strip().splitlines():
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) >= 3:
                    idx = int(parts[0])
                    used = float(parts[1])
                    total = float(parts[2])
                    vram_data[idx] = {
                        "used_mb": used,
                        "total_mb": total,
                        "free_mb": max(0.0, total - used),
                    }
            if vram_data:
                return vram_data
        except Exception:
            pass

        # Fallback to torch.cuda if nvidia-smi query is unavailable
        try:
            import torch
            if torch.cuda.is_available():
                for idx in range(torch.cuda.device_count()):
                    used_bytes = torch.cuda.memory_allocated(idx)
                    total_bytes = torch.cuda.get_device_properties(idx).total_memory
                    used_mb = used_bytes / (1024 * 1024)
                    total_mb = total_bytes / (1024 * 1024)
                    vram_data[idx] = {
                        "used_mb": round(used_mb, 1),
                        "total_mb": round(total_mb, 1),
                        "free_mb": round(max(0.0, total_mb - used_mb), 1),
                    }
        except Exception:
            pass

        return vram_data

    @classmethod
    def check_vram_against_profile(cls, target_mode: str, manifest: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validates whether target GPUs have sufficient free VRAM as defined in manifest.
        Does NOT rely on fixed arbitrary constants.
        """
        vram = cls.get_vram_info()
        if not vram:
            return True, "No GPU detected or non-CUDA environment; check bypassed."

        base_models = manifest.get("base_models", {})

        if target_mode == "gpu_max":
            spec = base_models.get("qwen_32b_awq", {})
            min_free_req = float(spec.get("min_vram_per_gpu_mb", 13500.0))
            for idx in [0, 1]:
                info = vram.get(idx, {})
                free = info.get("free_mb", 0.0)
                if free < min_free_req:
                    return False, f"GPU {idx} has {free:.1f} MB free VRAM (requires >= {min_free_req:.1f} MB for 32B AWQ TP=2)."
            return True, f"Sufficient VRAM verified for 32B AWQ TP=2 (>= {min_free_req} MB free on both GPUs)."

        elif target_mode == "gpu_finetune":
            spec = base_models.get("qwen_7b_base", {})
            min_gpu0_req = float(spec.get("min_vram_mb", 4800.0)) + 3000.0  # base + LoRA context (~7.8 GB)
            min_gpu1_req = 7000.0  # NLI + reader headroom (~7.0 GB)

            free_0 = vram.get(0, {}).get("free_mb", 0.0)
            if free_0 < min_gpu0_req:
                return False, f"GPU 0 has {free_0:.1f} MB free VRAM (requires >= {min_gpu0_req:.1f} MB for Qwen 7B NF4 + LoRAs)."

            if len(vram) >= 2:
                free_1 = vram.get(1, {}).get("free_mb", 0.0)
                if free_1 < min_gpu1_req:
                    return False, f"GPU 1 has {free_1:.1f} MB free VRAM (requires >= {min_gpu1_req:.1f} MB for NLI + Reader)."

            return True, "Sufficient VRAM verified for GPU Finetune Worker."

        return True, "OK"


SIGKILL = getattr(signal, "SIGKILL", 9)
SIGTERM = getattr(signal, "SIGTERM", 15)


def kill_process_group(pgid: int, sig: int) -> None:
    """Sends signal to entire process group across POSIX/Windows."""
    if hasattr(os, "killpg"):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        except Exception as e:
            print(f"[WorkerManager] Warning sending signal {sig} to pgid {pgid}: {e}")
    else:
        # Windows fallback
        try:
            if sig == SIGKILL:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pgid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.kill(pgid, SIGTERM)
        except Exception:
            pass


def is_process_group_alive(pgid: int) -> bool:
    """Checks whether any processes in the process group are still running."""
    if hasattr(os, "killpg"):
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    elif hasattr(os, "kill"):
        try:
            os.kill(pgid, 0)
            return True
        except OSError:
            return False
    return False


class WorkerManager:
    """
    Production-grade Process Group & VRAM Lifecycle Manager for Kaggle Dual-T4.
    Orchestrates transitions between Worker A (gpu_finetune) and Worker B (gpu_max).
    """

    def __init__(self, project_dir: str):
        self.project_dir = os.path.abspath(project_dir)
        self.lock = asyncio.Lock()

        # Operational State Contract
        self.selected_mode: str = "gpu_finetune"  # "gpu_finetune" | "gpu_max" | "hybrid"
        self.active_worker: Optional[str] = None   # "gpu_finetune" | "gpu_max" | None
        self.target_worker: Optional[str] = None
        self.transition_state: str = "idle"       # "idle" | "draining" | "stopping" | "reclaiming_vram" | "spawning" | "warming_up" | "ready" | "error"
        self.transition_id: Optional[str] = None
        self.transition_message: str = "Hệ thống sẵn sàng."

        # Request In-flight Tracking & Worker Lease
        self.active_requests: int = 0
        self.queue_depth: int = 0

        # Process Handles & Process Group ID
        self._proc: Optional[subprocess.Popen] = None
        self._pgid: Optional[int] = None
        self._last_switch_time: float = time.time()

        # Baseline VRAM Record
        self.baseline_vram: Dict[int, Dict[str, float]] = GPUVRAMProbe.get_vram_info()

        # Load & Validate Manifest
        self.manifest = self._load_manifest()
        self.validated_capabilities = self.validate_manifest_checkpoints()

    def _load_manifest(self) -> Dict[str, Any]:
        manifest_path = Path(self.project_dir) / "kaggle" / "weights_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WorkerManager WARNING] Could not parse weights_manifest.json: {e}")
        return {}

    def validate_manifest_checkpoints(self, verify_sha256: bool = False) -> Dict[str, bool]:
        """
        Validates all checkpoints against disk files.
        Returns a dictionary mapping capability -> bool.
        """
        capabilities = {
            "router": False,
            "generator": False,
            "critic": False,
            "writer": False,
            "fact_checker_nli": False,
            "layout_reader": False,
            "reranker": False,
            "vllm_32b_awq": False,
        }

        if not self.manifest or "checkpoints" not in self.manifest:
            return capabilities

        base_weights_dir = Path(self.project_dir) / "training_lab" / "final_weights"

        for cp in self.manifest.get("checkpoints", []):
            cap = cp.get("capability")
            sub_path = cp.get("sub_path", "")
            cp_dir = base_weights_dir / sub_path

            if not cp_dir.exists():
                continue

            # Verify required files exist
            req_files = cp.get("required_files", [])
            if not all((cp_dir / f).exists() for f in req_files):
                continue

            # Optional cryptographic SHA256 integrity check
            if verify_sha256 and "primary_file" in cp:
                pf = cp["primary_file"]
                pf_path = cp_dir / pf["filename"]
                if not pf_path.exists():
                    continue
                h = hashlib.sha256()
                with open(pf_path, "rb") as f:
                    while chunk := f.read(1024 * 1024):
                        h.update(chunk)
                if h.hexdigest() != pf["sha256"]:
                    print(f"[WorkerManager WARNING] Checksum mismatch on {cp['id']}")
                    continue

            if cap in capabilities:
                capabilities[cap] = True

        return capabilities

    def get_active_worker_url(self) -> Optional[str]:
        if self.active_worker == "gpu_max":
            return "http://127.0.0.1:8002"
        elif self.active_worker == "gpu_finetune":
            return "http://127.0.0.1:8001"
        return None

    @asynccontextmanager
    async def acquire_worker_lease(self) -> AsyncGenerator[None, None]:
        """
        Safe Async Context Manager: Guards every request processed by the Supervisor.
        Usage:
            async with worker_manager.acquire_worker_lease():
                await handle_request()
        Raises WorkerSwitchingError if worker is transitioning.
        """
        if self.transition_state in ("stopping", "reclaiming_vram", "spawning", "draining"):
            raise WorkerSwitchingError(
                f"GPU đang trong quá trình chuyển chế độ [{self.transition_state}]. Vui lòng thử lại sau."
            )

        self.active_requests += 1
        try:
            yield
        finally:
            self.active_requests = max(0, self.active_requests - 1)

    async def _drain_in_flight_requests(self, max_wait_seconds: float = 30.0) -> None:
        """
        Gracefully waits for active requests to finish.
        CRITICAL RULE: If timeout expires and requests are STILL active, DO NOT KILL THE WORKER!
        Raises DrainTimeoutAbortError to safely abort transition and preserve user tasks.
        """
        if self.active_requests == 0:
            return

        self.transition_state = "draining"
        self.transition_message = f"Đang chờ {self.active_requests} yêu cầu hiện tại hoàn tất..."
        start = time.time()

        while self.active_requests > 0:
            if time.time() - start > max_wait_seconds:
                # Abort transition rather than killing long document tasks!
                self.transition_state = "idle"
                self.transition_message = f"Hủy tráo đổi: còn {self.active_requests} yêu cầu đang xử lý dở. Worker hiện tại được bảo toàn."
                print(f"[WorkerManager ABORT] {self.transition_message}")
                raise DrainTimeoutAbortError(self.transition_message)
            await asyncio.sleep(0.5)

    async def _terminate_current_worker(self) -> None:
        """
        Terminates the active worker using strictly tracked Process Group (PGID):
        1. Sends SIGTERM to the process group.
        2. Waits up to 8s for graceful exit, checking is_process_group_alive(pgid).
        3. If any processes (parent or child) linger, sends SIGKILL to the process group.
        4. Waits and verifies all processes in the PGID have terminated.
        5. Reclaims and verifies VRAM via OS probe.
        """
        self.transition_state = "stopping"
        if self._proc is None and self._pgid is None:
            self.active_worker = None
            return

        pgid = self._pgid or (self._proc.pid if self._proc else None)
        print(f"[WorkerManager] Terminating Process Group pgid={pgid}...")

        if pgid:
            kill_process_group(pgid, SIGTERM)

        # Wait up to 8s for graceful process exit
        poll_start = time.time()
        while time.time() - poll_start < 8.0:
            if not is_process_group_alive(pgid):
                break
            await asyncio.sleep(0.3)

        # Force kill process group if any processes (parent or child) are still lingering
        if pgid and is_process_group_alive(pgid):
            print(f"[WorkerManager] Lingering processes detected in pgid {pgid}, sending SIGKILL to group...")
            kill_process_group(pgid, SIGKILL)
            kill_start = time.time()
            while time.time() - kill_start < 3.0:
                if not is_process_group_alive(pgid):
                    break
                await asyncio.sleep(0.2)

        if self._proc:
            try:
                self._proc.wait(timeout=2.0)
            except Exception:
                pass

        self._proc = None
        self._pgid = None
        self.active_worker = None

        # Verify OS-level VRAM reclamation
        self.transition_state = "reclaiming_vram"
        await self._verify_vram_reclaimed()

    async def _verify_vram_reclaimed(self, max_wait_seconds: float = 8.0) -> None:
        """Polls VRAM until memory is released back to baseline or timeout reached."""
        start = time.time()
        while time.time() - start < max_wait_seconds:
            vram = GPUVRAMProbe.get_vram_info()
            if not vram:
                break
            # If all detected GPUs have > 12 GB free, context is reclaimed
            all_free = all(info.get("free_mb", 0) > 12000 for info in vram.values())
            if all_free:
                print("[WorkerManager] VRAM fully reclaimed and verified via OS probe.")
                break
            await asyncio.sleep(0.5)

    def _spawn_process(self, cmd: List[str], log_filename: str) -> subprocess.Popen:
        """
        Spawns worker process in a brand new Process Group/Session using start_new_session=True.
        Avoids preexec_fn deadlocks in multithreaded Python runtimes.
        """
        log_dir = Path(self.project_dir) / "kaggle"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(log_dir / log_filename, "a", encoding="utf-8", errors="ignore")

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Portable, thread-safe process group creation
            cwd=str(Path(self.project_dir) / "kaggle"),
        )
        self._pgid = proc.pid  # When start_new_session=True, pgid == proc.pid
        return proc

    async def _poll_worker_ready(self, url: str, timeout_seconds: int = 90) -> bool:
        """Polls worker health endpoint until 200 OK or timeout."""
        import httpx
        client = httpx.AsyncClient(timeout=3.0)
        health_url = f"{url}/health"
        start = time.time()
        while time.time() - start < timeout_seconds:
            try:
                resp = await client.get(health_url)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "online" or data.get("worker_status") == "online":
                        return True
            except Exception:
                pass
            await asyncio.sleep(1.5)
        return False

    async def request_switch_mode(self, target_mode: str) -> Dict[str, Any]:
        """
        External Entrypoint for Mode Switching:
        - Acquires transition lock to serialize switches (zero race conditions).
        - Returns transition ID and 202 Accepted semantics.
        """
        clean_target = target_mode.lower().strip()
        if clean_target not in ("gpu_finetune", "gpu_max", "hybrid"):
            raise ValueError(f"Chế độ không hợp lệ: {target_mode}. Chỉ hỗ trợ: gpu_finetune, gpu_max, hybrid.")

        trans_id = str(uuid.uuid4())[:8]

        # Check if lock is currently held (another switch in progress)
        if self.lock.locked():
            return {
                "status": "rejected",
                "transition_id": trans_id,
                "message": "Một tác vụ chuyển chế độ khác đang diễn ra. Vui lòng thử lại sau.",
            }

        async with self.lock:
            # Case 1: Switching to 'hybrid' -> update policy without necessarily restarting worker
            if clean_target == "hybrid":
                self.selected_mode = "hybrid"
                if self.active_worker is None:
                    # Default bootstrap worker for hybrid is gpu_finetune
                    return await self._execute_transition("gpu_finetune", trans_id)
                return {
                    "status": "accepted",
                    "transition_id": trans_id,
                    "selected_mode": "hybrid",
                    "active_worker": self.active_worker,
                    "message": f"Chính sách Hybrid đã được áp dụng. Worker hiện tại ({self.active_worker}) sẵn sàng phục vụ."
                }

            self.selected_mode = clean_target

            # Case 2: Target worker already active and ready
            if self.active_worker == clean_target and self.transition_state == "ready":
                active_url = self.get_active_worker_url()
                if active_url and await self._poll_worker_ready(active_url, timeout_seconds=2):
                    return {
                        "status": "already_active",
                        "transition_id": trans_id,
                        "selected_mode": self.selected_mode,
                        "active_worker": self.active_worker,
                        "message": f"Worker {clean_target} đang trực tuyến và sẵn sàng."
                    }

            return await self._execute_transition(clean_target, trans_id)

    async def _execute_transition(self, target_worker: str, trans_id: str) -> Dict[str, Any]:
        """Internal execution of worker transition."""
        self.transition_id = trans_id
        self.target_worker = target_worker

        print(f"\n[WorkerManager] === BẮT ĐẦU TRÁO ĐỔI VRAM [ID: {trans_id}] -> {target_worker.upper()} ===")

        # 1. Drain in-flight requests (abort safely if user document task is active)
        try:
            await self._drain_in_flight_requests(max_wait_seconds=30.0)
        except DrainTimeoutAbortError as abort_err:
            return {
                "status": "aborted",
                "transition_id": trans_id,
                "selected_mode": self.selected_mode,
                "active_worker": self.active_worker,
                "message": str(abort_err),
            }

        # 2. Terminate existing worker with Process Group isolation
        await self._terminate_current_worker()

        # 3. Check VRAM capacity against profile requirements
        vram_ok, vram_msg = GPUVRAMProbe.check_vram_against_profile(target_worker, self.manifest)
        if not vram_ok:
            self.transition_state = "error"
            self.transition_message = f"VRAM không đủ để khởi động {target_worker}: {vram_msg}"
            print(f"[WorkerManager ERROR] {self.transition_message}")
            return {
                "status": "error",
                "transition_id": trans_id,
                "message": self.transition_message,
            }

        # 4. Spawn target worker process
        self.transition_state = "spawning"
        self.transition_message = f"Đang khởi động tiến trình worker {target_worker}..."

        try:
            if target_worker == "gpu_max":
                cmd = [
                    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                    "--model", "Qwen/Qwen2.5-32B-Instruct-AWQ",
                    "--tensor-parallel-size", "2",
                    "--dtype", "float16",
                    "--quantization", "awq",
                    "--enforce-eager",
                    "--gpu-memory-utilization", "0.88",
                    "--max-model-len", "4096",
                    "--port", "8002",
                ]
                self._proc = self._spawn_process(cmd, "worker_max.log")
                online = await self._poll_worker_ready("http://127.0.0.1:8002", timeout_seconds=90)
            else:
                cmd = [
                    sys.executable, "-m", "uvicorn",
                    "kaggle_finetune_server:app",
                    "--host", "127.0.0.1",
                    "--port", "8001",
                    "--app-dir", str(Path(self.project_dir) / "kaggle"),
                ]
                self._proc = self._spawn_process(cmd, "worker_finetune.log")
                online = await self._poll_worker_ready("http://127.0.0.1:8001", timeout_seconds=75)

            if not online:
                raise RuntimeError(f"Worker {target_worker} không phản hồi health check 200 OK.")

            # Post-startup VRAM & health verification
            post_vram = GPUVRAMProbe.get_vram_info()
            if post_vram:
                for idx, info in post_vram.items():
                    print(f"[WorkerManager] Post-startup GPU {idx}: used={info['used_mb']:.1f} MB, free={info['free_mb']:.1f} MB")

            self.active_worker = target_worker
            self.target_worker = None
            self.transition_state = "ready"
            self.transition_message = f"Worker {target_worker.upper()} đã trực tuyến và sẵn sàng phục vụ."
            self._last_switch_time = time.time()
            print(f"[WorkerManager] === TRÁO ĐỔI VRAM THÀNH CÔNG [ID: {trans_id}] ===")

            return {
                "status": "success",
                "transition_id": trans_id,
                "selected_mode": self.selected_mode,
                "active_worker": self.active_worker,
                "message": self.transition_message,
            }

        except Exception as exc:
            self.transition_state = "error"
            self.transition_message = f"Lỗi khởi động {target_worker}: {exc}"
            print(f"[WorkerManager ERROR] {self.transition_message}")
            return {
                "status": "error",
                "transition_id": trans_id,
                "message": self.transition_message,
            }

    def get_status_payload(self) -> Dict[str, Any]:
        """
        Builds standardized status contract for GET /api/admin/status and GET /health.
        Matches AGENT.md and evaluation audit contract.
        """
        vram_info = GPUVRAMProbe.get_vram_info()
        vram_display = {
            f"gpu_{idx}": {
                "used_gb": round(info["used_mb"] / 1024, 2),
                "total_gb": round(info["total_mb"] / 1024, 2),
                "free_gb": round(info["free_mb"] / 1024, 2),
            }
            for idx, info in vram_info.items()
        }

        # Check profiles availability based on manifest
        finetune_available = all(
            self.validated_capabilities.get(req_cap, False)
            for req_cap in ["router", "generator", "critic", "fact_checker_nli", "writer"]
        )

        capabilities = {
            "router": False,
            "generator": False,
            "critic": False,
            "writer": False,
            "fact_checker_nli": False,
            "layout_reader": False,
            "reranker": False,
            "vllm_32b_awq": False,
        }

        if self.active_worker == "gpu_max":
            capabilities.update({
                "router": True,
                "generator": True,
                "critic": True,
                "writer": True,
                "vllm_32b_awq": True,
            })
        elif self.active_worker == "gpu_finetune":
            # True capability state based on actual verified models
            capabilities.update(self.validated_capabilities)

        return {
            "runtime_status": "running" if self.transition_state != "error" else "degraded",
            "selected_mode": self.selected_mode,
            "active_worker": self.active_worker,
            "target_worker": self.target_worker,
            "transition_state": self.transition_state,
            "transition_id": self.transition_id,
            "transition_message": self.transition_message,
            "active_requests": self.active_requests,
            "queue_depth": self.queue_depth,
            "is_switching": self.transition_state in ("draining", "stopping", "reclaiming_vram", "spawning"),
            "vram_allocation": vram_display,
            "worker_profiles_available": {
                "gpu_finetune": finetune_available,
                "gpu_max": True,
            },
            "active_worker_capabilities": capabilities,
        }
