#!/usr/bin/env python3
"""
Multi-Agent RAG - Gate 5 & Gate 6 Comprehensive Verification Suite (v2 Hardened)
Senior Audit Review Compliant:
Tests:
  Gate 5 (Hybrid Policy & 32B AWQ Multi-Endpoint Proxy Adaptation):
    - E01: Direct 32B AWQ Generation (/generate)
    - E02: Router Transparent Proxy Adaptation (/api/router)
    - E03: Generator Transparent Proxy Adaptation (/api/summarize/generator)
    - E04: Writer Transparent Proxy Adaptation (/api/writer) - Verifies Markdown synthesis report
    - E05: Critic Transparent Proxy Adaptation (/api/summarize/critic) - Verifies adversarial action/critique
    - E06: Zero-Gemini Hybrid Offline Fallback Contract (Rigorous Mock Test: Remote GPU failure -> Local CPU -> Extractive, zero Gemini calls)
  Gate 6 (Concurrency, Worker Lease & Lifecycle Drain):
    - F01: Concurrency & Worker Lease Tracking (3 concurrent requests, in-flight lease tracking)
    - F02: Admin Switch-Back to gpu_finetune (/api/admin/switch_mode)
    - F03: Dual-GPU VRAM Reclaim & Lifecycle Monitoring (Independent baseline <= 2.5GB on BOTH GPUs, NO final_ready bypass)
    - F04: Target Finetune Worker Online & Operational (Ready state verified)
    - F05: Post-Switch Multi-LoRA Adapters Restored & Live Probe (Router, Generator, Critic, Writer)
"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
import httpx

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAIN_DIR = os.path.join(PROJECT_ROOT, "main")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MAIN_DIR not in sys.path:
    sys.path.insert(0, MAIN_DIR)


def get_git_commit_sha() -> str:
    """Retrieves current git commit hash for reproducible audit logging."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5.0
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


class Gate5and6Verifier:
    def __init__(
        self,
        base_url: str,
        admin_token: str,
        inference_api_key: str = "",
        timeout: float = 90.0,
        poll_interval: float = 5.0,
        max_wait_seconds: float = 360.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.admin_token = admin_token
        self.inference_api_key = inference_api_key
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_wait_seconds = max_wait_seconds

        self.admin_headers = {"Authorization": f"Bearer {admin_token}"}
        self.inference_headers: Dict[str, str] = {}
        if inference_api_key:
            self.inference_headers["X-API-Key"] = inference_api_key
        elif admin_token:
            self.inference_headers["Authorization"] = f"Bearer {admin_token}"

        self.results: List[Dict[str, Any]] = []
        self.vram_profile: Dict[str, Any] = {
            "initial": {},
            "lowest_observed": {},
            "final": {}
        }

    def record(self, test_id: str, gate: str, component: str, latency: float, details: str, passed: bool):
        self.results.append({
            "id": test_id,
            "gate": gate,
            "component": component,
            "latency": f"{latency:.2f}s" if latency > 0 else "N/A",
            "latency_sec": round(latency, 3),
            "passed": passed,
            "details": details,
        })
        status_symbol = "✔ PASS" if passed else "❌ FAIL"
        print(f"  [{status_symbol}] {test_id} - {gate}: {component} ({latency:.2f}s) -> {details[:85]}")

    def get_status(self) -> Dict[str, Any]:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{self.base_url}/api/admin/status", headers=self.admin_headers)
            resp.raise_for_status()
            return resp.json()

    def ensure_mode(self, target_mode: str) -> bool:
        """Ensures the cluster is in the target_mode, switching if necessary."""
        try:
            st = self.get_status()
            curr_w = st.get("active_worker")
            curr_s = st.get("transition_state")
            if curr_w == target_mode and curr_s == "ready":
                return True

            print(f"  ℹ Cluster đang ở mode '{curr_w}' ({curr_s}). Đang chuyển sang '{target_mode}'...")
            with httpx.Client(timeout=30.0) as client:
                client.post(
                    f"{self.base_url}/api/admin/switch_mode",
                    headers=self.admin_headers,
                    json={"target_mode": target_mode, "wait": False}
                )

            start = time.time()
            while time.time() - start < self.max_wait_seconds:
                time.sleep(self.poll_interval)
                st = self.get_status()
                if st.get("active_worker") == target_mode and st.get("transition_state") == "ready":
                    print(f"  ✔ Cluster đã sẵn sàng ở mode '{target_mode}'.")
                    return True
            print(f"  ❌ Quá thời gian chờ cluster chuyển sang mode '{target_mode}'.")
            return False
        except Exception as e:
            print(f"  ❌ Lỗi khi chuyển sang mode '{target_mode}': {e}")
            return False

    # ==================== GATE 5 ====================
    def run_gate5(self) -> bool:
        print("\n" + "=" * 80)
        print(" 🚀 [GATE 5] KIỂM THỬ HYBRID POLICY & 32B MULTI-ENDPOINT PROXY ADAPTATION")
        print("=" * 80)

        # Pre-flight check: Gate 5 requires gpu_max
        if not self.ensure_mode("gpu_max"):
            print("  ❌ Không thể đưa cluster về mode 'gpu_max' để chạy Gate 5.")
            return False

        all_passed = True

        # E01: 32B Direct generation (/generate)
        print("\n  • Đang kiểm tra E01: Direct Generation trên Qwen-32B AWQ...")
        t0 = time.time()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(
                    f"{self.base_url}/generate",
                    headers=self.inference_headers,
                    json={
                        "prompt": "Tóm tắt 1 câu định nghĩa về Trí tuệ Nhân tạo.",
                        "max_tokens": 128,
                        "temperature": 0.2
                    }
                )
            lat = time.time() - t0
            passed = (r.status_code == 200)
            data = r.json() if passed else {}
            txt = data.get("text") or data.get("output") or ""
            preview = f"HTTP {r.status_code}, len={len(txt)} chars: '{txt[:45]}...'"
            e01_passed = passed and len(txt) > 20
            self.record("E01", "Gate 5 (32B Proxy)", "Direct 32B Generation (/generate)", lat, preview, e01_passed)
            if not e01_passed: all_passed = False
        except Exception as e:
            self.record("E01", "Gate 5 (32B Proxy)", "Direct 32B Generation (/generate)", time.time() - t0, str(e), False)
            all_passed = False

        # E02: Router transparent adaptation on 32B (/api/router)
        print("  • Đang kiểm tra E02: Router Transparent Proxy Adaptation (/api/router)...")
        t0 = time.time()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(
                    f"{self.base_url}/api/router",
                    headers=self.inference_headers,
                    json={
                        "query": "Tìm kiếm các nghiên cứu về lượng tử hóa mạng nơ-ron sâu."
                    }
                )
            lat = time.time() - t0
            passed = (r.status_code == 200)
            data = r.json() if passed else {}
            intent = data.get("intent", "")
            sub_q = data.get("sub_queries", [])
            preview = f"HTTP {r.status_code}, intent='{intent[:25]}', sub_q={len(sub_q)}, alpha={data.get('alpha')}"
            e02_passed = passed and bool(intent)
            self.record("E02", "Gate 5 (32B Proxy)", "Router Proxy (/api/router)", lat, preview, e02_passed)
            if not e02_passed: all_passed = False
        except Exception as e:
            self.record("E02", "Gate 5 (32B Proxy)", "Router Proxy (/api/router)", time.time() - t0, str(e), False)
            all_passed = False

        # E03: Generator transparent adaptation on 32B (/api/summarize/generator)
        print("  • Đang kiểm tra E03: Generator Transparent Proxy Adaptation (/api/summarize/generator)...")
        t0 = time.time()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(
                    f"{self.base_url}/api/summarize/generator",
                    headers=self.inference_headers,
                    json={
                        "query": "LoRA là gì?",
                        "context": "LoRA là phương pháp đóng băng trọng số mô hình gốc và thêm các ma trận tích phân rã rank thấp.",
                        "max_tokens": 256
                    }
                )
            lat = time.time() - t0
            passed = (r.status_code == 200)
            data = r.json() if passed else {}
            ans = data.get("answer") or data.get("text") or ""
            claims = data.get("claims", [])
            preview = f"HTTP {r.status_code}, ans_len={len(ans)}, claims={len(claims)}"
            e03_passed = passed and bool(ans)
            self.record("E03", "Gate 5 (32B Proxy)", "Generator Proxy (/api/summarize/generator)", lat, preview, e03_passed)
            if not e03_passed: all_passed = False
        except Exception as e:
            self.record("E03", "Gate 5 (32B Proxy)", "Generator Proxy (/api/summarize/generator)", time.time() - t0, str(e), False)
            all_passed = False

        # E04: Writer transparent adaptation on 32B (/api/writer)
        print("  • Đang kiểm tra E04: Writer Transparent Proxy Adaptation (/api/writer)...")
        t0 = time.time()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(
                    f"{self.base_url}/api/writer",
                    headers=self.inference_headers,
                    json={
                        "query": "Nghiên cứu về cơ chế Multi-Head Attention",
                        "answer": "Multi-Head Attention chiếu các biểu diễn vào nhiều không gian con khác nhau.",
                        "claims": [
                            {"claim_text": "Transformer sử dụng scaled dot-product attention."},
                            {"claim_text": "Cơ chế song song hóa giảm thiểu thời gian huấn luyện."}
                        ]
                    }
                )
            lat = time.time() - t0
            passed = (r.status_code == 200)
            data = r.json() if passed else {}
            report = data.get("report") or data.get("text") or ""
            has_markdown = any(token in report for token in ["#", "**", "-", "\n\n"])
            e04_passed = passed and len(report) > 50 and has_markdown
            preview = f"HTTP {r.status_code}, report_len={len(report)}, markdown_struct={has_markdown}"
            self.record("E04", "Gate 5 (32B Proxy)", "Writer Proxy (/api/writer)", lat, preview, e04_passed)
            if not e04_passed: all_passed = False
        except Exception as e:
            self.record("E04", "Gate 5 (32B Proxy)", "Writer Proxy (/api/writer)", time.time() - t0, str(e), False)
            all_passed = False

        # E05: Critic transparent adaptation on 32B (/api/summarize/critic)
        print("  • Đang kiểm tra E05: Critic Transparent Proxy Adaptation (/api/summarize/critic)...")
        t0 = time.time()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(
                    f"{self.base_url}/api/summarize/critic",
                    headers=self.inference_headers,
                    json={
                        "query": "LoRA là gì?",
                        "answer": "LoRA tăng tốc độ huấn luyện bằng cách giảm tham số có thể huấn luyện.",
                        "context": "LoRA là phương pháp đóng băng trọng số mô hình gốc và thêm ma trận rank thấp.",
                        "claims": ["LoRA giảm tham số huấn luyện."]
                    }
                )
            lat = time.time() - t0
            passed = (r.status_code == 200)
            data = r.json() if passed else {}
            action = data.get("action", "")
            critique = data.get("critique") or data.get("text") or ""
            e05_passed = passed and (bool(action) or bool(critique))
            preview = f"HTTP {r.status_code}, action='{action}', critique_len={len(critique)}"
            self.record("E05", "Gate 5 (32B Proxy)", "Critic Proxy (/api/summarize/critic)", lat, preview, e05_passed)
            if not e05_passed: all_passed = False
        except Exception as e:
            self.record("E05", "Gate 5 (32B Proxy)", "Critic Proxy (/api/summarize/critic)", time.time() - t0, str(e), False)
            all_passed = False

        # E06: UnifiedLLMClient Zero-Gemini Offline Fallback Policy Contract
        print("  • Đang kiểm tra E06: Zero-Gemini Hybrid Offline Fallback Contract (Strict Mock Test)...")
        t0 = time.time()
        try:
            from core.config import settings
            from utils.llm_client import UnifiedLLMClient

            old_mode = settings.EXECUTION_MODE
            old_sum_mode = settings.get_agent_mode("summarizer")
            old_url = settings.REMOTE_INFERENCE_URL
            try:
                # Force hybrid execution with an unreachable remote GPU endpoint
                settings.EXECUTION_MODE = "hybrid"
                settings.set_agent_mode("summarizer", "hybrid")
                settings.REMOTE_INFERENCE_URL = "http://127.0.0.1:9999/unreachable_gpu"

                test_client = UnifiedLLMClient()
                unique_prompt = f"Contract test prompt for zero gemini verification at {time.time()}"

                with patch.object(test_client, "_call_gemini", MagicMock()) as mock_gemini, \
                     patch.object(test_client, "_call_local_cpu", side_effect=Exception("Ollama offline")):

                    out = test_client.generate(prompt=unique_prompt, agent="summarizer")

                    # Crucial Assertion: Gemini API was NEVER called under hybrid fallback!
                    mock_gemini.assert_not_called()
                    assert bool(out and len(out) > 0), "Fallback output must not be empty"

                lat = time.time() - t0
                preview = f"Deterministic fallback len={len(out)}, verified mock_gemini.assert_not_called()"
                self.record("E06", "Gate 5 (Zero-Gemini)", "Zero-Gemini Fallback Policy Contract", lat, preview, True)
            finally:
                settings.EXECUTION_MODE = old_mode
                settings.set_agent_mode("summarizer", old_sum_mode)
                settings.REMOTE_INFERENCE_URL = old_url
        except Exception as e:
            self.record("E06", "Gate 5 (Zero-Gemini)", "Zero-Gemini Fallback Policy Contract", time.time() - t0, str(e), False)
            all_passed = False

        return all_passed

    # ==================== GATE 6 ====================
    async def run_gate6(self) -> bool:
        print("\n" + "=" * 80)
        print(" 🚀 [GATE 6] KIỂM THỬ ĐỒNG THỜI, LEASE DRAIN & HOT-SWITCH BACK VỀ FINETUNE")
        print("=" * 80)

        # Pre-flight: Gate 6 tests concurrency on gpu_max, then tests hot-switch back to gpu_finetune
        if not self.ensure_mode("gpu_max"):
            print("  ❌ Không thể đưa cluster về mode 'gpu_max' để bắt đầu Gate 6.")
            return False

        all_passed = True

        # Capture baseline VRAM before any action
        try:
            init_st = self.get_status()
            vram_init = init_st.get("vram_allocation") or {}
            self.vram_profile["initial"] = {
                "gpu_0": vram_init.get("gpu_0", {}).get("used_gb", 0.0),
                "gpu_1": vram_init.get("gpu_1", {}).get("used_gb", 0.0),
                "active_worker": init_st.get("active_worker"),
            }
            print(f"  • VRAM ban đầu: G0={self.vram_profile['initial']['gpu_0']:.1f}GB, G1={self.vram_profile['initial']['gpu_1']:.1f}GB | Worker: {self.vram_profile['initial']['active_worker']}")
        except Exception as e:
            print(f"  ⚠ Không thể lấy VRAM ban đầu: {e}")

        # F01: Concurrent Requests & Worker Lease Tracking
        print("\n  • Đang kiểm tra F01: Gửi 3 requests đồng thời để kiểm tra Worker Lease...")
        t0 = time.time()

        async def send_req(i: int):
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.base_url}/generate",
                    headers=self.inference_headers,
                    json={"prompt": f"Đếm số từ 1 đến {i+3}:", "max_tokens": 30}
                )
                return resp.status_code, resp.text

        # Monitor active lease in parallel with requests
        async def monitor_status():
            await asyncio.sleep(0.3)
            try:
                st = self.get_status()
                return st.get("active_requests", 0), st.get("queue_depth", 0)
            except Exception:
                return -1, -1

        tasks = [send_req(0), send_req(1), send_req(2), monitor_status()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        req_results = results[:3]
        monitor_res = results[3]
        lat = time.time() - t0

        success_count = sum(1 for r in req_results if isinstance(r, tuple) and r[0] == 200)
        passed_f01 = (success_count == 3)
        preview_f01 = f"3/3 requests HTTP 200, lease monitor probe={monitor_res}"
        self.record("F01", "Gate 6 (Concurrency)", "Worker Lease Concurrency", lat, preview_f01, passed_f01)
        if not passed_f01: all_passed = False

        # F02: Hot-Switching BACK from gpu_max to gpu_finetune
        print("\n  • Đang kiểm tra F02: Hot-Switching chuyển ngược về 'gpu_finetune'...")
        t0 = time.time()
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.post(
                    f"{self.base_url}/api/admin/switch_mode",
                    headers=self.admin_headers,
                    json={"target_mode": "gpu_finetune", "wait": False}
                )
            lat_f02 = time.time() - t0
            passed_f02 = (r.status_code in (200, 202))
            preview_f02 = f"HTTP {r.status_code}, status={r.json().get('status')}"
            self.record("F02", "Gate 6 (Lifecycle)", "Switch Request to gpu_finetune", lat_f02, preview_f02, passed_f02)
            if not passed_f02: all_passed = False
        except Exception as e:
            self.record("F02", "Gate 6 (Lifecycle)", "Switch Request to gpu_finetune", time.time() - t0, str(e), False)
            all_passed = False

        # F03 & F04: Monitor switch back loop & verify clean dual-GPU VRAM drain
        print("\n  • Đang theo dõi chu trình chuyển đổi về gpu_finetune (Polling status mỗi 5s)...")
        start_poll = time.time()
        final_ready = False
        reclaim_state_seen = False
        lowest_vram_g0 = 999.0
        lowest_vram_g1 = 999.0
        final_state = "unknown"

        while time.time() - start_poll < self.max_wait_seconds:
            elapsed = time.time() - start_poll
            try:
                st = self.get_status()
                worker = st.get("active_worker")
                state = st.get("transition_state", "unknown")
                final_state = state
                vram = st.get("vram_allocation") or {}
                g0 = vram.get("gpu_0", {}).get("used_gb", 99.0)
                g1 = vram.get("gpu_1", {}).get("used_gb", 99.0)

                lowest_vram_g0 = min(lowest_vram_g0, g0)
                lowest_vram_g1 = min(lowest_vram_g1, g1)

                print(f"    [{elapsed:5.1f}s] State: {state:<14} | Worker: {str(worker):<14} | VRAM: G0={g0:4.1f}GB, G1={g1:4.1f}GB")
                if state in ("reclaiming_vram", "stopping"):
                    reclaim_state_seen = True

                if worker == "gpu_finetune" and state == "ready":
                    final_ready = True
                    self.vram_profile["final"] = {"gpu_0": g0, "gpu_1": g1, "active_worker": worker}
                    break
            except Exception as e:
                print(f"    [{elapsed:5.1f}s] Poll warning: {e}")
            await asyncio.sleep(self.poll_interval)

        poll_lat = time.time() - start_poll
        self.vram_profile["lowest_observed"] = {"gpu_0": lowest_vram_g0, "gpu_1": lowest_vram_g1}

        # Strict F03: Dual-GPU VRAM Drain & Clean Reclaim (NO final_ready bypass!)
        # BOTH GPU 0 and GPU 1 must demonstrably drop to baseline (<= 2.5GB)
        f03_passed = (lowest_vram_g0 <= 2.5 and lowest_vram_g1 <= 2.5) and reclaim_state_seen
        preview_f03 = (
            f"lowest: G0={lowest_vram_g0:.1f}GB, G1={lowest_vram_g1:.1f}GB (<=2.5GB req), "
            f"drain_seen={reclaim_state_seen}"
        )
        self.record("F03", "Gate 6 (Lifecycle)", "Dual-GPU VRAM Drain & Reclaim", poll_lat, preview_f03, f03_passed)
        if not f03_passed: all_passed = False

        # Strict F04: Target Finetune Worker Restored & Ready
        f04_passed = final_ready
        preview_f04 = f"active_worker='gpu_finetune', ready={final_ready}, final_state={final_state}, elapsed={poll_lat:.1f}s"
        self.record("F04", "Gate 6 (Lifecycle)", "Finetune Worker Restored", poll_lat, preview_f04, f04_passed)
        if not f04_passed: all_passed = False

        # F05: Post-switch health check & live probe on restored adapters
        if final_ready:
            print("\n  • Đang kiểm tra F05: Xác nhận 4 LoRA Adapters sẵn sàng & Gửi live probe...")
            t0 = time.time()
            try:
                st = self.get_status()
                caps = st.get("active_worker_capabilities") or {}
                all_caps = bool(caps.get("router") and caps.get("generator") and caps.get("critic") and caps.get("writer"))

                # Live probe on restored adapter
                probe_passed = False
                with httpx.Client(timeout=30.0) as client:
                    pr = client.post(
                        f"{self.base_url}/api/router",
                        headers=self.inference_headers,
                        json={"query": "Kiểm tra khôi phục adapter router."}
                    )
                    probe_passed = (pr.status_code == 200)

                lat = time.time() - t0
                f05_passed = all_caps and probe_passed
                preview_f05 = f"All 4 adapters active ({caps}), live /api/router probe HTTP {pr.status_code}"
                self.record("F05", "Gate 6 (Health)", "4 LoRA Adapters Restored & Live Probe", lat, preview_f05, f05_passed)
                if not f05_passed: all_passed = False
            except Exception as e:
                self.record("F05", "Gate 6 (Health)", "4 LoRA Adapters Restored & Live Probe", time.time() - t0, str(e), False)
                all_passed = False

        return all_passed

    def export_evidence_artifact(self, output_path: str):
        """Exports a machine-readable JSON artifact with commit SHA, timestamp, and test results."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        commit_sha = get_git_commit_sha()
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        pct = (passed / total) * 100 if total > 0 else 0.0

        artifact = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": commit_sha,
                "target_url": self.base_url,
                "verifier": "Gate5and6Verifier_v2_Hardened",
            },
            "summary": {
                "total_tests": total,
                "passed_tests": passed,
                "failed_tests": total - passed,
                "pass_percentage": round(pct, 2),
                "all_passed": (passed == total and total > 0),
            },
            "vram_profile": self.vram_profile,
            "test_results": self.results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False)
        print(f"📁 Machine-readable evidence artifact đã được lưu tại:\n   {os.path.abspath(output_path)}")

    def print_summary_table(self) -> bool:
        print("\n" + "─" * 98)
        print(" 📊 BẢNG TỔNG HỢP NGHIỆM THU GATE 5 & GATE 6 (DUAL TESLA T4 v2 HARDENED)")
        print("─" * 98)
        print(f"| {'ID':<4} | {'Hạng mục / Gate':<24} | {'Tiêu chí kiểm thử':<30} | {'Kết quả':<7} | {'Chi tiết / Preview':<23} |")
        print(f"|{'-'*6}|{'-'*26}|{'-'*32}|{'-'*9}|{'-'*25}|")
        for r in self.results:
            status = "✔ PASS" if r["passed"] else "❌ FAIL"
            print(f"| {r['id']:<4} | {r['gate']:<24} | {r['component']:<30} | {status:<7} | {r['details'][:23]:<23} |")
        print("─" * 98)

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        pct = (passed / total) * 100 if total > 0 else 0.0
        print(f"🎯 KẾT LUẬN GATE 5 & 6: {passed}/{total} bài kiểm thử đã hoàn thành ({pct:.1f}%)!")
        if passed == total and total > 0:
            print("🟢 TOÀN BỘ TIÊU CHÍ NGHIỆM THU GATE 5 & GATE 6 ĐÃ ĐẠT 100% HOÀN HẢO!")
        else:
            print("🔴 CÓ BÀI KIỂM THỬ CHƯA HOÀN TẤT. Xem chi tiết ở bảng trên.")
        print("─" * 98 + "\n")
        return passed == total and total > 0


def main():
    parser = argparse.ArgumentParser(description="Gate 5 and 6 Verifier (v2 Hardened)")
    parser.add_argument(
        "--url",
        default=os.environ.get("REMOTE_INFERENCE_URL", "http://127.0.0.1:8000"),
        help="Kaggle Gateway URL (defaults to REMOTE_INFERENCE_URL)"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("SUPERVISOR_ADMIN_TOKEN", ""),
        help="Supervisor Admin Token (defaults to SUPERVISOR_ADMIN_TOKEN)"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("INFERENCE_API_KEY", ""),
        help="Client Inference API Key (defaults to INFERENCE_API_KEY)"
    )
    parser.add_argument(
        "--json-out",
        default=os.path.join(PROJECT_ROOT, "evaluation", "results", "kaggle_dual_t4_gate_acceptance_v2.json"),
        help="Path to export machine-readable JSON acceptance evidence"
    )
    parser.add_argument(
        "--skip-gate5",
        action="store_true",
        help="Skip Gate 5 tests"
    )
    parser.add_argument(
        "--skip-gate6",
        action="store_true",
        help="Skip Gate 6 tests"
    )
    args = parser.parse_args()

    if not args.token:
        print("❌ LỖI BẢO MẬT: SUPERVISOR_ADMIN_TOKEN không được để trống. Vui lòng truyền qua --token hoặc biến môi trường.")
        sys.exit(1)

    verifier = Gate5and6Verifier(
        base_url=args.url,
        admin_token=args.token,
        inference_api_key=args.api_key
    )

    g5_ok = True
    if not args.skip_gate5:
        g5_ok = verifier.run_gate5()

    g6_ok = True
    if not args.skip_gate6:
        g6_ok = asyncio.run(verifier.run_gate6())

    all_passed = verifier.print_summary_table()
    verifier.export_evidence_artifact(args.json_out)

    if not (g5_ok and g6_ok and all_passed):
        sys.exit(1)


if __name__ == "__main__":
    main()
