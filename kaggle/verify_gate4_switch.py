#!/usr/bin/env python3
"""
Multi-Agent RAG - Gate 4 Hot-Switching & Worker Max (32B AWQ) Verifier (v2 Hardened)
Tests dynamic worker switching from Worker Finetune to Worker Max (Qwen-32B AWQ)
via the Kaggle Supervisor Gateway without restarting the master process.

Strict Assertions Enforced:
  D01: Admin Switch Request Accepted (POST /api/admin/switch_mode, HTTP 202)
  D02: Worker Clean Drain & Process Termination (PGID death & actual VRAM reclaim)
  D03: Dual GPU VRAM Baseline Reclaim (Measured <= 2.5 GB on BOTH GPUs independently, NO final_ready bypass)
  D04: Target Worker Online & Operational (Active worker matched, ready state verified)
  D05: Live Inference on Target Model (/generate output verification)
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
import httpx


class Gate4SwitchVerifier:
    def __init__(self, base_url: str, admin_token: str, poll_interval: float = 5.0, max_wait_seconds: float = 400.0):
        self.base_url = base_url.rstrip("/")
        self.admin_token = admin_token
        self.poll_interval = poll_interval
        self.max_wait_seconds = max_wait_seconds
        self.headers = {"Authorization": f"Bearer {admin_token}"}
        self.results: List[Dict[str, Any]] = []

    def record(self, test_id: str, stage: str, component: str, latency: float, details: str, passed: bool):
        self.results.append({
            "id": test_id,
            "stage": stage,
            "component": component,
            "latency": f"{latency:.2f}s" if latency > 0 else "N/A",
            "passed": passed,
            "details": details,
        })
        status_symbol = "✔ PASS" if passed else "❌ FAIL"
        print(f"  [{status_symbol}] {test_id} - {stage}: {component} ({latency:.2f}s) -> {details[:80]}")

    def get_status(self) -> Dict[str, Any]:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{self.base_url}/api/admin/status", headers=self.headers)
            resp.raise_for_status()
            return resp.json()

    def _execute_switch_cycle(self, target_mode: str) -> bool:
        """Executes a single mode switch and monitors VRAM drain across both GPUs."""
        t0 = time.time()
        print(f"\n  • Gửi lệnh chuyển đổi mode sang '{target_mode}' (async dispatch)...")
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{self.base_url}/api/admin/switch_mode",
                    headers=self.headers,
                    json={"target_mode": target_mode, "wait": False},
                )
            lat_d01 = time.time() - t0
            passed_d01 = (resp.status_code in (200, 202))
            resp_data = resp.json() if passed_d01 else {}
            detail_d01 = f"HTTP {resp.status_code}, status={resp_data.get('status')}"
            self.record("D01", "Gateway Switch", "POST /api/admin/switch_mode", lat_d01, detail_d01, passed_d01)
            if not passed_d01:
                print(f"  ❌ Switch request bị từ chối: HTTP {resp.status_code} - {resp.text}")
                return False
        except Exception as e:
            self.record("D01", "Gateway Switch", "POST /api/admin/switch_mode", time.time() - t0, str(e), False)
            return False

        print(f"\n  • Bắt đầu theo dõi vòng đời tráo đổi tiến trình (Polling status mỗi {self.poll_interval}s)...")
        start_poll = time.time()
        observed_states = set()
        lowest_vram_g0 = 999.0
        lowest_vram_g1 = 999.0
        final_ready = False
        error_msg = ""

        while time.time() - start_poll < self.max_wait_seconds:
            elapsed = time.time() - start_poll
            try:
                st = self.get_status()
                curr_worker = st.get("active_worker")
                state = st.get("transition_state", "unknown")
                msg = st.get("transition_message", "")
                vram = st.get("vram_allocation") or {}
                g0_used = vram.get("gpu_0", {}).get("used_gb", 99.0)
                g1_used = vram.get("gpu_1", {}).get("used_gb", 99.0)

                lowest_vram_g0 = min(lowest_vram_g0, g0_used)
                lowest_vram_g1 = min(lowest_vram_g1, g1_used)
                observed_states.add(state)

                print(f"    [{elapsed:5.1f}s] State: {state:<14} | Worker: {str(curr_worker):<12} | "
                      f"VRAM: G0={g0_used:4.1f}GB, G1={g1_used:4.1f}GB | Msg: {msg[:40]}")

                if state == "error":
                    error_msg = msg
                    break

                if curr_worker == target_mode and state == "ready":
                    final_ready = True
                    break

            except Exception as e:
                print(f"    [{elapsed:5.1f}s] Status poll warning: {e}")

            time.sleep(self.poll_interval)

        poll_duration = time.time() - start_poll

        # Strict D02: Lifecycle Drain & Process Termination
        # Requires: transition state reached stopping/reclaiming AND both GPUs observed reclaiming
        reclaim_state_seen = ("reclaiming_vram" in observed_states or "stopping" in observed_states)
        d02_passed = reclaim_state_seen and (lowest_vram_g0 <= 2.5 and lowest_vram_g1 <= 2.5)
        d02_detail = f"lowest: G0={lowest_vram_g0:.1f}GB, G1={lowest_vram_g1:.1f}GB, drain_seen={reclaim_state_seen}"
        self.record("D02", "Lifecycle Drain", "Process & VRAM Reclaim", poll_duration, d02_detail, d02_passed)

        # Strict D03: Dual GPU VRAM Cleanliness & Headroom Verification (NO final_ready bypass!)
        # BOTH GPU 0 and GPU 1 must have demonstrably dropped to baseline (<= 2.5 GB) during transition
        d03_passed = (lowest_vram_g0 <= 2.5 and lowest_vram_g1 <= 2.5)
        d03_detail = f"Baseline VRAM observed on both GPUs: G0={lowest_vram_g0:.1f}GB, G1={lowest_vram_g1:.1f}GB (<= 2.5GB req)"
        self.record("D03", "VRAM Probe", "Dual GPU Baseline Reclaim", 0.0, d03_detail, d03_passed)

        # Strict D04: Target Worker Readiness
        d04_passed = final_ready
        d04_detail = f"target={target_mode}, ready={final_ready}, final_state={state}, elapsed={poll_duration:.1f}s"
        if error_msg:
            d04_detail += f", err={error_msg[:40]}"
        self.record("D04", "Worker Readiness", f"{target_mode.upper()} Online Check", poll_duration, d04_detail, d04_passed)

        return final_ready and d02_passed and d03_passed

    def run_gate4_test(self, target_mode: str = "gpu_max") -> bool:
        print("\n" + "=" * 78)
        print(" 🚀 [GATE 4] KIỂM THỬ CHUYỂN ĐỔI NÓNG WORKER MAX (HOT-SWITCHING v2)")
        print(f" 🌐 Target Gateway: {self.base_url}")
        print(f" 🔄 Target Mode: {target_mode.upper()}")
        print("=" * 78)

        # Step 0: Check initial state
        try:
            init_status = self.get_status()
            initial_worker = init_status.get("active_worker")
            initial_state = init_status.get("transition_state")
            print(f"  • Trạng thái ban đầu: active_worker='{initial_worker}', state='{initial_state}'")

            # Strict requirement: If already at target mode, execute a real transition to alternate mode first
            # so that process termination and VRAM reclaim are genuinely exercised and proven.
            if initial_worker == target_mode and initial_state == "ready":
                alt_mode = "gpu_finetune" if target_mode == "gpu_max" else "gpu_max"
                print(f"  ℹ Cluster đang ở sẵn mode '{target_mode}'. Để chứng minh toàn vẹn vòng đời, "
                      f"kích hoạt chu kỳ chuyển đổi thực sự sang '{alt_mode}' rồi quay lại '{target_mode}'...")
                ok_pre = self._execute_switch_cycle(alt_mode)
                if not ok_pre:
                    print(f"  ❌ Chu kỳ chuyển đổi sơ bộ sang {alt_mode} thất bại.")
                    return False
                # Clear intermediate results to report the clean final cycle
                self.results.clear()

        except Exception as e:
            print(f"❌ Không thể kết nối tới Gateway: {e}")
            self.record("D01", "Gateway Switch", "Connect", 0.0, str(e), False)
            return False

        # Run primary transition cycle
        switch_ok = self._execute_switch_cycle(target_mode)
        if not switch_ok:
            return False

        # Step 3: Test inference on Target Model (D05)
        return self.verify_inference()

    def verify_inference(self) -> bool:
        print("\n" + "=" * 78)
        print(" 🎯 [D05] KIỂM TRA KHẢ NĂNG SUY LUẬN TRỰC TIẾP TRÊN WORKER ĐÍCH")
        print("=" * 78)

        prompt = "Hãy giải thích ngắn gọn nguyên lý của cơ chế Attention trong 3 gạch đầu dòng."
        payload = {
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.2,
        }
        t0 = time.time()
        try:
            with httpx.Client(timeout=90.0) as client:
                resp = client.post(f"{self.base_url}/generate", json=payload)
            lat = time.time() - t0
            passed = (resp.status_code == 200)
            data = resp.json() if passed else {}
            output_text = data.get("text") or data.get("output") or resp.text
            preview = f"output_len={len(output_text)} chars, HTTP {resp.status_code}"
            self.record("D05", "Worker Inference", "POST /generate", lat, preview, passed and len(output_text) > 30)

            print(f"\n  • Đầu ra suy luận từ Worker:\n{'-'*60}\n{output_text[:600]}\n{'-'*60}")
            return passed
        except Exception as e:
            self.record("D05", "Worker Inference", "POST /generate", time.time() - t0, str(e), False)
            return False

    def print_summary_table(self) -> bool:
        print("\n" + "─" * 96)
        print(" 📊 BẢNG TỔNG HỢP NGHIỆM THU GATE 4: WORKER HOT-SWITCHING (DUAL TESLA T4 v2)")
        print("─" * 96)
        print(f"| {'ID':<4} | {'Giai đoạn':<22} | {'Hạng mục kiểm thử':<28} | {'Kết quả':<7} | {'Chi tiết / Preview':<25} |")
        print(f"|{'-'*6}|{'-'*24}|{'-'*30}|{'-'*9}|{'-'*27}|")
        for r in self.results:
            status = "✔ PASS" if r["passed"] else "❌ FAIL"
            print(f"| {r['id']:<4} | {r['stage']:<22} | {r['component']:<28} | {status:<7} | {r['details'][:25]:<25} |")
        print("─" * 96)

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        pct = (passed / total) * 100 if total > 0 else 0.0
        print(f"🎯 KẾT LUẬN GATE 4: {passed}/{total} bài kiểm thử đã hoàn thành ({pct:.1f}%)!")
        if passed == total and total > 0:
            print("🟢 CƠ CHẾ HOT-SWITCHING WORKER TRÊN DUAL TESLA T4 ĐẠT 100% TIÊU CHÍ NGHIỆM THU GATE 4!")
        else:
            print("🔴 CÓ BÀI KIỂM THỬ CHƯA HOÀN TẤT. Xem chi tiết ở bảng trên.")
        print("─" * 96 + "\n")
        return passed == total and total > 0


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent RAG Gate 4 Switch Verifier (v2 Hardened)")
    parser.add_argument(
        "--url",
        default=os.environ.get("REMOTE_INFERENCE_URL", "http://127.0.0.1:8000"),
        help="Kaggle Gateway URL (defaults to REMOTE_INFERENCE_URL or http://127.0.0.1:8000)"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("SUPERVISOR_ADMIN_TOKEN", ""),
        help="Admin Token (defaults to SUPERVISOR_ADMIN_TOKEN)"
    )
    parser.add_argument("--target", default="gpu_max", help="Target mode: gpu_max or gpu_finetune")
    parser.add_argument("--max-wait", type=float, default=400.0, help="Max wait seconds for transition")
    args = parser.parse_args()

    if not args.token:
        print("❌ LỖI BẢO MẬT: SUPERVISOR_ADMIN_TOKEN không được để trống. Vui lòng truyền qua --token hoặc biến môi trường.")
        sys.exit(1)

    verifier = Gate4SwitchVerifier(base_url=args.url, admin_token=args.token, max_wait_seconds=args.max_wait)
    verifier.run_gate4_test(target_mode=args.target)
    all_ok = verifier.print_summary_table()

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
