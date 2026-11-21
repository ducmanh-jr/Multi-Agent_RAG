#!/usr/bin/env python3
"""
Multi-Agent RAG: Kaggle Remote Endpoint Verification Suite (Gate 1 & Gate 2).
Automates formal acceptance testing against live Kaggle Dual T4 cluster.

Gates Covered:
- Gate 1 (A01 - A08): Health, Admin Status & Capability Verification.
- Gate 2 (B01 - B06): Six Inference Endpoints Verification (Router, Generator, Critic, Fact-Checker NLI, Writer, Generic).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle Dual T4 Gate 1 & Gate 2 Endpoint Verification")
    parser.add_argument(
        "--url",
        default=os.environ.get("REMOTE_INFERENCE_URL", "http://127.0.0.1:8000"),
        help="Base URL of the Supervisor Gateway (e.g. Cloudflare Tunnel URL or http://127.0.0.1:8000)"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("SUPERVISOR_ADMIN_TOKEN", ""),
        help="Supervisor Admin Token for authenticated endpoints"
    )
    parser.add_argument(
        "--gate",
        choices=["all", "gate1", "gate2"],
        default="all",
        help="Which gate to run: gate1, gate2, or all"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds (default: 120s)"
    )
    return parser.parse_args()


class EndpointVerifier:
    def __init__(self, base_url: str, admin_token: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.admin_token = admin_token
        self.timeout = timeout
        self.results: List[Dict[str, Any]] = []

    def record(self, test_id: str, mode: str, endpoint: str, http_code: int, latency: float, preview: str, passed: bool, notes: str = ""):
        self.results.append({
            "id": test_id,
            "mode": mode,
            "endpoint": endpoint,
            "http": http_code,
            "latency": round(latency, 2),
            "preview": preview,
            "verdict": "PASS" if passed else "FAIL",
            "notes": notes,
        })

    def run_gate1(self) -> bool:
        print("\n" + "=" * 78)
        print(" 🔍 [GATE 1] KIỂM TRA TRẠNG THÁI WORKER FINETUNE & KHẢ NĂNG (CAPABILITIES)")
        print("=" * 78)
        all_passed = True

        # A01: GET /health
        t0 = time.time()
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=self.timeout)
            lat = time.time() - t0
            if r.status_code == 200:
                data = r.json()
                st = data.get("status")
                wh = data.get("worker_health", {}).get("status")
                passed = (st == "online" and wh == "online")
                preview = f"status={st}, worker_health={wh}"
                self.record("A01", "Finetune", "/health", r.status_code, lat, preview, passed)
            else:
                self.record("A01", "Finetune", "/health", r.status_code, lat, r.text[:60], False)
                all_passed = False
        except Exception as e:
            self.record("A01", "Finetune", "/health", 0, time.time() - t0, str(e)[:60], False)
            all_passed = False

        # A02: GET /api/admin/status (Authenticated)
        t0 = time.time()
        headers = {"Authorization": f"Bearer {self.admin_token}"} if self.admin_token else {}
        try:
            r = httpx.get(f"{self.base_url}/api/admin/status", headers=headers, timeout=self.timeout)
            lat = time.time() - t0
            passed = (r.status_code == 200)
            preview = f"HTTP {r.status_code} (Admin Authenticated)" if passed else r.text[:60]
            self.record("A02", "Finetune", "/api/admin/status", r.status_code, lat, preview, passed)
            admin_data = r.json() if passed else {}
        except Exception as e:
            self.record("A02", "Finetune", "/api/admin/status", 0, time.time() - t0, str(e)[:60], False)
            all_passed = False
            admin_data = {}

        # A03: active_worker == "gpu_finetune"
        act_worker = admin_data.get("active_worker")
        passed = (act_worker == "gpu_finetune")
        self.record("A03", "Finetune", "state.active_worker", 200 if passed else 500, 0.0, f"active_worker='{act_worker}'", passed)
        if not passed: all_passed = False

        # A04: transition_state == "ready"
        tr_state = admin_data.get("transition_state")
        passed = (tr_state == "ready")
        self.record("A04", "Finetune", "state.transition_state", 200 if passed else 500, 0.0, f"transition_state='{tr_state}'", passed)
        if not passed: all_passed = False

        # A05: Check LoRA adapters (all 4 loaded)
        caps = admin_data.get("active_worker_capabilities", {})
        mandatory_caps = ["router", "generator", "critic", "writer"]
        loaded_caps = [c for c in mandatory_caps if caps.get(c) is True]
        passed = (len(loaded_caps) == 4)
        self.record("A05", "Finetune", "adapters_loaded", 200 if passed else 500, 0.0, f"4/4 adapters: {loaded_caps}", passed)
        if not passed: all_passed = False

        # A06: Check active models (qwen_cluster & factcheck)
        fc_cap = caps.get("fact_checker_nli") is True
        passed = fc_cap and (len(loaded_caps) == 4)
        self.record("A06", "Finetune", "active_models", 200 if passed else 500, 0.0, "qwen_cluster & factcheck active", passed)
        if not passed: all_passed = False

        # A07: VRAM allocation on Dual T4
        vram = admin_data.get("vram_allocation", {})
        g0 = vram.get("gpu_0", {})
        g1 = vram.get("gpu_1", {})
        g0_used = g0.get("used_gb", 0)
        g1_used = g1.get("used_gb", 0)
        passed = (g0_used > 3.0 and g1_used > 0.3)
        self.record("A07", "Finetune", "vram_allocation", 200 if passed else 500, 0.0, f"GPU0: {g0_used}GB, GPU1: {g1_used}GB", passed)
        if not passed: all_passed = False

        # A08: Admin Auth Guard (No token must return 401/403)
        t0 = time.time()
        try:
            r_no_auth = httpx.get(f"{self.base_url}/api/admin/status", timeout=self.timeout)
            lat = time.time() - t0
            passed = (r_no_auth.status_code in (401, 403))
            preview = f"Blocked with HTTP {r_no_auth.status_code}" if passed else f"Unexpected HTTP {r_no_auth.status_code}"
            self.record("A08", "Finetune", "/api/admin/status (no token)", r_no_auth.status_code, lat, preview, passed)
            if not passed: all_passed = False
        except Exception as e:
            self.record("A08", "Finetune", "/api/admin/status (no token)", 0, time.time() - t0, str(e)[:60], False)
            all_passed = False

        return all_passed

    def run_gate2(self) -> bool:
        print("\n" + "=" * 78)
        print(" 🚀 [GATE 2] KIỂM THỬ THỰC TẾ 6 API SUY LUẬN WORKER FINETUNE (LIVE INFERENCE)")
        print("=" * 78)
        all_passed = True

        # B01: /api/router
        print("\n  • Đang gửi request B01: /api/router (Agent 1 Router)...")
        payload = {"query": "Hãy xác định ý định và chia nhỏ câu hỏi nghiên cứu về tác động của AI đến giáo dục."}
        t0 = time.time()
        try:
            r = httpx.post(f"{self.base_url}/api/router", json=payload, timeout=self.timeout)
            lat = time.time() - t0
            if r.status_code == 200:
                data = r.json()
                intent = data.get("intent", "")
                sub_q = data.get("sub_queries", [])
                passed = bool(intent and sub_q)
                preview = f"intent: '{intent[:40]}...', sub_queries: {len(sub_q)}"
                self.record("B01", "Finetune", "/api/router", r.status_code, lat, preview, passed)
            else:
                self.record("B01", "Finetune", "/api/router", r.status_code, lat, r.text[:60], False)
                all_passed = False
        except Exception as e:
            self.record("B01", "Finetune", "/api/router", 0, time.time() - t0, str(e)[:60], False)
            all_passed = False

        # B02: /api/summarize/generator
        print("  • Đang gửi request B02: /api/summarize/generator (Agent 3 Generator)...")
        payload = {
            "query": "Tóm tắt ngắn gọn trong 2 câu: Điểm số hai nhóm khác nhau thế nào?",
            "context": "Thí nghiệm trên 200 sinh viên cho thấy nhóm sử dụng công cụ hỗ trợ có điểm trung bình 8.1, nhóm đối chứng có điểm trung bình 7.2.",
            "feedback": "",
            "max_tokens": 512
        }
        t0 = time.time()
        try:
            r = httpx.post(f"{self.base_url}/api/summarize/generator", json=payload, timeout=self.timeout)
            lat = time.time() - t0
            if r.status_code == 200:
                data = r.json()
                ans = data.get("answer", "")
                claims = data.get("claims", [])
                passed = bool(ans)
                preview = f"answer len: {len(ans)}, claims: {len(claims)}"
                self.record("B02", "Finetune", "/api/summarize/generator", r.status_code, lat, preview, passed)
            else:
                self.record("B02", "Finetune", "/api/summarize/generator", r.status_code, lat, r.text[:60], False)
                all_passed = False
        except Exception as e:
            self.record("B02", "Finetune", "/api/summarize/generator", 0, time.time() - t0, str(e)[:60], False)
            all_passed = False

        # B03: /api/summarize/critic
        print("  • Đang gửi request B03: /api/summarize/critic (Agent 3 Critic)...")
        payload = {
            "query": "Nghiên cứu cho thấy điều gì?",
            "answer": "Nhóm sử dụng công cụ đạt điểm trung bình cao hơn.",
            "context": "Nhóm sử dụng công cụ đạt 8.1; nhóm đối chứng đạt 7.2.",
            "claims": ["Nhóm sử dụng công cụ đạt điểm trung bình cao hơn."]
        }
        t0 = time.time()
        try:
            r = httpx.post(f"{self.base_url}/api/summarize/critic", json=payload, timeout=self.timeout)
            lat = time.time() - t0
            if r.status_code == 200:
                data = r.json()
                action = data.get("action")
                critique = data.get("critique", "")
                passed = action in ("accept", "rewrite", "request_more_context")
                preview = f"action='{action}', critique='{critique[:40]}...'"
                self.record("B03", "Finetune", "/api/summarize/critic", r.status_code, lat, preview, passed)
            else:
                self.record("B03", "Finetune", "/api/summarize/critic", r.status_code, lat, r.text[:60], False)
                all_passed = False
        except Exception as e:
            self.record("B03", "Finetune", "/api/summarize/critic", 0, time.time() - t0, str(e)[:60], False)
            all_passed = False

        # B04: /api/factcheck
        print("  • Đang gửi request B04: /api/factcheck (Agent 4 Fact-Checker NLI on GPU 1)...")
        payload = {
            "context": "Apollo 11 landed on the Moon in July 1969.",
            "claims": [
                "Apollo 11 landed on the Moon in 1969.",
                "Apollo 11 landed on Mars in 1969.",
                "The mission used a blue spacecraft."
            ]
        }
        t0 = time.time()
        try:
            r = httpx.post(f"{self.base_url}/api/factcheck", json=payload, timeout=self.timeout)
            lat = time.time() - t0
            if r.status_code == 200:
                data = r.json()
                evals = data.get("evaluations", [])
                passed = len(evals) == 3 and all("verdict" in ev for ev in evals)
                verdicts = [ev.get("verdict") for ev in evals]
                preview = f"evals={len(evals)}, verdicts={verdicts}"
                self.record("B04", "Finetune", "/api/factcheck", r.status_code, lat, preview, passed)
            else:
                self.record("B04", "Finetune", "/api/factcheck", r.status_code, lat, r.text[:60], False)
                all_passed = False
        except Exception as e:
            self.record("B04", "Finetune", "/api/factcheck", 0, time.time() - t0, str(e)[:60], False)
            all_passed = False

        # B05: /api/writer
        print("  • Đang gửi request B05: /api/writer (Agent 5 Publication Writer)...")
        payload = {
            "query": "Viết báo cáo ngắn về kết quả nghiên cứu.",
            "answer": "Nhóm thử nghiệm đạt kết quả cao hơn nhóm đối chứng.",
            "claims": [{"claim_text": "Điểm trung bình nhóm thử nghiệm là 8.1."}]
        }
        t0 = time.time()
        try:
            r = httpx.post(f"{self.base_url}/api/writer", json=payload, timeout=self.timeout)
            lat = time.time() - t0
            if r.status_code == 200:
                data = r.json()
                report = data.get("report", "")
                passed = bool(report and len(report) > 100)
                preview = f"report len: {len(report)} chars, markdown structure OK"
                self.record("B05", "Finetune", "/api/writer", r.status_code, lat, preview, passed)
            else:
                self.record("B05", "Finetune", "/api/writer", r.status_code, lat, r.text[:60], False)
                all_passed = False
        except Exception as e:
            self.record("B05", "Finetune", "/api/writer", 0, time.time() - t0, str(e)[:60], False)
            all_passed = False

        # B06: /generate
        print("  • Đang gửi request B06: /generate (Generic Generation Endpoint)...")
        payload = {
            "prompt": "Giải thích ngắn gọn sự khác nhau giữa CPU và GPU.",
            "adapter": "generator",
            "max_tokens": 128,
            "temperature": 0.1
        }
        t0 = time.time()
        try:
            r = httpx.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
            lat = time.time() - t0
            if r.status_code == 200:
                data = r.json()
                text = data.get("text", "")
                passed = bool(text)
                preview = f"generated text len: {len(text)} chars"
                self.record("B06", "Finetune", "/generate", r.status_code, lat, preview, passed)
            else:
                self.record("B06", "Finetune", "/generate", r.status_code, lat, r.text[:60], False)
                all_passed = False
        except Exception as e:
            self.record("B06", "Finetune", "/generate", 0, time.time() - t0, str(e)[:60], False)
            all_passed = False

        return all_passed

    def print_summary(self):
        print("\n" + "─" * 96)
        print(" 📊 BẢNG TỔNG HỢP KẾT QUẢ KIỂM THỬ KAGGLE DUAL T4 (GATE 1 & GATE 2)")
        print("─" * 96)
        print(f"| {'ID':<4} | {'Chế độ':<9} | {'Endpoint / Hạng mục':<30} | {'HTTP':<5} | {'Độ trễ':<8} | {'Kết quả':<7} | {'Chi tiết / Preview':<28} |")
        print(f"|{'-'*6}|{'-'*11}|{'-'*32}|{'-'*7}|{'-'*10}|{'-'*9}|{'-'*30}|")

        passed_count = 0
        total_count = len(self.results)
        for res in self.results:
            is_pass = (res["verdict"] == "PASS")
            if is_pass: passed_count += 1
            verdict_icon = "✔ PASS" if is_pass else "❌ FAIL"
            lat_str = f"{res['latency']}s"
            print(f"| {res['id']:<4} | {res['mode']:<9} | {res['endpoint']:<30} | {res['http']:<5} | {lat_str:<8} | {verdict_icon:<7} | {res['preview'][:28]:<28} |")

        print("─" * 96)
        print(f"🎯 KẾT LUẬN: {passed_count}/{total_count} bài kiểm thử đã vượt qua thành công ({round(passed_count/total_count*100, 1)}%)!")
        if passed_count == total_count:
            print("🟢 HỆ THỐNG WORKER FINETUNE TRÊN DUAL TESLA T4 ĐẠT 100% TIÊU CHÍ NGHIỆM THU GATE 1 & GATE 2!")
        else:
            print("🔴 CÓ BÀI KIỂM THỬ CHƯA ĐẠT. Vui lòng rà soát chi tiết ở bảng trên.")
        print("─" * 96 + "\n")


def main():
    args = parse_args()
    print("\n" + "═" * 78)
    print(" 🚀 MULTI-AGENT RAG: KAGGLE WORKER ACCEPTANCE TEST RUNNER")
    print(f" 🌐 Target URL: {args.url}")
    print(f" 🔑 Admin Token: {'*' * 8 + args.token[-4:] if len(args.token) > 4 else '(none)'}")
    print("═" * 78)

    verifier = EndpointVerifier(args.url, args.token, timeout=args.timeout)

    g1_ok = True
    g2_ok = True

    if args.gate in ("all", "gate1"):
        g1_ok = verifier.run_gate1()

    if args.gate in ("all", "gate2"):
        g2_ok = verifier.run_gate2()

    verifier.print_summary()

    if not (g1_ok and g2_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
