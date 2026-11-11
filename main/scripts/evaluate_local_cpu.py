import sys
sys.stdout.reconfigure(encoding="utf-8")

import datetime
import json
import os
import time
import urllib.request

BASE_API_URL = "http://127.0.0.1:8000"
CPU_SERVER_URL = "http://127.0.0.1:11434"

TEST_CASES = [
    {
        "id": "TC-01",
        "name": "Tra cứu sự kiện mốc thời gian cụ thể (Factual Grounding)",
        "query": "Báo Hà Nội mới ra số đầu tiên vào thời gian nào?",
        "expected_keywords": ["1957", "24/10", "Hà Nội", "đầu tiên"],
        "expect_grounded": True,
        "category": "Direct Fact Retrieval"
    },
    {
        "id": "TC-02",
        "name": "Phân tích vai trò và tôn chỉ chính trị (Thematic Synthesis)",
        "query": "Vai trò của báo Hà Nội mới đối với phong trào thi đua và nhân dân Thủ đô là gì?",
        "expected_keywords": ["cơ quan ngôn luận", "Thủ đô", "nhân dân", "tiếng nói"],
        "expect_grounded": True,
        "category": "Thematic Synthesis"
    },
    {
        "id": "TC-03",
        "name": "Phân tích bối cảnh thành lập sau giải phóng (Historical Context)",
        "query": "Bối cảnh ra đời của báo gắn liền với sự kiện giải phóng Thủ đô 1954 như thế nào?",
        "expected_keywords": ["1954", "giải phóng", "Thủ đô", "kiến thiết"],
        "expect_grounded": True,
        "category": "Contextual Analysis"
    },
    {
        "id": "TC-04",
        "name": "Chống ảo giác đối với thông tin ngoài tài liệu (Anti-Hallucination Guard)",
        "query": "Giá cổ phiếu VinFast trên sàn Nasdaq hôm nay là bao nhiêu?",
        "expected_keywords": ["không", "chưa", "tài liệu"],
        "expect_grounded": False,
        "category": "Hallucination Defense"
    },
    {
        "id": "TC-05",
        "name": "Độ trễ và đối soát trích dẫn nguyên văn (Exact Quote Verification)",
        "query": "Hãy trích dẫn nguyên văn câu văn nói về ý nghĩa của số báo đầu tiên?",
        "expected_keywords": ["báo", "số đầu tiên"],
        "expect_grounded": True,
        "category": "Quote Verification"
    }
]


def test_connectivity():
    print("=" * 70)
    print("KIEM TRA KET NOI LOCAL CPU ENGINE & API GATEWAY")
    print("=" * 70)

    # 1. Local CPU Server
    try:
        with urllib.request.urlopen(f"{CPU_SERVER_URL}/api/tags", timeout=3) as res:
            data = json.loads(res.read())
            models = [m.get("name") for m in data.get("models", [])]
            print(f"[OK] Local CPU Engine (11434): ONLINE (Model: {', '.join(models)})")
    except Exception as e:
        print(f"[FAIL] Local CPU Engine (11434) LOI: {e}")
        return False

    # 2. Main API Server
    try:
        with urllib.request.urlopen(f"{BASE_API_URL}/api/config", timeout=3) as res:
            data = json.loads(res.read())
            print(f"[OK] Main Multi-Agent API (8000): ONLINE (Modes: {data.get('agent_modes')})")
    except Exception as e:
        print(f"[FAIL] Main API Server (8000) LOI: {e}")
        return False

    return True


def run_test_case(tc):
    print(f"\n--- Đang thực thi {tc['id']}: {tc['name']} ---")
    print(f"Câu hỏi: \"{tc['query']}\"")
    
    start_time = time.perf_counter()
    payload = {
        "query": tc["query"],
        "document_ids": ["doc_166bea0a"],
        "max_iterations": 1
    }
    
    req = urllib.request.Request(
        f"{BASE_API_URL}/api/runs",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    with urllib.request.urlopen(req) as res:
        run_data = json.loads(res.read())
        run_id = run_data["id"]

    # Poll until completed
    max_wait = 25
    elapsed = 0
    final_result = None
    
    while elapsed < max_wait:
        time.sleep(1)
        elapsed += 1
        with urllib.request.urlopen(f"{BASE_API_URL}/api/runs/{run_id}") as r_res:
            r_data = json.loads(r_res.read())
            if r_data.get("status") in ("completed", "failed"):
                final_result = r_data
                break

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    
    if not final_result or final_result.get("status") != "completed":
        print(f"-> KẾT QUẢ: THẤT BẠI (Timeout hoặc Lỗi)")
        return {
            "id": tc["id"],
            "name": tc["name"],
            "status": "FAILED",
            "latency_ms": duration_ms,
            "passed": False,
            "evidence_count": 0,
            "details": "Timeout or failed execution"
        }

    report = final_result.get("report") or final_result.get("final_report") or ""
    evidence = final_result.get("evidence") or final_result.get("evidence_spans") or []
    claims = final_result.get("verified_claims") or []

    # Scoring criteria: Keyword match
    matched_kw = [kw for kw in tc["expected_keywords"] if kw.lower() in report.lower()]
    kw_coverage = len(matched_kw) / len(tc["expected_keywords"])

    is_passed = kw_coverage >= 0.5 and (len(evidence) > 0 if tc["expect_grounded"] else True)

    print(f"-> Thời gian phản hồi: {duration_ms} ms")
    print(f"-> Trích dẫn tìm thấy: {len(evidence)} nguồn")
    print(f"-> Mệnh đề đã đối soát: {len(claims)} mệnh đề")
    print(f"-> Độ khớp từ khóa: {int(kw_coverage * 100)}% ({matched_kw})")
    print(f"-> Trạng thái: {'[ĐẠT / PASS]' if is_passed else '[KHÔNG ĐẠT / FAIL]'}")

    return {
        "id": tc["id"],
        "name": tc["name"],
        "category": tc["category"],
        "latency_ms": duration_ms,
        "evidence_count": len(evidence),
        "claims_count": len(claims),
        "kw_coverage_pct": int(kw_coverage * 100),
        "passed": is_passed,
        "sample_output": report[:150] + "..." if len(report) > 150 else report
    }


def main():
    if not test_connectivity():
        sys.exit(1)

    results = []
    print("\n" + "=" * 70)
    print("BAT DAU DANH GIA 5 KICH BAN THU NGHIEM LOCAL CPU RAG")
    print("=" * 70)

    for tc in TEST_CASES:
        res = run_test_case(tc)
        results.append(res)

    print("\n" + "=" * 70)
    print("BANG TONG HOP DANH GIA HIEU NANG LOCAL CPU MULTI-AGENT")
    print("=" * 70)
    print(f"{'Mã TC':<7} | {'Kịch Bản':<26} | {'Độ Trễ':<9} | {'Trích Dẫn':<9} | {'Kết Quả'}")
    print("-" * 70)

    total_latency = 0
    passed_count = 0
    for r in results:
        status_str = "PASS [OK]" if r["passed"] else "FAIL [X]"
        print(f"{r['id']:<7} | {r['name'][:24]:<26} | {str(r['latency_ms']) + 'ms':<9} | {str(r['evidence_count']) + ' ev':<9} | {status_str}")
        total_latency += r["latency_ms"]
        if r["passed"]:
            passed_count += 1

    avg_latency = round(total_latency / len(results), 2)
    pass_rate = round((passed_count / len(results)) * 100, 1)

    print("-" * 70)
    print(f"TONG KET: Ty le dat {pass_rate}% ({passed_count}/{len(results)} test cases)")
    print(f"DO TRE TRUNG BINH TREN CPU: {avg_latency} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
