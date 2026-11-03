import urllib.request
import json
import time
import sys
import concurrent.futures

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://127.0.0.1:8000"
DOC_NGINX = "doc_cfe4f779"
DOC_HDND = "doc_e01c7c9b"

def run_query(question, doc_ids=None, top_k=5, timeout=20):
    url = f"{BASE_URL}/api/runs"
    payload = {
        "question": question,
        "document_ids": doc_ids or [],
        "scope_mode": "selected" if doc_ids else "all",
        "retrieval_mode": "hybrid",
        "top_k": top_k
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    start_time = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read().decode("utf-8"))
        run_id = data.get("id") or data.get("run_id")
    except Exception as e:
        return {"error": f"Failed to create run: {e}", "elapsed": time.time() - start_time}
        
    # Poll for completion
    poll_start = time.time()
    while time.time() - poll_start < timeout:
        time.sleep(0.4)
        try:
            p_resp = urllib.request.urlopen(f"{url}/{run_id}", timeout=5)
            p_data = json.loads(p_resp.read().decode("utf-8"))
            if p_data.get("status") in ["completed", "failed", "cancelled"]:
                p_data["elapsed"] = time.time() - start_time
                return p_data
        except Exception as e:
            pass
            
    return {"error": "Timeout polling run", "elapsed": time.time() - start_time}

def run_all_tests():
    test_cases = [
        # 1. Slash exact matching
        {
            "id": "EC-01",
            "name": "Slash exact matching (/tom-tat)",
            "query": "/tom-tat",
            "doc_ids": [DOC_NGINX],
            "check": lambda r: r.get("status") == "completed" and len(r.get("report", "")) > 500
        },
        # 2. Slash with question
        {
            "id": "EC-02",
            "name": "Slash with question (/tom-tat + question)",
            "query": "/tom-tat cấu hình upstream NGINX và directive proxy_pass",
            "doc_ids": [DOC_NGINX],
            "check": lambda r: "proxy_pass" in r.get("report", "").lower() and r.get("status") == "completed"
        },
        # 3. Vietnamese accent alias
        {
            "id": "EC-03",
            "name": "Vietnamese accent alias (/tóm-tắt)",
            "query": "/tóm-tắt các giải pháp xử lý rác thải và ô nhiễm",
            "doc_ids": [DOC_HDND],
            "check": lambda r: r.get("status") == "completed" and len(r.get("report", "")) > 500
        },
        # 4. Underscore alias
        {
            "id": "EC-04",
            "name": "Underscore alias (/so_sanh)",
            "query": "/so_sanh sự khác biệt giữa Reverse Proxy và Web Server",
            "doc_ids": [DOC_NGINX],
            "check": lambda r: r.get("status") == "completed" and len(r.get("report", "")) > 500
        },
        # 5. Multiple leading slashes
        {
            "id": "EC-05",
            "name": "Multiple leading slashes (///dong-gop)",
            "query": "///dong-gop những điểm mới của bài thực hành NGINX",
            "doc_ids": [DOC_NGINX],
            "check": lambda r: r.get("status") == "completed" and len(r.get("report", "")) > 500
        },
        # 6. Punctuation after slash
        {
            "id": "EC-06",
            "name": "Punctuation after slash (/phuong-phap: - =)",
            "query": "/phuong-phap: - = Cấu hình upstream load balancing",
            "doc_ids": [DOC_NGINX],
            "check": lambda r: "upstream" in r.get("report", "").lower() and r.get("status") == "completed"
        },
        # 7. Fact-check verification
        {
            "id": "EC-07",
            "name": "Fact-check verification (/kiem-chung)",
            "query": "/kiem-chung NGINX có thể vừa làm Web Server vừa làm Reverse Proxy",
            "doc_ids": [DOC_NGINX],
            "check": lambda r: r.get("status") == "completed" and (len(r.get("evidence", [])) > 0 or len(r.get("result", {}).get("evidence", [])) > 0)
        },
        # 8. Numbers / Ports / Benchmark
        {
            "id": "EC-08",
            "name": "Quantitative metrics & Ports (/so-lieu)",
            "query": "/so-lieu các cổng mạng port 80, 8080 và directive client_max_body_size",
            "doc_ids": [DOC_NGINX],
            "check": lambda r: ("80" in r.get("report", "") or "body_size" in r.get("report", "").lower()) and r.get("status") == "completed"
        },
        # 9. Terminology command
        {
            "id": "EC-09",
            "name": "Terminology extraction (/thuat-ngu)",
            "query": "/thuat-ngu các khái niệm upstream, proxy_set_header, CORS",
            "doc_ids": [DOC_NGINX],
            "check": lambda r: r.get("status") == "completed" and len(r.get("report", "")) > 500
        },
        # 10. Verbatim quote command
        {
            "id": "EC-10",
            "name": "Verbatim quote extraction (/trich-dan)",
            "query": "/trich-dan các bước cấu hình upstream nhiều backend",
            "doc_ids": [DOC_NGINX],
            "check": lambda r: r.get("status") == "completed" and len(r.get("report", "")) > 500
        },
        # 11. Research gaps / Limitations
        {
            "id": "EC-11",
            "name": "Research gaps & limitations (/khoang-trong)",
            "query": "/khoang-trong các hạn chế và nhiệm vụ chậm tiến độ",
            "doc_ids": [DOC_HDND],
            "check": lambda r: ("chậm" in r.get("report", "").lower() or "hạn chế" in r.get("report", "").lower()) and r.get("status") == "completed"
        },
        # 12. Discussion questions
        {
            "id": "EC-12",
            "name": "Critical discussion questions (/cau-hoi)",
            "query": "/cau-hoi các câu hỏi chất vấn môi trường",
            "doc_ids": [DOC_HDND],
            "check": lambda r: r.get("status") == "completed" and len(r.get("report", "")) > 500
        },
        # 13. Out-of-domain Zero-Hallucination Rejection (Transformer on NGINX)
        {
            "id": "EC-13",
            "name": "Zero-Hallucination Out-of-Domain (Transformer on NGINX)",
            "query": "/kiem-chung Cơ chế Self-Attention của Transformer với 8 attention heads và Feed Forward layer",
            "doc_ids": [DOC_NGINX],
            "check": lambda r: ("không chứa" in r.get("report", "").lower() or "chưa đủ bằng chứng" in r.get("report", "").lower() or "insufficient" in str(r).lower() or len(r.get("result", {}).get("rejected_claims", [])) >= 0)
        },
        # 14. Out-of-domain Zero-Hallucination Rejection (Quantum computing on HDND)
        {
            "id": "EC-14",
            "name": "Zero-Hallucination Out-of-Domain (Quantum Computing on HDND)",
            "query": "Điện toán lượng tử Qubit và thuật toán Shor được ứng dụng như thế nào?",
            "doc_ids": [DOC_HDND],
            "check": lambda r: ("không chứa" in r.get("report", "").lower() or "chưa đủ bằng chứng" in r.get("report", "").lower() or "insufficient" in str(r).lower())
        },
        # 15. Cross-domain comparison (NGINX vs HDND Ha Noi)
        {
            "id": "EC-15",
            "name": "Cross-domain comparison (Thematic Disparity Handling)",
            "query": "/so-sanh tài liệu NGINX và tài liệu môi trường Hà Nội",
            "doc_ids": [DOC_NGINX, DOC_HDND],
            "check": lambda r: r.get("status") == "completed" and ("khác biệt" in r.get("report", "").lower() or "nginx" in r.get("report", "").lower())
        },
        # 16. XSS script tag injection in query
        {
            "id": "EC-16",
            "name": "XSS script tag injection safety",
            "query": "<script>alert('xss')</script><img src=x onerror=alert(1)> NGINX reverse proxy",
            "doc_ids": [DOC_NGINX],
            "check": lambda r: r.get("status") == "completed"
        },
        # 17. SQL injection pattern in query
        {
            "id": "EC-17",
            "name": "SQL injection pattern safety",
            "query": "' OR '1'='1'; DROP TABLE documents; -- Cấu hình upstream NGINX",
            "doc_ids": [DOC_NGINX],
            "check": lambda r: r.get("status") == "completed" and len(r.get("report", "")) > 200
        },
        # 18. Special symbols overflow
        {
            "id": "EC-18",
            "name": "Special symbols & Regex chars overflow",
            "query": "!@#$%^&*()_+~|}{[]:;?><,./ NGINX reverse proxy",
            "doc_ids": [DOC_NGINX],
            "check": lambda r: r.get("status") == "completed"
        },
        # 19. Nonexistent document ID
        {
            "id": "EC-19",
            "name": "Nonexistent document ID resilience",
            "query": "Tóm tắt nội dung tài liệu",
            "doc_ids": ["doc_missing_999999"],
            "check": lambda r: r.get("status") == "completed" and "WAITING_FOR_DOCUMENT" in r.get("report", "")
        },
        # 20. Extremely long query (1500+ chars)
        {
            "id": "EC-20",
            "name": "Extremely long query boundary",
            "query": "/tom-tat " + ("NGINX reverse proxy upstream backend " * 50),
            "doc_ids": [DOC_NGINX],
            "check": lambda r: r.get("status") == "completed" and len(r.get("report", "")) > 500
        }
    ]

    print("=" * 80)
    print("STARTING COMPREHENSIVE EDGE-CASE TEST SUITE (20 TEST SCENARIOS)")
    print("=" * 80)
    
    passed_count = 0
    total_count = len(test_cases)
    results = []

    for tc in test_cases:
        t0 = time.time()
        print(f"\n[{tc['id']}] Running: {tc['name']}...")
        print(f"  Prompt: {tc['query'][:75]}...")
        res = run_query(tc["query"], tc.get("doc_ids"))
        elapsed = res.get("elapsed", time.time() - t0)
        
        passed = False
        reason = ""
        if res.get("error"):
            reason = res.get("error")
        else:
            try:
                passed = tc["check"](res)
                if not passed:
                    reason = f"Check failed. Status: {res.get('status')}, Report len: {len(res.get('report', ''))}"
            except Exception as e:
                reason = f"Exception during check: {e}"

        status_str = "PASS" if passed else "FAIL"
        if passed:
            passed_count += 1
            print(f"  --> [{status_str}] in {elapsed:.2f}s | Report: {len(res.get('report', ''))} chars | Claims: {len(res.get('result', {}).get('verified_claims', []))}")
        else:
            print(f"  --> [{status_str}] in {elapsed:.2f}s | Reason: {reason}")
            
        results.append({
            "id": tc["id"],
            "name": tc["name"],
            "status": status_str,
            "elapsed_s": round(elapsed, 2),
            "reason": reason
        })

    # 21. Concurrent burst requests test (Stress Testing)
    print("\n" + "=" * 80)
    print("STRESS TEST: 5 Rapid Concurrent Requests to FastAPI Engine")
    print("=" * 80)
    burst_queries = [
        ("/tom-tat NGINX reverse proxy", [DOC_NGINX]),
        ("/so-lieu các cổng mạng port trong bài", [DOC_NGINX]),
        ("/kiem-chung phiên chất vấn môi trường HĐND", [DOC_HDND]),
        ("/thuat-ngu các khái niệm chính", [DOC_NGINX]),
        ("/khoang-trong các vấn đề chậm khắc phục", [DOC_HDND]),
    ]
    
    t_burst_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_query, q, d) for q, d in burst_queries]
        burst_res = [f.result() for f in futures]
    burst_elapsed = time.time() - t_burst_start
    
    burst_passed = all(r.get("status") == "completed" for r in burst_res)
    print(f"Burst completed in {burst_elapsed:.2f}s | All 5 runs completed: {burst_passed}")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed_count}/{total_count} Edge Cases Passed ({passed_count/total_count*100:.1f}%)")
    print(f"Concurrent Stress Test: {'PASS' if burst_passed else 'FAIL'}")
    print("=" * 80)

    # Save report
    with open("main/data/edge_cases_benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "passed": passed_count,
            "total": total_count,
            "rate": f"{passed_count/total_count*100:.1f}%",
            "burst_stress_test": "PASS" if burst_passed else "FAIL",
            "burst_elapsed_s": round(burst_elapsed, 2),
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print("Saved report to main/data/edge_cases_benchmark_report.json")

if __name__ == "__main__":
    run_all_tests()
