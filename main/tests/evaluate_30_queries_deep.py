import sys, os, glob, re
sys.path.insert(0, 'main')
sys.stdout.reconfigure(encoding='utf-8')

from core.graph import build_research_rag_graph

pdf_path = glob.glob('main/uploads/*NGINX*.pdf')[0]
print(f"Target Document: {os.path.basename(pdf_path)}\n")

test_cases = [
    # 15 IN-DOMAIN QUESTIONS
    {
        "id": 1,
        "query": "Chuẩn bị thuyết trình và demo cần gì",
        "type": "in_domain",
        "expected_keywords": ["thuyết trình", "demo", "slide"],
    },
    {
        "id": 2,
        "query": "Upstream là gì",
        "type": "in_domain",
        "expected_keywords": ["upstream", "backend"],
    },
    {
        "id": 3,
        "query": "Reverse proxy là gì",
        "type": "in_domain",
        "expected_keywords": ["reverse proxy", "proxy"],
    },
    {
        "id": 4,
        "query": "Tắt tính năng tự sinh file hosts trong WSL bằng cách nào",
        "type": "in_domain",
        "expected_keywords": ["generatehosts", "wsl.conf"],
    },
    {
        "id": 5,
        "query": "Cấu hình generateHosts = false ở đâu",
        "type": "in_domain",
        "expected_keywords": ["wsl.conf", "network"],
    },
    {
        "id": 6,
        "query": "Lệnh openssl tạo chứng chỉ ssl tự ký là gì",
        "type": "in_domain",
        "expected_keywords": ["openssl req", "myapp.key"],
    },
    {
        "id": 7,
        "query": "Proxy_pass dùng để làm gì",
        "type": "in_domain",
        "expected_keywords": ["proxy_pass", "chuyển tiếp"],
    },
    {
        "id": 8,
        "query": "Tại sao không sửa file etc hosts trong WSL ở trường hợp 2",
        "type": "in_domain",
        "expected_keywords": ["windows", "trình duyệt"],
    },
    {
        "id": 9,
        "query": "Các thuật toán cân bằng tải của NGINX là gì",
        "type": "in_domain",
        "expected_keywords": ["round-robin", "least_conn"],
    },
    {
        "id": 10,
        "query": "Load balancing nhiều backend để giải quyết vấn đề gì",
        "type": "in_domain",
        "expected_keywords": ["chịu tải", "backend"],
    },
    {
        "id": 11,
        "query": "Task người thực hiện ước lượng point trong buổi 8",
        "type": "in_domain",
        "expected_keywords": ["task", "point", "người thực hiện"],
    },
    {
        "id": 12,
        "query": "Nghiên cứu cấu hình thực tế container FE và BE cần chú ý gì",
        "type": "in_domain",
        "expected_keywords": ["client_max_body_size", "container"],
    },
    {
        "id": 13,
        "query": "/tom-tat",
        "type": "in_domain",
        "expected_keywords": ["nginx", "reverse proxy"],
    },
    {
        "id": 14,
        "query": "Tóm tắt nội dung tài liệu",
        "type": "in_domain",
        "expected_keywords": ["nginx", "reverse proxy"],
    },
    {
        "id": 15,
        "query": "Client_max_body_size có tác dụng gì",
        "type": "in_domain",
        "expected_keywords": ["body", "1mb", "413"],
    },

    # 15 OUT-OF-DOMAIN QUESTIONS
    {
        "id": 16,
        "query": "bác hồ sinh năm bao nhiêu",
        "type": "out_of_domain",
        "expected_keywords": ["hoàn toàn không được đề cập"],
    },
    {
        "id": 17,
        "query": "Bảo tàng Dân tộc học Việt Nam được thành lập khi nào ?",
        "type": "out_of_domain",
        "expected_keywords": ["hoàn toàn không được đề cập"],
    },
    {
        "id": 18,
        "query": "Thủ đô của nước Pháp là gì",
        "type": "out_of_domain",
        "expected_keywords": ["hoàn toàn không được đề cập"],
    },
    {
        "id": 19,
        "query": "Định luật bảo toàn năng lượng là gì",
        "type": "out_of_domain",
        "expected_keywords": ["hoàn toàn không được đề cập"],
    },
    {
        "id": 20,
        "query": "Kafka broker hoạt động như thế nào",
        "type": "out_of_domain",
        "expected_keywords": ["hoàn toàn không được đề cập"],
    },
    {
        "id": 21,
        "query": "Kubernetes pods giao tiếp với nhau qua đâu",
        "type": "out_of_domain",
        "expected_keywords": ["hoàn toàn không được đề cập"],
    },
    {
        "id": 22,
        "query": "Cách nấu phở bò Hà Nội",
        "type": "out_of_domain",
        "expected_keywords": ["hoàn toàn không được đề cập"],
    },
    {
        "id": 23,
        "query": "Ai là tác giả của Truyện Kiều",
        "type": "out_of_domain",
        "expected_keywords": ["hoàn toàn không được đề cập"],
    },
    {
        "id": 24,
        "query": "Dân số thế giới hiện nay là bao nhiêu",
        "type": "out_of_domain",
        "expected_keywords": ["hoàn toàn không được đề cập"],
    },
    {
        "id": 25,
        "query": "Mô hình mạng nơ-ron Transformer ra đời năm nào",
        "type": "out_of_domain",
        "expected_keywords": ["hoàn toàn không được đề cập"],
    },
    {
        "id": 26,
        "query": "Triệu chứng của bệnh tiểu đường là gì",
        "type": "out_of_domain",
        "expected_keywords": ["hoàn toàn không được đề cập"],
    },
    {
        "id": 27,
        "query": "Hồ Gươm nằm ở quận nào của Hà Nội",
        "type": "out_of_domain",
        "expected_keywords": ["hoàn toàn không được đề cập"],
    },
    {
        "id": 28,
        "query": "Chiến thắng Điện Biên Phủ diễn ra vào năm nào",
        "type": "out_of_domain",
        "expected_keywords": ["hoàn toàn không được đề cập"],
    },
    {
        "id": 29,
        "query": "Nguyên lý hoạt động của động cơ đốt trong",
        "type": "out_of_domain",
        "expected_keywords": ["hoàn toàn không được đề cập"],
    },
    {
        "id": 30,
        "query": "Lập trình hướng đối tượng OOP gồm những tính chất nào",
        "type": "out_of_domain",
        "expected_keywords": ["hoàn toàn không được đề cập"],
    },
]

def evaluate_mode(mode_name: str):
    print("=" * 80)
    print(f"DEEP EVALUATION OF MODE: {mode_name.upper()}")
    print("=" * 80)
    
    graph = build_research_rag_graph(mode_name)
    passed_count = 0
    failures = []
    
    for tc in test_cases:
        tc_id = tc["id"]
        q = tc["query"]
        q_type = tc["type"]
        exp_kws = tc["expected_keywords"]
        
        try:
            res = graph.invoke({
                "query": q,
                "pdf_path": pdf_path,
                "retry_count": 0,
                "max_reflection_loops": 1
            })
            
            final_report = res.get("final_report") or res.get("answer") or ""
            verified_claims = res.get("verified_claims") or []
            evidence_spans = res.get("evidence_spans") or []
            
            report_l = final_report.lower()
            
            if q_type == "out_of_domain":
                # Must be honest refusal
                is_honest_refusal = "hoàn toàn không được đề cập" in report_l or "không được đề cập" in report_l
                has_zero_evidence = len(evidence_spans) == 0
                has_no_hallucination = "upstream và nhiều backend" not in report_l and "tự sinh này bằng cách" not in report_l
                
                success = is_honest_refusal and has_zero_evidence and has_no_hallucination
                status = "PASS" if success else "FAIL"
                
                if success:
                    passed_count += 1
                else:
                    failures.append((tc_id, q, "OOD check failed or hallucinated technical excerpt", final_report[:200]))
                
                print(f"[{status}] Q{tc_id:02d} (OOD): {q[:45]:<45} | Refusal={is_honest_refusal} | EvCount={len(evidence_spans)}")
                
            else: # in_domain
                is_honest_refusal = "hoàn toàn không được đề cập" in report_l
                has_stub = "chưa thể tổng hợp" in report_l or "chế độ dự phòng" in report_l
                matched_kws = [kw for kw in exp_kws if kw.lower() in report_l]
                
                success = (not is_honest_refusal) and (not has_stub) and (len(matched_kws) > 0)
                status = "PASS" if success else "FAIL"
                
                if success:
                    passed_count += 1
                else:
                    failures.append((tc_id, q, f"In-domain query missed expected keywords {exp_kws}", final_report[:200]))
                    
                print(f"[{status}] Q{tc_id:02d} (IN ): {q[:45]:<45} | Found={matched_kws} | Len={len(final_report)}")
                
        except Exception as e:
            failures.append((tc_id, q, f"Exception: {e}", ""))
            print(f"[FAIL] Q{tc_id:02d}: {q} -> Error: {e}")
            
    print(f"\n--- MODE {mode_name.upper()} RESULT: {passed_count}/{len(test_cases)} PASSED ({passed_count/len(test_cases)*100:.1f}%) ---\n")
    if failures:
        print("Failures:")
        for fid, fq, freason, fpreview in failures:
            print(f"  - Q{fid}: '{fq}' -> {freason}")
            if fpreview:
                print(f"    Preview: {fpreview}")
    return passed_count == len(test_cases)

if __name__ == "__main__":
    modes_to_test = ["cpu_min", "cpu_max", "kaggle_fn", "kaggle_max"]
    all_success = True
    for m in modes_to_test:
        ok = evaluate_mode(m)
        if not ok:
            all_success = False
            
    print("=" * 80)
    if all_success:
        print("ALL 4 MODES PASSED 100% OF 30 TEST CASES PERFECTLY!")
    else:
        print("SOME MODES HAVE FAILURES TO FIX!")
    print("=" * 80)
