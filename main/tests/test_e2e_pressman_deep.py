import sys
import time
import os
import json

sys.path.insert(0, os.path.abspath("main"))
from core.config import settings
from core.graph import build_research_rag_graph

pdf_path = os.path.abspath("uploads/softwareengineering-apractitionersapproach-rogerspressman-110630132301-phpapp01.pdf")

print("=" * 80)
print("DEEP ACCURACY BENCHMARK ON 888-PAGE PRESSMAN TEXTBOOK")
print(f"Target PDF: {pdf_path}")
print("=" * 80)

# Configure mode to gemini to test full LLM reasoning + fact-checking
settings.EXECUTION_MODE = "gemini"
for a in ["retrieval", "reader", "summarizer", "factchecker", "writer"]:
    settings.set_agent_mode(a, "gemini")

app_graph = build_research_rag_graph("gemini")

test_cases = [
    {
        "id": "TC1_SPIRAL_MODEL",
        "question": "Mô hình xoắn ốc (Spiral Model) được đề xuất bởi ai, và kết hợp những đặc tính nào giữa prototyping và linear sequential model?",
        "expected_author": "Boehm",
        "expected_page": 64,
    },
    {
        "id": "TC2_CYCLOMATIC_COMPLEXITY",
        "question": "Độ phức tạp Cyclomatic (Cyclomatic Complexity) do ai phát triển, dựa trên lý thuyết nào và 3 công thức tính toán là gì?",
        "expected_author": "McCabe",
        "expected_pages": [476, 557],
    },
    {
        "id": "TC3_WHITE_VS_BLACK_BOX",
        "question": "Sự khác biệt cốt lõi giữa kiểm thử hộp trắng (white-box testing) và kiểm thử hộp đen (black-box testing) theo Roger Pressman là gì?",
        "expected_keywords": ["white-box", "black-box"],
        "expected_page": 500,
    }
]

benchmark_results = []

for idx, tc in enumerate(test_cases, 1):
    q = tc["question"]
    print(f"\n[{idx}/3] TESTING QUERY: {q}")
    
    initial_state = {
        "query": q,
        "question": q,
        "pdf_path": pdf_path,
        "pdf_paths": [pdf_path],
        "retry_count": 0,
        "max_reflection_loops": 2,
        "chat_history": [],
    }
    
    t0 = time.time()
    try:
        final_state = app_graph.invoke(initial_state)
        elapsed = time.time() - t0
        
        report = final_state.get("final_report", "")
        verified_claims = final_state.get("verified_claims", [])
        evidence_spans = final_state.get("evidence_spans", [])
        fact_check_label = final_state.get("fact_check_label", "unknown")
        retrieved_docs = final_state.get("retrieved_docs", [])
        
        pages_found = sorted(list({d.get("page_no") for d in retrieved_docs if d.get("page_no")}))
        
        print(f"-> Time: {elapsed:.2f}s | Fact-Check Label: {fact_check_label}")
        print(f"-> Pages retrieved: {pages_found}")
        print(f"-> Verified Claims count: {len(verified_claims)}")
        print(f"-> Evidence Spans count: {len(evidence_spans)}")
        print(f"-> FINAL REPORT SNIPPET:\n{report[:400]}...\n")
        
        benchmark_results.append({
            "test_id": tc["id"],
            "question": q,
            "elapsed_seconds": round(elapsed, 2),
            "fact_check_label": fact_check_label,
            "pages_retrieved": pages_found,
            "report": report,
            "verified_claims_count": len(verified_claims),
            "status": "PASS" if (len(report) > 100 and "không được đề cập" not in report) else "FAIL"
        })
    except Exception as e:
        print(f"-> ERROR: {e}")
        benchmark_results.append({
            "test_id": tc["id"],
            "question": q,
            "error": str(e),
            "status": "ERROR"
        })

print("\n" + "=" * 80)
print("SUMMARY OF BENCHMARK RESULTS:")
for r in benchmark_results:
    print(f"- [{r['status']}] {r['test_id']}: {r.get('elapsed_seconds', 'N/A')}s | Pages: {r.get('pages_retrieved', [])}")
print("=" * 80)

# Save result to artifact/json
with open("main/data/pressman_888p_benchmark.json", "w", encoding="utf-8") as f:
    json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
