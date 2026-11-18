import sys
import time
import os
import json

sys.path.insert(0, os.path.abspath("main"))
from core.config import settings
from core.graph import build_research_rag_graph

pdf_path = os.path.abspath("uploads/softwareengineering-apractitionersapproach-rogerspressman-110630132301-phpapp01.pdf")

test_queries = [
    {
        "id": "Q1_SPIRAL_MODEL",
        "question": "Mô hình xoắn ốc (Spiral Model) được đề xuất bởi ai, và kết hợp những đặc tính nào giữa prototyping và linear sequential model?",
        "expected_page": 64
    },
    {
        "id": "Q2_CYCLOMATIC_COMPLEXITY",
        "question": "Độ phức tạp Cyclomatic (Cyclomatic Complexity) do ai phát triển, dựa trên lý thuyết nào và 3 công thức tính toán là gì?",
        "expected_page": 476
    },
    {
        "id": "Q3_WHITE_VS_BLACK_BOX",
        "question": "Sự khác biệt cốt lõi giữa kiểm thử hộp trắng (white-box testing) và kiểm thử hộp đen (black-box testing) theo Roger Pressman là gì?",
        "expected_page": 500
    }
]

modes_to_test = ["cpu_min", "cpu_max", "gemini"]
comparison_matrix = {}

print("=" * 85)
print("DEEP 3-MODE BENCHMARK: GEMINI vs CPU MIN vs CPU MAX")
print(f"Dataset: Pressman Software Engineering (888 Pages, 2,003 Chunks)")
print("=" * 85)

for mode in modes_to_test:
    print(f"\n==================== RUNNING MODE: {mode.upper()} ====================")
    settings.EXECUTION_MODE = mode
    for a in ["retrieval", "reader", "summarizer", "factchecker", "writer"]:
        settings.set_agent_mode(a, mode)
    
    app_graph = build_research_rag_graph(mode)
    comparison_matrix[mode] = []
    
    for q_idx, q_item in enumerate(test_queries, 1):
        q_text = q_item["question"]
        print(f"\n[{mode}] [{q_idx}/3] Query: {q_text[:75]}...")
        
        initial_state = {
            "query": q_text,
            "question": q_text,
            "pdf_path": pdf_path,
            "pdf_paths": [pdf_path],
            "retry_count": 0,
            "max_reflection_loops": 2,
            "chat_history": [],
        }
        
        t0 = time.time()
        try:
            res = app_graph.invoke(initial_state)
            elapsed = time.time() - t0
            
            report = res.get("final_report", "")
            verified_claims = res.get("verified_claims", [])
            retrieved_docs = res.get("retrieved_docs", [])
            fact_label = res.get("fact_check_label", "unknown")
            pages = sorted(list({d.get("page_no") for d in retrieved_docs if d.get("page_no")}))
            
            print(f"  -> Latency: {elapsed:.2f}s | Fact-Check: {fact_label} | Verified Claims: {len(verified_claims)} | Pages: {pages}")
            
            comparison_matrix[mode].append({
                "query_id": q_item["id"],
                "question": q_text,
                "latency_sec": round(elapsed, 2),
                "fact_check_label": fact_label,
                "verified_claims_count": len(verified_claims),
                "pages_retrieved": pages,
                "report_snippet": report[:300].strip(),
                "status": "PASS" if (len(report) > 50 and "hoàn toàn không được đề cập" not in report) else "FAIL"
            })
        except Exception as e:
            print(f"  -> ERROR in {mode}: {e}")
            comparison_matrix[mode].append({
                "query_id": q_item["id"],
                "question": q_text,
                "error": str(e),
                "status": "ERROR"
            })

print("\n" + "=" * 85)
print("FINAL BENCHMARK COMPARISON MATRIX:")
print(f"{'Mode':<12} | {'Query ID':<25} | {'Latency':<8} | {'Fact-Check':<12} | {'Pages':<18} | {'Status'}")
print("-" * 85)
for m, runs in comparison_matrix.items():
    for r in runs:
        pages_str = str(r.get("pages_retrieved", []))[:16]
        print(f"{m:<12} | {r['query_id']:<25} | {str(r.get('latency_sec'))+'s':<8} | {str(r.get('fact_check_label')):<12} | {pages_str:<18} | {r['status']}")

with open("main/data/comparison_gemini_cpu_min_cpu_max.json", "w", encoding="utf-8") as f:
    json.dump(comparison_matrix, f, ensure_ascii=False, indent=2)

print("\nBenchmark saved to: main/data/comparison_gemini_cpu_min_cpu_max.json")
