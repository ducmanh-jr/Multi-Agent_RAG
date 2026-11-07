import sys
import time
import os

sys.path.insert(0, os.path.abspath("main"))
from core.state import AgentState
from modes.hybrid.retrieval import HybridRetrievalAgent
from core.supervisor import supervisor_planner

pdf_path = "uploads/softwareengineering-apractitionersapproach-rogerspressman-110630132301-phpapp01.pdf"

agent = HybridRetrievalAgent()

test_queries = [
    "Mô hình thác nước waterfall model gồm những giai đoạn nào?",
    "Kiểm thử hộp đen black-box testing và kiểm thử hộp trắng white-box testing khác nhau thế nào?",
    "Độ phức tạp Cyclomatic complexity của Thomas McCabe",
    "Mô hình xoắn ốc spiral model Barry Boehm"
]

for q in test_queries:
    print("=" * 80)
    print(f"TESTING HYBRID RETRIEVAL AGENT FOR: {q}")
    state: AgentState = {
        "query": q,
        "question": q,
        "pdf_path": pdf_path,
        "pdf_paths": [pdf_path],
        "retry_count": 0,
        "seen_chunks": [],
        "evidence_gaps": [],
    }
    
    t0 = time.time()
    out = agent.run(state)
    t1 = time.time()
    
    docs = out.get("retrieved_docs") or []
    print(f"Time taken: {t1 - t0:.3f}s | Retrieved docs count: {len(docs)}")
    for i, d in enumerate(docs[:3]):
        print(f"  [{i+1}] Page {d.get('page_no')} (ID: {d.get('id')})")
        print(f"      Text: {d.get('text')[:180].replace(chr(10), ' ')}...")
