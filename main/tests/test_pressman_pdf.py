import sys
import time
import os

sys.path.insert(0, os.path.abspath("main"))
from core.tools import load_or_parse_selected_pdf
from agents.agent1_retrieval.tools.fts5_search import SQLiteFTS5Searcher
from agents.agent1_retrieval.tools.bm25 import SimpleBM25
from utils.bilingual import expand_bilingual_tokens, BILINGUAL_PHRASE_MAP

pdf_path = "uploads/softwareengineering-apractitionersapproach-rogerspressman-110630132301-phpapp01.pdf"

print("1. Testing PDF Parsing on 888 pages...")
t0 = time.time()
chunks = load_or_parse_selected_pdf(pdf_path)
t1 = time.time()
print(f"-> Parsed 888 pages into {len(chunks)} chunks in {t1 - t0:.2f}s")

print("\n2. Testing C-Engine Indexing (SQLite FTS5 + BM25)...")
fts = SQLiteFTS5Searcher()
bm25 = SimpleBM25()
t0 = time.time()
fts.fit(chunks)
bm25.fit(chunks)
t1 = time.time()
print(f"-> Indexed {len(chunks)} chunks in {t1 - t0:.3f}s")

test_cases = [
    {
        "query": "Mô hình thác nước waterfall model gồm những bước nào?",
        "keywords": ["waterfall", "model"]
    },
    {
        "query": "Mô hình xoắn ốc spiral model của ai và có đặc điểm gì?",
        "keywords": ["spiral", "model", "boehm"]
    },
    {
        "query": "Kiểm thử hộp đen black-box testing và kiểm thử hộp trắng white-box testing",
        "keywords": ["black-box", "white-box", "testing"]
    },
    {
        "query": "Độ phức tạp Cyclomatic complexity của Thomas McCabe",
        "keywords": ["cyclomatic", "complexity", "mccabe"]
    }
]

print("\n3. Testing Cross-Lingual Vietnamese Query Retrieval on English Textbook:")
for tc in test_cases:
    q = tc["query"]
    print("=" * 70)
    print(f"Query (Vietnamese): {q}")
    
    # Expand tokens
    base_tokens = set(q.lower().split())
    expanded = expand_bilingual_tokens(q, base_tokens)
    search_q = " ".join(list(expanded)[:12])
    
    t_search0 = time.time()
    results = fts.search(search_q, top_k=3)
    if not results:
        results = bm25.search(search_q, top_k=3)
    t_search1 = time.time()
    
    print(f"Search time: {(t_search1 - t_search0)*1000:.2f} ms")
    if results:
        top = results[0]
        print(f"-> Top Match: Page {top.get('page_no')} (ID: {top.get('id')})")
        print(f"-> Text preview:\n{top.get('text')[:350]}...\n")
    else:
        print("-> No match found.")
