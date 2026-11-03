import sys, glob
sys.path.insert(0, 'main')
from core.graph import build_research_rag_graph

graph = build_research_rag_graph(mode='cpu_min')

pdf_bao = 'bao/Báo Hà Nội mới ra số đầu tiên (1).pdf'
pdf_env = 'bao/Thường trực Hội đồng nhân dân thành phố Hà Nội tổ chức phiên chất vấn về lĩnh vực môi trường (1).pdf'
nginx_files = glob.glob('main/uploads/*cfe4f779*.pdf')
pdf_nginx = nginx_files[0] if nginx_files else pdf_bao

tests = [
    {
        'id': 'TC-01',
        'desc': 'Specific date question (Báo Thủ đô)',
        'query': 'Báo Thủ đô ra số đầu tiên vào ngày nào ?',
        'pdf_path': pdf_bao,
        'pdf_paths': [pdf_bao],
        'expected_kw': ['24-10-1957', 'Báo Thủ đô'],
    },
    {
        'id': 'TC-02',
        'desc': 'Specific year question (Báo Thủ đô ra đời năm nào)',
        'query': 'báo thủ đô ra đời vào năm nào ?',
        'pdf_path': pdf_bao,
        'pdf_paths': [pdf_bao],
        'expected_kw': ['1957', 'Báo Thủ đô'],
    },
    {
        'id': 'TC-03',
        'desc': 'Specific list question (5 nhóm vấn đề)',
        'query': '5 nhóm vấn đề môi trường là gì ?',
        'pdf_path': pdf_env,
        'pdf_paths': [pdf_env],
        'expected_kw': ['1.', '2.', '3.', '4.', '5.', 'chất thải rắn'],
    },
    {
        'id': 'TC-04',
        'desc': 'Full guide summary covering all pages (NGINX)',
        'query': 'tóm tắt tài liệu này',
        'pdf_path': pdf_nginx,
        'pdf_paths': [pdf_nginx],
        'expected_kw': ['1.', '2.', '3.', '4.', '5.', 'Trang 1', 'Trang 2', 'Trang 3', 'Trang 4', 'Trang 5'],
    },
    {
        'id': 'TC-05',
        'desc': 'Multi-document summary (Tóm tắt 2 bài báo này)',
        'query': 'Tóm tắt 2 bài báo này',
        'pdf_path': pdf_bao,
        'pdf_paths': [pdf_bao, pdf_env],
        'expected_kw': ['Bài báo 1', 'Bài báo 2'],
    },
    {
        'id': 'TC-06',
        'desc': 'Out-of-domain query (Cầu Tứ Liên)',
        'query': 'Cầu Tứ Liên khi nào khởi công ?',
        'pdf_path': pdf_bao,
        'pdf_paths': [pdf_bao],
        'expected_kw': ['hoàn toàn không được đề cập'],
    },
    {
        'id': 'TC-07',
        'desc': 'Mismatched doc (Báo Thủ đô on environmental PDF)',
        'query': 'Báo Thủ đô ra số đầu tiên vào ngày nào ?',
        'pdf_path': pdf_env,
        'pdf_paths': [pdf_env],
        'expected_kw': ['hoàn toàn không được đề cập'],
    },
    {
        'id': 'TC-08',
        'desc': 'Mismatched doc (5 nhóm môi trường on Báo Hà Nội mới PDF)',
        'query': '5 nhóm vấn đề môi trường là gì ?',
        'pdf_path': pdf_bao,
        'pdf_paths': [pdf_bao],
        'expected_kw': ['hoàn toàn không được đề cập'],
    },
]

passed = 0
for t in tests:
    print("\n" + "="*70)
    print(f"RUNNING {t['id']}: {t['desc']}")
    print(f"Query: '{t['query']}' | PDFs: {len(t['pdf_paths'])}")
    print("="*70)
    res = graph.invoke({
        'query': t['query'],
        'pdf_path': t['pdf_path'],
        'pdf_paths': t['pdf_paths'],
        'mode': 'cpu_min',
    })
    ans = res.get('answer', '')
    print("STATUS LABEL:", res.get('fact_check_label'))
    print("ANSWER PREVIEW:\n", ans[:400], "\n...")
    
    # Verify expected keywords
    ok = all(kw.lower() in ans.lower() for kw in t['expected_kw'])
    # Verify no raw <br> in answer
    no_br = '<br>' not in ans and '<br/>' not in ans
    
    if ok and no_br:
        print(f">>> RESULT: PASSED! (All keywords present, zero <br>)")
        passed += 1
    else:
        print(f">>> RESULT: FAILED! (Missing: {[kw for kw in t['expected_kw'] if kw.lower() not in ans.lower()]}, no_br={no_br})")

print("\n" + "="*70)
print(f"BENCHMARK SCORE: {passed}/{len(tests)} TESTS PASSED ({passed/len(tests)*100:.1f}%)")
print("="*70)
