import sys, glob, time, json
sys.path.insert(0, 'main')
from core.graph import build_research_rag_graph

pdf_bao = 'bao/Báo Hà Nội mới ra số đầu tiên (1).pdf'
pdf_env = 'bao/Thường trực Hội đồng nhân dân thành phố Hà Nội tổ chức phiên chất vấn về lĩnh vực môi trường (1).pdf'
nginx_files = glob.glob('main/uploads/*cfe4f779*.pdf')
pdf_nginx = nginx_files[0] if nginx_files else pdf_bao

tests_30 = [
    # --- Nhóm 1: Câu hỏi Ngày / Tháng / Năm (Dates & Years) ---
    ('TC-01', 'Báo Thủ đô ra số đầu tiên vào ngày nào ?', [pdf_bao], True, ['24', '10', '1957']),
    ('TC-02', 'Báo Thủ đô ra đời vào năm nào ?', [pdf_bao], True, ['1957']),
    ('TC-03', 'Chiến thắng Sông Lô diễn ra vào năm nào ?', [pdf_bao], True, ['1947']),
    ('TC-04', 'Hiến chương Liên Hợp Quốc có hiệu lực vào ngày nào ?', [pdf_bao], True, ['24.10.1945']),
    ('TC-05', 'Thị trường chứng khoán Mỹ chao đảo Thứ Năm đen tối vào năm nào ?', [pdf_bao], True, ['1929']),

    # --- Nhóm 2: Câu hỏi Danh sách & Nhóm vấn đề (Lists & Groups) ---
    ('TC-06', '5 nhóm vấn đề môi trường là gì ?', [pdf_env], True, ['chất thải rắn', 'không khí']),
    ('TC-07', 'Chủ tịch HĐND đề nghị tập trung vào 3 nội dung gì ?', [pdf_env], True, ['3 nội dung', 'thực trạng', 'nguyên nhân']),
    ('TC-08', 'Một server block chuẩn trong NGINX thường gồm những gì ?', [pdf_nginx], True, ['listen', 'server_name', 'location']),
    ('TC-09', 'Các header quan trọng nào cần thiết lập khi proxy_set_header ?', [pdf_nginx], True, ['Host', 'X-Real-IP', 'X-Forwarded-For']),

    # --- Nhóm 3: Thông số kỹ thuật & Mã lỗi (Technical & Configs) ---
    ('TC-10', 'Giá trị mặc định của client_max_body_size trong NGINX là bao nhiêu ?', [pdf_nginx], True, ['1MB']),
    ('TC-11', 'Khi request vượt quá dung lượng body sẽ gặp lỗi HTTP nào ?', [pdf_nginx], True, ['413']),
    ('TC-12', 'Làm thế nào để tắt tính năng tự sinh file /etc/hosts của WSL2 ?', [pdf_nginx], True, ['generateHosts = false']),
    ('TC-13', 'Công cụ nào dùng để lấy chứng chỉ SSL miễn phí trên server thật ?', [pdf_nginx], True, ['Certbot', 'Let\'s Encrypt']),
    ('TC-14', 'Cấu hình NGINX trỏ vào container Backend trong Docker network như thế nào ?', [pdf_nginx], True, ['proxy_pass', 'backend']),

    # --- Nhóm 4: Định nghĩa & Khái niệm cốt lõi (Core Concepts) ---
    ('TC-15', 'Mục tiêu của Buổi 8 là gì ?', [pdf_nginx], True, ['Web Server', 'Reverse Proxy']),
    ('TC-16', 'Upstream trong NGINX dùng để làm gì ?', [pdf_nginx], True, ['cân bằng tải', 'backend']),
    ('TC-17', 'Cấu hình NGINX làm Reverse Proxy như thế nào để tránh lỗi CORS ?', [pdf_nginx], True, ['CORS', 'origin', 'api']),
    ('TC-18', 'Vì sao chứng chỉ tự ký (self-signed) bị trình duyệt cảnh báo Not Secure ?', [pdf_nginx], True, ['CA', 'cảnh báo']),
    ('TC-19', 'Chuỗi tin cậy (chain of trust) của SSL hoạt động như thế nào ?', [pdf_nginx], True, ['Root CA', 'xác minh']),

    # --- Nhóm 5: Tóm tắt Đơn & Đa tài liệu (Summaries) ---
    ('TC-20', 'Tóm tắt nội dung tài liệu Buổi 8 NGINX', [pdf_nginx], True, ['Trang 1', 'Trang 2']),
    ('TC-21', 'Tóm tắt bài báo về phiên chất vấn môi trường', [pdf_env], True, ['môi trường', 'chất vấn']),
    ('TC-22', 'Tóm tắt 2 bài báo này', [pdf_bao, pdf_env], True, ['Bài báo 1', 'Bài báo 2']),

    # --- Nhóm 6: Tiếng Việt Không Dấu (Unaccented Queries) ---
    ('TC-23', 'bao thu do ra doi vao nam nao ?', [pdf_bao], True, ['1957']),
    ('TC-24', '5 nhom van de moi truong la gi ?', [pdf_env], True, ['chất thải rắn', 'không khí']),
    ('TC-25', 'gia tri mac dinh cua client_max_body_size la bao nhieu ?', [pdf_nginx], True, ['1MB']),

    # --- Nhóm 7: Trường hợp Biên & Ngoài phạm vi (Out-of-Domain & Cross-Doc) ---
    ('TC-26', 'Apache Kafka được dùng để làm gì trong Buổi 8 ?', [pdf_nginx], False, ['hoàn toàn không được đề cập']),
    ('TC-27', 'Cầu Tứ Liên khi nào khởi công xây dựng ?', [pdf_bao], False, ['hoàn toàn không được đề cập']),
    ('TC-28', 'Mô hình Transformer và cơ chế Self-Attention hoạt động ra sao ?', [pdf_nginx], False, ['hoàn toàn không được đề cập']),
    ('TC-29', 'Báo Thủ đô ra số đầu tiên vào ngày nào ?', [pdf_env], False, ['hoàn toàn không được đề cập']),
    ('TC-30', '5 nhóm vấn đề môi trường là gì ?', [pdf_bao], False, ['hoàn toàn không được đề cập']),
]

graph = build_research_rag_graph(mode='cpu_max')
results = []
passed_count = 0

print(f"=== BAT DAU BENCHMARK CPU MAX (30 CA KIEM THU) ===")

for tc_id, query, paths, in_doc, expected in tests_30:
    t0 = time.time()
    try:
        res = graph.invoke({
            'query': query,
            'pdf_path': paths[0],
            'pdf_paths': paths,
        })
        elapsed = round(time.time() - t0, 2)
        ans = res.get('final_report', '') or res.get('answer', '')
        
        # Check formatting bugs
        has_br = '<br>' in ans
        has_glued = any(g in ans for g in ['BáoThủ', 'BáoHà', 'trênđường'])
        
        # Check expected keywords
        missing_kw = [kw for kw in expected if kw.lower() not in ans.lower()]
        
        passed = (len(missing_kw) == 0) and (not has_br) and (not has_glued)
        if passed:
            passed_count += 1
            
        preview = ans.replace('\n', ' ').strip()[:100]
        results.append({
            'id': tc_id,
            'query': query,
            'in_doc': in_doc,
            'elapsed_sec': elapsed,
            'passed': passed,
            'has_br': has_br,
            'has_glued': has_glued,
            'missing_kw': missing_kw,
            'preview': preview,
        })
        status_str = "PASS" if passed else "FAIL"
        print(f"[{status_str}] {tc_id} ({elapsed}s) | Query: '{query[:40]}' | Missing: {missing_kw} | <br>: {has_br}")
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        print(f"[ERROR] {tc_id} ({elapsed}s) | Query: '{query[:40]}' | Error: {e}")
        results.append({
            'id': tc_id,
            'query': query,
            'in_doc': in_doc,
            'elapsed_sec': elapsed,
            'passed': False,
            'error': str(e),
        })

print("\n" + "="*70)
print(f"TONG KET BENCHMARK CPU MAX: {passed_count}/{len(tests_30)} PASSED ({round(passed_count/len(tests_30)*100, 1)}%)")
print("="*70)

with open('main/data/cpu_max_benchmark_report.json', 'w', encoding='utf-8') as f:
    json.dump({
        'total': len(tests_30),
        'passed': passed_count,
        'rate': round(passed_count/len(tests_30)*100, 1),
        'details': results
    }, f, ensure_ascii=False, indent=2)

print("Saved report to main/data/cpu_max_benchmark_report.json")
