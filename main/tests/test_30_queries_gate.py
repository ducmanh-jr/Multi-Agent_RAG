import sys, glob, re, unicodedata
sys.stdout.reconfigure(encoding='utf-8')
import pypdf

pdf = glob.glob('main/uploads/*NGINX*.pdf')[0]
reader = pypdf.PdfReader(pdf)
source_units = [{'id': f'u_{i+1}', 'page_no': i+1, 'text': p.extract_text(), 'doc_id': 'nginx'} for i, p in enumerate(reader.pages)]
raw_doc_text = ' '.join(str(u.get('text', '')) for u in source_units)

def strip_accents(text: str) -> str:
    if not text:
        return ''
    norm = unicodedata.normalize('NFD', str(text))
    s = re.sub(r'[\u0300-\u036f]', '', norm)
    return s.replace('đ', 'd').replace('Đ', 'D')

doc_l = raw_doc_text.lower()
doc_unaccent = strip_accents(doc_l)

def is_unaccented_query(q: str) -> bool:
    """Checks if the user typed Vietnamese without diacritics."""
    return bool(re.search(r'[a-zA-Z]', q) and not re.search(r'[à-ỹÀ-Ỹ]', q))

def has_phrase(p: str, query_has_accents: bool) -> bool:
    if not p: return False
    p_c = re.escape(p.lower().strip())
    pat1 = r'(?<![\wà-ỹÀ-Ỹ])' + p_c + r'(?![\wà-ỹÀ-Ỹ])'
    if re.search(pat1, doc_l):
        return True
    # Only fall back to unaccented match if the query itself was typed without accents
    if not query_has_accents:
        p_u = re.escape(strip_accents(p.lower().strip()))
        pat2 = r'(?<!\w)' + p_u + r'(?!\w)'
        if re.search(pat2, doc_unaccent):
            return True
    return False

# Generic cross-domain verbs, adjectives, and question connectors
GENERIC_COMMON_WORDS = {
    'là', 'gì', 'thế', 'nào', 'sao', 'khi', 'bao', 'nhiêu', 'mấy', 'ai', 'ở', 'đâu',
    'được', 'có', 'của', 'cho', 'với', 'trong', 'về', 'từ', 'đến', 'và', 'hoặc', 'hay',
    'những', 'các', 'một', 'này', 'đó', 'thì', 'mà', 'lại', 'ra', 'vào', 'bởi', 'do',
    'hãy', 'biết', 'giúp', 'tôi', 'bạn', 'làm', 'cách', 'như', 'tại', 'thực', 'hiện',
    'gồm', 'dùng', 'để', 'tác', 'dụng', 'ý', 'nghĩa', 'qua', 'lại', 'cùng', 'nhau',
    'bằng', 'năm', 'ngày', 'thời', 'điểm', 'sinh', 'nay', 'hiện', 'nguyên', 'lý',
    'hoạt', 'động', 'giao', 'tiếp', 'tính', 'chất', 'thành', 'lập', 'đời', 'bước',
    'mục', 'phần', 'nội', 'dung', 'khái', 'niệm', 'vấn', 'đề', 'chi', 'tiết',
    'hoạt động', 'giao tiếp', 'nguyên lý', 'tính chất', 'thành lập', 'ra đời',
    'hiện nay', 'sinh năm', 'bao nhiêu', 'thế nào', 'như thế nào', 'khi nào',
    'là gì', 'để làm gì', 'có tác dụng gì', 'ý nghĩa gì', 'gồm những gì'
}

VN_ASCII_WORDS = {
    'cho', 'voi', 'khi', 'bao', 'nam', 'sinh', 'nhu', 'the', 'nao', 'sao', 'lam',
    'cac', 'mot', 'nay', 'trong', 'duoc', 'giao', 'tiep', 'nhau', 'qua', 'lai',
    'cung', 'chay', 'cach', 'theo', 'bang', 'thuc', 'hien', 'dung', 'de', 'khong',
    'phan', 'biet', 'nguoi', 'nhom', 'truong', 'buoc', 'dong', 'them', 'sua'
}

def extract_core_subject_phrases(query: str) -> list[str]:
    q_clean = query.strip().rstrip('?.!')
    
    # 1. Technical identifiers / code / English terms (e.g. generateHosts, proxy_pass, client_max_body_size, nginx, upstream, wsl, openssl, kafka, oop)
    raw_tech = re.findall(r'\b[a-zA-Z0-9_.-]{3,}\b', q_clean)
    tech = [t for t in raw_tech if t.lower() not in VN_ASCII_WORDS]
    
    # 2. Quoted phrases
    quotes = re.findall(r'[\"\'\`“](.*?)[\"\'\`”]', q_clean)
    
    # 3. Capitalized proper nouns (2 or more capitalized words)
    caps = re.findall(r'\b[A-ZĐ][a-zà-ỹ0-9_-]+(?:\s+[A-ZĐ][a-zà-ỹ0-9_-]+)+\b', q_clean)
    
    # 4. Multi-word compound phrases (bigrams and trigrams)
    words = re.findall(r'\w+', q_clean.lower())
    compounds = []
    for i in range(len(words) - 1):
        bg = f"{words[i]} {words[i+1]}"
        if bg not in GENERIC_COMMON_WORDS and words[i] not in GENERIC_COMMON_WORDS and words[i+1] not in GENERIC_COMMON_WORDS:
            compounds.append(bg)
        if i < len(words) - 2:
            tg = f"{words[i]} {words[i+1]} {words[i+2]}"
            if tg not in GENERIC_COMMON_WORDS and words[i] not in GENERIC_COMMON_WORDS and words[i+2] not in GENERIC_COMMON_WORDS:
                compounds.append(tg)
                
    phrases = []
    phrases.extend(tech)
    phrases.extend(quotes)
    phrases.extend(caps)
    phrases.extend(compounds)
    
    unique = []
    for p in sorted(phrases, key=len, reverse=True):
        p_strip = p.strip()
        if p_strip.lower() not in [u.lower() for u in unique] and len(p_strip) >= 3:
            unique.append(p_strip)
    return unique

test_qs = [
    ('Chuẩn bị thuyết trình và demo cần gì', False),
    ('Upstream là gì', False),
    ('Reverse proxy là gì', False),
    ('Tắt tính năng tự sinh file hosts trong WSL bằng cách nào', False),
    ('Cấu hình generateHosts = false ở đâu', False),
    ('Lệnh openssl tạo chứng chỉ ssl tự ký là gì', False),
    ('Proxy_pass dùng để làm gì', False),
    ('Tại sao không sửa file etc hosts trong WSL ở trường hợp 2', False),
    ('Các thuật toán cân bằng tải của NGINX là gì', False),
    ('Load balancing nhiều backend để giải quyết vấn đề gì', False),
    ('Task người thực hiện ước lượng point trong buổi 8', False),
    ('Nghiên cứu cấu hình thực tế container FE và BE cần chú ý gì', False),
    ('/tom-tat', False),
    ('Tóm tắt nội dung tài liệu', False),
    ('Client_max_body_size có tác dụng gì', False),
    ('bác hồ sinh năm bao nhiêu', True),
    ('Bảo tàng Dân tộc học Việt Nam được thành lập khi nào ?', True),
    ('Thủ đô của nước Pháp là gì', True),
    ('Định luật bảo toàn năng lượng là gì', True),
    ('Kafka broker hoạt động như thế nào', True),
    ('Kubernetes pods giao tiếp với nhau qua đâu', True),
    ('Cách nấu phở bò Hà Nội', True),
    ('Ai là tác giả của Truyện Kiều', True),
    ('Dân số thế giới hiện nay là bao nhiêu', True),
    ('Mô hình mạng nơ-ron Transformer ra đời năm nào', True),
    ('Triệu chứng của bệnh tiểu đường là gì', True),
    ('Hồ Gươm nằm ở quận nào của Hà Nội', True),
    ('Chiến thắng Điện Biên Phủ diễn ra vào năm nào', True),
    ('Nguyên lý hoạt động của động cơ đốt trong', True),
    ('Lập trình hướng đối tượng OOP gồm những tính chất nào', True)
]

passed = 0
for q, exp_ood in test_qs:
    if q.startswith('/') or any(w in q.lower() for w in ['tóm tắt', 'tổng quan']):
        is_ood = False
    else:
        query_has_accents = bool(re.search(r'[à-ỹÀ-Ỹ]', q))
        phrases = extract_core_subject_phrases(q)
        
        # Check if query contains any prominent foreign / English term (e.g. Transformer, Kubernetes, Kafka)
        foreign_terms = [
            t for t in re.findall(r'\b[A-Za-z0-9_-]{4,}\b', q)
            if t.lower() not in VN_ASCII_WORDS and not re.search(r'[à-ỹÀ-Ỹ]', t) and t.lower() not in GENERIC_COMMON_WORDS
        ]
        missing_foreign = [t for t in foreign_terms if not has_phrase(t, query_has_accents)]
        if foreign_terms and len(missing_foreign) == len(foreign_terms):
            is_ood = True
        else:
            matched_phrases = [p for p in phrases if has_phrase(p, query_has_accents)]
            is_ood = (len(matched_phrases) == 0)
    
    ok = (is_ood == exp_ood)
    if ok: passed += 1
    tag = 'PASS' if ok else 'FAIL'
    print(f'[{tag}] {q[:50]:<50} -> OOD: {str(is_ood):<5} (Expected: {exp_ood})')

print(f'\nTOTAL RESULT: {passed}/{len(test_qs)} passed ({passed/len(test_qs)*100:.1f}%)')
