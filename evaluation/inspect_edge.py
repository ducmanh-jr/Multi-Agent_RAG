import json

for m in ['gemini', 'cpu_min', 'hybrid', 'cpu_max']:
    path = f'evaluation/results/{m}.json'
    data = json.load(open(path, encoding='utf-8'))
    for r in data.get('runs', []):
        if r['test_id'] in ('EDGE-03', 'GQA-03'):
            print(f"[{m}] {r['test_id']} Status={r['status']} Score={r['scores']['total_score']} Intent={r['detected_intent']}")
            print(f"   Notes: {r.get('notes')}")
            print(f"   Answer: {r['answer'][:120]}...")
