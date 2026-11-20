import json

for mode in ['gemini', 'cpu_min', 'hybrid', 'cpu_max']:
    path = f'evaluation/results/{mode}.json'
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Cannot load {path}: {e}")
        continue

    print("=" * 40, mode.upper(), "=" * 40)
    print(f"Status: {data.get('status')} | Total: {data['metrics']['total_run']} | Passed: {data['metrics']['passed']}")
    for c in data.get('executed_cases', []):
        notes = f" | Notes: {c.get('notes')}" if c.get('notes') else ""
        print(f"[{c['test_id']}] ({c['category']}) Passed={c['passed']} Latency={c['latency_sec']}s{notes}")
        sample = c.get('answer_sample', '').replace('\n', ' ')
        print(f"   Answer: {sample[:120]}...")
