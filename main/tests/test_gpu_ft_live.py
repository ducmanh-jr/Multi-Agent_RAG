import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if main_dir not in sys.path:
    sys.path.insert(0, main_dir)

from core.graph import build_research_rag_graph

print("=" * 60)
print("🔍 BẮT ĐẦU KIỂM TRA CHẾ ĐỘ gpu_ft")
print("=" * 60)

g_ft = build_research_rag_graph(mode='gpu_ft')
test_state = {
    'query': 'What is quantum computing overview?',
    'pdf_path': None,
    'retry_count': 0,
    'reasoning_trace': [],
}
result = g_ft.invoke(test_state)
report = result.get('final_report', '')
print("[OK] gpu_ft đã chạy hoàn tất thành công!")
print(f"  - Độ dài báo cáo sinh ra: {len(report)} ký tự")
print(f"  - Số claims xác minh: {len(result.get('verified_claims', []))}")
print(f"  - Nhãn Fact-Check: {result.get('fact_check_label')}")
print("=" * 60)
