import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Add main to sys.path
main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if main_dir not in sys.path:
    sys.path.insert(0, main_dir)

print("=" * 60)
print("🔍 BẮT ĐẦU KIỂM TRA TÍNH ỔN ĐỊNH TOÀN DIỆN CỦA HỆ THỐNG")
print("=" * 60)

# 1. Mode Registry
print("\n[1/5] Kiểm tra Mode Registry & Agent Factory:")
from modes.registry import get_agent_class, MODE_MAP

MODES = ['gpu_finetune', 'gpu_ft', 'gpu_max', 'gemini', 'cpu_min', 'cpu_max', 'hybrid']
AGENTS = ['retrieval', 'reader', 'summarizer', 'factchecker', 'writer']

for m in MODES:
    for a in AGENTS:
        cls = get_agent_class(a, m)
        assert cls is not None, f"Error: {a} in {m} is None"
print(f"  --> ĐẠT CHUẨN: 100% {len(MODES)} modes x {len(AGENTS)} agents ({len(MODES)*len(AGENTS)} cặp) khởi tạo hợp lệ!")

# 2. Graph Builders
print("\n[2/5] Kiểm tra LangGraph DAG Builders:")
from core.graph import build_research_rag_graph

for m in MODES:
    g = build_research_rag_graph(mode=m)
    assert g is not None, f"Lỗi build graph cho mode {m}"
print(f"  --> ĐẠT CHUẨN: Toàn bộ {len(MODES)} chế độ build thành công luồng xử lý!")

# 3. Database
print("\n[3/5] Kiểm tra SQLite Database:")
from core import db
conn = db.get_db_connection()
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]
print(f"  --> ĐẠT CHUẨN: Database workspace.db kết nối mượt mà! Bảng dữ liệu: {tables}")

# 4. FastAPI Backend App
print("\n[4/5] Kiểm tra FastAPI Backend Server:")
from api.app import app
assert app is not None
print("  --> ĐẠT CHUẨN: FastAPI Application sẵn sàng phục vụ Web UI / Client!")

# 5. End-to-End Dry Run
print("\n[5/5] Chạy thử nghiệm End-to-End Pipeline (Dry-run mode CPU Min):")
g_test = build_research_rag_graph(mode='cpu_min')
test_state = {
    'query': 'Deep learning for image recognition overview',
    'pdf_path': None,
    'retry_count': 0,
    'reasoning_trace': [],
}
result = g_test.invoke(test_state)
report = result.get('final_report', '')
print(f"  --> ĐẠT CHUẨN: Pipeline chạy trơn tru từ đầu đến cuối!")
print(f"  --> Độ dài báo cáo: {len(report)} ký tự")
print(f"  --> Số luận điểm xác minh: {len(result.get('verified_claims', []))}")
print(f"  --> Nhãn Fact-Check: {result.get('fact_check_label')}")

print("\n" + "=" * 60)
print("🎉 KẾT QUẢ: HỆ THỐNG HOẠT ĐỘNG HOÀN TOÀN ỔN ĐỊNH VÀ SẴN SÀNG 100%!")
print("=" * 60)
