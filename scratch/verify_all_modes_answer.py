import os
import sys
from pathlib import Path

_curr_dir = str(Path(__file__).resolve().parent.parent / "main")
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.graph import build_research_rag_graph
from core.config import settings

settings.REMOTE_INFERENCE_URL = "https://assumed-somerset-couples-washing.trycloudflare.com"

pdf_path = os.path.join(_curr_dir, "uploads", "doc_581b5793_Buổi 8 Cấu hình NGINX reverse proxy _ Upstream nhiều backend.pdf")
query = "upstream là gì"

modes = ["cpu_min", "cpu_max", "gemini", "hybrid", "gpu_finetune"]
report_results = []

for m in modes:
    print(f"\n>>> ĐANG KIỂM TRA CHẾ ĐỘ: {m.upper()}...")
    settings.EXECUTION_MODE = m
    try:
        graph = build_research_rag_graph(mode=m)
        out = graph.invoke({"query": query, "pdf_path": pdf_path, "retry_count": 0, "reasoning_trace": []})
        ans = out.get("answer") or ""
        
        # Check criteria
        has_def = "upstream" in ans.lower() and any(w in ans.lower() for w in ["nhóm", "backend", "chịu tải", "song song", "cân bằng"])
        has_code = "```" in ans or "{" in ans
        has_mech = any(w in ans.lower() for w in ["round-robin", "cân bằng tải", "least_conn", "proxy_pass"])
        has_noise = any(w in ans.lower() for w in ["người thực hiện", "chuẩn bị slide", "luyện tập thuyết trình", "bài viết: các bài", "video: các video"])
        
        is_pass = has_def and not has_noise
        status_label = "✅ ĐẠT (ĐÚNG & ĐỦ)" if is_pass else "⚠️ CẦN XEM LẠI"
        
        report_results.append({
            "mode": m,
            "status": status_label,
            "has_def": has_def,
            "has_code": has_code,
            "has_mech": has_mech,
            "has_noise": has_noise,
            "answer_preview": ans.strip()[:260] + "..."
        })
        print(f"Status: {status_label}")
        print(f"Preview:\n{ans.strip()[:260]}...\n")
    except Exception as exc:
        print(f"Error in mode {m}: {exc}")
        report_results.append({
            "mode": m,
            "status": f"❌ LỖI: {str(exc)[:60]}",
            "has_def": False,
            "has_code": False,
            "has_mech": False,
            "has_noise": False,
            "answer_preview": str(exc)[:200]
        })

print("\n" + "="*80)
print("KẾT QUẢ TỔNG QUAN KIỂM THỬ TẤT CẢ CÁC TÙY CHỌN THEO TIÊU CHÍ 'ĐÚNG & ĐỦ':")
print("="*80)
for r in report_results:
    print(f"• Chế độ {r['mode']:<14} | Trạng thái: {r['status']:<18} | Định nghĩa: {r['has_def']} | Code/Config: {r['has_code']} | Cơ chế liên quan: {r['has_mech']} | Rác/Nhiễu: {r['has_noise']}")
