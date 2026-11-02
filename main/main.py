import os
import sys
from pathlib import Path
_curr_dir = str(Path(__file__).resolve().parent)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

import argparse
import json
import sys

# Đảm bảo console UTF-8 trên Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.graph import build_research_rag_graph
from core.config import settings


def run_cli(query: str, pdf: str = None, test_retry: bool = False, mode: str = None):
    if test_retry:
        query = f"[test_retry] {query}"

    print("\n" + "=" * 60)
    print("KHỞI ĐỘNG HỆ THỐNG MULTI-AGENT RESEARCH RAG")
    print(f"Chế độ thực thi: {settings.EXECUTION_MODE.upper()}")
    print(f"Câu truy vấn: {query}")
    print(f"File PDF kèm theo: {pdf}")
    print("=" * 60 + "\n")

    app = build_research_rag_graph(mode=mode)

    initial_state = {
        "query": query,
        "pdf_path": pdf,
        "retry_count": 0,
        "reasoning_trace": [],
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print("KẾT QUẢ CUỐI CÙNG (FINAL REPORT)")
    print("=" * 60)
    print(final_state.get("final_report", "Không có báo cáo được tạo ra."))

    print("\n" + "=" * 60)
    print("NHẬT KÝ SUY LUẬN TỔNG HỢP (REASONING TRACE)")
    print("=" * 60)
    for idx, step in enumerate(final_state.get("reasoning_trace", []), 1):
        print(f"[{idx}] {json.dumps(step, ensure_ascii=False)}")
    print("=" * 60 + "\n")


def run_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    print("\n" + "=" * 60)
    print(f"KHỞI ĐỘNG WEB WORKSPACE & FASTAPI SERVER TẠI:")
    print(f"http://localhost:{port}")
    print(f"Chế độ thực thi: {settings.EXECUTION_MODE.upper()}")
    print("=" * 60 + "\n")
    uvicorn.run("api.app:app", host=host, port=port, reload=True)


def main():
    parser = argparse.ArgumentParser(description="Unified Multi-Agent Research RAG System")
    parser.add_argument("--query", type=str, default="What is LoRA and how does it reduce memory?", help="Câu hỏi nghiên cứu")
    parser.add_argument("--pdf", type=str, default=None, help="Đường dẫn file PDF nghiên cứu (nếu có)")
    parser.add_argument("--test-retry", action="store_true", help="Kích hoạt kiểm thử vòng lặp retry 3 lần và escalation")
    parser.add_argument("--serve", action="store_true", help="Khởi động máy chủ Web Workspace UI trên localhost:8000")
    parser.add_argument("--port", type=int, default=8000, help="Cổng chạy server (mặc định 8000)")
    parser.add_argument("--mode", type=str, default=None, choices=["gemini", "cpu_min", "cpu_max", "kaggle_fn", "kaggle_max"], help="Chế độ thực thi 5 agent (gemini, cpu_min, cpu_max, kaggle_fn, kaggle_max)")
    args = parser.parse_args()

    if args.serve:
        run_server(port=args.port)
    else:
        run_cli(query=args.query, pdf=args.pdf, test_retry=args.test_retry, mode=args.mode)


if __name__ == "__main__":
    main()
