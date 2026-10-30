"""
File chạy thử nghiệm hệ thống Multi-Agent RAG v2 (API-Driven).
Không cần GPU, không cần model fine-tune nặng nề.
"""
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Đảm bảo đường dẫn import
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from core.graph import build_research_rag_graph

console = Console()

def main():
    console.print(Panel.fit("[bold green]HỆ THỐNG MULTI-AGENT RAG V2 (API-DRIVEN)[/bold green]\n"
                            "[dim]Kiến trúc không cần Fine-tune | Chạy mượt mà trên Laptop[/dim]"))
    
    app = build_research_rag_graph()
    
    sample_query = "Low-Rank Adaptation (LoRA) giúp tối ưu hoá bộ nhớ và tham số huấn luyện như thế nào?"
    
    initial_state = {
        "query": sample_query,
        "pdf_path": None,
        "retry_count": 0,
        "reasoning_trace": []
    }
    
    console.print(f"\n[bold yellow]Câu hỏi thử nghiệm:[/bold yellow] {sample_query}\n")
    console.print("[cyan]Đang khởi chạy luồng điều phối LangGraph qua các Agent...[/cyan]")
    
    try:
        final_state = app.invoke(initial_state)
        
        console.print("\n[bold green]ĐIỀU PHỐI HOÀN TẤT![/bold green]\n")
        console.print(f"[bold cyan]Kết quả Fact-Check:[/bold cyan] {final_state.get('fact_check_label')}")
        console.print(f"[bold cyan]Độ tin cậy:[/bold cyan] {final_state.get('confidence')}")
        console.print(f"[bold cyan]Số bước suy luận:[/bold cyan] {len(final_state.get('reasoning_trace', []))}")
        
        console.print("\n" + "="*70)
        console.print(Panel(Markdown(final_state.get("final_report", "Không có báo cáo")), title="BÁO CÁO NGHIÊN CỨU CUỐI CÙNG"))
        console.print("="*70 + "\n")
        
    except Exception as e:
        console.print(f"[bold red]Lỗi thực thi:[/bold red] {e}")
        console.print("[yellow]Gợi ý:[/yellow] Bạn đã cung cấp GEMINI_API_KEY trong file .env chưa?")

if __name__ == "__main__":
    main()
