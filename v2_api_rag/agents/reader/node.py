from typing import Any
from pathlib import Path
import fitz  # PyMuPDF
from core.base_agent import BaseAgent
from core.state import AgentState

class ReaderAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ReaderAgent")

    def run(self, state: AgentState) -> dict[str, Any]:
        pdf_path = state.get("pdf_path")
        if not pdf_path or not Path(pdf_path).exists():
            self.logger.warning(f"[{self.name}] Không tìm thấy file PDF: {pdf_path}")
            return {
                "document_schema": {"pages": [], "total_blocks": 0},
                "reasoning_trace": [{"node": self.name, "action": "skip", "reason": "No PDF provided"}]
            }

        self.logger.info(f"[{self.name}] Bóc tách cấu trúc file PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        pages_schema = []
        total_blocks = 0

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            blocks = page.get_text("blocks")
            page_blocks = []
            for b in blocks:
                text = b[4].strip()
                if text:
                    page_blocks.append({
                        "bbox": [b[0], b[1], b[2], b[3]],
                        "text": text,
                        "type": "Title" if len(text) < 60 and b[1] < 150 else "Paragraph"
                    })
                    total_blocks += 1
            pages_schema.append({
                "page_no": page_idx,
                "sections": page_blocks
            })
        doc.close()

        schema = {
            "pdf_path": pdf_path,
            "pages": pages_schema,
            "total_blocks": total_blocks
        }
        trace = {
            "node": self.name,
            "action": "extracted_pdf_blocks",
            "total_pages": len(pages_schema),
            "total_blocks": total_blocks
        }
        return {
            "document_schema": schema,
            "reasoning_trace": [trace]
        }
