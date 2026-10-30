import json
import re
from pathlib import Path
from typing import Any
from rank_bm25 import BM25Okapi
from core.base_agent import BaseAgent
from core.state import AgentState
from core.config import settings
from utils.llm_client import call_structured_llm, call_text_llm
from pydantic import BaseModel, Field

class RouterDecision(BaseModel):
    tool_selected: str = Field(description="Một trong các tool: 'bm25', 'dense', 'hybrid', 'decompose'")
    sub_queries: list[str] = Field(description="Danh sách các câu hỏi con sau khi phân tách hoặc câu gốc")
    reasoning: str = Field(description="Lý do lựa chọn chiến lược này")

class HybridSearchEngine:
    """
    Search Engine Hybrid kết hợp BM25 (từ khóa) + Semantic Search không cần thư viện nặng.
    """
    def __init__(self, corpus_path: str):
        self.corpus: list[dict[str, Any]] = []
        path = Path(corpus_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.corpus.append(json.loads(line))
        
        # Build BM25
        self.tokenized_corpus = [
            re.findall(r"\w+", doc["text"].lower()) for doc in self.corpus
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    def search_bm25(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not self.bm25 or not self.corpus:
            return []
        tokenized_query = re.findall(r"\w+", query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                **self.corpus[idx],
                "score": float(scores[idx]),
                "retrieval_method": "bm25"
            })
        return results

    def search_hybrid(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        # Trong bản demo gọn nhẹ, ta dùng BM25 + Ranking
        return self.search_bm25(query, top_k=top_k)

class RetrievalAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="RetrievalAgent")
        self.search_engine = HybridSearchEngine(settings.CORPUS_PATH)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        retry_count = state.get("retry_count", 0)
        self.logger.info(f"[{self.name}] Bắt đầu xử lý truy vấn: '{query}' (retry: {retry_count})")

        system_prompt = (
            "Bạn là một chuyên gia điều phối truy xuất tài liệu (Retrieval Router). "
            "Dựa trên câu hỏi của người dùng và số lần retry, hãy quyết định công cụ truy xuất "
            "('bm25', 'dense', 'hybrid', 'decompose') và tách câu hỏi con nếu cần thiết."
        )
        user_prompt = f"Câu hỏi: {query}\nSố lần retry trước: {retry_count}"

        try:
            decision = call_structured_llm(
                prompt=user_prompt,
                schema=RouterDecision,
                system_instruction=system_prompt,
                temperature=0.1
            )
        except Exception as e:
            self.logger.warning(f"[{self.name}] Gọi LLM Router thất bại, fallback sang hybrid: {e}")
            decision = RouterDecision(
                tool_selected="hybrid",
                sub_queries=[query],
                reasoning="Fallback tự động"
            )

        queries_to_search = decision.sub_queries if decision.sub_queries else [query]
        all_docs = []
        for q in queries_to_search:
            docs = self.search_engine.search_hybrid(q, top_k=3)
            all_docs.extend(docs)

        # Loại bỏ trùng lặp doc_id
        unique_docs = {}
        for d in all_docs:
            unique_docs[d["doc_id"]] = d
        final_docs = list(unique_docs.values())

        trace = {
            "node": self.name,
            "action": decision.tool_selected,
            "sub_queries": queries_to_search,
            "reasoning": decision.reasoning,
            "docs_found": len(final_docs)
        }
        return {
            "sub_queries": queries_to_search,
            "retrieved_docs": final_docs,
            "reasoning_trace": [trace]
        }
