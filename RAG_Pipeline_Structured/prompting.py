from transformers import AutoTokenizer
from .config import RAGConfig

class PromptBuilder:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.generation_model)

    def build_prompt(self, query: str, candidates: list) -> str:
        # Reserve tokens for system and query
        system = "You are an AI assistant. Answer the question using ONLY the provided context. Do not use outside knowledge. If the context does not contain enough information, respond with: 'I don't know from the document.'"
        system_tokens = len(self.tokenizer.encode(system))
        query_tokens = len(self.tokenizer.encode(query))
        # Remaining for context
        budget = self.config.context_token_budget - system_tokens - query_tokens - 50  # safety margin
        # Build context by adding chunks until budget reached
        context_parts = []
        current_tokens = 0
        for c in candidates:
            chunk_text = f"[Page {c.chunk.page_start}] {c.chunk.text}"
            chunk_tokens = len(self.tokenizer.encode(chunk_text))
            if current_tokens + chunk_tokens > budget:
                break
            context_parts.append(chunk_text)
            current_tokens += chunk_tokens
        context = "\n---\n".join(context_parts)
        prompt = f"{system}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        return prompt