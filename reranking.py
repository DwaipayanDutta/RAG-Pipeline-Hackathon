from sentence_transformers import CrossEncoder
from .models import RetrievalCandidate
from .config import RAGConfig

class Reranker:
    def __init__(self, config: RAGConfig):
        self.config = config
        if config.use_reranker and config.reranker_model:
            self.model = CrossEncoder(config.reranker_model)
        else:
            self.model = None

    def rerank(self, query: str, candidates: list) -> list:
        if not self.model:
            return candidates
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.model.predict(pairs)
        for i, sc in enumerate(scores):
            candidates[i].reranker_score = float(sc)
        # Sort by reranker score
        candidates.sort(key=lambda x: x.reranker_score or 0, reverse=True)
        return candidates[:self.config.final_top_k]