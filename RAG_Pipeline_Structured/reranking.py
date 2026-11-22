import logging

from .config import RAGConfig

logger = logging.getLogger("rag_pipeline")


class Reranker:
    """Optional cross-encoder reranking; degrades to pass-through if the model cannot be loaded."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        if config.use_reranker and config.reranker_model:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(config.reranker_model)
            except Exception as e:
                logger.warning("Reranker %s unavailable (%s); continuing without reranking",
                               config.reranker_model, e)

    def rerank(self, query: str, candidates: list) -> list:
        if not self.model:
            return candidates[:self.config.final_top_k]
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.model.predict(pairs)
        for i, sc in enumerate(scores):
            candidates[i].reranker_score = float(sc)
        candidates.sort(key=lambda x: x.reranker_score or 0, reverse=True)
        return candidates[:self.config.final_top_k]
