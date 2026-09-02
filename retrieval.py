import numpy as np
from typing import List, Tuple
from .models import Chunk, RetrievalCandidate
from .config import RAGConfig
from .indexing import IndexManager

def dense_retrieval(query: str, index_manager: IndexManager, top_k: int) -> List[Tuple[Chunk, float]]:
    embedder = index_manager.embedder
    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb).astype('float32')
    faiss.normalize_L2(q_emb)
    scores, indices = index_manager.dense_index.search(q_emb, top_k)
    candidates = []
    for idx, score in zip(indices[0], scores[0]):
        if idx >= 0 and idx < len(index_manager.chunks):
            candidates.append((index_manager.chunks[idx], float(score)))
    return candidates

def lexical_retrieval(query: str, index_manager: IndexManager, top_k: int) -> List[Tuple[Chunk, float]]:
    bm25 = index_manager.bm25_index
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    # Get top_k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    candidates = []
    for idx in top_indices:
        if idx < len(index_manager.chunks):
            candidates.append((index_manager.chunks[idx], float(scores[idx])))
    return candidates

def reciprocal_rank_fusion(dense_cands: List[Tuple[Chunk, float]],
                           lexical_cands: List[Tuple[Chunk, float]],
                           rrf_k: int = 60) -> List[Tuple[Chunk, float]]:
    # Create rank maps
    chunk_to_score = {}
    for rank, (chunk, _) in enumerate(dense_cands, start=1):
        chunk_to_score[chunk.chunk_id] = chunk_to_score.get(chunk.chunk_id, 0) + 1 / (rrf_k + rank)
    for rank, (chunk, _) in enumerate(lexical_cands, start=1):
        chunk_to_score[chunk.chunk_id] = chunk_to_score.get(chunk.chunk_id, 0) + 1 / (rrf_k + rank)
    # Sort by score
    sorted_chunks = sorted(chunk_to_score.items(), key=lambda x: x[1], reverse=True)
    # Return list of (chunk, rrf_score)
    # We need to retrieve the chunk objects
    chunk_dict = {c.chunk_id: c for c in dense_cands + lexical_cands}
    result = []
    for cid, score in sorted_chunks:
        if cid in chunk_dict:
            result.append((chunk_dict[cid], score))
    return result

def retrieve(query: str, index_manager: IndexManager, config: RAGConfig) -> List[RetrievalCandidate]:
    # Dense
    dense_cands = dense_retrieval(query, index_manager, config.dense_top_k)
    # Lexical (if enabled)
    if config.use_lexical:
        lexical_cands = lexical_retrieval(query, index_manager, config.lexical_top_k)
        # RRF
        fused = reciprocal_rank_fusion(dense_cands, lexical_cands, config.rrf_k)
    else:
        fused = dense_cands

    # Convert to RetrievalCandidate
    candidates = []
    for chunk, score in fused:
        rc = RetrievalCandidate(chunk=chunk, rrf_score=score)
        # Store dense/lexical scores if we have them (we lost them but can re‑search)
        # For simplicity, we'll keep only rrf_score.
        candidates.append(rc)

    # Optional reranker
    if config.use_reranker:
        # (we'll implement reranking in reranking.py)
        pass

    # Return top final_top_k
    return candidates[:config.final_top_k]