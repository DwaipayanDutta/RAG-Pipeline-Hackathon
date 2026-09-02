from .models import RetrievalCandidate

def generate_citations(candidates: list) -> list:
    citations = []
    for c in candidates:
        citations.append({
            "page": c.chunk.page_start,
            "chunk_id": c.chunk.chunk_id,
            "source_type": c.chunk.source_type
        })
    return citations