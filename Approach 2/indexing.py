import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pathlib import Path
from typing import Tuple, List
from .models import Chunk
from .config import RAGConfig
from .fingerprint import get_index_metadata, compute_document_fingerprint

class IndexManager:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.index_dir = Path(config.index_dir)
        self.index_dir.mkdir(exist_ok=True)
        self.embedder = SentenceTransformer(config.embedding_model)
        self.embedder.to(config.device)
        self.dense_index = None
        self.bm25_index = None
        self.chunks = []

    def build_or_load(self, chunks: List[Chunk], force_rebuild: bool = False):
        fingerprint = compute_document_fingerprint(self.config.pdf_path)
        meta = get_index_metadata(self.config.pdf_path, self.config)
        # Build fingerprint string for filenames
        fp_str = f"{fingerprint[:12]}_{meta['embedding_model'].replace('/', '_')}"
        dense_path = self.index_dir / f"dense_{fp_str}.faiss"
        bm25_path = self.index_dir / f"bm25_{fp_str}.pkl"
        meta_path = self.index_dir / f"meta_{fp_str}.json"

        if not force_rebuild and dense_path.exists() and bm25_path.exists() and meta_path.exists():
            with open(meta_path, "r") as f:
                saved_meta = json.load(f)
            if saved_meta == meta:
                # Load dense
                self.dense_index = faiss.read_index(str(dense_path))
                # Load BM25 (simplistic, we'll store tokenized corpus)
                with open(bm25_path, "rb") as f:
                    import pickle
                    self.bm25_index = pickle.load(f)
                self.chunks = chunks  # assume order same
                return

        # Build fresh
        self.chunks = chunks
        texts = [c.text for c in chunks]
        # Dense
        embeddings = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        embeddings = np.asarray(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.dense_index = faiss.IndexFlatIP(dim)
        self.dense_index.add(embeddings)

        # BM25
        tokenized_corpus = [doc.split() for doc in texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)

        # Save
        faiss.write_index(self.dense_index, str(dense_path))
        with open(bm25_path, "wb") as f:
            import pickle
            pickle.dump(self.bm25_index, f)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def get_dense_index(self):
        return self.dense_index

    def get_bm25(self):
        return self.bm25_index