from .config import RAGConfig
from .ingestion import extract_pages
from .chunking import chunk_document
from .indexing import IndexManager
from .retrieval import retrieve
from .reranking import Reranker
from .query_router import classify_query
from .prompting import PromptBuilder
from .generation import Generator
from .validation import validate_answer
from .citations import generate_citations
from .observability import RequestContext

class RAGPipeline:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.index_manager = IndexManager(config)
        self.reranker = Reranker(config)
        self.generator = Generator(config)
        self.prompt_builder = PromptBuilder(config)
        # Build index if not loaded
        pages = extract_pages(config.pdf_path)
        document_id = "titan_ar_2026"
        chunks = chunk_document(pages, document_id, config)
        self.index_manager.build_or_load(chunks, force_rebuild=False)

    def process(self, query: str):
        ctx = RequestContext()
        ctx.stage_start("retrieval")
        candidates = retrieve(query, self.index_manager, self.config)
        ctx.stage_end("retrieval")

        ctx.stage_start("reranking")
        candidates = self.reranker.rerank(query, candidates)
        ctx.stage_end("reranking")

        ctx.stage_start("prompt")
        prompt = self.prompt_builder.build_prompt(query, candidates)
        ctx.stage_end("prompt")

        ctx.stage_start("generation")
        answer_text = self.generator.generate(prompt)
        ctx.stage_end("generation")

        # Validation
        ctx.stage_start("validation")
        validation = validate_answer(answer_text, candidates)
        ctx.stage_end("validation")

        citations = generate_citations(candidates)

        result = {
            "query": query,
            "answer": answer_text if validation.status == "grounded" else "I don't know from the document.",
            "citations": citations,
            "validation": validation.status,
            "retrieved_chunks": [
                {
                    "page": c.chunk.page_start,
                    "chunk_id": c.chunk.chunk_id,
                    "source_type": c.chunk.source_type,
                    "score": c.rrf_score or 0.0
                } for c in candidates
            ]
        }
        ctx.log({"query_type": classify_query(query), "validation_status": validation.status})
        return result