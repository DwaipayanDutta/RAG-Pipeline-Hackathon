<!-- Append to README.md -->

## Graph-RAG pipeline (`rag_pipeline_graph.py`)

A third, standalone variant next to `rag_pipeline.py` and `rag_pipeline_jev.py`. It targets the enterprise
scenario from the Graph-RAG tutorial (live database, wiki pages, PDFs) rather than the Titan annual report, and
resolves conflicts between sources with **priority rules in the prompt**, re-checked after generation.
The graph is held in-process (`QuadStore`); no graph server is required.

| Tier | Source | Stored as | Context tag | Role |
| ---- | ------ | --------- | ----------- | ---- |
| Priority 1 | SQLite `services` table | SPOC quads | `cmdb` | Absolute, always-current facts |
| Priority 2 | Markdown wiki pages | SPOC quads (regex, LLM fallback) | `wiki` | Supplementary, possibly stale |
| Priority 3 | PDFs | ChromaDB chunks | n/a | Narrative context / vector fallback |

### Comparison

| | Standard RAG | Jev + RAG | Graph-RAG |
| - | ------------ | --------- | --------- |
| File | `rag_pipeline.py` | `rag_pipeline_jev.py` | `rag_pipeline_graph.py` |
| Target data | Annual report PDF | Annual report PDF | Database + wiki + PDFs |
| Index | FAISS + BM25 | FAISS + BM25 | QuadStore + ChromaDB |
| Retrieval | Dense + BM25 + RRF | Dense + BM25 + RRF, Jev gate | Entity match over quads, graph expansion, vector fallback |
| Conflict handling | None | None | Priority rules in prompt + contradiction check |
| Refusal | Grounding validation | Jev gate + grounding validation | Priority-1 contradiction, entity and coverage checks |
| Extra dependencies | n/a | `jev-reranker` (optional) | `chromadb`, `spacy` |

### Usage

```bash
pip install -r requirements_graph.txt
python -m spacy download en_core_web_sm

python rag_pipeline_graph.py --seed-demo demo_graph_data        # CMDB says Amara Chen, wiki says Jordan Blake
python rag_pipeline_graph.py --query "Who is on-call for payments-service?" --config config_graph.json
python rag_pipeline_graph.py --queries queries_graph.json --output results_graph.json --rebuild-index
pytest test_rag_pipeline_graph.py -v
```

Output uses the same fields as the other pipelines (`answer`, `grounded`, `citations`, `retrieved_chunks`,
`validation`, `refusal_reason`) plus `priority_tier`, `vector_fallback_used`, `entities` and `conflicts`.

### Known limitations of the Graph-RAG variant

1. Graph construction is offline. A change to any source triggers a rebuild (fingerprint-based cache);
   incremental updates, for example LightRAG-style, are a future enhancement.
2. Wiki facts use the page filename as the subject (`payments-service.md` -> `payments-service`).
3. The PDF tier is vector-only; PDF text does not yet produce graph quads.
4. `vector_max_distance` is not tuned; adjust it on real data.
5. The tests use a scripted generator, so they do not measure how well FLAN-T5-base follows the priority
   rules. `RUN_LLM_TESTS=1 pytest -k real_flan` runs a real-model check.

### Roadmap entry

#### Phase 7: Graph-RAG (parallel track)

- Query entity extraction (spaCy + known-entity list)
- Three-tier ingestion: database facts, wiki facts, PDF vector fallback
- Priority-fused retrieval with graph expansion
- Prompt-enforced conflict resolution and contradiction validation
- Future: incremental graph updates, graph quads from PDF text, evaluation against a golden set
