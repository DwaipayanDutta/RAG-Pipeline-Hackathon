#!/usr/bin/env bash
# End-to-end check of the Graph RAG pipeline. Run from the repository root:
#   bash RAG_Pipeline_Graph/Verify.sh
set -euo pipefail

# 1. Install (paths are resolved relative to the requirements file, so any cwd works)
pip install -r RAG_Pipeline_Graph/requirements_graph.txt

# 2. Unit + integration tests (offline: hash embeddings, no model downloads).
#    Expected: all pass; 2 skipped (spaCy model test, opt-in real FLAN-T5 test).
python -m pytest RAG_Pipeline_Graph -q

# 3. Build the index from the committed sample data and show what was ingested
python RAG_Pipeline_Graph/rag_pipeline_graph.py --stats

# 4. The canonical conflict question (live DB says Amara Chen, stale wiki + 2024 PDF say Jordan Blake)
python RAG_Pipeline_Graph/rag_pipeline_graph.py --query "Who is on-call for payments-service?" --format text

# 5. Score all sample questions (exit code 1 if any fails)
python RAG_Pipeline_Graph/rag_pipeline_graph.py --evaluate RAG_Pipeline_Graph/sample_data/queries_graph.json

# 6. Same data from a real SQLite file in a temp folder
DEMO_DIR="$(mktemp -d)"
python RAG_Pipeline_Graph/rag_pipeline_graph.py --seed-demo "$DEMO_DIR"
python RAG_Pipeline_Graph/rag_pipeline_graph.py \
  --db "$DEMO_DIR/cmdb.sqlite" --wiki-dir "$DEMO_DIR/wiki" --pdf-dir "$DEMO_DIR/pdfs" \
  --index-dir "$DEMO_DIR/index" --query "Who manages the team that owns payments-service?" --format text
