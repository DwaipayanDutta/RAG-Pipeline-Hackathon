"""Command-line interface (entry point: RAG_Pipeline_Graph/rag_pipeline_graph.py)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from .config import LLM_BACKENDS, EMBEDDING_BACKENDS, VECTOR_BACKENDS, load_config, load_dotenv_if_available
from .demo_data import SAMPLE_DIR, generate_sample_pdfs, seed_demo_data
from .evaluation import evaluate, format_report, load_queries
from .models import TIER_LABEL, QueryResult
from .pipeline import GraphRAGPipeline


def format_result(res: QueryResult) -> str:
    """Human-readable report: answer, evidence, provenance, conflicts, trace."""
    out = [f"Q: {res.query}", f"A: {res.answer}", ""]
    if res.refusal_reason:
        out.append(f"Refusal reason: {res.refusal_reason}")
    out.append(f"Source tier: {TIER_LABEL.get(res.priority_tier, res.priority_tier)}   "
               f"confidence: {res.confidence}   grounded: {res.grounded}")
    if res.source_explanation:
        out.append(f"Provenance: {res.source_explanation}")
    if res.citations:
        out.append("Citations:")
        for c in res.citations:
            if c.get("fact"):
                where = c["fact"]
            else:
                where = f"({c['tier']}) {c.get('title') or c['source']}" + (
                    f" p.{c['page']}" if c.get("page") else f" / {c.get('section', '')}")
            out.append(f"  [{c['id']}] {where}  <{c['source']}>")
    if res.conflicts:
        out.append("Conflicts resolved by priority (* = affects this answer):")
        for c in res.conflicts:
            mark = "*" if c.get("affects_answer") else "-"
            out.append(f"  {mark} {c['authoritative']}  WINS OVER  {c['overridden']}  [{c['overridden_source']}]")
    t = res.trace or {}
    out.append(f"Trace: entities={t.get('entities')} predicates={t.get('predicates')} "
               f"counts={t.get('counts')} mode={t.get('generation_mode')}")
    for h in (t.get("hops") or [])[:5]:
        out.append(f"  hop {h['hop']}: {h['fact']}")
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag_pipeline_graph.py",
        description="Graph-RAG pipeline: live DB (P1) > wiki/runbooks (P2) > PDFs (P3), with "
                    "priority-based conflict resolution.")
    mode = p.add_argument_group("what to run (default: interactive prompt)")
    mode.add_argument("--query", type=str, help="Answer a single question")
    mode.add_argument("--queries", type=str, help="JSON file: list of questions or {'query': ...} objects")
    mode.add_argument("--evaluate", type=str, metavar="FILE", help="Score answers against 'expected' in FILE")
    mode.add_argument("--build-only", action="store_true", help="Build/refresh the index and exit")
    mode.add_argument("--stats", action="store_true", help="Print index statistics and exit")
    mode.add_argument("--seed-demo", type=str, metavar="DIR", help="Copy the sample data to DIR (SQLite file) and exit")
    mode.add_argument("--generate-sample-pdfs", action="store_true",
                      help=f"Regenerate the sample PDFs in {SAMPLE_DIR / 'pdfs'} (needs reportlab) and exit")
    src = p.add_argument_group("sources and settings")
    src.add_argument("--config", type=str, help="JSON config file (e.g. RAG_Pipeline_Graph/config_graph.json)")
    src.add_argument("--db", type=str, help="Priority-1 database: .sqlite/.db file, .sql script or sqlite:/// URL")
    src.add_argument("--wiki-dir", type=str, help="Priority-2 Markdown wiki/runbook folder")
    src.add_argument("--pdf-dir", type=str, help="Priority-3 PDF folder")
    src.add_argument("--index-dir", type=str, help="Where the graph + vector index is cached")
    src.add_argument("--llm", choices=LLM_BACKENDS, help="Answer generator (default: extractive)")
    src.add_argument("--embedding-backend", choices=EMBEDDING_BACKENDS)
    src.add_argument("--vector-backend", choices=VECTOR_BACKENDS)
    src.add_argument("--rebuild-index", action="store_true", help="Force a full rebuild of graph + vector index")
    out = p.add_argument_group("output")
    out.add_argument("--output", type=str, help="Write JSON results to this file")
    out.add_argument("--format", choices=("json", "text"), default="json", help="stdout format (default: json)")
    out.add_argument("--show-prompt", action="store_true", help="Include the fused prompt in the output")
    out.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ...")
    return p


def _emit(results: List[QueryResult], args) -> None:
    data = [r.to_dict() for r in results]
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logging.getLogger("graph_rag").info("Results written to %s", args.output)
    if args.format == "text":
        print("\n\n".join(format_result(r) for r in results))
    elif not args.output:
        print(json.dumps(data, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv_if_available()

    if args.seed_demo:
        print(json.dumps(seed_demo_data(args.seed_demo), indent=2))
        return 0
    if args.generate_sample_pdfs:
        for path in generate_sample_pdfs():
            print(path)
        return 0

    config = load_config(args.config, db_path=args.db, wiki_dir=args.wiki_dir, pdf_dir=args.pdf_dir,
                         index_dir=args.index_dir, llm_backend=args.llm,
                         embedding_backend=args.embedding_backend, vector_backend=args.vector_backend)
    if args.show_prompt:
        config["include_prompt"] = True
    logging.basicConfig(level=(args.log_level or config.get("log_level", "INFO")).upper(),
                        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s", stream=sys.stderr)

    pipe = GraphRAGPipeline(config).build(rebuild=args.rebuild_index)
    if args.build_only or args.stats:
        print(json.dumps(pipe.stats(), indent=2, default=str))
        return 0
    if args.evaluate:
        report = evaluate(pipe, load_queries(args.evaluate))
        print(format_report(report))
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        return 0 if report["passed"] == report["scored"] else 1
    if args.query:
        _emit([pipe.ask(args.query)], args)
        return 0
    if args.queries:
        _emit(pipe.ask_many(item["query"] for item in load_queries(args.queries)), args)
        return 0

    print("Graph-RAG interactive mode. Type a question, or 'exit' to quit.", file=sys.stderr)
    results = []
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if q.lower() in ("exit", "quit"):
            break
        if q:
            res = pipe.ask(q)
            results.append(res)
            print(format_result(res) + "\n")
    if args.output and results:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
    return 0
