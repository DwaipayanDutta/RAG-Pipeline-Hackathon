"""CLI for the structured pipeline: `python -m RAG_Pipeline_Structured {query,evaluate} ...`."""
import argparse
import json
import logging
from typing import List, Optional

from .config import RAGConfig


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m RAG_Pipeline_Structured")
    parser.add_argument("--config", default=None, help="JSON file with RAGConfig overrides")
    parser.add_argument("--pdf", default=None, help="PDF to index (default: the bundled Titan report)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Answer every query in queries.json")
    eval_parser.add_argument("--queries", default=None)
    eval_parser.add_argument("--output", default=None)

    query_parser = subparsers.add_parser("query", help="Answer one question")
    query_parser.add_argument("--question", required=True)

    args = parser.parse_args(argv)
    config = RAGConfig.from_json(args.config)
    if args.pdf:
        config.pdf_path = args.pdf
    logging.basicConfig(level=config.log_level)

    # Imported here so `--help` works without the heavy ML dependencies.
    from .evaluation import run_evaluation
    from .pipeline import RAGPipeline

    if args.command == "evaluate":
        if args.queries:
            config.queries_path = args.queries
        if args.output:
            config.output_path = args.output
        results = run_evaluation(config)
        print(f"Wrote {len(results)} results to {config.output_path}")
    else:
        result = RAGPipeline(config).process(args.question)
        print(json.dumps(result, indent=2))
    return 0
