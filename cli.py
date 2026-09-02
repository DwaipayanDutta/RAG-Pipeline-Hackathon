# cli.py (in rag_pipeline package)
import argparse
from .config import RAGConfig
from .evaluation import run_evaluation
from .pipeline import RAGPipeline
import json

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--config", default="configs/default.json")

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("--question", required=True)
    query_parser.add_argument("--config", default="configs/default.json")

    args = parser.parse_args()
    # Load config (simplified)
    config = RAGConfig()  # would load from file

    if args.command == "evaluate":
        run_evaluation(config)
    elif args.command == "query":
        pipeline = RAGPipeline(config)
        result = pipeline.process(args.question)
        print(json.dumps(result, indent=2))