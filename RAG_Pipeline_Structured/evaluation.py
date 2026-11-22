import json
from typing import List, Dict
from .pipeline import RAGPipeline
from .config import RAGConfig
from .observability import logger

def run_evaluation(config: RAGConfig) -> List[Dict]:
    pipeline = RAGPipeline(config)
    with open(config.queries_path, "r") as f:
        queries_data = json.load(f)
    results = []
    for item in queries_data:
        query = item["query"]
        result = pipeline.process(query)
        results.append(result)
    # Save
    with open(config.output_path, "w") as f:
        json.dump(results, f, indent=2)
    return results