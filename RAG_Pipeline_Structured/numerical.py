import re
from typing import List, Dict, Any

def extract_numbers(text: str) -> List[Dict[str, Any]]:
    # Find all numbers with optional commas, decimals, and units like crore, lakh, %
    pattern = r"(\d+[\.,]?\d*)\s*(crore|lakh|million|billion|%)?"
    matches = re.findall(pattern, text)
    results = []
    for num, unit in matches:
        # Clean number
        num_clean = num.replace(",", "")
        if "." in num_clean:
            val = float(num_clean)
        else:
            val = int(num_clean)
        results.append({"value": val, "unit": unit if unit else "number"})
    return results

def normalize_crore(val, unit):
    if unit == "crore":
        return val * 1e7
    elif unit == "lakh":
        return val * 1e5
    elif unit == "million":
        return val * 1e6
    elif unit == "billion":
        return val * 1e9
    else:
        return val  # assume raw

def filter_by_threshold(candidates: List[Dict], threshold: float, metric: str = "revenue") -> List:
    """Not implemented in this package (see handle_threshold_query in RAG_Pipeline_Simple/rag_pipeline.py)."""
    raise NotImplementedError("Threshold filtering is implemented in RAG_Pipeline_Simple/rag_pipeline.py "
                              "(handle_threshold_query); the structured package does not use it yet.")