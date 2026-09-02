import re
from typing import Dict

def classify_query(query: str) -> str:
    # Heuristic classification
    query_lower = query.lower()
    if "exceeding" in query_lower or "greater than" in query_lower or "above" in query_lower:
        return "threshold"
    if "growth" in query_lower or "increase" in query_lower or "%" in query:
        return "numerical"
    if "compare" in query_lower or "vs" in query_lower or "versus" in query_lower:
        return "comparison"
    if "table" in query_lower or "financial" in query_lower:
        return "table"
    if "policy" in query_lower or "governance" in query_lower or "risk" in query_lower:
        return "policy"
    if "how many" in query_lower or "how much" in query_lower:
        return "factual"
    if "what were" in query_lower or "what is" in query_lower:
        return "semantic"
    return "semantic"