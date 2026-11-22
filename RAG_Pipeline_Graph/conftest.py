"""Make `graph_rag`, `rag_pipeline_graph` and the tests' helpers importable from any working directory."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
