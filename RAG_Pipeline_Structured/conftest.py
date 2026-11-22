"""Put the repository root on sys.path so `import RAG_Pipeline_Structured` works from any directory."""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
