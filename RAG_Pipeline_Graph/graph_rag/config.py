"""
Configuration: defaults, JSON config files, environment overrides and path resolution.

Precedence (lowest to highest):
    DEFAULT_CONFIG  <  --config JSON file  <  RAG_GRAPH_* environment variables  <  CLI flags

Relative paths are resolved so the pipeline behaves the same from any working directory:
    * defaults                    -> relative to the RAG_Pipeline_Graph/ folder
    * values from a config file   -> relative to that config file's folder
    * environment / CLI overrides -> relative to the current working directory
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

#: The RAG_Pipeline_Graph/ folder (parent of this package).
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

#: Config keys that hold filesystem paths (resolved to absolute paths on load).
PATH_KEYS = ("db_path", "wiki_dir", "pdf_dir", "index_dir")

LLM_BACKENDS = ("extractive", "flan-t5")
EMBEDDING_BACKENDS = ("auto", "sentence-transformers", "hash")
VECTOR_BACKENDS = ("auto", "chroma", "memory")

DEFAULT_CONFIG: Dict[str, Any] = {
    # ---- sources ---------------------------------------------------------------------------
    # Priority 1: a SQLite file (.sqlite/.db), a SQL seed script (.sql, loaded into an in-memory
    # SQLite database on every read), or a sqlite:/// URL.
    "db_path": "sample_data/cmdb.sql",
    # Tables to turn into SPOC quads. "Entity" tables: one subject per row, one fact per column.
    # "Edge" tables: one relationship per row (edge_predicate + object_column).
    "db_tables": [
        {"table": "services", "key_column": "service_name", "entity_type": "service",
         "timestamp_column": "updated_at"},
        {"table": "teams", "key_column": "team_name", "entity_type": "team",
         "timestamp_column": "updated_at"},
        {"table": "service_dependencies", "key_column": "service_name",
         "edge_predicate": "depends_on", "object_column": "depends_on_service"},
    ],
    # Legacy single-table settings (used only when db_tables is empty / null).
    "db_table": "services",
    "db_key_column": "service_name",
    "live_db_refresh": True,            # re-read Priority-1 rows for query entities at query time
    # Priority 2
    "wiki_dir": "sample_data/wiki",
    "index_wiki_passages": True,        # also embed wiki/runbook sections for semantic search
    "use_llm_wiki_extraction": False,   # FLAN-T5 triple extraction for pages with no parsable facts
    # Priority 3
    "pdf_dir": "sample_data/pdfs",
    "extract_pdf_facts": True,          # entity mentions + relation patterns from PDF text -> quads
    "chunk_size": 120,                  # words per chunk
    "chunk_overlap": 30,                # words shared by consecutive chunks of a page

    # ---- index -----------------------------------------------------------------------------
    "index_dir": ".graph_index",
    "embedding_backend": "auto",        # auto | sentence-transformers | hash
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "vector_backend": "auto",           # auto | chroma | memory

    # ---- retrieval -------------------------------------------------------------------------
    "known_entities": [],               # extra entity names to match in queries (jargon, aliases)
    "use_spacy": True,                  # optional spaCy NER on the query (if installed)
    "spacy_model": "en_core_web_sm",
    "graph_top_k": 12,                  # max facts per tier
    "graph_hops": 2,                    # multi-hop expansion depth (0 disables)
    "traversal_predicates": ["on_call_engineer", "owner_team", "depends_on", "member",
                             "manager", "escalation_contact"],
    "vector_fallback_top_k": 3,         # passages returned by semantic search
    "vector_max_distance": 0.75,        # cosine distance cutoff (0 = identical, 1 = orthogonal)
    "min_passage_overlap": 1,           # passage must share >= N content words with the query
    "dedupe_jaccard": 0.8,              # drop passages this similar to an already-selected one

    # ---- fusion ----------------------------------------------------------------------------
    "priority_weights": {"cmdb": 1.0, "wiki": 0.6, "pdf": 0.3},
    # Predicates that can hold only one true value per subject (disagreement => conflict).
    "single_valued_predicates": [
        "owner_team", "on_call_engineer", "status", "version", "environment", "sla_tier",
        "last_deployed_at", "manager", "slack_channel", "escalation_contact", "tier", "type",
    ],

    # ---- generation ------------------------------------------------------------------------
    "llm_backend": "extractive",        # extractive (deterministic, offline) | flan-t5
    "generation_model": "google/flan-t5-base",
    "entity_extraction_model": "google/flan-t5-base",
    "max_context_tokens": 512,
    "max_answer_tokens": 150,

    # ---- validation / output ---------------------------------------------------------------
    "grounding_min_coverage": 0.8,
    "include_prompt": False,
    "log_level": "INFO",
}

#: Environment variable -> (config key, type).
ENV_OVERRIDES = {
    "RAG_GRAPH_DB_PATH": ("db_path", str),
    "RAG_GRAPH_WIKI_DIR": ("wiki_dir", str),
    "RAG_GRAPH_PDF_DIR": ("pdf_dir", str),
    "RAG_GRAPH_INDEX_DIR": ("index_dir", str),
    "RAG_GRAPH_LLM_BACKEND": ("llm_backend", str),
    "RAG_GRAPH_GENERATION_MODEL": ("generation_model", str),
    "RAG_GRAPH_EMBEDDING_BACKEND": ("embedding_backend", str),
    "RAG_GRAPH_EMBEDDING_MODEL": ("embedding_model", str),
    "RAG_GRAPH_VECTOR_BACKEND": ("vector_backend", str),
    "RAG_GRAPH_LOG_LEVEL": ("log_level", str),
}


def _is_url(value: str) -> bool:
    return "://" in value


def resolve_path(value: Optional[str], base: Path) -> Optional[str]:
    """Make a relative filesystem path absolute against `base` (URLs and empty values pass through)."""
    if not value or _is_url(str(value)):
        return value
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else (base / p).resolve())


def _resolve_paths(values: Dict[str, Any], base: Path) -> Dict[str, Any]:
    out = dict(values)
    for key in PATH_KEYS:
        if key in out:
            out[key] = resolve_path(out[key], base)
    return out


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Raise ValueError for invalid enum-style settings and obviously wrong numbers."""
    for key, allowed in (("llm_backend", LLM_BACKENDS), ("embedding_backend", EMBEDDING_BACKENDS),
                         ("vector_backend", VECTOR_BACKENDS)):
        if config.get(key) not in allowed:
            raise ValueError(f"{key}={config.get(key)!r} is invalid; choose one of {allowed}")
    if int(config["chunk_size"]) <= 0 or not 0 <= int(config["chunk_overlap"]) < int(config["chunk_size"]):
        raise ValueError("chunk_size must be > 0 and 0 <= chunk_overlap < chunk_size")
    return config


def load_config(config_path: Optional[str] = None, use_env: bool = True,
                **overrides: Any) -> Dict[str, Any]:
    """
    Build the effective configuration.

    Args:
        config_path: optional JSON file whose keys override the defaults. A missing file raises
            FileNotFoundError (silently ignoring a typo'd path hides real mistakes).
        use_env: apply RAG_GRAPH_* environment overrides.
        **overrides: final overrides (paths relative to the current working directory).
    """
    config = _resolve_paths(copy.deepcopy(DEFAULT_CONFIG), PACKAGE_ROOT)
    if config_path:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(path, "r", encoding="utf-8") as f:
            config.update(_resolve_paths(json.load(f), path.resolve().parent))
    if use_env:
        for env, (key, typ) in ENV_OVERRIDES.items():
            if os.environ.get(env):
                config[key] = typ(os.environ[env])
        config = _resolve_paths(config, Path.cwd())
    if overrides:
        config.update(_resolve_paths({k: v for k, v in overrides.items() if v is not None}, Path.cwd()))
    return validate_config(config)


def load_dotenv_if_available(path: Optional[Path] = None) -> bool:
    """Load RAG_Pipeline_Graph/.env when python-dotenv is installed (optional convenience)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return bool(load_dotenv(path or PACKAGE_ROOT / ".env", override=False))
