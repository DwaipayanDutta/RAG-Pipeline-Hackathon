"""
Priority 1: structured service registry (CMDB) -> SPOC quads with context="cmdb".

`DatabaseSource` is the abstraction; `SQLiteSource` is the implementation used locally. A
PostgreSQL (or any DB-API) source only needs to implement `connect()` and `placeholder` -
see README "Adding a PostgreSQL source". Rows can also be re-read at query time
(`facts_for`) so Priority-1 answers reflect the live database, not the cached index.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..graph_store import QuadStore
from ..models import CTX_CMDB, Quad
from .wiki import normalize_key

logger = logging.getLogger("graph_rag.database")

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _ident(name: str) -> str:
    """Validate a SQL identifier (table/column names cannot be bound as parameters)."""
    if not name or not _IDENT_RE.fullmatch(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return name


class DatabaseSource(ABC):
    """A read-only relational source of Priority-1 facts."""

    #: DB-API parameter placeholder ("?" for sqlite3, "%s" for psycopg).
    placeholder = "?"

    def __init__(self, label: str):
        self.label = label

    @abstractmethod
    def connect(self):
        """Return a new DB-API 2.0 connection."""

    #: Path of the file backing this source (used for index fingerprinting), if any.
    file_path: Optional[str] = None

    def fetch_rows(self, table: str, where_column: Optional[str] = None,
                   values: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        sql = f"SELECT * FROM {_ident(table)}"
        params: List[Any] = []
        if where_column is not None:
            values = list(values or [])
            if not values:
                return []
            sql += f" WHERE LOWER({_ident(where_column)}) IN ({', '.join([self.placeholder] * len(values))})"
            params = [str(v).lower() for v in values]
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        finally:
            conn.close()


class SQLiteSource(DatabaseSource):
    """
    SQLite database. Accepts a `.sqlite`/`.db` file, a `sqlite:///path` URL, or a `.sql`
    script that is executed into a fresh in-memory database on every connection (handy for
    committed, diff-able sample data: edit the script and the next query sees the change).
    """

    def __init__(self, path: str):
        path = path[len("sqlite:///"):] if path.startswith("sqlite:///") else path
        self.path = path
        self.file_path = path
        self.is_script = path.lower().endswith(".sql")
        if not Path(path).is_file():
            raise FileNotFoundError(f"Database not found: {path}")
        super().__init__(Path(path).name)

    def connect(self):
        if self.is_script:
            conn = sqlite3.connect(":memory:")
            conn.executescript(Path(self.path).read_text(encoding="utf-8"))
            return conn
        return sqlite3.connect(f"{Path(self.path).resolve().as_uri()}?mode=ro", uri=True)


def open_database(db_path: str) -> DatabaseSource:
    """Factory: pick a DatabaseSource implementation from a path or URL."""
    if db_path.startswith(("postgres://", "postgresql://")):
        raise NotImplementedError(
            "PostgreSQL is not implemented yet. Subclass graph_rag.sources.database.DatabaseSource "
            "(implement connect() with psycopg and set placeholder='%s'); see README.")
    if "://" in db_path and not db_path.startswith("sqlite:///"):
        raise ValueError(f"Unsupported database URL: {db_path}")
    return SQLiteSource(db_path)


def table_specs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """`db_tables` from config, or the legacy single-table (`db_table`, `db_key_column`) form."""
    specs = config.get("db_tables")
    if specs:
        return specs
    return [{"table": config.get("db_table", "services"), "key_column": config.get("db_key_column", "service_name")}]


def rows_to_quads(spec: Dict[str, Any], rows: List[Dict[str, Any]], source_label: str) -> List[Quad]:
    """
    Convert rows of one table into quads.

    Entity table: subject = key column; one quad per non-null column (column name normalised
    to a predicate); optional `entity_type` adds a `type` quad; `timestamp_column` becomes the
    quad timestamp instead of a fact. Edge table: one `edge_predicate` quad per row.
    """
    key = spec["key_column"]
    ts_col = spec.get("timestamp_column")
    edge = spec.get("edge_predicate")
    src = f"{source_label}:{spec['table']}"
    out: List[Quad] = []
    for row in rows:
        if row.get(key) is None:
            continue
        subject = str(row[key])
        ts = str(row.get(ts_col) or "") if ts_col else ""
        if edge:
            obj = row.get(spec["object_column"])
            if obj is not None:
                out.append(Quad(subject, edge, str(obj), CTX_CMDB, src, timestamp=ts,
                                evidence=f"{spec['table']} row"))
            continue
        if spec.get("entity_type"):
            out.append(Quad(subject, "type", spec["entity_type"], CTX_CMDB, src, timestamp=ts))
        for col, val in row.items():
            if col in (key, ts_col) or val is None or str(val).strip() == "":
                continue
            out.append(Quad(subject, normalize_key(col), str(val), CTX_CMDB, src, timestamp=ts,
                            evidence=f"{spec['table']}.{col}"))
    return out


def ingest_database(db_path: str, store: QuadStore, table: str = "services",
                    key_column: str = "service_name",
                    tables: Optional[List[Dict[str, Any]]] = None) -> int:
    """Load every configured table into the store. Returns the number of new quads."""
    source = open_database(db_path)
    specs = tables or [{"table": table, "key_column": key_column}]
    added = 0
    for spec in specs:
        rows = source.fetch_rows(spec["table"])
        added += store.add_many(rows_to_quads(spec, rows, source.label))
        logger.info("cmdb: %s rows from %s", len(rows), spec["table"])
    logger.info("cmdb: %d quads total", added)
    return added


def facts_for(source: DatabaseSource, specs: List[Dict[str, Any]], subjects: Iterable[str]) -> Dict[str, List[Quad]]:
    """Live lookup: current Priority-1 quads for the given subjects, keyed by subject (lowercase)."""
    subjects = [s for s in subjects if s]
    out: Dict[str, List[Quad]] = {}
    for spec in specs:
        try:
            rows = source.fetch_rows(spec["table"], spec["key_column"], subjects)
        except Exception as e:                       # a missing table must not break querying
            logger.warning("live refresh of %s failed: %s", spec["table"], e)
            continue
        for q in rows_to_quads(spec, rows, source.label):
            out.setdefault(q.subject.lower(), []).append(q)
    return out
