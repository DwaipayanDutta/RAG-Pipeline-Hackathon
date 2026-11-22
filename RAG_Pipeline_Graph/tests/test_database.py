"""Priority 1: SQLite service registry ingestion and live lookups."""
import sqlite3

import pytest

from graph_rag import QuadStore
from graph_rag.config import DEFAULT_CONFIG
from graph_rag.demo_data import SAMPLE_DIR
from graph_rag.sources.database import facts_for, ingest_database, open_database, rows_to_quads, table_specs

SPECS = DEFAULT_CONFIG["db_tables"]


def test_sql_script_and_sqlite_file_give_the_same_quads(cfg):
    a, b = QuadStore(), QuadStore()
    ingest_database(str(SAMPLE_DIR / "cmdb.sql"), a, tables=SPECS)
    ingest_database(cfg["db_path"], b, tables=SPECS)                    # seeded cmdb.sqlite
    strip = lambda s: {(q.subject, q.predicate, q.object) for q in s}   # noqa: E731
    assert strip(a) == strip(b) and len(a) > 30


def test_entity_rows_edges_types_and_timestamps(cfg):
    s = QuadStore()
    ingest_database(cfg["db_path"], s, tables=SPECS)
    pay = {(q.predicate, q.object) for q in s.about("payments-service", context="cmdb")}
    assert {("on_call_engineer", "Amara Chen"), ("status", "degraded"), ("owner_team", "Payments Platform"),
            ("type", "service"), ("depends_on", "ledger-service"), ("depends_on", "fraud-service")} <= pay
    assert not any(p == "updated_at" for p, _ in pay)                   # timestamp column is metadata
    q = s.about("payments-service", "on_call_engineer")[0]
    assert q.timestamp == "2026-11-21T08:15:00Z" and q.source == "cmdb.sqlite:services" and q.context == "cmdb"
    assert s.about("Payments Platform", "manager")[0].object == "Priya Nair"


def test_legacy_single_table_config(tmp_path):
    db = tmp_path / "x.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE services (service_name TEXT, on_call_engineer TEXT, status TEXT)")
    conn.execute("INSERT INTO services VALUES ('svc', 'Ann Lee', NULL)")
    conn.commit()
    conn.close()
    s = QuadStore()
    assert ingest_database(str(db), s, "services", "service_name") == 1  # NULL columns are skipped
    assert table_specs({"db_tables": None, "db_table": "t", "db_key_column": "k"}) == [{"table": "t", "key_column": "k"}]


def test_live_lookup_is_parameterised_and_case_insensitive(cfg):
    db = open_database(cfg["db_path"])
    live = facts_for(db, SPECS, ["PAYMENTS-SERVICE", "nope'; DROP TABLE services; --"])
    assert {q.object for q in live["payments-service"] if q.predicate == "on_call_engineer"} == {"Amara Chen"}
    assert len(db.fetch_rows("services")) == 4                         # injection attempt did nothing


def test_unsafe_identifiers_and_unsupported_databases(cfg, tmp_path):
    with pytest.raises(ValueError):
        ingest_database(cfg["db_path"], QuadStore(), "services; DROP TABLE x", "service_name")
    with pytest.raises(NotImplementedError):
        open_database("postgresql://user@host/db")
    with pytest.raises(FileNotFoundError):
        open_database(str(tmp_path / "missing.sqlite"))


def test_rows_to_quads_edge_table():
    spec = {"table": "deps", "key_column": "a", "edge_predicate": "depends_on", "object_column": "b"}
    qs = rows_to_quads(spec, [{"a": "x", "b": "y"}, {"a": "x", "b": None}], "db")
    assert [(q.subject, q.predicate, q.object) for q in qs] == [("x", "depends_on", "y")]
