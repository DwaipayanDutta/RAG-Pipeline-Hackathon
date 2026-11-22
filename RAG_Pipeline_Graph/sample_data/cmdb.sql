-- Priority 1 sample data: the live service registry (CMDB).
-- Loaded into an in-memory SQLite database on every read, so edits here are picked up by the
-- next query (live refresh) and trigger an index rebuild on the next start (fingerprint change).
-- `python RAG_Pipeline_Graph/rag_pipeline_graph.py --seed-demo DIR` writes the same data to a
-- real SQLite file (DIR/cmdb.sqlite).

CREATE TABLE teams (
    team_name      TEXT PRIMARY KEY,
    manager        TEXT,
    slack_channel  TEXT,
    updated_at     TEXT
);

INSERT INTO teams VALUES
    ('Payments Platform', 'Priya Nair',  '#payments-platform', '2026-11-20T09:00:00Z'),
    ('Web Commerce',      'Daniel Kim',  '#web-commerce',      '2026-11-18T14:30:00Z'),
    ('Core Ledger',       'Sofia Rossi', '#core-ledger',       '2026-11-02T11:00:00Z');

CREATE TABLE services (
    service_name      TEXT PRIMARY KEY,
    owner_team        TEXT REFERENCES teams(team_name),
    on_call_engineer  TEXT,
    status            TEXT,
    version           TEXT,
    environment       TEXT,
    sla_tier          TEXT,
    last_deployed_at  TEXT,
    updated_at        TEXT
);

INSERT INTO services VALUES
    ('payments-service', 'Payments Platform', 'Amara Chen',     'degraded', '2.14.1', 'production', 'tier-1', '2026-11-19T16:42:00Z', '2026-11-21T08:15:00Z'),
    ('checkout-service', 'Web Commerce',      'Luis Ortega',    'healthy',  '5.3.0',  'production', 'tier-1', '2026-11-20T10:05:00Z', '2026-11-20T10:05:00Z'),
    ('ledger-service',   'Core Ledger',       'Amara Chen',     'healthy',  '1.9.4',  'production', 'tier-2', '2026-11-12T09:30:00Z', '2026-11-12T09:30:00Z'),
    ('fraud-service',    'Payments Platform', 'Kenji Watanabe', 'healthy',  '0.8.2',  'staging',    'tier-3', '2026-11-17T13:20:00Z', '2026-11-17T13:20:00Z');

CREATE TABLE service_dependencies (
    service_name        TEXT REFERENCES services(service_name),
    depends_on_service  TEXT REFERENCES services(service_name)
);

INSERT INTO service_dependencies VALUES
    ('payments-service', 'ledger-service'),
    ('payments-service', 'fraud-service'),
    ('checkout-service', 'payments-service');
