---
entity: payments-service
title: Payments Service
last_updated: 2025-01-10
---
# Payments Service

- Owner team: Payments Platform
- On-call engineer: Jordan Blake
- Status: healthy
- Escalation contact: Priya Nair
- Depends on: ledger-service, fraud-service
- Tier: 1

The payments-service authorises card payments for checkout and records every capture in the
ledger. This page was last reviewed in January 2025; the service registry is the source of truth
for on-call and status.

## Runbook: rolling back a bad deploy

To roll back a bad deploy of payments-service, first freeze deploys in the release channel.
Then run the rollback job for the previous release tag and watch the error-rate dashboard
for ten minutes. If the error rate does not recover, page the escalation contact and open an
incident.

## Known issues

Connection pool exhaustion shows up as rising checkout latency. The fix is to restart the
payments workers and confirm the database connection pool size is at least 50.
