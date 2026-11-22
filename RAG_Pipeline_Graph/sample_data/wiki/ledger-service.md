---
entity: ledger-service
title: Ledger Service
last_updated: 2026-03-01
---
# Ledger Service

This page is written as prose on purpose. It has no key/value lines, so facts are taken from
relation patterns in the text instead.

The ledger-service is owned by the Core Ledger team. Amara Chen is the on-call engineer for
ledger-service this quarter. The ledger-service depends on fraud-service for risk scores.

## Runbook: replaying ledger events

To replay ledger events after an outage, pause the consumers, reset the consumer offset to the
last checkpoint, and resume. Balances are recomputed from the event log, so replays are safe.
