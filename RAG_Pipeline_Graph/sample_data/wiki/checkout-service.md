---
entity: checkout-service
title: Checkout Service
last_updated: 2026-06-02
---
# Checkout Service

**Owner team:** Web Commerce
**On-call engineer:** Luis Ortega
**Version:** 5.2.0
**Depends on:** payments-service

The checkout-service renders the basket and hands payment authorisation to payments-service.

## Runbook: clearing a stuck basket cache

To clear a stuck basket cache on checkout-service, drain one node at a time from the load
balancer, flush its basket cache, and add it back once health checks pass. Never flush every
node at once because the cold cache doubles payments traffic.
