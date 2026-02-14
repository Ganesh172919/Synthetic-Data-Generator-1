# Security Baseline (Current Implementation)

Last updated: 2026-02-14

This file documents what is actually implemented in the current API/worker stack.

## Implemented Controls

## 1) Input Validation

`POST /api/generate` validates:
- `domain` against allowlist:
  - `financial|healthcare|legal|technology|science|education|custom`
- `targetCount` within configured min/max
- `batchSize` within configured min/max
- `outputFormat` in `jsonl|csv|json`
- `provider` in `mock|openai|huggingface`
- `parseMode` in `qa|text|json`

Domain endpoints validate required fields for persistence.

## 2) Rate Limiting

Express rate limiting is enabled with separate buckets:
- General `/api/*`
- `POST /api/generate`
- `GET /api/downloads/:jobId/:format`

All limits are environment-configurable.

## 3) Optional API-Key Auth

Auth modes:
- `AUTH_MODE=none`
- `AUTH_MODE=api_key`

When API-key mode is enabled:
- Accepts `x-api-key` or `Authorization: Bearer <key>`
- Keys come from `API_KEYS` (comma-separated)
- Localhost requests are allowed without a key (for local dev)

## 4) Request Size Limits

JSON body parser limit is configured via:
- `MAX_BODY_SIZE` (default `50kb`)

## 5) Safe Artifact Downloads

`GET /api/downloads/:jobId/:format`:
- Whitelists format (`jsonl|csv|json`)
- Resolves output path from known job metadata
- Verifies resolved paths remain inside `OUTPUTS_DIR`
- Streams files from disk (no in-memory synthetic payloads)

## 6) Durable Audit Trail

SQLite tables:
- `jobs`
- `job_events`
- `domains`

Job progress/status changes are persisted and exposed via SSE.

## 7) Structured Logging + Request IDs

API uses:
- `pino`
- `pino-http`

Request IDs are generated (or propagated from `x-request-id`).

## 8) Retention Cleanup

Old terminal jobs (`completed|failed|stopped`) are pruned based on:
- `JOB_RETENTION_DAYS` (default `7`)

Cleanup removes DB rows and artifact directories.

## Remaining Gaps / Next Hardening Steps

- No user accounts or RBAC yet (API-key mode is coarse-grained)
- No TLS termination in-repo (expected at ingress/proxy layer)
- No centralized SIEM integration
- No automated secret scanning in runtime path
- No immutable object storage for artifacts yet

## Reporting

If you find a vulnerability, open a private security advisory or issue with:
- reproduction steps
- expected vs actual behavior
- impact scope
