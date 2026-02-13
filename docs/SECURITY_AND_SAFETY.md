# Security & Safety — Synthetic Data Generator

This document complements `SECURITY.md` with a reality-aligned threat model and
synthetic-data safety guidance.

## Reality-aligned scope

Today:
- `website/server/index.js` is a demo API that stores jobs/domains in memory.
- The Python scripts in `Pre-Work/` can generate real text datasets.

This means:
- web risk is mostly “demo API” risk (no auth, in-memory state)
- generation risk is mostly “content safety” + “prompting hygiene”

## Threat model (high level)

### If you run the web server locally

Primary risks:
- unexpected exposure if you bind to `0.0.0.0` and share the port
- denial-of-service (too many jobs) because everything is in memory
- untrusted input (domain configs) stored without sanitization beyond basic checks

### If you generate datasets for sensitive domains

Primary risks:
- generating harmful advice (medical/legal/financial)
- including personal data (PII) by accident
- producing biased or discriminatory content
- producing content that looks authoritative but is wrong

## Practical safety guidance

1. **Treat outputs as untrusted**
   - Validate schemas, scan for red flags, and sample manually.
2. **Add “do not provide advice” constraints**
   - Especially for healthcare/legal/finance domains.
3. **Filter for PII**
   - Names, phone numbers, addresses, emails, SSNs, etc.
4. **Track provenance**
   - Save model/provider and parameters in `metadata`.
5. **Avoid copyrighted prompts**
   - Don’t paste books/articles verbatim into prompts.

## Reconciling with `SECURITY.md`

`SECURITY.md` contains recommendations and analysis items (rate limiting, path traversal prevention, auth).
Some are **not implemented** in the demo server. Use this checklist if you harden the server:

- Add auth (JWT or API keys)
- Add durable storage (DB)
- Add rate limiting (Redis-backed for multi-instance)
- Add safe file streaming for downloads (no arbitrary paths)
- Add structured logging and monitoring

## Ethical note

Synthetic data is not automatically “safe”. It can still encode real patterns and harm.
Use the generator responsibly and document dataset intent and limitations.

