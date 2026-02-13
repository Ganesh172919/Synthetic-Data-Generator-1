# Architecture — How the System Fits Together

This document explains the architecture “as implemented” and “as intended”.

## High-level mental model

There are two major parts:

1. **Python generators** (`Pre-Work/`) — real dataset generation logic
2. **Web platform** (`website/`) — a UI + API that currently behaves like a demo

### Reality-aligned diagram (current state)

```
┌─────────────────────────┐     HTTP (Vite dev proxy)     ┌──────────────────────────┐
│   Browser (React UI)     │  /api/* → http://localhost:3001 │  Express API (demo)      │
│ website/client           ├──────────────────────────────►│ website/server/index.js  │
└─────────────────────────┘                                └──────────────┬──────────┘
                                                                         (mock)
                                                                          │
                                                                          ▼
                                                           ┌──────────────────────────┐
                                                           │ generateMockDataset(...) │
                                                           │ (returns sample Q&A)     │
                                                           └──────────────────────────┘
```

### Intended diagram (extension point)

If you want “real generation” behind the web API, the natural extension is:

```
┌─────────────────────────┐      HTTP      ┌──────────────────────────┐
│   Browser (React UI)     ├──────────────►│  Express API              │
└─────────────────────────┘               └──────────────┬───────────┘
                                                        spawn/queue
                                                          │
                                                          ▼
                                             ┌──────────────────────────┐
                                             │ Python generator          │
                                             │ Pre-Work/universal_...py  │
                                             └──────────────┬──────────┘
                                                            writes
                                                              │
                                                              ▼
                                             ┌──────────────────────────┐
                                             │ Dataset files (JSONL/CSV) │
                                             └──────────────────────────┘
```

## Components and responsibilities

### React client (`website/client/`)
- Renders pages:
  - Dashboard: start generation, show progress, download when complete
  - Templates: browse template catalog (server-backed, with fallback)
  - Domain Builder: create a “custom domain” config (server stores it in memory)
  - Documentation: built-in docs view (static content in UI)
- Uses `fetch('/api/...')` with Vite proxy to avoid CORS in development.

### Express server (`website/server/index.js`)
- Implements:
  - `/api/templates` (static list)
  - `/api/generate` (creates a job + starts a simulated progress loop)
  - `/api/jobs/*` (job status/list/stop/delete)
  - `/api/downloads/:jobId/:format` (returns a mock dataset payload)
  - `/api/domains` (store custom domains in memory)
- Stores job state in `Map` and progress timers in another `Map`.

### Python generators (`Pre-Work/`)
- `universal_dataset_generator.py`:
  - provider abstraction + generation loop + async output writer
  - checkpointing and resume
  - deduplication
- `financial_education_generator_ultra.py`:
  - tuned prompting and throughput features for a finance Q&A dataset target
  - emergency save/download helpers

## Data flow (web UI, current demo)

1. UI sends `POST /api/generate` with domain, targetCount, batchSize, outputFormat.
2. Server creates a job and starts a timer that increments `generated`.
3. UI polls `GET /api/jobs/:jobId` (or simulates progress in demo mode).
4. When job is `completed`, UI downloads `GET /api/downloads/:jobId/:format`.

## Job state machine (web server)

```
             ┌──────────┐   stop   ┌──────────┐
POST generate│ running  ├─────────►│ stopped  │
────────────►└────┬─────┘          └──────────┘
                  │
                  │ generated >= targetCount
                  ▼
             ┌──────────┐
             │ completed │
             └──────────┘
```

## What’s missing (intentionally)

The demo server is not a production job runner. Missing pieces are documented
in `docs/WEB_PLATFORM.md` and `docs/SECURITY_AND_SAFETY.md`:

- durable storage for jobs/domains
- authentication/authorization
- a real queue/worker model
- real dataset generation integration with Python

