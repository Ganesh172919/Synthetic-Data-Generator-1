# Architecture — How the System Fits Together

This document describes the architecture **as implemented** (v2.0).

## High-level mental model

There are three processes:

1. **React client** (`website/client/`) — UI on port 5173
2. **Express API** (`website/server/`) — Control plane on port 3001, SQLite-backed
3. **Python Worker** (`worker/`) — Execution plane, polls SQLite for jobs

Plus the **Provider Abstraction Layer** (`Pre-Work/providers/`) which gives
the generator access to 10 different LLM backends through a unified interface.

## Architecture diagram

```
┌─────────────────────────┐     HTTP + SSE      ┌──────────────────────────────┐
│   Browser (React UI)     │  /api/* → :3001      │  Express API (Control Plane) │
│ website/client           ├─────────────────────►│ website/server/src/server.js │
│   :5173                  │◄─────────────────────┤   :3001                      │
└─────────────────────────┘   SSE progress stream │                              │
                                                   │  SQLite (better-sqlite3)     │
                                                   │  - jobs table                │
                                                   │  - job_events table          │
                                                   │  - domains table             │
                                                   │  - _migrations table         │
                                                   └──────────────┬───────────────┘
                                                                  │
                                                    SQLite (shared filesystem)
                                                                  │
                                                                  ▼
                                                   ┌──────────────────────────────┐
                                                   │  Python Worker (Execution)    │
                                                   │  worker/main.py               │
                                                   │  Concurrent: 1-16 threads     │
                                                   │                               │
                                                   │  ┌─────────────────────────┐  │
                                                   │  │ Provider Factory        │  │
                                                   │  │ Pre-Work/providers/     │  │
                                                   │  │                         │  │
                                                   │  │ mock        │ huggingface│  │
                                                   │  │ openai      │ anthropic  │  │
                                                   │  │ google      │ ollama     │  │
                                                   │  │ azure_openai│ groq       │  │
                                                   │  │ together    │ custom     │  │
                                                   │  └─────────────────────────┘  │
                                                   │                               │
                                                   │  ┌─────────────────────────┐  │
                                                   │  │ UniversalGenerator      │  │
                                                   │  │ - PromptBuilder         │  │
                                                   │  │ - ResponseParser        │  │
                                                   │  │ - AsyncFileWriter       │  │
                                                   │  │ - ThreadSafeSet (dedup) │  │
                                                   │  │ - Checkpoint/Resume     │  │
                                                   │  └─────────────────────────┘  │
                                                   └──────────────┬───────────────┘
                                                                  │ writes
                                                                  ▼
                                                   ┌──────────────────────────────┐
                                                   │  Dataset files               │
                                                   │  website/server/data/outputs/ │
                                                   │  *.{jsonl, csv, json}        │
                                                   └──────────────────────────────┘
```

## Components and responsibilities

### React client (`website/client/`)
- Renders pages:
  - **Dashboard**: start generation with provider/language/mode selection, real-time SSE progress, download
  - **Templates**: browse 18 pre-built templates across 17 domains
  - **Domain Builder**: create/edit/delete custom domain configurations
  - **Documentation**: built-in docs view
- Uses `fetch('/api/...')` with Vite proxy to avoid CORS in development.

### Express API (`website/server/`)
- 21 REST endpoints covering:
  - Health, templates, providers (with health checks)
  - Job lifecycle: create, list, get, stop, retry, delete
  - SSE event streaming for real-time progress
  - Download and preview of generated datasets
  - Domain CRUD (create, read, update, delete)
  - Aggregate metrics
- SQLite with WAL mode for concurrent reads between API and Worker
- Database migrations system for schema versioning
- Rate limiting (3 tiers: general, generate, download)
- Optional API key authentication

### Python Worker (`worker/`)
- Polls SQLite every 1 second for `queued` jobs
- Claims jobs using `BEGIN IMMEDIATE` transactions (safe concurrency)
- Supports configurable concurrent execution (1-16 threads via `MAX_CONCURRENT_JOBS`)
- Recovers stale `running` jobs on restart
- Language instruction injection for non-English generation

### Provider Abstraction Layer (`Pre-Work/providers/`)
- `BaseProvider` ABC with `generate()`, `health_check()`, `get_models()`
- Factory pattern with `@register_provider` decorator
- 10 providers: mock, openai, huggingface, anthropic, google, ollama, azure_openai, groq, together, custom
- Custom endpoint support for any OpenAI-compatible API (vLLM, TGI, llama.cpp)

### Generator (`Pre-Work/universal_dataset_generator.py`)
- Backward-compatible `ModelProvider` enum (3 legacy + 7 new providers)
- `ProviderBackend` wrapper for new providers via the abstraction layer
- 11 parse modes: qa, text, json, instruction, conversation, classification, ner, summarization, translation, code, reasoning
- Checkpoint/resume, hash-based dedup, emergency save handlers

## Data flow (real generation)

1. UI sends `POST /api/generate` with domain, provider, language, targetCount, parseMode, etc.
2. API validates input, inserts job into SQLite with status `queued`.
3. Worker polls SQLite, claims the job, sets status to `running`.
4. Worker loads the appropriate provider via factory, builds prompt, generates in batches.
5. Worker updates SQLite progress + inserts events; API streams events via SSE to client.
6. Client renders real-time progress (generated count, rate, ETA, quality score).
7. When complete, client downloads via `GET /api/downloads/:jobId/:format`.

## Job state machine

```
                    ┌──────────┐   stop   ┌──────────┐
  POST /api/generate│  queued  ├─────────►│ stopped  │
  ─────────────────►└────┬─────┘          └────┬─────┘
                         │ worker claims        │ retry
                         ▼                      │
                    ┌──────────┐                │
                    │  running ├────────────────┘  (stop_requested flag)
                    └────┬─────┘
                         │
                         │ completed / failed
                         ▼
                    ┌──────────┐
                    │completed │  or  │ failed │
                    └──────────┘      └────────┘
```

## Key design decisions

- **SQLite with WAL mode**: enables concurrent read access between API and Worker without locking
- **Shared filesystem**: Docker volume mount (or local paths) for outputs and database
- **Job claiming with `BEGIN IMMEDIATE`**: prevents two workers from claiming the same job
- **SSE via polling**: API polls `job_events` table (simple, durable, no in-memory pub/sub needed)
- **Provider factory**: new LLM providers can be added by creating a single file in `Pre-Work/providers/`
- **Frozen config**: `config.js` exports an immutable object to prevent accidental mutation
