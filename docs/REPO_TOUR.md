# Repository Tour — What's Here and What You Learn

This is a file-by-file tour of **Synthetic Data Generator 1**. Use it as a learning map.

## Top-level files

- `Readme.md` — Main project overview (architecture, quick start, API reference, roadmap)
- `IMPROVEMENT_PLAN.md` — Feature roadmap and implementation plan (Phase 1–7)
- `QUICKSTART.md` — A fast "get it running" guide
- `SETUP.md` — More detailed setup guide (local + Colab)
- `SECURITY.md` — Security posture and recommendations
- `docker-compose.yml` — Docker orchestration for API + Worker services
- `LICENSE` — MIT License

## Python generators (`Pre-Work/`)

These scripts are where "real" dataset generation happens.

- `Pre-Work/universal_dataset_generator.py`
  - What you learn:
    - Provider abstraction (10 LLM providers via factory pattern)
    - Batching and throughput vs quality tradeoffs
    - Checkpointing + resume logic
    - Async buffered file writing
    - Deduplication (hash-based)
    - 11 parse modes (qa, text, json, instruction, conversation, etc.)
  - Typical usage:
    - Interactive CLI prompts
    - Or `--prompt`, `--size`, `--mode`, `--format`, `--provider` CLI args

- `Pre-Work/financial_education_generator_ultra.py`
  - What you learn:
    - High-throughput "mega-batch" prompting strategies
    - Emergency-save patterns for Colab sessions
    - Aggressive performance tuning knobs

- `Pre-Work/providers/` — **LLM Provider Abstraction Layer**
  - `base.py` — `BaseProvider` ABC with `GenerationRequest`, `GenerationResponse`, `ProviderHealth` dataclasses and custom exceptions
  - `factory.py` — Provider registry with `@register_provider` decorator and `get_provider()` factory
  - `mock.py` — Deterministic test provider (no API key needed)
  - `openai_provider.py` — OpenAI GPT models via `openai` SDK
  - `anthropic.py` — Anthropic Claude models via `anthropic` SDK
  - `google.py` — Google Gemini models via `google-genai` SDK
  - `huggingface.py` — Local HuggingFace models via `transformers`
  - `ollama.py` — Local Ollama inference server
  - `azure_openai.py` — Azure-hosted OpenAI models
  - `groq.py` — Groq ultra-fast inference API
  - `together.py` — Together.ai open model hosting
  - `custom.py` — Any OpenAI-compatible endpoint (vLLM, TGI, llama.cpp)
  - What you learn:
    - Factory pattern with decorator-based registration
    - Unified interface across diverse LLM APIs
    - Provider-specific error handling and retry semantics

- `Pre-Work/OPTIMIZATION_GUIDE.md`
  - What you learn:
    - Why batching and async I/O matter for LLM throughput
    - How to tune speed vs quality by hardware tier

## Python worker (`worker/`)

The execution plane — polls SQLite for queued jobs and runs generation.

- `worker/main.py`
  - What you learn:
    - SQLite job claiming with `BEGIN IMMEDIATE` transactions
    - Concurrent job execution with `ThreadPoolExecutor` (1–16 threads)
    - Stale job recovery on worker restart
    - Language instruction injection for multilingual generation
    - Progress callback pattern for real-time updates
  - Key functions:
    - `worker_loop()` — Main entry, dispatches to single or concurrent mode
    - `claim_next_job()` / `claim_next_jobs()` — Transactional job claiming
    - `run_job()` — Full job lifecycle (load provider, generate, write, report)
    - `recover_stale_running_jobs()` — Re-queues orphaned running jobs

- `worker/requirements.txt` — Python dependencies including all provider SDKs
- `worker/Dockerfile` — Container for the worker service

## Web platform (`website/`)

A full React + Express application with SQLite-backed durable job management.

### Backend API (`website/server/`)

- `website/server/index.js`
  - Entry point — just imports `start()` from `src/server.js` and calls it
  - Includes uncaught exception and unhandled rejection handlers

- `website/server/src/server.js` — **Main API server (~1100 lines)**
  - 21 REST endpoints:
    - Health, templates, providers (with health checks)
    - Job lifecycle: create, list, get, stop, retry, delete
    - SSE event streaming for real-time progress
    - Download and preview of generated datasets
    - Domain CRUD (create, read, update, delete)
    - Aggregate metrics
  - What you learn:
    - Express middleware composition (pino-http, cors, auth, rate limiting)
    - SSE implementation with database polling
    - `asyncHandler` wrapper for async route error handling
    - Path-traversal-safe file serving
    - Input validation patterns

- `website/server/src/config.js` — Environment config parsing
  - What you learn:
    - `parseIntEnv` with min/max clamping
    - `parseCsvEnv` for comma-separated values
    - `Object.freeze()` for immutable config

- `website/server/src/db.js` — SQLite helpers and mappers
  - What you learn:
    - `better-sqlite3` synchronous API
    - WAL mode for concurrent reads
    - Row-to-API-object mapping (`toApiJob`, `toApiDomain`)
    - Path-traversal-safe directory deletion

- `website/server/src/migrations.js` — Database migration runner
  - What you learn:
    - Version-tracked schema migrations
    - `_migrations` table pattern
    - Transactional migration execution

- `website/server/src/templates.js` — 18 pre-built domain templates

- `website/server/src/__tests__/` — Test suite (63 tests)
  - `config.test.js` — Config parsing, clamping, validation, freeze
  - `db.test.js` — Database init, helpers, mappers, artifact deletion
  - `migrations.test.js` — Migration runner, schema creation, idempotency
  - `server.test.js` — All 21 API routes (integration tests using http.createServer)

- `website/server/package.json` — Dependencies: express, better-sqlite3, pino, cors, express-rate-limit

### Frontend client (`website/client/`)

- `website/client/src/main.jsx` — React entry point
- `website/client/src/App.jsx` — Router + providers wiring (ThemeProvider, ToastProvider, BrowserRouter)
- `website/client/src/pages/`
  - `LandingPage.jsx` — Marketing page with feature overview
  - `Dashboard.jsx` — Job management with provider/language/mode selectors, SSE progress, download
  - `Templates.jsx` — Template browser with 18 templates
  - `DomainBuilder.jsx` — Custom domain creation with CRUD
  - `Documentation.jsx` — Built-in API docs
  - `NotFound.jsx` — 404 page
- `website/client/src/components/`
  - `Navbar.jsx` — Responsive nav with theme toggle
  - `Footer.jsx` — Footer with links
  - `ui/` — Reusable components: Button, Card, Input, Modal, Toast, Badge, Progress, Skeleton
- `website/client/src/services/api.js` — API client with methods for all 21+ endpoints
- `website/client/src/hooks/`
  - `useTheme.jsx` — Dark/light theme context
  - `useIntersectionObserver.jsx` — Scroll-triggered animations
- `website/client/src/styles/` — Design tokens, animations, component CSS

## Infrastructure

- `docker-compose.yml` — Two services: `api` (Node.js on port 3001) and `worker` (Python), sharing `/data` volume
- `.github/workflows/`
  - `client.yml` — Builds React app on push/PR
  - `server.yml` — Runs `npm test` on push/PR
  - `python.yml` — Compile check + mock smoke test on push/PR

## Where to go next (exercises)

1. Run all three services (API, worker, client) and inspect the network calls in DevTools.
2. Add a new provider by creating a file in `Pre-Work/providers/` — see how the factory auto-discovers it.
3. Add a new template to `website/server/src/templates.js` and watch it appear in the UI.
4. Write a new test for an untested edge case in `server.test.js`.
5. Try generating a dataset in a non-English language and observe the language instruction injection.
