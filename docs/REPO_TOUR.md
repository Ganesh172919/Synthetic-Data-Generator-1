# Repository Tour — What’s Here and What You Learn

This is a file-by-file tour of **Synthetic Data Generator 1**. Use it as a learning map.

## Top-level files

- `Readme.md` — Main project overview (marketing + quick start + links)
- `QUICKSTART.md` — A fast “get it running” guide (web + CLI)
- `SETUP.md` — More detailed setup guide (local + Colab)
- `SECURITY.md` — Security posture and recommendations (includes some aspirational items)
- `LICENSE` — MIT License

## Python generators (`Pre-Work/`)

These scripts are where “real” dataset generation happens.

- `Pre-Work/universal_dataset_generator.py`
  - What you learn:
    - Provider abstraction (HuggingFace vs OpenAI vs mock)
    - Batching and throughput vs quality tradeoffs
    - Checkpointing + resume logic
    - Async buffered file writing
    - Deduplication (hash-based)
  - Typical usage:
    - Interactive CLI prompts
    - Or `--prompt`, `--size`, `--mode`, `--format` CLI args

- `Pre-Work/financial_education_generator_ultra.py`
  - What you learn:
    - High-throughput “mega-batch” prompting strategies
    - Parallel generation patterns
    - Emergency-save patterns for Colab sessions
    - Aggressive performance tuning knobs

- `Pre-Work/OPTIMIZATION_GUIDE.md`
  - What you learn:
    - Why batching and async I/O matter for LLM throughput
    - How to tune speed vs quality by hardware tier

## Web platform (`website/`)

This is a full React + Express demo application.

- `website/README.md` — How to run the web platform
- `website/.gitignore` — Web-specific ignore patterns and why they exist

### Backend API (`website/server/`)

- `website/server/index.js`
  - What you learn:
    - Minimal Express API design for “jobs”
    - Input validation patterns
    - In-memory job state using `Map`
    - Simulated progress using `setInterval`
  - Reality note:
    - This is a **demo backend** — it does not call the Python generators yet.

- `website/server/package.json` / `website/server/package-lock.json`
  - What you learn:
    - Dependency graph and script entrypoints
    - What lockfiles are and why they’re tool-managed

### Frontend client (`website/client/`)

- `website/client/src/main.jsx` — React entry point
- `website/client/src/App.jsx` — Router + providers wiring
- `website/client/src/pages/*` — Pages (Dashboard, Templates, Domain Builder, etc.)
- `website/client/src/components/*` — Shared components (Navbar, Footer)
- `website/client/src/components/ui/*` — Reusable UI components
- `website/client/src/services/api.js` — API wrapper (fetch + error handling)
- `website/client/src/hooks/*` — Custom hooks (theme + scroll animations)
- `website/client/src/styles/*` — Design tokens and reusable CSS

## Where to go next (exercises)

1. Run the web UI + server and inspect the network calls in DevTools.
2. Modify the demo backend to persist jobs to disk (simple JSON file) and document the tradeoffs.
3. Replace the backend mock dataset generator with a call to `Pre-Work/universal_dataset_generator.py`
   using a job queue (even a minimal one) and observe how the UI needs to change to poll real progress.

