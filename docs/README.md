# Docs — Synthetic Data Generator (Educational Notes)

This `docs/` folder is an **educational companion** to the repository. The goal is to explain:

- What each part of the system does (Python generators + web demo)
- How data flows end-to-end
- What assumptions are baked into the current implementation
- How to extend the project safely and realistically

## Suggested reading order

1. `docs/REPO_TOUR.md` — File-by-file map of the project
2. `docs/ARCHITECTURE.md` — How the pieces fit together (with diagrams)
3. `docs/WEB_PLATFORM.md` — React + Vite client and Express API server (as implemented)
4. `docs/PYTHON_GENERATORS.md` — Universal generator + finance-optimized generator deep dive
5. `docs/DATASET_SCHEMA.md` — Output formats and schema examples (JSONL/JSON/CSV)
6. `docs/TROUBLESHOOTING.md` — Common setup/runtime issues (Windows, GPU, Colab)
7. `docs/SECURITY_AND_SAFETY.md` — Threat model + synthetic data safety guidance

## Important “reality-aligned” note

This repo includes:

- **Real generators** in `Pre-Work/` (Python scripts that can generate datasets)
- A **web platform demo** in `website/`:
  - `website/client/` is a real React app
  - `website/server/index.js` is a **demo API** that simulates jobs and returns mock datasets

In other words: **the web UI is real, but the backend generation is mocked** right now.
The docs call this out explicitly and show where you would connect the Python generator
if you wanted “production-like” behavior.

