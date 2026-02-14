# Setup Guide (Current Architecture)

Last updated: 2026-02-14

## Architecture

The repository is now structured as:
- `website/client/` - React + Vite UI
- `website/server/` - Express API, SQLite schema, download streaming, SSE
- `worker/` - Python polling worker that claims jobs and runs generation
- `Pre-Work/universal_dataset_generator.py` - integrated generator (config-driven, callback-ready)

SQLite is the source of truth for jobs/domains/events:
- `website/server/data/synthgen.sqlite`

Artifacts are written to:
- `website/server/data/outputs/<jobId>/`

## Local Setup

## 1) API
```bash
cd website/server
npm install
npm start
```

## 2) Worker
```bash
cd worker
python main.py
```

Worker defaults:
- `MAX_CONCURRENT_JOBS=1`
- `POLL_INTERVAL_MS=1000`
- `PROGRESS_UPDATE_INTERVAL_MS=2000`

## 3) Client
```bash
cd website/client
npm install
npm run dev
```

## Environment Variables

Common API env vars:
- `PORT` (default `3001`)
- `DATA_DIR` (default `website/server/data`)
- `SQLITE_PATH` (default `<DATA_DIR>/synthgen.sqlite`)
- `OUTPUTS_DIR` (default `<DATA_DIR>/outputs`)
- `MAX_BODY_SIZE` (default `50kb`)
- `AUTH_MODE` = `none|api_key` (default `none`)
- `API_KEYS` (comma-separated when `AUTH_MODE=api_key`)
- `JOB_RETENTION_DAYS` (default `7`)

Validation bounds:
- `TARGET_COUNT_MIN` (default `100`)
- `TARGET_COUNT_MAX` (default `100000`)
- `BATCH_SIZE_MIN` (default `1`)
- `BATCH_SIZE_MAX` (default `50`)

Rate limiting:
- `RATE_LIMIT_WINDOW_MS`, `RATE_LIMIT_MAX`
- `GENERATE_RATE_LIMIT_WINDOW_MS`, `GENERATE_RATE_LIMIT_MAX`
- `DOWNLOAD_RATE_LIMIT_WINDOW_MS`, `DOWNLOAD_RATE_LIMIT_MAX`

Worker env vars:
- `DATA_DIR`, `SQLITE_PATH`, `OUTPUTS_DIR`
- `POLL_INTERVAL_MS`
- `PROGRESS_UPDATE_INTERVAL_MS`
- `MAX_CONCURRENT_JOBS`

Generator install policy:
- Runtime dependency auto-install is **off by default**
- Enable only if needed via `SYNTHGEN_AUTO_INSTALL=1` or `--auto-install`

## Docker Compose

Use the included compose file for reproducible local deployment:

```bash
docker compose up --build
```

Services:
- `api` (port `3001`)
- `worker` (shared `/data` mount)

## Verification Checklist

1. API health:
```bash
curl http://localhost:3001/api/health
```

2. Queue a job:
```bash
curl -X POST http://localhost:3001/api/generate \
  -H "Content-Type: application/json" \
  -d '{"domain":"technology","targetCount":100,"batchSize":10,"outputFormat":"jsonl","provider":"mock","parseMode":"qa"}'
```

3. Wait for completion and download:
```bash
curl http://localhost:3001/api/jobs/<jobId>
curl -O http://localhost:3001/api/downloads/<jobId>/jsonl
```
