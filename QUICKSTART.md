# Quick Start (Durable API + Worker)

This project now runs as a two-service backend:
- `website/server` (Node/Express API + SQLite metadata)
- `worker/main.py` (Python job worker + real artifact generation)

The frontend remains at `website/client`.

## 1) Prerequisites

- Node.js 20+
- Python 3.10+
- npm

Optional:
- Docker + Docker Compose

## 2) Install Dependencies

```bash
# API
cd website/server
npm install

# Frontend
cd ../client
npm install

# Worker runtime deps (optional for mock-only local runs)
cd ../../
pip install -r worker/requirements.txt
```

## 3) Run Locally (3 Terminals)

Terminal A (API):
```bash
cd website/server
npm start
```

Terminal B (Worker):
```bash
cd worker
python main.py
```

Terminal C (Client):
```bash
cd website/client
npm run dev
```

Open:
- UI: `http://localhost:5173`
- API: `http://localhost:3001/api`

## 4) Minimal API Flow

Start a job:
```bash
curl -X POST http://localhost:3001/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "technology",
    "targetCount": 1000,
    "batchSize": 25,
    "outputFormat": "jsonl",
    "provider": "mock",
    "parseMode": "qa"
  }'
```

Check status:
```bash
curl http://localhost:3001/api/jobs/<jobId>
```

Download artifact:
```bash
curl -O http://localhost:3001/api/downloads/<jobId>/jsonl
```

## 5) Docker Compose (Production-like Local Stack)

```bash
docker compose up --build
```

Shared data is persisted under:
- `website/server/data/synthgen.sqlite`
- `website/server/data/outputs/<jobId>/...`

## 6) Supported Output Formats

- `jsonl`
- `csv`
- `json`

`parquet` is intentionally not exposed in this cycle.
