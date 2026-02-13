# Web Platform (React + Vite + Express) — Reality-Aligned Guide

This doc focuses on what the web app actually does today, and where to extend it.

## Layout

```
website/
├── client/   # React + Vite frontend
└── server/   # Express backend (demo job API)
```

## Development flow

1. Start the API server:
   - `cd website/server`
   - `npm start`
   - API: `http://localhost:3001/api`

2. Start the React app:
   - `cd website/client`
   - `npm run dev`
   - UI: `http://localhost:5173`

3. Vite proxy:
   - The client calls `/api/...`
   - Vite proxies those requests to `http://localhost:3001`
   - Config: `website/client/vite.config.js`

## Backend API endpoints (as implemented)

### Health
- `GET /api/health` → `{ status: "ok", timestamp: "..." }`

### Templates
- `GET /api/templates` → `{ templates: [...] }`
- `GET /api/templates/:id` → template or 404

### Generation jobs (demo)
- `POST /api/generate`
  - Validates `domain`, `targetCount`, `batchSize`, `outputFormat`
  - Creates a job in memory
  - Starts simulated progress via `setInterval`
- `GET /api/jobs/:jobId` → job object
- `GET /api/jobs` → list
- `POST /api/jobs/:jobId/stop` → stops the timer and marks job `stopped`
- `DELETE /api/jobs/:jobId` → deletes job

### Downloads (mock payload)
- `GET /api/downloads/:jobId/:format`
  - Only allowed if job is `completed`
  - `format` supports `jsonl`, `csv`, otherwise returns JSON

### Custom domains (in-memory demo)
- `POST /api/domains` → stores a custom domain config in memory
- `GET /api/domains` / `GET /api/domains/:id`

## Frontend data flow

The UI mostly follows this pattern:

```
User clicks “Start Generation”
  → POST /api/generate
  → store jobId in state
  → poll GET /api/jobs/:jobId (or simulate in UI)
  → when completed, GET /api/downloads/:jobId/:format
```

## Demo-mode behaviors to know

Some pages/components include fallbacks:

- If template fetch fails, `Templates.jsx` uses `defaultTemplates`.
- `Dashboard.jsx` has a “simulation” interval that updates progress client-side.

These are helpful for UX demos but can drift from the backend’s real job state.

## Where to extend (connecting Python generation)

The cleanest approach is to keep the web API stable while swapping internals:

1. In `POST /api/generate`, instead of `simulateProgress(jobId)`:
   - enqueue a job in a real queue (BullMQ/Redis or even a simple local queue)
   - spawn a worker process that runs the Python generator
2. Persist job state (SQLite/Postgres) rather than `Map`
3. Make `/api/downloads/...` stream the real output file

This repo intentionally stops short of implementing those pieces, but the docs
call out the design boundaries.

