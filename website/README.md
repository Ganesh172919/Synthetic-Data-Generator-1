# 🧬 Synthetic Data Generator - Web Platform

A modern, full-stack web application for the Synthetic Data Generator platform.

## 🛠️ Tech Stack

- **Frontend**: React + Vite + TailwindCSS
- **Backend**: Node.js + Express
- **Icons**: Lucide React
- **Routing**: React Router

## 📁 Project Structure

```
website/
├── client/                 # React frontend
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Page components
│   │   ├── services/       # API service layer
│   │   ├── App.jsx         # Main app component
│   │   └── main.jsx        # Entry point
│   ├── package.json
│   └── vite.config.js
│
├── server/                 # Express backend
│   ├── index.js            # API server
│   └── package.json
│
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

1. **Install frontend dependencies:**

   ```bash
   cd website/client
   npm install
   ```

2. **Install backend dependencies:**
   ```bash
   cd website/server
   npm install
   ```

### Running the Application

**Option 1: Run both servers separately**

Terminal 1 - Backend:

```bash
cd website/server
npm start
```

Terminal 2 - Frontend:

```bash
cd website/client
npm run dev
```

**Option 2: Using concurrent processes**

```bash
# From website directory
npm run dev
```

The frontend will be available at `http://localhost:5173` and the API at `http://localhost:3001`.

## 🌐 Features

### Landing Page

- Hero section with key value propositions
- Feature highlights
- Performance benchmarks
- Testimonials
- Call-to-action sections

### Dashboard

- Real-time generation monitoring
- Progress tracking with live stats
- Job management
- Configuration panel
- Export functionality

### Templates

- Pre-built domain templates
- Search and filtering
- Template ratings and download counts
- Featured templates section

### Domain Builder

- Custom domain configuration
- Topic and subtopic management
- Question type selection
- Output settings customization
- Live preview panel

### Documentation

- Getting started guide
- Quick start instructions
- API reference
- Configuration options
- Output format specification

## 📡 API Endpoints

| Method | Endpoint                        | Description                         |
| ------ | ------------------------------- | ----------------------------------- |
| GET    | `/api/health`                   | Health check                        |
| GET    | `/api/templates`                | List all templates                  |
| GET    | `/api/templates/:id`            | Get template by ID                  |
| POST   | `/api/generate`                 | Start generation job                |
| GET    | `/api/jobs/:jobId`              | Get job status                      |
| GET    | `/api/jobs`                     | List all jobs                       |
| POST   | `/api/jobs/:jobId/stop`         | Stop a running job                  |
| DELETE | `/api/jobs/:jobId`              | Delete a job                        |
| GET    | `/api/downloads/:jobId/:format` | Download dataset (jsonl, csv, json) |
| POST   | `/api/domains`                  | Save custom domain                  |
| GET    | `/api/domains/:id`              | Get domain by ID                    |
| GET    | `/api/domains`                  | List all domains                    |

## 🎨 UI Components

- **Navbar**: Responsive navigation with mobile menu
- **Footer**: Links and social icons
- **Cards**: Feature, template, and stat cards
- **Forms**: Configuration forms with validation
- **Progress**: Animated progress bars
- **Code Blocks**: Syntax highlighted with copy functionality

## 📦 Production Build

```bash
# Build frontend
cd website/client
npm run build

# The build output will be in dist/ directory
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the server directory:

```env
PORT=3001
NODE_ENV=development
```

### Vite Configuration

The frontend is configured to proxy API requests to the backend during development. See `vite.config.js` for details.

## 📄 License

MIT License - see the [LICENSE](../LICENSE) file for details.

## Educational Notes (Added)

### What this file is for

This README documents the **web platform** portion of the repository:

- `website/client` — React + Vite frontend
- `website/server` — Express backend API (demo implementation)

### Reality-aligned updates (paths, current behavior)

There is no `website/package.json` in this repo. That means:

- You cannot run `npm install` or `npm run dev` from `website/` as a single workspace command.
- Install and run **separately** in `website/server` and `website/client`.

The backend is a **demo**:
- jobs/domains are stored in memory (lost on restart)
- progress is simulated
- dataset downloads return mock payloads (useful for UI testing)

### API contract (as implemented)

The “job” object returned by the server typically contains:

- `id`
- `domain`
- `targetCount`, `batchSize`, `outputFormat`
- `status`: `running` | `stopped` | `completed`
- `generated`
- `createdAt`, `updatedAt`

Error responses are usually:

```json
{ "error": "Human readable message" }
```

### ASCII sequence diagram (UI → API)

```
User clicks "Start Generation"
  │
  ├─► POST /api/generate  (create job)
  │       │
  │       └─► jobId returned
  │
  ├─► GET /api/jobs/:jobId (poll)
  │       │
  │       └─► status running → completed
  │
  └─► GET /api/downloads/:jobId/:format (download mock dataset)
```

### Extension points (how to make it “real”)

If you want the backend to generate real datasets:

1. Replace the `simulateProgress(...)` loop with a real job runner (queue + worker).
2. Call a Python generator from `Pre-Work/` and write outputs to disk.
3. Update `/api/downloads/...` to stream the generated file instead of synthesizing payloads in memory.

For more detailed guidance, see:
- `docs/ARCHITECTURE.md`
- `docs/WEB_PLATFORM.md`

### Next steps / exercises

1. Open `website/server/index.js` and annotate each endpoint with request/response examples.
2. Add persistence (SQLite) and document what changes in the job lifecycle.
