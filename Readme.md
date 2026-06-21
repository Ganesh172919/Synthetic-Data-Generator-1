<div align="center">

# 🧬 Synthetic Data Generator

### _Enterprise-Grade AI Dataset Generation Platform_

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Node.js 20+](https://img.shields.io/badge/Node.js-20+-339933?style=for-the-badge&logo=node.js&logoColor=white)](https://nodejs.org)
[![React 19](https://img.shields.io/badge/React-19+-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=for-the-badge&logo=google-colab)](https://colab.research.google.com/)
[![GPU Optimized](https://img.shields.io/badge/GPU-Optimized-76B900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

<br/>

**Generate high-quality synthetic datasets at unprecedented speed using state-of-the-art LLMs.**

_From 30,000 Q&A pairs in 3 hours on a FREE Google Colab T4 GPU—to unlimited possibilities._

<br/>

[🚀 Quick Start](#-quick-start) •
[📖 Documentation](#-full-documentation) •
[🌐 Web Platform](#-full-stack-web-platform) •
[🔌 API Reference](#-api-reference) •
[🤝 Contributing](#-contributing)

---

</div>

## Educational Notes (Added)

### What this file is for

This `Readme.md` is the **front door** to the repository. It gives a high-level overview, quick start steps, and a marketing-style feature list. The goal of the educational notes here is to:

- make the setup instructions match the **actual folders on disk**
- clearly separate the **real Python generators** from the **web demo backend**
- give you a safe “learning path” through the repo and `docs/`

### Architecture (current state)

```
Browser (React UI)  ──→  Express API (website/server)  ──→  Python Worker (worker/main.py)
     :5173                     :3001                            │
                               │ SQLite (jobs, events)          │ Pre-Work/providers/ (LLM abstraction)
                               │                                │ Pre-Work/universal_dataset_generator.py
                               ▼                                ▼
                          SSE progress  ◀────────────────  Real generation (Mock, OpenAI, HuggingFace,
                          Download files                   Anthropic, Google, Ollama, Groq, Together, etc.)
```

**Three-process architecture:**
1. **React Client** (`website/client/`) — UI on port 5173, proxies API calls to port 3001
2. **Express API** (`website/server/`) — Control plane on port 3001, SQLite-backed job management
3. **Python Worker** (`worker/`) — Execution plane, polls SQLite for queued jobs, runs generation

### Quick start (local)

**1) Start the API server**
```bash
cd website/server
npm install
npm start
```

**2) Start the Python worker**
```bash
cd worker
pip install -r requirements.txt
python main.py
```

**3) Start the frontend**
```bash
cd website/client
npm install
npm run dev
```

Open `http://localhost:5173` — the UI drives real generation through the worker.

### Dataset schema overview (what to expect)

Most examples use a Q&A dataset record with fields like:

```json
{
  "id": "item_abc123_0001",
  "topic": "Technology",
  "question": "What is a React Hook?",
  "answer": "A React Hook is a function that lets you...",
  "difficulty": "beginner",
  "created_at": "2026-01-29T00:00:00.000Z"
}
```

For a deeper guide (including JSONL vs CSV tradeoffs), see `docs/DATASET_SCHEMA.md`.

### Real-world examples

1. **Support chat data for internal tooling**
   - Generate “customer issue → agent response” pairs for training an intent classifier.
2. **Template-driven Q&A**
   - Start from “Technology” or “Financial” templates in the UI, then tweak topics and size.
3. **Structured extraction datasets**
   - Use the universal generator’s `json` mode to create records like:
     - `{ "input": "...", "output": "...", "label": "..." }`

### Edge cases & failure modes

- **Hallucinations / unsafe advice**: synthetic content can look confident but be wrong (especially finance/medical/legal).
- **PII leakage**: prompts can accidentally produce personal info; treat outputs as untrusted.
- **Duplicates**: high-volume generation often repeats phrasing; dedup helps but isn’t semantic.
- **GPU OOM**: large `max_new_tokens` and batch sizes can exceed VRAM; tune down for stability.
- **Long installs**: the Python scripts may install large dependencies (Torch/Transformers).

### Troubleshooting

- UI can’t reach API: confirm `website/server` runs on port 3001 and Vite proxy is active (`website/client/vite.config.js`).
- “Module not found”: run `npm install` in both `website/server` and `website/client`.
- Python generation is slow: reduce batch size / token limits; verify GPU availability.

### Learning notes

- **Why JSONL is popular**: it streams well, supports large datasets, and plays nicely with data tooling.
- **Why the web backend is “mock”**: it keeps the UI functional without requiring GPU/LLM runtime in the server.
- **Why checkpointing matters**: long runs fail; resume support prevents losing progress.

### Next steps / exercises

Follow the learning path in `docs/README.md`, then:

1. Trace the full web flow in DevTools Network (generate → jobs → download).
2. Read the generator run loop in `Pre-Work/universal_dataset_generator.py`.
3. Implement a “real job runner” behind the API (document-only in this repo; see `docs/WEB_PLATFORM.md`).

## 🎯 Overview

**Synthetic Data Generator** is a production-ready, full-stack platform for generating large-scale, domain-specific datasets for machine learning and AI training. Featuring a **beautiful React dashboard**, **RESTful API**, and **high-performance Python generators**, our system leverages cutting-edge LLM technology to produce high-quality, validated synthetic data.

### ✨ Key Features

| Feature                       | Description                                                |
| ----------------------------- | ---------------------------------------------------------- |
| 💸 **Zero Cost**              | Runs entirely on Google Colab Free Tier (T4 GPU)           |
| ⚡ **Blazing Fast**           | Up to **167 Q&A pairs/minute** with MEGA batch processing  |
| 🌐 **Web Dashboard**          | Beautiful React UI with real-time progress tracking        |
| 🔌 **RESTful API**            | Full API access for programmatic generation                |
| 🧠 **Intelligent Generation** | 25 Q&A pairs per LLM call with smart prompting             |
| 🛡️ **Bulletproof Safety**     | Emergency save handlers, auto-download on Colab disconnect |
| ✅ **Quality Assured**        | Built-in pattern matching and content validation           |
| 🔄 **Resume Support**         | Checkpoint-based resume for interrupted sessions           |
| 🌍 **18 Templates**           | Pre-built templates across 17 domains + custom builder     |
| 📂 **Multi-Format Output**    | Export to JSONL, CSV, or JSON formats                      |
| 🤖 **10 LLM Providers**       | OpenAI, Anthropic, Google, HuggingFace, Ollama, Groq, etc. |
| 🌐 **20 Languages**           | Generate datasets in English, Spanish, Chinese, Hindi, etc. |
| 👥 **Concurrent Jobs**        | Configurable parallel job execution (1-16 workers)         |

---

## 🌐 Full-Stack Web Platform

> ✅ **FULLY AVAILABLE** — Production-ready web platform for synthetic data generation!

We've built a complete full-stack application with an intuitive web interface for dataset generation.

### 🖥️ Platform Pages

| Page                  | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| **🏠 Landing Page**   | Beautiful hero section with platform overview and features   |
| **📊 Dashboard**      | Real-time generation monitoring with live stats and progress |
| **📚 Templates**      | 6 pre-configured templates for common domains                |
| **🔧 Domain Builder** | Visual interface to create custom dataset domains            |
| **📖 Documentation**  | Complete API docs and usage guides                           |

### 🎨 Features

| Feature                      | Status       |
| ---------------------------- | ------------ |
| **Interactive Dashboard**    | ✅ Available |
| **Real-time Progress**       | ✅ Available |
| **Custom Domain Builder**    | ✅ Available |
| **6 Pre-built Templates**    | ✅ Available |
| **Job Management**           | ✅ Available |
| **RESTful API**              | ✅ Available |
| **Multiple Export Formats**  | ✅ Available |
| **Rate Limiting & Security** | ✅ Available |
| **Checkpoint Resume**        | ✅ Available |
| **Dark Mode UI**             | ✅ Available |

### 🛠️ Tech Stack

```
Frontend:     React 19 + Vite 7 + Tailwind CSS 4
Backend:      Node.js 20+ + Express 4 + SQLite (better-sqlite3)
Worker:       Python 3.10+ + provider abstraction layer
Providers:    OpenAI, Anthropic, Google, HuggingFace, Ollama, Groq, Together, Azure, Custom
Storage:      SQLite + local filesystem (Docker-ready)
Real-time:    Server-Sent Events (SSE) for progress streaming
```

### 📸 Screenshots

|            Dashboard            |        Templates         |    Domain Builder     |
| :-----------------------------: | :----------------------: | :-------------------: |
| Real-time generation monitoring | Pre-configured templates | Create custom domains |

---

## 🚀 Quick Start

### Prerequisites

| Requirement          | Details                           |
| -------------------- | --------------------------------- |
| **Node.js**          | Version 20 or higher              |
| **Python**           | Version 3.10 or higher            |
| **npm**              | Comes with Node.js                |
| **GPU** _(Optional)_ | NVIDIA T4, RTX 3090/4090, or A100 |

### ⚡ 3-Step Setup

**Step 1: Start the API Server**

```bash
cd website/server
npm install
npm start
```

> API runs on `http://localhost:3001`

**Step 2: Start the Python Worker**

```bash
cd worker
pip install -r requirements.txt
python main.py
```

> Worker polls SQLite for queued jobs and runs generation

**Step 3: Start the Frontend**

```bash
cd website/client
npm install
npm run dev
```

> Frontend runs on `http://localhost:5173`

### 🎉 You're Ready!

Open **http://localhost:5173** in your browser and start generating datasets!

---

## 📁 Project Structure

```
Synthetic-Data-Generator-1/
│
├── 📂 Pre-Work/                              # Core Python generators
│   ├── universal_dataset_generator.py           # 🌍 Universal domain generator (~1300 lines)
│   ├── financial_education_generator_ultra.py   # 🚀 Finance-optimized generator
│   ├── providers/                               # 🤖 LLM provider abstraction layer
│   │   ├── base.py                              #    BaseProvider ABC + exceptions
│   │   ├── factory.py                           #    Provider registry & factory
│   │   ├── mock.py                              #    Mock provider (testing)
│   │   ├── openai_provider.py                   #    OpenAI GPT provider
│   │   ├── anthropic.py                         #    Anthropic Claude provider
│   │   ├── google.py                            #    Google Gemini provider
│   │   ├── huggingface.py                       #    HuggingFace local models
│   │   ├── ollama.py                            #    Ollama local inference
│   │   ├── azure_openai.py                      #    Azure OpenAI provider
│   │   ├── groq.py                              #    Groq fast inference
│   │   ├── together.py                          #    Together.ai provider
│   │   └── custom.py                            #    Custom OpenAI-compatible endpoints
│   └── OPTIMIZATION_GUIDE.md                    # 📖 Performance tuning guide
│
├── 📂 worker/                                # 🔄 Python job worker
│   ├── main.py                                  # Polling worker with concurrent execution
│   ├── requirements.txt                         # Python dependencies
│   └── Dockerfile                               # Worker container
│
├── 📂 website/                               # 🌐 Web Platform
│   ├── server/                                  # Express API backend
│   │   ├── index.js                             # Entry point
│   │   ├── package.json                         # Node.js dependencies
│   │   ├── Dockerfile                           # API container
│   │   └── src/
│   │       ├── server.js                        # Express routes & middleware (~950 lines)
│   │       ├── config.js                        # Environment config (frozen)
│   │       ├── db.js                            # SQLite helpers & mappers
│   │       ├── migrations.js                    # Database migration runner
│   │       ├── templates.js                     # 18 pre-built domain templates
│   │       └── __tests__/                       # Test suite (63 tests)
│   │           ├── config.test.js
│   │           ├── db.test.js
│   │           ├── migrations.test.js
│   │           └── server.test.js
│   └── client/                                  # React frontend
│       ├── src/
│       │   ├── pages/                           # Landing, Dashboard, Templates, etc.
│       │   ├── components/                      # Navbar, Footer, ui/
│       │   ├── services/api.js                  # API client
│       │   ├── hooks/                           # useTheme, useIntersectionObserver
│       │   └── App.jsx                          # Router + providers
│       ├── package.json
│       └── vite.config.js
│
├── 📂 docs/                                  # 📖 Documentation
├── 📄 docker-compose.yml                     # Docker orchestration
├── 📄 IMPROVEMENT_PLAN.md                    # 🗺️ Feature roadmap
├── 📄 Readme.md                              # You are here!
└── 📄 LICENSE                                # MIT License
```

---

## 🔌 API Reference

The backend provides a comprehensive RESTful API for programmatic dataset generation.

### Endpoints Overview

| Method   | Endpoint                          | Description              |
| -------- | --------------------------------- | ------------------------ |
| `GET`    | `/api/health`                     | Server health check      |
| `GET`    | `/api/templates`                  | List 18 templates        |
| `GET`    | `/api/templates/:id`              | Get template by ID       |
| `GET`    | `/api/providers`                  | List 10 LLM providers    |
| `GET`    | `/api/providers/:name`            | Provider details         |
| `GET`    | `/api/providers/:name/health`     | Provider health check    |
| `POST`   | `/api/generate`                   | Start generation job     |
| `GET`    | `/api/jobs`                       | List all jobs (paginated)|
| `GET`    | `/api/jobs/:jobId`                | Get job status           |
| `GET`    | `/api/jobs/:jobId/events`         | SSE progress stream      |
| `POST`   | `/api/jobs/:jobId/stop`           | Stop a running job       |
| `POST`   | `/api/jobs/:jobId/retry`          | Retry failed job         |
| `DELETE` | `/api/jobs/:jobId`                | Delete terminal job      |
| `GET`    | `/api/downloads/:jobId/:format`   | Download dataset         |
| `GET`    | `/api/jobs/:jobId/preview`        | Preview generated data   |
| `POST`   | `/api/domains`                    | Create custom domain     |
| `GET`    | `/api/domains`                    | List custom domains      |
| `GET`    | `/api/domains/:id`                | Get domain               |
| `PUT`    | `/api/domains/:id`                | Update domain            |
| `DELETE` | `/api/domains/:id`                | Delete domain            |
| `GET`    | `/api/metrics`                    | Aggregate metrics        |

### Quick API Examples

**Start a Generation Job:**

```bash
curl -X POST http://localhost:3001/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "technology",
    "targetCount": 1000,
    "batchSize": 25,
    "outputFormat": "jsonl",
    "domainDescription": "Python programming tutorials",
    "topics": ["Functions", "Classes", "Async/Await", "Decorators"]
  }'
```

**Response:**

```json
{
  "jobId": "gen_a1b2c3d4",
  "status": "initializing",
  "message": "Generation job started"
}
```

**Check Job Status:**

```bash
curl http://localhost:3001/api/jobs/gen_a1b2c3d4
```

**Response:**

```json
{
  "id": "gen_a1b2c3d4",
  "status": "running",
  "generated": 450,
  "targetCount": 1000,
  "progress": 45.0,
  "rate": 12.5,
  "estimatedTimeRemaining": 44
}
```

**Download Completed Dataset:**

```bash
curl -O http://localhost:3001/api/downloads/gen_a1b2c3d4/dataset_gen_a1b2c3d4.jsonl
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      WEB BROWSER (Client)                           │
│                React 19 + Vite 7 + Tailwind CSS 4                   │
│                    http://localhost:5173                            │
│                                                                     │
│  • Landing Page    • Dashboard      • Templates    • Domain Builder │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP/REST + SSE
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   EXPRESS API SERVER (Control Plane)                │
│             Node.js 20 + Express 4 + SQLite + Rate Limiting        │
│                    http://localhost:3001/api                        │
│                                                                     │
│  • Job CRUD (21 endpoints)  • 18 Templates  • SSE Progress         │
│  • Provider Registry        • Domain CRUD    • Metrics             │
│  • Database Migrations      • File Downloads • Auth (optional)     │
└────────────────────────────┬────────────────────────────────────────┘
                             │ SQLite (shared via filesystem)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│               PYTHON WORKER (Execution Plane)                       │
│                    worker/main.py                                   │
│              Concurrent job execution (1-16 threads)                │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Provider   │───▶│   Prompt     │───▶│   Response   │          │
│  │   Factory    │    │   Builder    │    │   Parser     │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                                          │                 │
│         ▼                                          ▼                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  10 LLM      │    │    Dedup     │    │  Async File  │          │
│  │  Providers   │    │   Engine     │    │   Writer     │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │ Save to disk
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GENERATED DATASETS                               │
│          website/server/data/outputs/*.{jsonl,csv,json}             │
└─────────────────────────────────────────────────────────────────────┘
```

### 🔧 Core Components

| Component                 | Purpose                                                 |
| ------------------------- | ------------------------------------------------------- |
| **🤖 Provider Factory**   | 10 LLM providers behind a unified abstraction           |
| **✍️ AsyncFileWriter**    | High-performance async file writer with buffering       |
| **🔐 ThreadSafeSet**      | Lock-free deduplication using MD5 hashing               |
| **⚡ AtomicCounter**      | Thread-safe progress tracking                           |
| **🛡️ Emergency Handlers** | Auto-save on interrupt, timeout, or crash               |
| **📊 Migration Runner**   | Schema versioning with automatic migration              |

---

## ⚙️ Configuration

### Generation Parameters

| Parameter      | Range          | Default | Description                             |
| -------------- | -------------- | ------- | --------------------------------------- |
| `targetCount`  | 100-100,000    | 1000    | Number of items to generate             |
| `batchSize`    | 5-50           | 25      | Items per LLM call (higher = faster)    |
| `temperature`  | 0.0-2.0        | 0.8     | Model creativity (lower = more factual) |
| `outputFormat` | jsonl/csv/json | jsonl   | Output file format                      |

### Python Config Class

```python
@dataclass
class ExtremeSpeedConfig:
    # Model Settings
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    use_quantization: bool = True      # 4-bit for T4 GPU
    use_flash_attention: bool = True   # FlashAttention 2

    # Generation Settings
    batch_size: int = 25               # Q&A pairs per LLM call
    target_count: int = 30000          # Total pairs to generate
    save_interval: int = 200           # Buffer flush interval

    # Quality Settings
    min_answer_length: int = 40        # Minimum answer chars
    output_file: str = "dataset.jsonl" # Output filename
```

---

## 📊 Performance Benchmarks

| Hardware                 | Rate     | 1k Dataset | 10k Dataset | 30k Dataset |
| ------------------------ | -------- | ---------- | ----------- | ----------- |
| **T4 GPU** (Colab Free)  | ~100/min | ~10 min    | ~1.7 hours  | ~5 hours    |
| **A100 GPU** (Colab Pro) | ~200/min | ~5 min     | ~50 min     | ~2.5 hours  |
| **RTX 3090/4090**        | ~150/min | ~7 min     | ~1.1 hours  | ~3.5 hours  |
| **CPU Only**             | ~10/min  | ~1.7 hours | ~16 hours   | ~50 hours   |

---

## 💾 Output Formats

### JSONL (Recommended for ML Training)

```json
{
  "id": "fin_Per_Bud_1234_56789",
  "topic": "Personal Finance",
  "subtopic": "Budgeting Basics",
  "question": "What is the 50/30/20 rule in budgeting?",
  "answer": "The 50/30/20 rule is a simple budgeting framework that suggests allocating 50% of after-tax income to needs, 30% to wants, and 20% to savings and debt repayment...",
  "difficulty": "beginner",
  "question_type": "definition",
  "created_at": "2026-01-30T10:30:00.000000"
}
```

### CSV (For Spreadsheets & Analysis)

```csv
id,question,answer,topic,created_at
fin_Per_Bud_1234,"What is the 50/30/20 rule?","The 50/30/20 rule is...",Personal Finance,2026-01-30T10:30:00
```

### JSON (For Web Apps & APIs)

```json
[
  {
    "id": "fin_Per_Bud_1234",
    "question": "What is the 50/30/20 rule?",
    "answer": "The 50/30/20 rule is...",
    "topic": "Personal Finance"
  }
]
```

---

## 📚 Pre-Built Templates (18)

| Template                    | Category        | Description                                                 |
| --------------------------- | --------------- | ----------------------------------------------------------- |
| 💰 **Financial Education**  | Financial       | Personal finance, investing, budgeting, credit management   |
| 🏥 **Clinical Knowledge**   | Healthcare      | Medical terminology, symptoms, treatments, procedures       |
| ⚖️ **Legal Documents**      | Legal           | Contracts, compliance, legal terms, case law                |
| 💻 **Programming Q&A**      | Technology      | Code explanations, debugging, best practices, algorithms    |
| 🔬 **Scientific Research**  | Science         | Research methodology, experiments, scientific concepts      |
| 📚 **Educational Tutoring** | Education       | Math, science, english, history tutoring content            |
| 🎧 **Customer Support**     | Customer Support| Help desk, troubleshooting, returns, account issues         |
| 🛒 **E-commerce Products**  | E-commerce      | Product descriptions, reviews, FAQ, catalog data            |
| 🏠 **Real Estate**          | Real Estate     | Property listings, market analysis, neighborhood guides     |
| 🎮 **Gaming NPC Dialogue**  | Gaming          | NPC conversations, quests, item lore, world building        |
| 📣 **Marketing Copy**       | Marketing       | Ad copy, social media, email campaigns, brand messaging     |
| 👥 **HR Data**              | HR              | Job descriptions, interview questions, performance reviews  |
| 📰 **News & Journalism**    | News            | Article summaries, fact-checking, headline generation       |
| 🔒 **Cybersecurity**        | Cybersecurity   | Threat analysis, incident reports, vulnerability data       |
| ✈️ **Travel Guides**        | Travel          | Destination guides, hotel reviews, itinerary planning       |
| 🍳 **Food & Recipes**       | Food            | Recipes, restaurant reviews, nutrition, cooking techniques  |
| 💻 **Code Generation**      | Technology      | Code snippets, code review, bug fixes, API patterns         |
| 🏥 **Medical Research**     | Healthcare      | Clinical trials, drug interactions, case studies            |

---

## 🛡️ Security Features

### Built-in Protections

| Feature               | Description                                                       |
| --------------------- | ----------------------------------------------------------------- |
| **Rate Limiting**     | API abuse prevention (100 req/15min general, 10/15min generation) |
| **Input Validation**  | Comprehensive parameter validation                                |
| **SIGINT/SIGTERM**    | Graceful shutdown with full data save                             |
| **Colab Disconnect**  | Auto-download before session timeout                              |
| **Checkpoint Resume** | Restart from exact last position                                  |
| **Crash Recovery**    | Emergency buffer flush on any error                               |

### Emergency Save & Recovery

```python
# Force save at any time (works in Colab!)
force_save_and_download()

# Resume from checkpoint
generator.resume_from_checkpoint("checkpoint.json")
```

---

## ⚠️ Troubleshooting

<details>
<summary><b>🔴 Out of Memory (OOM)</b></summary>

```python
batch_size: int = 15        # Reduce from 25
max_new_tokens: int = 2000  # Reduce from 2500
clear_cache_interval: int = 15  # More frequent clearing
```

</details>

<details>
<summary><b>🟡 Slow Generation</b></summary>

1. Verify GPU is active: `torch.cuda.is_available()`
2. Check quantization is enabled
3. Ensure FlashAttention is installed
4. Try reducing batch size for better throughput

</details>

<details>
<summary><b>🟠 Session Timeout (Colab)</b></summary>

- Keep browser tab active
- Use `force_save_and_download()` periodically
- Enable auto-save: `auto_save_interval: int = 180`

</details>

<details>
<summary><b>🔵 Port Already in Use</b></summary>

```bash
# Find process using port 3001
lsof -i :3001

# Kill the process
kill -9 <PID>

# Or use different port
PORT=3002 npm start
```

</details>

<details>
<summary><b>🟣 Python Not Found</b></summary>

```bash
# Set Python path environment variable
export PYTHON_PATH=/usr/bin/python3

# Or on Windows
set PYTHON_PATH=C:\Python311\python.exe
```

</details>

---

## 🗺️ Roadmap

### ✅ Completed (v2.0)

- [x] Extreme speed batch processing (167 Q&A/min)
- [x] Emergency save handlers & crash recovery
- [x] Checkpoint-based resume
- [x] Full-stack web application with React 19 dashboard
- [x] RESTful API with 21 endpoints + SSE progress streaming
- [x] 18 pre-built domain templates across 17 categories
- [x] Custom domain builder with CRUD operations
- [x] Multi-format export (JSONL, CSV, JSON)
- [x] **10 LLM providers** (OpenAI, Anthropic, Google, HuggingFace, Ollama, Groq, Together, Azure, Custom)
- [x] **Provider abstraction layer** with unified interface
- [x] **Multi-language support** (20 languages)
- [x] **Concurrent job execution** (configurable 1-16 workers)
- [x] **Database migrations** system
- [x] **Domain CRUD** (create, read, update, delete)
- [x] **Test suite** (63 tests — config, db, migrations, all API routes)
- [x] **9 parse modes** (qa, text, json, instruction, conversation, classification, summarization, code, reasoning)

### 🔜 Coming Soon (Phase 3-5)

- [ ] Dataset quality scoring & bias detection
- [ ] Parquet, Arrow, HuggingFace Datasets output formats
- [ ] User authentication (JWT/OAuth2) + RBAC
- [ ] Cloud storage integration (S3, GCS, Azure Blob)
- [ ] RAG-grounded generation from uploaded documents
- [ ] Dataset augmentation & transformation tools
- [ ] Interactive dataset editor
- [ ] Fine-tuning pipeline integration

### 📋 Future Plans (Phase 6-7)

- [ ] Kubernetes deployment + Helm chart
- [ ] PostgreSQL support for multi-instance deployments
- [ ] Plugin system for custom generators
- [ ] Visual workflow builder (drag-and-drop pipelines)
- [ ] Multi-modal data support (images, audio, video)
- [ ] Dataset marketplace for sharing
- [ ] Enterprise integrations (Slack, Teams, Snowflake, MLflow)

---

## 📖 Full Documentation

| Resource                                                | Description                              |
| ------------------------------------------------------- | ---------------------------------------- |
| [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md)              | Feature roadmap & implementation plan    |
| [QUICKSTART.md](QUICKSTART.md)                          | 3-minute quick start guide               |
| [SETUP.md](SETUP.md)                                    | Complete installation & setup guide      |
| [SECURITY.md](SECURITY.md)                              | Security considerations & best practices |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)            | System architecture deep dive            |
| [OPTIMIZATION_GUIDE.md](Pre-Work/OPTIMIZATION_GUIDE.md) | Detailed performance tuning guide        |

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Getting Started

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Contribution Areas

| Area                 | Examples                                         |
| -------------------- | ------------------------------------------------ |
| **🌐 Frontend**      | React components, UI/UX improvements, animations |
| **🔧 Backend**       | API endpoints, performance optimizations         |
| **🐍 Python**        | Generator improvements, new models               |
| **📝 Templates**     | New domain templates, question types             |
| **📚 Documentation** | Tutorials, guides, examples                      |
| **🐛 Bug Fixes**     | Issue resolution, error handling                 |

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### 💫 Star Us on GitHub!

If you find this project useful, please consider giving it a ⭐

<br/>

**Built with ❤️ by the Synthetic Data Generator Team**

_Last Updated: January 2026_

---

[⬆ Back to Top](#-synthetic-data-generator)

</div>
