# Quick Start Guide - Universal Synthetic Dataset Generator

## 🎯 What This Platform Does

Generate high-quality synthetic datasets for ANY domain using Mistral AI, with:
- **Web Interface**: Beautiful React dashboard
- **API**: RESTful backend for automation
- **Zero Cost**: Runs on free Google Colab or local GPU
- **Fast**: Up to 100+ items/minute on T4 GPU
- **Secure**: Rate-limited, validated, production-ready

## 📋 3-Minute Setup

### Step 1: Install Dependencies

```bash
# Backend dependencies
cd server
npm install

# Frontend dependencies
cd ../website/client
npm install

# Python dependencies
cd ../../Pre-Work
pip install transformers accelerate bitsandbytes torch tqdm
```

### Step 2: Start the Platform

**Terminal 1 - Backend Server:**
```bash
cd server
npm start
```

**Terminal 2 - Frontend:**
```bash
cd website/client
npm run dev
```

**Terminal 3 - Test (Optional):**
```bash
cd server
npm test
```

### Step 3: Open Browser

Navigate to: **http://localhost:5173**

## 🎨 Using the Web Interface

### Option 1: Quick Start with Templates

1. Click **"Templates"** in navigation
2. Browse 6 pre-configured templates:
   - Financial Education
   - Healthcare/Clinical
   - Legal Documents
   - Programming Q&A
   - Scientific Research
   - Educational Tutoring
3. Click **"Use Template"** on any template
4. Adjust settings (size, batch, format)
5. Click **"Start Generation"**
6. Monitor real-time progress
7. Download when complete

### Option 2: Custom Domain Builder

1. Click **"Domain Builder"** in navigation
2. Fill in the form:
   - **Domain Name**: e.g., "Machine Learning Tutorials"
   - **Description**: Brief description of content
   - **Topics**: Add topics (Python, PyTorch, Training, etc.)
   - **Output Format**: JSONL, CSV, or JSON
3. Click **"Save Domain"**
4. Use saved domain in generation

### Option 3: Dashboard (Quick Generation)

1. Go to **"Dashboard"**
2. Configure generation:
   - **Domain**: Select from dropdown
   - **Target Size**: 100 - 100,000 items
   - **Batch Size**: 5 - 50 items per batch
   - **Format**: JSONL, CSV, or JSON
3. Click **"Start Generation"**
4. View live progress:
   - Items generated
   - Generation rate
   - Time remaining
   - Progress percentage
5. Click **"Download"** when complete

## 🔌 Using the API

### Start Generation

```bash
curl -X POST http://localhost:3001/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "technology",
    "targetCount": 1000,
    "batchSize": 25,
    "outputFormat": "jsonl",
    "domainDescription": "Python programming tutorials and examples",
    "topics": ["Functions", "Classes", "Decorators", "Async/Await"]
  }'
```

Response:
```json
{
  "jobId": "gen_a1b2c3d4",
  "status": "initializing",
  "message": "Generation job started"
}
```

### Check Job Status

```bash
curl http://localhost:3001/api/jobs/gen_a1b2c3d4
```

Response:
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

### Download Dataset

```bash
curl -O http://localhost:3001/api/downloads/gen_a1b2c3d4/dataset_gen_a1b2c3d4.jsonl
```

### List All Jobs

```bash
curl http://localhost:3001/api/jobs
```

### Stop a Job

```bash
curl -X POST http://localhost:3001/api/jobs/gen_a1b2c3d4/stop
```

## 📊 Example: Generate 5,000 Q&A Dataset

**Goal**: Create a dataset of 5,000 Q&A pairs about web development

**Method 1: Web Interface**

1. Open http://localhost:5173
2. Go to Dashboard
3. Set:
   - Domain: Technology
   - Target: 5000
   - Batch: 25
   - Description: "Web development Q&A covering HTML, CSS, JavaScript, React"
   - Topics: ["HTML Basics", "CSS Flexbox", "JavaScript ES6", "React Hooks"]
4. Click "Start Generation"
5. Wait ~40-50 minutes (on T4 GPU at ~100 items/min)
6. Download JSONL file

**Method 2: API**

```bash
curl -X POST http://localhost:3001/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "technology",
    "targetCount": 5000,
    "batchSize": 25,
    "outputFormat": "jsonl",
    "domainDescription": "Web development Q&A covering HTML, CSS, JavaScript, React",
    "topics": [
      "HTML Basics",
      "CSS Flexbox and Grid",
      "JavaScript ES6 Features",
      "React Hooks and Components",
      "RESTful API Design"
    ]
  }'
```

## 🎯 Output Format Examples

### JSONL (Default - Recommended)

Each line is a JSON object:
```json
{"id":"item_abc123_1234","question":"What is a React Hook?","answer":"React Hooks are functions that let you use state and other React features without writing a class...","metadata":{"source_prompt":"Web development Q&A","parse_mode":"qa"},"created_at":"2026-01-29T00:00:00.000000"}
{"id":"item_def456_5678","question":"What is CSS Flexbox?","answer":"Flexbox is a CSS layout module that makes it easier to design flexible responsive layout structures...","metadata":{"source_prompt":"Web development Q&A","parse_mode":"qa"},"created_at":"2026-01-29T00:00:01.000000"}
```

**Use for**: ML training, LLM fine-tuning, database imports

### CSV

```csv
id,question,answer,created_at
item_abc123_1234,"What is a React Hook?","React Hooks are functions...",2026-01-29T00:00:00
item_def456_5678,"What is CSS Flexbox?","Flexbox is a CSS layout...",2026-01-29T00:00:01
```

**Use for**: Excel, data analysis, spreadsheets

### JSON

```json
[
  {
    "id": "item_abc123_1234",
    "question": "What is a React Hook?",
    "answer": "React Hooks are functions...",
    "created_at": "2026-01-29T00:00:00"
  },
  {
    "id": "item_def456_5678",
    "question": "What is CSS Flexbox?",
    "answer": "Flexbox is a CSS layout...",
    "created_at": "2026-01-29T00:00:01"
  }
]
```

**Use for**: Web apps, APIs, single-file processing

## ⚙️ Configuration Options

### Generation Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `targetCount` | 100-100,000 | 1000 | Number of items to generate |
| `batchSize` | 5-50 | 25 | Items per LLM call (higher = faster) |
| `temperature` | 0.0-2.0 | 0.8 | Model creativity (lower = factual) |
| `outputFormat` | jsonl/csv/json | jsonl | Output file format |

### Supported Models

- `mistralai/Mistral-7B-Instruct-v0.2` (Default, recommended)
- `microsoft/phi-2`
- `google/gemma-2b`

### Performance Expectations

| Hardware | Rate | 1k Dataset | 10k Dataset |
|----------|------|------------|-------------|
| T4 GPU (Free Colab) | ~100/min | ~10 min | ~1.7 hours |
| RTX 3090/4090 | ~150/min | ~7 min | ~1.1 hours |
| A100 GPU | ~200/min | ~5 min | ~50 min |

## 🐛 Troubleshooting

### "Python not available"

```bash
# Install Python 3.8+
# Set PYTHON_PATH environment variable
export PYTHON_PATH=/usr/bin/python3
```

### "Port already in use"

```bash
# Kill existing process
lsof -i :3001
kill -9 <PID>

# Or use different port
PORT=3002 npm start
```

### "Out of memory" during generation

**Solution**: Reduce batch size
```json
{
  "batchSize": 10  // Instead of 25
}
```

### Generation is slow

**Check**:
1. GPU is enabled: `nvidia-smi` (should show GPU)
2. Quantization is on: `"useQuantization": true`
3. Using appropriate batch size

## 📚 Additional Resources

- **Setup Guide**: See [SETUP.md](SETUP.md)
- **Security**: See [SECURITY.md](SECURITY.md)
- **Backend API**: See [server/README.md](server/README.md)
- **Main README**: See [Readme.md](Readme.md)

## 🎉 You're Ready!

Start generating synthetic datasets for:
- 🎓 Training AI models
- 📊 Data augmentation
- 🧪 Testing and validation
- 🔬 Research and experimentation
- 💼 Demos and prototypes

**Happy Dataset Generating!** 🚀
