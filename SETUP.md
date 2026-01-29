# Full-Stack Synthetic Data Generator - Setup Guide

## 🎯 Overview

This guide will help you set up the complete full-stack application:
- **Frontend**: React + Vite web interface
- **Backend**: Node.js API server  
- **Generator**: Python-based dataset generation

## 📋 Prerequisites

### Required Software

- **Node.js** 16+ and npm
- **Python** 3.8+
- **Git**

### Optional but Recommended

- **GPU**: NVIDIA T4 or better (for local generation)
- **CUDA**: For GPU acceleration
- **Google Colab**: Alternative to local GPU

## 🚀 Quick Start (3 Steps)

### Step 1: Install Python Dependencies

```bash
cd Pre-Work
pip install transformers accelerate bitsandbytes torch tqdm
```

### Step 2: Start Backend Server

```bash
cd server
npm install
npm start
```

Server will run on `http://localhost:3001`

### Step 3: Start Frontend

```bash
cd website/client
npm install
npm run dev
```

Frontend will run on `http://localhost:5173`

## 🌐 Accessing the Application

Open your browser and navigate to:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:3001/api

## 📚 Detailed Setup

### Backend Server Setup

The backend handles:
- API endpoints
- Python subprocess management
- File storage
- Job tracking

```bash
cd server

# Install dependencies
npm install

# Start server (development)
npm run dev

# Start server (production)
npm start
```

**Configuration Options:**

- `PORT`: Server port (default: 3001)
- `PYTHON_PATH`: Path to Python executable (default: python3)

### Frontend Setup

```bash
cd website/client

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Python Generator Setup

The Python generators are in `Pre-Work/`:

```bash
cd Pre-Work

# Install required packages
pip install transformers accelerate bitsandbytes torch tqdm

# Test standalone (optional)
python universal_dataset_generator.py
```

## 🎨 Using the Web Interface

### 1. Choose a Template

- Navigate to **Templates** page
- Browse pre-configured templates for different domains
- Click on a template to see details

### 2. Configure Generation

On the **Dashboard** page:

1. **Select Domain**: Financial, Healthcare, Technology, etc.
2. **Set Target Size**: Number of items (100-100,000)
3. **Batch Size**: Items per generation batch (5-50)
4. **Output Format**: JSONL, CSV, or JSON
5. **Optional**: Add custom topics and description

### 3. Start Generation

1. Click **Start Generation**
2. Monitor real-time progress
3. View generation rate and ETA
4. Download when complete

### 4. Custom Domains

Use the **Domain Builder** to create custom datasets:

1. Define domain name and description
2. Add topics and subtopics
3. Set difficulty levels
4. Configure output settings
5. Save and use in generation

## 🔧 API Usage

### Start Generation

```bash
curl -X POST http://localhost:3001/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "technology",
    "targetCount": 1000,
    "batchSize": 25,
    "outputFormat": "jsonl",
    "domainDescription": "Programming tutorials",
    "topics": ["Python", "JavaScript"]
  }'
```

### Check Job Status

```bash
curl http://localhost:3001/api/jobs/{jobId}
```

### Download Dataset

```bash
curl -O http://localhost:3001/api/downloads/{jobId}/{filename}
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client Browser                       │
│                    (React + Vite)                       │
│                  http://localhost:5173                  │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/REST API
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Express API Server                     │
│                  (Node.js + Express)                    │
│                  http://localhost:3001                  │
└────────────────────┬────────────────────────────────────┘
                     │ Subprocess (child_process.spawn)
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Generator Runner (Python)                  │
│                generator_runner.py                      │
└────────────────────┬────────────────────────────────────┘
                     │ Import & Execute
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Universal Dataset Generator                  │
│         universal_dataset_generator.py                  │
│                                                         │
│  ┌─────────────────────────────────────────────┐       │
│  │         Mistral-7B-Instruct-v0.2            │       │
│  │         (4-bit quantized)                   │       │
│  │         Local or Google Colab               │       │
│  └─────────────────────────────────────────────┘       │
└────────────────────┬────────────────────────────────────┘
                     │ Generate & Save
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Generated Datasets                         │
│              (JSONL / CSV / JSON)                       │
│              server/data/outputs/                       │
└─────────────────────────────────────────────────────────┘
```

## 🐛 Troubleshooting

### Backend won't start

**Issue**: `Error: Cannot find module 'express'`

**Solution**:
```bash
cd server
npm install
```

### Python not found

**Issue**: `Error: Python not available`

**Solution**:
- Install Python 3.8+
- Set `PYTHON_PATH` environment variable:
  ```bash
  export PYTHON_PATH=/usr/bin/python3
  ```

### CORS errors

**Issue**: Frontend can't connect to backend

**Solution**:
- Ensure backend is running on port 3001
- Check Vite proxy configuration in `website/client/vite.config.js`

### Generation fails

**Issue**: Python generator crashes

**Solutions**:
1. Check Python dependencies are installed
2. Verify GPU/CUDA setup (or use CPU mode)
3. Reduce batch size if out of memory
4. Check logs in `server/data/` directory

### Port already in use

**Issue**: `Error: listen EADDRINUSE: address already in use :::3001`

**Solution**:
```bash
# Find process using port 3001
lsof -i :3001

# Kill the process
kill -9 <PID>

# Or use different port
PORT=3002 npm start
```

## 📊 Performance Tips

### For Google Colab (Free Tier)

- Target: ~100 items/minute on T4 GPU
- Batch size: 25 items
- Use 4-bit quantization
- Enable FlashAttention 2

### For Local GPU (RTX 3090/4090)

- Target: ~150 items/minute
- Batch size: 25-30 items
- Full or 8-bit quantization
- Increase parallel workers

### For CPU Only

- Target: ~10 items/minute
- Batch size: 5-10 items
- Disable quantization
- Smaller target sizes recommended

## 🔒 Security Considerations

### Production Deployment

1. **Authentication**: Add user authentication (JWT, OAuth)
2. **Rate Limiting**: Prevent API abuse
3. **Input Validation**: Already implemented, but review for your use case
4. **File Storage**: Use cloud storage (S3) instead of local filesystem
5. **HTTPS**: Enable SSL/TLS
6. **Environment Variables**: Use `.env` files for secrets

### Recommended `.env` File

```bash
PORT=3001
PYTHON_PATH=/usr/bin/python3
NODE_ENV=production
MAX_GENERATION_SIZE=10000
RATE_LIMIT_WINDOW=900000
RATE_LIMIT_MAX=10
```

## 📁 Project Structure

```
Synthetic-Data-Generator-1/
├── Pre-Work/                          # Python generators
│   ├── universal_dataset_generator.py
│   └── financial_education_generator_ultra.py
│
├── server/                            # Backend API
│   ├── server.js                      # Express server
│   ├── generator_runner.py            # Python wrapper
│   ├── package.json
│   └── data/                          # Generated data (gitignored)
│       ├── outputs/
│       ├── checkpoints/
│       └── configs/
│
├── website/
│   ├── client/                        # Frontend React app
│   │   ├── src/
│   │   │   ├── pages/                 # Page components
│   │   │   ├── components/            # Reusable components
│   │   │   ├── services/              # API client
│   │   │   └── App.jsx
│   │   ├── package.json
│   │   └── vite.config.js
│   │
│   └── server/                        # Legacy server (deprecated)
│       └── index.js
│
└── SETUP.md                           # This file
```

## 🚢 Deployment

### Docker Deployment (Recommended)

Coming soon - Docker compose configuration for easy deployment.

### Manual Deployment

1. **Backend**:
   ```bash
   cd server
   npm install --production
   PORT=3001 npm start
   ```

2. **Frontend**:
   ```bash
   cd website/client
   npm install
   npm run build
   # Serve dist/ with nginx or similar
   ```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/Ganesh172919/Synthetic-Data-Generator-1/issues)
- **Documentation**: See README.md and inline code comments
- **Examples**: Check `examples/` directory (coming soon)

---

**Built with ❤️ for the AI/ML community**
