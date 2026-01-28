<div align="center">

# 🧬 Synthetic Data Generator

### _Enterprise-Grade AI Dataset Generation Platform_

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=for-the-badge&logo=google-colab)](https://colab.research.google.com/)
[![GPU Optimized](https://img.shields.io/badge/GPU-Optimized-76B900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

<br/>

**Generate high-quality synthetic datasets at unprecedented speed using state-of-the-art LLMs.**

_From 30,000 Q&A pairs in 3 hours on a FREE Google Colab T4 GPU—to unlimited possibilities._

<br/>

[🚀 Quick Start](#-quick-start) •
[📖 Documentation](#-documentation) •
[🌐 Website (Coming Soon)](#-upcoming-website) •
[🤝 Contributing](#-contributing)

---

</div>

## 🎯 Overview

**Synthetic Data Generator** is a high-performance, production-ready Python platform designed to generate large-scale, domain-specific datasets for machine learning and AI training. Built with **extreme speed optimizations**, our system leverages cutting-edge LLM technology to produce high-quality, validated synthetic data.

### ✨ What Makes Us Different

| Feature                       | Description                                                |
| ----------------------------- | ---------------------------------------------------------- |
| 💸 **Zero Cost**              | Runs entirely on Google Colab Free Tier (T4 GPU)           |
| ⚡ **Blazing Fast**           | Up to **167 Q&A pairs/minute** with MEGA batch processing  |
| 🧠 **Intelligent Generation** | 25 Q&A pairs per LLM call with smart prompting             |
| 🛡️ **Bulletproof Safety**     | Emergency save handlers, auto-download on Colab disconnect |
| ✅ **Quality Assured**        | Built-in pattern matching and content validation           |
| 🔄 **Resume Support**         | Checkpoint-based resume for interrupted sessions           |
| 🌍 **Universal Templates**    | Generate datasets for ANY domain, not just finance         |
| 📂 **ML-Ready Output**        | Industry-standard JSONL format for training pipelines      |

---

## 🌐 Upcoming Website

> 🚧 **Coming Soon** — A beautiful, interactive web platform!

We're building a modern web application to make synthetic data generation accessible to everyone. Here's what's planned:

### 🎨 Website Features

| Feature                   | Description                                     | Status     |
| ------------------------- | ----------------------------------------------- | ---------- |
| **Interactive Dashboard** | Real-time generation monitoring with live stats | 🔜 Planned |
| **Custom Domain Builder** | Visual interface to define any dataset domain   | 🔜 Planned |
| **Cloud Generation**      | Run generation jobs in the cloud without Colab  | 🔜 Planned |
| **Dataset Marketplace**   | Share and discover community-generated datasets | 🔜 Planned |
| **API Access**            | RESTful API for programmatic dataset generation | 🔜 Planned |
| **Export Options**        | One-click export to Hugging Face, CSV, Parquet  | 🔜 Planned |
| **Premium Templates**     | Pre-built templates for popular ML use cases    | 🔜 Planned |
| **Team Collaboration**    | Share projects and datasets with your team      | 🔜 Planned |

### 🛠️ Tech Stack (Planned)

```
Frontend:     React + Vite + TailwindCSS
Backend:      Node.js / FastAPI
Database:     PostgreSQL + Redis
Cloud:        AWS / GCP (GPU instances)
Auth:         NextAuth / Clerk
Deployment:   Vercel + Docker
```

### 📅 Development Timeline

| Phase       | Milestone                        | Target  |
| ----------- | -------------------------------- | ------- |
| **Phase 1** | Landing Page & Documentation     | Q1 2026 |
| **Phase 2** | Core Web UI + Local Generation   | Q2 2026 |
| **Phase 3** | Cloud Generation + API           | Q3 2026 |
| **Phase 4** | Marketplace + Community Features | Q4 2026 |

> 💡 **Want to contribute to the website?** See our [Contributing](#-contributing) section!

---

## 🚀 Quick Start

### Prerequisites

- A Google Account (for Google Colab)
- Basic familiarity with Python/Jupyter Notebooks
- **Optional:** Local GPU (RTX 3090/4090 or better recommended)

### ⚡ Option 1: Google Colab (Recommended - Free!)

```bash
# 1. Upload the generator script to Colab
# 2. Configure runtime: Runtime > Change runtime type > T4 GPU
# 3. Run the generator:

!python financial_education_generator_ultra.py
```

> ⏱️ **First run:** ~5 minutes to install dependencies and download model (~5GB)

### 💻 Option 2: Local Machine with GPU

```bash
# Clone the repository
git clone https://github.com/yourusername/synthetic-data-generator.git
cd synthetic-data-generator

# Install dependencies
pip install transformers accelerate bitsandbytes torch tqdm

# Run the generator
python Pre-Work/financial_education_generator_ultra.py
```

### 🌍 Option 3: Universal Dataset Generator

For generating datasets in **any domain** (not just finance):

```bash
python Pre-Work/universal_dataset_generator.py
```

---

## 📁 Project Structure

```
Synthetic Data Generator/
│
├── 📂 Pre-Work/                              # Core generation scripts
│   ├── financial_education_generator_ultra.py   # 🚀 EXTREME SPEED (30k in 3hrs)
│   ├── universal_dataset_generator.py           # 🌍 Universal domain generator
│   └── OPTIMIZATION_GUIDE.md                    # 📖 Performance tuning guide
│
├── 📂 website/                               # 🔜 Coming Soon - Web Platform
│   ├── client/                                  # React frontend
│   └── server/                                  # Backend API
│
├── 📄 Readme.md                              # You are here!
└── 📄 LICENSE                                # MIT License
```

### 📊 Generator Comparison

| Version             | Speed  | Target     | Best For                           |
| ------------------- | ------ | ---------- | ---------------------------------- |
| **Ultra (Finance)** | 🚀🚀🚀 | 30k/3hrs   | Maximum speed, financial education |
| **Universal**       | 🌍⚡⚡ | Any domain | Custom datasets, research          |

---

## 🏗️ System Architecture

Our generator employs multiple cutting-edge optimization strategies:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SYNTHETIC DATA GENERATOR                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│    │   Topic      │───▶│   Batch      │───▶│    LLM       │        │
│    │  Selection   │    │   Prompt     │    │  Inference   │        │
│    └──────────────┘    └──────────────┘    └──────────────┘        │
│                                                   │                 │
│                                                   ▼                 │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│    │    Async     │◀───│    Dedup     │◀───│  Validation  │        │
│    │  File Write  │    │   Engine     │    │   Pipeline   │        │
│    └──────────────┘    └──────────────┘    └──────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 🔧 Core Components

| Component                 | Purpose                                                 |
| ------------------------- | ------------------------------------------------------- |
| **🔧 ExtremeSpeedConfig** | Configurable parameters for speed/quality tradeoffs     |
| **✍️ UltraAsyncWriter**   | High-performance async file writer with 200-item buffer |
| **🔐 ThreadSafeSet**      | Lock-free deduplication using MD5 hashing               |
| **⚡ AtomicCounter**      | Thread-safe progress tracking                           |
| **🛡️ Emergency Handlers** | Auto-save on interrupt, timeout, or crash               |

---

## ⚙️ Configuration

Customize the generator by editing the config class:

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

| Hardware                 | Rate     | 10k Dataset | 30k Dataset |
| ------------------------ | -------- | ----------- | ----------- |
| **T4 GPU** (Colab Free)  | ~100/min | ~1.7 hours  | ~5 hours    |
| **A100 GPU** (Colab Pro) | ~200/min | ~50 min     | ~2.5 hours  |
| **RTX 3090/4090**        | ~150/min | ~1.1 hours  | ~3.5 hours  |
| **CPU Only**             | ~10/min  | ~16 hours   | ~50 hours   |

---

## 💾 Output Format

Generated datasets are saved in ML-ready JSONL format:

```json
{
  "id": "fin_Per_Bud_1234_56789",
  "topic": "Personal Finance",
  "subtopic": "Budgeting Basics",
  "question": "What is the 50/30/20 rule in budgeting?",
  "answer": "The 50/30/20 rule is a simple budgeting framework that suggests allocating 50% of after-tax income to needs, 30% to wants, and 20% to savings and debt repayment...",
  "difficulty": "beginner",
  "question_type": "definition",
  "created_at": "2026-01-28T10:30:00.000000"
}
```

### 📚 Available Domains

<table>
<tr>
<td>

**💰 Financial Education**

- Personal Finance
- Credit & Debt
- Investing
- Banking
- Real Estate
- Tax Planning
- Retirement
- Insurance

</td>
<td>

**🌍 Universal (Any Domain)**

- Healthcare
- Legal
- Education
- Technology
- Science
- History
- Custom Topics
- And more...

</td>
</tr>
</table>

---

## 🛡️ Data Safety Features

### Emergency Save & Recovery

```python
# Force save at any time (works in Colab!)
force_save_and_download()

# Resume from checkpoint
generator.resume_from_checkpoint("checkpoint.json")
```

### Built-in Protections

| Feature               | Protection                            |
| --------------------- | ------------------------------------- |
| **SIGINT/SIGTERM**    | Graceful shutdown with full data save |
| **Colab Disconnect**  | Auto-download before session timeout  |
| **Checkpoint Resume** | Restart from exact last position      |
| **Crash Recovery**    | Emergency buffer flush on any error   |

---

## 🛠️ Technology Stack

| Layer             | Technology                                       |
| ----------------- | ------------------------------------------------ |
| **LLM**           | Mistral-7B-Instruct-v0.2 (4-bit quantized)       |
| **Inference**     | Hugging Face Transformers + BitsAndBytes         |
| **Optimization**  | FlashAttention 2, CUDA acceleration              |
| **Validation**    | Pattern matching, length checks, quality filters |
| **Deduplication** | MD5 hash-based (100x faster than embeddings)     |
| **I/O**           | Async buffered writer with threading             |

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

---

## 🗺️ Roadmap

### ✅ Completed

- [x] Extreme speed batch processing
- [x] Emergency save handlers
- [x] Colab auto-download
- [x] Checkpoint-based resume
- [x] Universal domain support

### 🔜 In Progress

- [ ] **Website Development** (see [Upcoming Website](#-upcoming-website))
- [ ] Interactive web dashboard
- [ ] Cloud-based generation
- [ ] API access

### 📋 Planned

- [ ] Multi-language dataset support
- [ ] Custom domain templates
- [ ] RAG-based factual grounding
- [ ] Fine-tuning integration
- [ ] Dataset marketplace
- [ ] Team collaboration features

---

## 📖 Documentation

| Resource                                                           | Description                        |
| ------------------------------------------------------------------ | ---------------------------------- |
| [OPTIMIZATION_GUIDE.md](Pre-Work/OPTIMIZATION_GUIDE.md)            | Detailed performance tuning guide  |
| [Universal Generator](Pre-Work/universal_dataset_generator.py)     | Generate datasets for any domain   |
| [Ultra Generator](Pre-Work/financial_education_generator_ultra.py) | Maximum speed financial generation |

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Code Contributions

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Contribution Ideas

| Area                 | Examples                                      |
| -------------------- | --------------------------------------------- |
| **🌐 Website**       | React components, UI/UX design, API endpoints |
| **📝 Templates**     | New domain templates, question types          |
| **⚡ Performance**   | Speed optimizations, memory efficiency        |
| **📚 Documentation** | Tutorials, guides, examples                   |
| **🐛 Bug Fixes**     | Issue resolution, error handling              |

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
