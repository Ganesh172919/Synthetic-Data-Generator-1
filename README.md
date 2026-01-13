# 🤖 SynthAgent Engine

> **A Multi-Agent Synthetic Data Generator for High-Quality Financial Q&A Pairs**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Open Source LLM](https://img.shields.io/badge/LLM-Mistral--7B-orange.svg)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Overview

**SynthAgent Engine** is an autonomous multi-agent system designed to generate high-quality synthetic datasets for training AI models. Built with **LangChain** and **LangGraph**, it leverages open-source LLMs (Mistral-7B-Instruct) with 4-bit quantization to run entirely on **Google Colab Free Tier** (under 12GB RAM).

The engine generates **10,000+ financial question-answer pairs** across multiple categories and difficulty levels, incorporating built-in quality control loops and human-in-the-loop (HITL) checkpoints.

---

## ✨ Key Features

| Feature                         | Description                                                                   |
| ------------------------------- | ----------------------------------------------------------------------------- |
| 🧠 **Multi-Agent Architecture** | Modular agents for parsing, context building, generation, and quality control |
| 💰 **100% Free**                | Runs on Google Colab Free Tier with open-source models                        |
| 📊 **Financial Domain Focus**   | Covers 14+ financial categories with expert-curated knowledge base            |
| 🎯 **Quality Control Loop**     | Automated quality scoring with retry mechanisms                               |
| 👤 **HITL Checkpoints**         | Human-in-the-loop review points at configurable intervals                     |
| 📈 **Difficulty Distribution**  | Configurable beginner to expert content generation                            |
| 💾 **Memory Efficient**         | 4-bit quantization keeps RAM usage under 12GB                                 |
| 📦 **Export Ready**             | Outputs clean CSV with full metadata                                          |

---

## 🏗️ Architecture

### Agent System

```
┌─────────────────────────────────────────────────────────────────┐
│                    SynthAgent Engine Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   [User Input]                                                  │
│        │                                                        │
│        v                                                        │
│   ┌─────────────────────┐                                       │
│   │ RequirementParser   │ ──> Parse natural language request    │
│   │       Agent         │                                       │
│   └─────────────────────┘                                       │
│        │                                                        │
│        v                                                        │
│   ┌─────────────────────┐                                       │
│   │  ContextBuilder     │ ──> Build domain-specific context     │
│   │       Agent         │                                       │
│   └─────────────────────┘                                       │
│        │                                                        │
│        v                                                        │
│   ┌─────────────────────┐                                       │
│   │ MasterDataGenerator │ ──> Generate Q&A pairs in batches     │
│   │       Agent         │                                       │
│   └─────────────────────┘                                       │
│        │                                                        │
│        v                                                        │
│   ┌─────────────────────┐      ┌──────────────┐                 │
│   │  QualityController  │ ──> │ Score < 7.0? │ ──> Retry        │
│   │       Agent         │      └──────────────┘                 │
│   └─────────────────────┘              │                        │
│        │                               │ Pass                   │
│        v                               v                        │
│   ┌─────────────────────┐   ┌─────────────────────┐             │
│   │ DatasetAggregator   │   │   HITL Checkpoint   │             │
│   └─────────────────────┘   └─────────────────────┘             │
│        │                                                        │
│        v                                                        │
│   ┌─────────────────────┐                                       │
│   │   ExporterAgent     │ ──> CSV + Metadata JSON               │
│   └─────────────────────┘                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### LangGraph Workflow

The engine uses **LangGraph's StateGraph** for orchestration:

```
parse → context → generate → quality → aggregate → export
                      ↑                    │
                      └────── retry ───────┘
```

---

## 📚 Financial Categories

The knowledge base covers **14 financial domains**:

| Category                 | Topics                                                       |
| ------------------------ | ------------------------------------------------------------ |
| 📈 **Investing**         | Stocks, bonds, ETFs, mutual funds, portfolio diversification |
| 🏦 **Banking**           | Savings accounts, CDs, interest rates, FDIC insurance        |
| 📋 **Taxation**          | Income tax, capital gains, deductions, tax brackets          |
| 🎯 **Retirement**        | 401(k), IRA, Roth IRA, pension, Social Security              |
| 💵 **Personal Finance**  | Budgeting, emergency fund, credit score, debt management     |
| 📊 **Stock Markets**     | NYSE, NASDAQ, orders, trading strategies                     |
| 🏢 **Corporate Finance** | Financial statements, EBITDA, M&A                            |
| ⚠️ **Risk Management**   | Diversification, hedging, VaR                                |
| 🏠 **Real Estate**       | Property investment, mortgages                               |
| 🔐 **Insurance**         | Life, health, property insurance                             |
| 💼 **Accounting**        | GAAP, auditing, bookkeeping                                  |
| 🪙 **Cryptocurrency**    | Bitcoin, blockchain, DeFi                                    |
| 📜 **Derivatives**       | Options, futures, swaps                                      |
| ⚖️ **Regulations**       | SEC, compliance, financial laws                              |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Colab account (recommended) or local GPU with 12GB+ VRAM
- HuggingFace account (for model access)

### Installation

#### Option 1: Google Colab (Recommended)

1. Open `synthagent_engine.ipynb` in Google Colab
2. Run cells sequentially
3. Generated dataset will be saved to `/content/output/`

#### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/your-username/synthetic-data-generator.git
cd synthetic-data-generator

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.36.0 accelerate>=0.25.0 bitsandbytes>=0.41.0
pip install langchain>=0.3.0 langchain-community>=0.3.0 langchain-huggingface>=0.1.0
pip install langgraph>=0.2.0 pydantic>=2.0.0 pandas numpy tqdm rich

# Run the engine
python synthagent_engine.py
```

### Basic Usage

```python
from synthagent_engine import run_generation, preview_samples

# Quick test (2-5 minutes, 10 samples)
df, stats, csv_path = run_generation(target_samples=10)

# Full generation (4-6 hours, 10000 samples)
df, stats, csv_path = run_generation(target_samples=10000)

# Preview generated samples
preview_samples(df, n=5)
```

---

## ⚙️ Configuration

All settings are controlled via the `Config` class:

```python
class Config:
    # Model settings
    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
    USE_4BIT = True
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.7

    # Generation settings
    TARGET_SAMPLES = 10000
    BATCH_SIZE = 20
    SAMPLES_PER_LLM_CALL = 5

    # Quality thresholds
    MIN_QUALITY_SCORE = 7.0
    MAX_CORRECTION_ROUNDS = 2
    MAX_CONSECUTIVE_FAILURES = 5

    # HITL settings
    HITL_CHECKPOINT_INTERVAL = 500
    ENABLE_HITL = True

    # Output settings
    OUTPUT_DIR = "/content/output"
    CSV_FILENAME = "financial_qa_dataset.csv"
```

---

## 📊 Output Format

### CSV Structure

| Column       | Type   | Description                                  |
| ------------ | ------ | -------------------------------------------- |
| `question`   | string | The financial question                       |
| `answer`     | string | Comprehensive answer (2-4 sentences)         |
| `category`   | string | Financial domain category                    |
| `difficulty` | string | beginner/intermediate/advanced/expert        |
| `keywords`   | list   | Key financial concepts mentioned             |
| `reasoning`  | string | Chain-of-thought reasoning (when applicable) |

### Metadata JSON

```json
{
  "generated_at": "2026-01-13T21:00:00",
  "model_used": "mistralai/Mistral-7B-Instruct-v0.2",
  "statistics": {
    "total": 10000,
    "avg_quality": 7.8,
    "by_category": {"investing": 1500, "banking": 1200, ...},
    "by_difficulty": {"beginner": 2500, "intermediate": 4000, ...}
  },
  "config": {
    "batch_size": 20,
    "quality_threshold": 7.0,
    "temperature": 0.7
  }
}
```

---

## 🔄 Quality Control System

The engine implements a **multi-stage quality assurance pipeline**:

1. **Automated Scoring** - Each batch is scored on 5 dimensions:

   - Coherence (logical structure)
   - Accuracy (factual correctness)
   - Completeness (answer thoroughness)
   - Clarity (language quality)
   - Relevance (domain appropriateness)

2. **Quality Threshold** - Samples must score ≥ 7.0/10 to pass

3. **Retry Mechanism** - Failed batches are regenerated (up to 2 attempts)

4. **Consecutive Failure Handling** - Generation stops after 5 consecutive failures

5. **HITL Checkpoints** - Manual review points every 500 samples

---

## 📁 Project Structure

```
Synthetic Data Generator 1/
├── README.md                  # This file
├── synthagent_engine.py       # Main Python script
├── synthagent_engine.ipynb    # Jupyter Notebook version
└── .git/                      # Git repository
```

---

## 🛠️ Technical Stack

| Component           | Technology                       |
| ------------------- | -------------------------------- |
| **LLM**             | Mistral-7B-Instruct-v0.2         |
| **Quantization**    | 4-bit NF4 (BitsAndBytes)         |
| **Orchestration**   | LangGraph StateGraph             |
| **LLM Integration** | LangChain + HuggingFace Pipeline |
| **Data Validation** | Pydantic v2                      |
| **Data Processing** | Pandas + NumPy                   |
| **UI/Progress**     | Rich + tqdm                      |

---

## 📈 Performance

| Metric                   | Value                   |
| ------------------------ | ----------------------- |
| **Memory Usage**         | ~8-10 GB GPU RAM        |
| **Generation Rate**      | ~0.5-1.0 samples/second |
| **Time for 10K samples** | 4-6 hours               |
| **Quality Pass Rate**    | ~85-90%                 |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Mistral AI](https://mistral.ai/) for the open-source Mistral-7B model
- [LangChain](https://langchain.com/) for the LLM orchestration framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) for the workflow graph engine
- [HuggingFace](https://huggingface.co/) for model hosting and transformers library

---

<div align="center">

**Made with ❤️ for the AI community**

[⬆ Back to Top](#-synthagent-engine)

</div>
