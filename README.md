# SynthAgent Engine 🤖

> **Enterprise-Grade Synthetic Data Generation on Google Colab (Free Tier)**

**SynthAgent Engine** is a cutting-edge, agentic AI system designed to democratize high-quality synthetic data generation. Built on **LangGraph** and **LangChain**, it orchestrates a team of specialized AI agents to produce large-scale, domain-specific datasets (with a focus on Finance) significantly cheaper than proprietary APIs—**at zero cost** using Google Colab's Free Tier.

## 🌟 Key Features

- **💸 Zero Cost Architecture**: Optimized to run entirely on a standard Google Colab T4 GPU instance. No paid APIs required.
- **🧠 Multi-Agent Orchestration**: Uses a directed graph of agents (Requirement Parser, Context Builder, Generator, Quality Critic) to ensure high fidelity.
- **⚡ High-Speed Batching**: Implements smart batching (generating 5+ samples per inference set) to maximize throughput on limited hardware.
- **🛡️ Robust Validation**: Integrated **Pydantic** validation ensures all outputs adhere to strict schemas (JSON/CSV).
- **📂 Open & Portable**: Uses open-source weights (`Llama-3-8B-Instruct-bnb-4bit`) and standard Python libraries.

## 🚀 Quick Start Guide

### Prerequisites

- A Google Account (for Google Colab).
- Basic familiarity with Jupyter Notebooks.

### Installation & Execution

1.  **Download**: Get the [SynthAgent_Engine.ipynb](SynthAgent_Engine.ipynb) file from this repository.
2.  **Upload to Colab**:
    - Go to [https://colab.research.google.com/](https://colab.research.google.com/).
    - Click `File` > `Upload notebook`.
    - Select the `.ipynb` file.
3.  **Configure Runtime**:
    - In the Colab menu, go to `Runtime` > `Change runtime type`.
    - Set **Hardware accelerator** to **T4 GPU**.
    - Click **Save**.
4.  **Run**:
    - Click `Runtime` > `Run all` (or press `Ctrl+F9`).
    - _Note_: The first run will take a few minutes to install dependencies and download the model (~5GB).

### Customizing the Generation

Scroll down to **Cell 8 (Run Production Engine)** in the notebook. You will see a variable `USER_REQUEST`.

```python
USER_REQUEST = """
I need a dataset of 50 complex investment banking Q&A pairs.
Focus on M&A, DCF analysis, and LBO models.
The questions should be suitable for a senior analyst interview.
"""
```

Edit this string to change the domain, task type, or difficulty of the data you want to generate.

## 🏗 System Architecture

The engine functions as a state machine where data flows through distinct processing nodes:

1.  **🔍 RequirementParser**: Analyzes your natural language prompt to understand the domain, valid output formats, and constraints.
2.  **🌍 ContextBuilder**: "Hallucinates" rich, realistic scenarios, background documents, or personas to ground the synthetic data generation, ensuring it's not generic.
3.  **⚙️ MasterGenerator**: The core LLM worker. It takes the context and requirements and generates data in batches.
4.  **⚖️ QualityController**: (Optional/Simulated) A critic node that evaluates samples against quality metrics (correctness, style, adherence to constraints).
5.  **💾 Exporter**: Aggregates valid samples and saves them to `synth_data.csv`.

## 🛠 Technology Stack

- **Orchestration**: [LangChain](https://www.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/)
- **LLM Inference**: [Hugging Face Transformers](https://huggingface.co/docs/transformers/) & [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) (4-bit quantization)
- **Base Model**: `unsloth/llama-3-8b-Instruct-bnb-4bit` (Llama 3 optimized)
- **Validation**: [Pydantic v2](https://docs.pydantic.dev/)
- **Data Handling**: Pandas, Rich (logging)

## ⚠️ Performance & Limitations

- **Speed**: On a T4 GPU, generation speed is approximately **1-2 seconds per sample**. Generating the full 10,000 dataset will take several hours.
- **VRAM Usage**: The system is tuned to use ~8GB VRAM. If you modify the code to use larger batch sizes or longer contexts, you may trigger Out-Of-Memory (OOM) errors.
- **Session Timeout**: Google Colab Free Tier may disconnect active sessions after a period of inactivity. Keep the tab open or use a browser extension to prevent timeout for long runs.

## 🔮 Roadmap

- [ ] **Vector Database Integration**: Add FAISS/Chroma for RAG-based generation.
- [ ] **Fine-Tuning Loop**: specific node to train a small adapter on high-quality synthetic data validation.
- [ ] **Multi-Modal Support**: Capability to generate chart description data.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements, especially for new "Agent Strategies" or "Domain Templates".

---

**License**: MIT
**Author**: Antigravity (Google DeepMind)
