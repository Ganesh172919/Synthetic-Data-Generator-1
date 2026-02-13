"""
Financial Education Dataset Generator - EXTREME SPEED VERSION
===============================================================
TARGET: 30,000 Q&A pairs in 3 hours (~167 Q&A/minute)

EXTREME Optimizations:
1. MEGA batches (25 Q&A per LLM call)
2. PARALLEL generation with ThreadPoolExecutor
3. Flash Attention 2 for maximum throughput
4. Continuous KV-cache optimization
5. Minimal prompt tokens, maximum output
6. Ultra-fast hash deduplication
7. Aggressive memory management
8. Async I/O with massive buffers
9. Optional: Faster Phi-3-mini model support

🛡️ EMERGENCY SAVE & DOWNLOAD PROTECTION:
- Auto-saves data every 3 minutes (more frequent)
- Catches Ctrl+C, Colab disconnect, errors
- Download anytime with: force_save_and_download() or download_now()

Reality-aligned notes (important for learners):

- This script is optimized for *throughput*, not for “perfect” output quality.
  It intentionally relaxes some checks so it can hit high Q&A/minute targets.
- It auto-installs dependencies on import. This is convenient on Google Colab, but
  in controlled environments you typically want explicit, pinned installs.
- `flash-attn` (Flash Attention 2) can fail to install depending on CUDA/toolchain
  and often fails on Windows. The script continues without it.
- Finance content can look authoritative while being wrong. Treat outputs as
  synthetic educational text, not financial advice.

Conceptual flow (simplified):

  install deps
    → load quantized model (optionally Flash Attention)
    → generate mega-batches (25 Q&A per call)
    → fast parse + lightweight validation
    → hash dedup (exact-ish)
    → async buffered writer
    → periodic checkpoint + emergency-save handlers

Usage:
!python financial_education_generator_ultra.py
"""

# ============================================================================
# SECTION 1: INSTALLATION
# ============================================================================

import subprocess
import sys

def install_dependencies():
    """
    Install required packages.

    Educational note:
    - This script is often run in Colab, so it tries to self-bootstrap dependencies.
    - `flash-attn` is optional and may fail to build; failures are intentionally ignored.
    - If you want reproducibility, create a pinned requirements file instead.
    """
    packages = [
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "torch>=2.0.0",
        "tqdm",
        "flash-attn --no-build-isolation",  # Flash Attention 2
    ]
    
    print("📦 Installing dependencies...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + package.split())
        except:
            pass
    print("✅ Dependencies ready!\n")

install_dependencies()

# ============================================================================
# SECTION 2: IMPORTS
# ============================================================================

import os
import json
import time
import random
import hashlib
import threading
import queue
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async CUDA operations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging as transformers_logging
)
transformers_logging.set_verbosity_error()

from tqdm import tqdm

# ============================================================================
# SECTION 2.5: SIGNAL HANDLING & EMERGENCY SAVE/DOWNLOAD
# ============================================================================

import signal
import atexit

# Global reference for emergency save
_global_generator = None
_global_writer = None
_emergency_save_done = False

def force_save_and_download():
    """Force save current progress and trigger download - CALL THIS ANYTIME to get your file!"""
    global _global_writer, _emergency_save_done
    
    print("\n" + "="*60)
    print("📥 FORCE SAVE & DOWNLOAD TRIGGERED")
    print("="*60)
    
    # Flush the writer
    if _global_writer is not None:
        try:
            _global_writer.stop()
            print(f"✅ Writer flushed! Written: {_global_writer.get_written():,} records")
        except Exception as e:
            print(f"⚠️ Writer flush warning: {e}")
    
    # Save checkpoint
    if _global_generator is not None:
        try:
            _global_generator.save_checkpoint()
            print("✅ Checkpoint saved!")
        except:
            pass
    
    # Try to download in Colab
    output_file = "financial_education_dataset_30k.jsonl"
    checkpoint_file = "ultra_checkpoint.json"
    
    if os.path.exists(output_file):
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"📁 Dataset file: {output_file} ({size_mb:.2f} MB)")
        
        # Count records
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                record_count = sum(1 for _ in f)
            print(f"📊 Total records in file: {record_count:,}")
        except:
            pass
        
        # Try Colab download
        try:
            from google.colab import files
            print("\n📥 DOWNLOADING FILE NOW...")
            files.download(output_file)
            print("✅ Download started! Check your browser.")
        except ImportError:
            print("\n📋 Not running in Colab. File saved locally.")
            print(f"   Path: {os.path.abspath(output_file)}")
        except Exception as e:
            print(f"⚠️ Download error: {e}")
            print(f"📋 File is saved at: {os.path.abspath(output_file)}")
    else:
        print("⚠️ No output file found yet!")
    
    _emergency_save_done = True
    print("="*60 + "\n")


def emergency_save_handler(signum=None, frame=None):
    """Handle interrupts gracefully with data preservation."""
    global _emergency_save_done
    
    if _emergency_save_done:
        return
    
    signal_name = signal.Signals(signum).name if signum else "UNKNOWN"
    print(f"\n\n🚨 SIGNAL {signal_name} RECEIVED - EMERGENCY SAVE STARTING...")
    force_save_and_download()
    
    # Exit after emergency save
    if signum in (signal.SIGTERM, signal.SIGINT):
        print("👋 Exiting gracefully after save...")
        sys.exit(0)


def setup_emergency_handlers():
    """Setup all emergency save handlers."""
    print("🛡️ Setting up emergency save handlers...")
    
    # Register atexit handler (runs on normal exit or unhandled exceptions)
    atexit.register(force_save_and_download)
    
    # Register signal handlers
    try:
        signal.signal(signal.SIGINT, emergency_save_handler)  # Ctrl+C
        print("   ✓ SIGINT (Ctrl+C) handler")
    except:
        pass
    
    try:
        signal.signal(signal.SIGTERM, emergency_save_handler)  # Termination
        print("   ✓ SIGTERM handler")
    except:
        pass
    
    # Unix-specific signals
    if hasattr(signal, 'SIGHUP'):
        try:
            signal.signal(signal.SIGHUP, emergency_save_handler)  # Terminal closed
            print("   ✓ SIGHUP handler")
        except:
            pass
    
    print("✅ Emergency handlers ready!\n")
    print("💡 TIP: Call force_save_and_download() at ANY time to get your file!\n")


# ============================================================================
# SECTION 3: EXTREME SPEED CONFIGURATION
# ============================================================================

@dataclass
class ExtremeSpeedConfig:
    """
    Extreme speed configuration for 30k in 3 hours.

    Educational note (how to read these knobs):
    - `qa_per_generation` drives how many Q&As you ask for per model call.
      Higher = fewer model calls (faster) but higher risk of format drift.
    - `parallel_workers` suggests parallelism, but on a single GPU true parallel
      generation is often limited; see `ParallelBatchProcessor` notes.
    - `save_interval` and the async writer buffer help keep generation compute-bound.
    - `use_flash_attention` can improve throughput but is optional and environment-dependent.

    Practical tuning examples (starting points):
    - Colab T4: qa_per_generation=10–20, parallel_workers=1–2, max_new_tokens=1200–1800
    - RTX 3090/4090: qa_per_generation=20–30, parallel_workers=1–2, max_new_tokens=1800–2500
    - CPU-only: not recommended for this “ultra” profile; use smaller targets or the universal generator.
    """
    
    # Model - Options for speed (uncomment preferred)
    # Option 1: Mistral-7B (better quality, slower)
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    # Option 2: Phi-3 mini (faster, good quality) - uncomment to use
    # model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    # Option 3: Gemma-2B (fastest, decent quality) - uncomment to use
    # model_name: str = "google/gemma-2b-it"
    
    use_quantization: bool = True
    use_flash_attention: bool = True  # Flash Attention 2 for speed
    max_new_tokens: int = 2500  # Larger for mega batches
    temperature: float = 0.8
    top_p: float = 0.95
    
    # EXTREME BATCH SETTINGS - KEY FOR SPEED
    target_dataset_size: int = 30000  # 30k target
    qa_per_generation: int = 25  # Generate 25 Q&A per LLM call!
    parallel_workers: int = 2  # Parallel generation threads
    save_interval: int = 200  # Buffer 200 before writing
    
    # Output
    output_file: str = "financial_education_dataset_30k.jsonl"
    checkpoint_file: str = "ultra_checkpoint.json"
    
    # Quality (relaxed for speed)
    min_answer_length: int = 40  # Lower threshold
    
    # Memory - aggressive cleanup
    clear_cache_interval: int = 25  # Clear every 25 batches
    auto_save_interval: int = 180  # Auto-save every 3 minutes


CONFIG = ExtremeSpeedConfig()

# Practical tuning quick guide (non-executable notes):
#
# 1) If you hit CUDA OOM:
#    - reduce CONFIG.qa_per_generation
#    - reduce CONFIG.max_new_tokens
#    - increase CONFIG.clear_cache_interval frequency (smaller number)
#
# 2) If output quality is low / format drifts:
#    - reduce CONFIG.temperature
#    - reduce CONFIG.qa_per_generation
#    - tighten prompt rules in `UltraFastQAGenerator.generate_mega_batch`
#
# 3) If generation is slow but stable:
#    - increase CONFIG.qa_per_generation gradually
#    - ensure quantization is enabled (CONFIG.use_quantization=True)

# ============================================================================
# SECTION 4: EXPANDED CURRICULUM (More subtopics = more diversity)
# ============================================================================

FINANCIAL_CURRICULUM = {
    "Personal Finance Basics": {
        "subtopics": ["Budgeting Methods", "Emergency Funds", "Debt Management", "Credit Scores", 
                      "Saving Strategies", "Financial Goals", "Net Worth", "Cash Flow",
                      "Income Management", "Expense Tracking", "Financial Planning"],
        "difficulties": ["beginner", "intermediate"]
    },
    "Insurance & Protection": {
        "subtopics": ["Life Insurance", "Health Insurance", "Auto Insurance", 
                      "Homeowners Insurance", "Disability Insurance", "Umbrella Policies",
                      "Term vs Whole Life", "Insurance Deductibles", "Claims Process"],
        "difficulties": ["beginner", "intermediate"]
    },
    "Retirement Planning": {
        "subtopics": ["401k Plans", "IRA Accounts", "Roth vs Traditional", "Social Security",
                      "Pension Plans", "Early Retirement", "Retirement Income", "Required Distributions",
                      "Catch-up Contributions", "Retirement Calculators"],
        "difficulties": ["intermediate", "advanced"]
    },
    "Tax Planning": {
        "subtopics": ["Income Tax Basics", "Tax Deductions", "Tax Credits", "Capital Gains",
                      "Tax-Advantaged Accounts", "Estate Taxes", "Tax Filing", "Tax Brackets",
                      "Tax Loss Harvesting", "Estimated Taxes", "Tax Forms"],
        "difficulties": ["beginner", "intermediate", "advanced"]
    },
    "Stock Market Investing": {
        "subtopics": ["Stock Basics", "How Stocks Work", "Stock Valuation", "Dividends",
                      "Growth vs Value", "Stock Picking", "Market Orders", "Blue Chips",
                      "Stock Splits", "Earnings Reports", "Market Indices"],
        "difficulties": ["beginner", "intermediate", "advanced"]
    },
    "Bond Investing": {
        "subtopics": ["Bond Basics", "Treasury Bonds", "Corporate Bonds", "Municipal Bonds",
                      "Bond Yields", "Bond Duration", "Bond Ratings", "Bond Funds",
                      "Bond Laddering", "Yield Curve", "TIPS"],
        "difficulties": ["intermediate", "advanced"]
    },
    "Mutual Funds & ETFs": {
        "subtopics": ["Mutual Fund Basics", "ETF Basics", "Index Funds", "Expense Ratios",
                      "Active vs Passive", "Fund Selection", "Sector Funds", "Target Date Funds",
                      "Fund Performance", "NAV Calculation"],
        "difficulties": ["beginner", "intermediate"]
    },
    "Portfolio Management": {
        "subtopics": ["Asset Allocation", "Diversification", "Rebalancing", "Risk Management",
                      "Modern Portfolio Theory", "Factor Investing", "Correlation",
                      "Risk Tolerance", "Investment Horizon", "Portfolio Optimization"],
        "difficulties": ["intermediate", "advanced"]
    },
    "Advanced Investing": {
        "subtopics": ["Options Basics", "Options Strategies", "Futures", "Short Selling",
                      "Margin Trading", "Derivatives", "Hedging", "Leverage",
                      "Volatility Trading", "Greeks in Options"],
        "difficulties": ["advanced"]
    },
    "Real Estate": {
        "subtopics": ["Home Buying", "Mortgages", "REITs", "Rental Properties",
                      "Real Estate Investing", "Property Taxes", "Home Equity",
                      "Mortgage Types", "Down Payments", "Real Estate Markets"],
        "difficulties": ["beginner", "intermediate", "advanced"]
    },
    "Cryptocurrency": {
        "subtopics": ["Bitcoin", "Ethereum", "Blockchain", "Crypto Wallets",
                      "DeFi", "NFTs", "Staking", "Crypto Risks",
                      "Crypto Exchanges", "Altcoins", "Mining"],
        "difficulties": ["beginner", "intermediate", "advanced"]
    },
    "Economic Concepts": {
        "subtopics": ["Inflation", "Interest Rates", "GDP", "Monetary Policy",
                      "Fiscal Policy", "Business Cycles", "Recession", "Fed Policy",
                      "Unemployment", "Economic Indicators", "Supply and Demand"],
        "difficulties": ["intermediate", "advanced"]
    },
    "Behavioral Finance": {
        "subtopics": ["Cognitive Biases", "Loss Aversion", "Herd Mentality", 
                      "Overconfidence", "Anchoring", "Emotional Investing",
                      "Mental Accounting", "Confirmation Bias", "FOMO in Investing"],
        "difficulties": ["intermediate", "advanced"]
    },
    "Corporate Finance": {
        "subtopics": ["Financial Statements", "Ratio Analysis", "Cash Flow",
                      "Valuation", "Mergers Acquisitions", "IPOs", "Corporate Bonds",
                      "Capital Structure", "Working Capital", "Financial Modeling"],
        "difficulties": ["intermediate", "advanced"]
    },
    "International Finance": {
        "subtopics": ["Forex Basics", "Exchange Rates", "International Investing",
                      "Emerging Markets", "Currency Risk", "ADRs",
                      "Global Diversification", "Trade Balance", "Currency Hedging"],
        "difficulties": ["intermediate", "advanced"]
    },
    "Banking & Credit": {
        "subtopics": ["Checking Accounts", "Savings Accounts", "CDs", "Money Market",
                      "Credit Cards", "Loans", "Interest Calculations", "Banking Fees",
                      "Online Banking", "Credit Building"],
        "difficulties": ["beginner", "intermediate"]
    },
    "Financial Technology": {
        "subtopics": ["Robo-Advisors", "Mobile Banking", "Payment Apps", "Budgeting Apps",
                      "Investment Apps", "Digital Wallets", "Open Banking", "AI in Finance"],
        "difficulties": ["beginner", "intermediate"]
    }
}

QUESTION_TYPES = ["definition", "conceptual", "comparison", "example", "application",
                  "importance", "calculation", "strategy", "risk", "misconception",
                  "step-by-step", "pros-cons", "when-to-use"]

# ============================================================================
# SECTION 5: DATA STRUCTURES
# ============================================================================

@dataclass
class QAPair:
    """
    A single Q&A record emitted by the finance generator.

    Educational note:
    - Keeping a dedicated data structure makes it easier to evolve the schema over time.
    - `to_jsonl()` emits one JSON object suitable for JSON Lines (one record per line).
    """
    id: str
    topic: str
    subtopic: str
    question: str
    answer: str
    difficulty: str
    question_type: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_jsonl(self) -> str:
        return json.dumps(asdict(self))


class AtomicCounter:
    """Thread-safe counter."""
    def __init__(self, initial=0):
        self.value = initial
        self.lock = threading.Lock()
    
    def increment(self, amount=1):
        with self.lock:
            self.value += amount
            return self.value
    
    def get(self):
        with self.lock:
            return self.value


class ThreadSafeSet:
    """Thread-safe set for deduplication."""
    def __init__(self):
        self.data = set()
        self.lock = threading.Lock()
    
    def add(self, item) -> bool:
        with self.lock:
            if item in self.data:
                return False
            self.data.add(item)
            return True
    
    def bulk_check_and_add(self, items: List[str]) -> List[bool]:
        """Check and add multiple items at once - faster for batch operations."""
        with self.lock:
            results = []
            for item in items:
                if item in self.data:
                    results.append(False)
                else:
                    self.data.add(item)
                    results.append(True)
            return results
    
    def __len__(self):
        with self.lock:
            return len(self.data)


class UltraAsyncWriter:
    """
    Ultra high-performance async file writer with large buffers.

    Why this exists (educational note):
    - High-volume generation can become I/O-bound if every record is written synchronously.
    - We use a queue + background thread to batch writes and reduce syscalls.

    Edge cases:
    - If generation outpaces disk writes, the queue can fill and backpressure occurs.
    - Always call `stop()` (or rely on emergency handlers) to flush buffers on exit.
    """
    
    def __init__(self, filepath: str, buffer_size: int = 200):
        self.filepath = filepath
        self.buffer_size = buffer_size
        self.queue = queue.Queue(maxsize=10000)  # Larger queue
        self.stop_event = threading.Event()
        self.written_count = AtomicCounter()
        
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
    
    def _writer_loop(self):
        buffer = []
        last_flush = time.time()
        
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.1)
                buffer.append(item)
                
                # Flush on size or time
                if len(buffer) >= self.buffer_size or (time.time() - last_flush > 2 and buffer):
                    self._flush(buffer)
                    buffer = []
                    last_flush = time.time()
            except queue.Empty:
                if buffer and time.time() - last_flush > 1:
                    self._flush(buffer)
                    buffer = []
                    last_flush = time.time()
        
        if buffer:
            self._flush(buffer)
    
    def _flush(self, buffer):
        if not buffer:
            return
        try:
            mode = 'a' if os.path.exists(self.filepath) else 'w'
            with open(self.filepath, mode, encoding='utf-8', buffering=65536) as f:  # 64KB buffer
                f.write('\n'.join(buffer) + '\n')
            self.written_count.increment(len(buffer))
        except Exception as e:
            print(f"\n⚠️ Write error: {e}")
    
    def write(self, qa_pair: QAPair):
        try:
            self.queue.put_nowait(qa_pair.to_jsonl())
        except queue.Full:
            # Force flush if queue is full
            # Reality note: this is backpressure (we wait briefly, then block on `put`).
            time.sleep(0.1)
            self.queue.put(qa_pair.to_jsonl())
    
    def write_batch(self, qa_pairs: List[QAPair]):
        """Write multiple pairs at once."""
        for qa in qa_pairs:
            self.write(qa)
    
    def stop(self):
        self.stop_event.set()
        self.writer_thread.join(timeout=15)
    
    def get_written(self):
        return self.written_count.get()


# ============================================================================
# SECTION 6: EXTREME SPEED MODEL MANAGER
# ============================================================================

class ExtremeModelManager:
    """
    Ultra-optimized model manager with optional Flash Attention.

    Responsibilities:
    - load tokenizer + model (optionally quantized)
    - choose attention implementation (Flash Attention 2 if available)
    - provide a `generate(prompt)` method used by the Q&A generator

    Educational note:
    Even if you “parallelize” upstream (threads), most single-GPU inference ends up
    serialized by the model/generation lock. This script focuses on minimizing per-call
    overhead and maximizing output per call (mega-batching).
    """
    
    def __init__(self, config: ExtremeSpeedConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generation_lock = threading.Lock()  # For thread-safe generation
        
    def load_model(self):
        print(f"🔄 Loading: {self.config.model_name}")
        
        if self.device == "cuda":
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")
        else:
            print("⚠️ No GPU! This will be VERY slow. Use Google Colab with GPU.")
            # Educational note: this ultra profile is tuned for GPU throughput and intentionally
            # does not provide a robust CPU fallback path.
            return
        
        # Quantization config
        quant_config = None
        if self.config.use_quantization:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        
        # Check for Flash Attention
        attn_implementation = None
        if self.config.use_flash_attention:
            try:
                import flash_attn
                attn_implementation = "flash_attention_2"
                print("⚡ Flash Attention 2 enabled!")
            except ImportError:
                # Flash Attention is an optional acceleration. If it cannot be imported (common on
                # some platforms/toolchains), we fall back to default attention implementation.
                print("⚠️ Flash Attention not available, using default")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Model kwargs
        model_kwargs = {
            "quantization_config": quant_config,
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }
        
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        self.model.eval()
        
        # Compile for speed if possible (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.device == "cuda":
            try:
                # Use reduce-overhead for inference
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("⚡ Torch compile enabled!")
            except Exception as e:
                print(f"⚠️ Torch compile not applied: {e}")
        
        print(f"✅ Model loaded! VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        """Thread-safe generation with optimizations."""
        with self.generation_lock:
            try:
                # Minimal formatting
                formatted = f"[INST] {prompt} [/INST]"
                
                inputs = self.tokenizer(
                    formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=400,  # Shorter input = faster
                    padding=False
                ).to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,  # Greedy-ish for speed
                    early_stopping=False,
                    repetition_penalty=1.1  # Prevent loops
                )
                
                input_len = inputs["input_ids"].shape[1]
                text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                return text.strip()
                
            except Exception as e:
                print(f"\n❌ Generation error: {e}")
                return ""
    
    def clear_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# ============================================================================
# SECTION 7: ULTRA-FAST Q&A GENERATOR
# ============================================================================

class UltraFastQAGenerator:
    """
    Generates many Q&A pairs per LLM call with optimized prompts.

    Educational note:
    This class is the core throughput trick:
    - Instead of 1 model call per record, we ask for many Q&As in a strict format.
    - We then parse the raw text quickly and apply lightweight checks.

    Common failure modes:
    - Format drift: missing Q/A markers, merged answers, extra commentary.
    - Repetition: multiple Q&As that are near-duplicates within the same mega-batch.
    """
    
    def __init__(self, model: ExtremeModelManager, config: ExtremeSpeedConfig):
        self.model = model
        self.config = config
    
    def generate_mega_batch(self, topic: str, subtopic: str, difficulty: str, count: int) -> List[Dict]:
        """Generate many Q&A pairs in ONE LLM call with minimal prompt."""
        
        # Ultra-compact prompt for speed
        prompt = f"""Generate {count} educational Q&A about "{subtopic}" ({topic}).
Level: {difficulty}

Rules: 60-150 word answers, educational only, accurate.

Format exactly:
Q1: [question]
A1: [answer]

Q2: [question]
A2: [answer]

...up to Q{count}"""

        response = self.model.generate(prompt)
        return self._fast_parse(response, topic, subtopic, difficulty)
    
    def _fast_parse(self, response: str, topic: str, subtopic: str, difficulty: str) -> List[Dict]:
        """Ultra-fast parsing of LLM response."""
        qa_pairs = []
        
        current_q = None
        current_a_lines = []
        
        lines = response.split('\n')
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            upper = stripped.upper()
            
            # Detect question start (Q1:, Q2:, etc.)
            if len(upper) >= 2 and upper[0] == 'Q' and upper[1:3].lstrip('0123456789')[:1] in (':', ''):
                # Save previous Q&A
                if current_q and current_a_lines:
                    answer = ' '.join(current_a_lines).strip()
                    if len(answer) >= self.config.min_answer_length and len(current_q) > 8:
                        qa_pairs.append({
                            "topic": topic,
                            "subtopic": subtopic,
                            "question": current_q,
                            "answer": answer,
                            "difficulty": difficulty,
                            "question_type": self._quick_type(current_q)
                        })
                
                # Extract question text
                colon_idx = stripped.find(':')
                if colon_idx != -1:
                    current_q = stripped[colon_idx+1:].strip()
                else:
                    current_q = stripped[2:].strip()
                current_a_lines = []
            
            # Detect answer start (A1:, A2:, etc.)
            elif len(upper) >= 2 and upper[0] == 'A' and upper[1:3].lstrip('0123456789')[:1] in (':', ''):
                colon_idx = stripped.find(':')
                if colon_idx != -1:
                    text = stripped[colon_idx+1:].strip()
                    if text:
                        current_a_lines.append(text)
            
            # Continuation of answer
            elif current_q and not upper.startswith('Q'):
                current_a_lines.append(stripped)
        
        # Don't forget the last Q&A
        if current_q and current_a_lines:
            answer = ' '.join(current_a_lines).strip()
            if len(answer) >= self.config.min_answer_length and len(current_q) > 8:
                qa_pairs.append({
                    "topic": topic,
                    "subtopic": subtopic,
                    "question": current_q,
                    "answer": answer,
                    "difficulty": difficulty,
                    "question_type": self._quick_type(current_q)
                })
        
        return qa_pairs
    
    def _quick_type(self, q: str) -> str:
        """Ultra-fast question type detection."""
        ql = q.lower()
        if "what is" in ql or "define" in ql or "what are" in ql:
            return "definition"
        if "how" in ql:
            return "conceptual" if "how does" in ql else "step-by-step"
        if "difference" in ql or "compare" in ql or "vs" in ql:
            return "comparison"
        if "example" in ql:
            return "example"
        if "why" in ql:
            return "importance"
        if "risk" in ql or "danger" in ql:
            return "risk"
        if "calculate" in ql or "formula" in ql:
            return "calculation"
        if "advantage" in ql or "benefit" in ql or "pros" in ql:
            return "pros-cons"
        return random.choice(["conceptual", "application", "strategy"])


# ============================================================================
# SECTION 8: BATCH PROCESSOR WITH PARALLEL GENERATION
# ============================================================================

class ParallelBatchProcessor:
    """
    Processes multiple batches with a “parallel” API surface.

    Reality-aligned note:
    The current implementation is effectively sequential on a single GPU. True parallel
    generation typically requires either:
    - multiple GPUs
    - async batching at the inference server level
    - or model-level support for concurrent generation requests

    The educational value of this class is that it isolates the “batch spec planning”
    from the generation mechanics, making it easier to upgrade later.
    """
    
    def __init__(self, qa_generator: UltraFastQAGenerator, config: ExtremeSpeedConfig):
        self.qa_gen = qa_generator
        self.config = config
        
    def generate_parallel_batches(self, batch_specs: List[Tuple[str, str, str]]) -> List[List[Dict]]:
        """Generate multiple batches - currently sequential due to GPU constraint."""
        # Note: True parallel requires multiple GPUs or async inference
        # For single GPU, we optimize the sequential path
        results = []
        for topic, subtopic, difficulty in batch_specs:
            qa_list = self.qa_gen.generate_mega_batch(
                topic, subtopic, difficulty, self.config.qa_per_generation
            )
            results.append(qa_list)
        return results


# ============================================================================
# SECTION 9: MAIN EXTREME GENERATOR
# ============================================================================

class ExtremeFinancialGenerator:
    """
    Main ultra-fast generator orchestrator (target: 30k / ~3 hours).

    Responsibilities:
    - initialize emergency handlers (so you can always save/download)
    - load the model manager and Q&A generator
    - run the generate → validate/dedup → write loop
    - periodically checkpoint and clear GPU cache

    Educational note:
    This is a “single-process pipeline” optimized for simplicity. In production you’d
    usually separate concerns (API → queue → worker → storage) and add stronger safety
    filters for finance-related content.
    """
    
    def __init__(self, config: ExtremeSpeedConfig):
        self.config = config
        self.model = None
        self.qa_gen = None
        self.batch_processor = None
        self.writer = None
        self.hashes = ThreadSafeSet()
        
        self.generated = AtomicCounter()
        self.rejected = AtomicCounter()
        self.duplicates = AtomicCounter()
        self.batches = AtomicCounter()
        self.start_time = time.time()
        self.last_auto_save = time.time()
    
    def initialize(self):
        global _global_generator, _global_writer
        
        print("=" * 60)
        print("🚀 EXTREME SPEED FINANCIAL DATASET GENERATOR")
        print(f"🎯 Target: {self.config.target_dataset_size:,} Q&A pairs in 3 hours")
        print(f"⚡ Required rate: ~{self.config.target_dataset_size/180:.0f} Q&A/min")
        print("=" * 60)
        
        # Setup emergency handlers first
        setup_emergency_handlers()
        
        self.model = ExtremeModelManager(self.config)
        self.model.load_model()
        
        self.qa_gen = UltraFastQAGenerator(self.model, self.config)
        self.batch_processor = ParallelBatchProcessor(self.qa_gen, self.config)
        self.writer = UltraAsyncWriter(self.config.output_file, self.config.save_interval)
        
        # Register with global handlers for emergency save
        _global_generator = self
        _global_writer = self.writer
        
        print("✅ Ready to generate at EXTREME speed!")
        print("💡 TIP: Run force_save_and_download() anytime to get your file!\n")
    
    def get_hash(self, text: str) -> str:
        """Fast hash for deduplication."""
        return hashlib.md5(text.lower().encode()).hexdigest()[:16]
    
    def get_rate(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed > 5:  # Wait 5 seconds for stable rate
            return (self.generated.get() / elapsed) * 60
        return 0
    
    def get_eta(self) -> str:
        rate = self.get_rate()
        if rate > 0:
            remaining = self.config.target_dataset_size - self.generated.get()
            mins = remaining / rate
            if mins < 60:
                return f"{mins:.1f}m"
            return f"{mins/60:.1f}h"
        return "..."
    
    def process_batch(self, topic: str, subtopic: str, difficulty: str):
        """Process one mega batch of Q&A generation."""
        qa_list = self.qa_gen.generate_mega_batch(
            topic, subtopic, difficulty,
            count=self.config.qa_per_generation
        )
        
        # Batch hash check
        hashes = [self.get_hash(qa["question"]) for qa in qa_list]
        is_unique = self.hashes.bulk_check_and_add(hashes)
        
        valid_pairs = []
        for qa, unique in zip(qa_list, is_unique):
            if not unique:
                self.duplicates.increment()
                continue
            
            if len(qa["answer"]) < self.config.min_answer_length:
                self.rejected.increment()
                continue
            
            pair = QAPair(
                id=f"x_{self.generated.get()}_{time.time_ns()%1000000}",
                **qa
            )
            valid_pairs.append(pair)
            self.generated.increment()
        
        if valid_pairs:
            self.writer.write_batch(valid_pairs)
        
        self.batches.increment()
    
    def print_progress(self):
        gen = self.generated.get()
        target = self.config.target_dataset_size
        pct = (gen / target) * 100
        rate = self.get_rate()
        eta = self.get_eta()
        written = self.writer.get_written()
        
        # Projection for 3 hours
        projected_3h = rate * 180 if rate > 0 else 0
        
        print(f"\r🔥 {gen:,}/{target:,} ({pct:.1f}%) | ⚡{rate:.0f}/min | 📊3h→{projected_3h:,.0f} | "
              f"⏱️{eta} | 💾{written:,}", end="", flush=True)
    
    def save_checkpoint(self):
        data = {
            "generated": self.generated.get(),
            "batches": self.batches.get(),
            "duplicates": self.duplicates.get(),
            "rejected": self.rejected.get(),
            "time": time.time() - self.start_time,
            "rate_per_min": self.get_rate()
        }
        try:
            with open(self.config.checkpoint_file, 'w') as f:
                json.dump(data, f)
        except:
            pass
    
    def auto_save(self):
        """Check and perform auto-save every 3 minutes."""
        if time.time() - self.last_auto_save >= self.config.auto_save_interval:
            self.last_auto_save = time.time()
            self.save_checkpoint()
            gen = self.generated.get()
            rate = self.get_rate()
            print(f"\n\n💾 AUTO-SAVE: {gen:,} Q&A | Rate: {rate:.0f}/min | File: {self.config.output_file}")
            try:
                from google.colab import files
                print("   📥 Run force_save_and_download() to get file now")
            except:
                pass
            print()
    
    def generate(self):
        """Main generation loop with maximum speed."""
        topics = list(FINANCIAL_CURRICULUM.keys())
        all_subtopics = []
        
        for topic, data in FINANCIAL_CURRICULUM.items():
            for subtopic in data["subtopics"]:
                for diff in data["difficulties"]:
                    all_subtopics.append((topic, subtopic, diff))
        
        random.shuffle(all_subtopics)
        
        # Calculate iterations needed
        total_batches = (self.config.target_dataset_size // self.config.qa_per_generation) + 500
        batches_per_cycle = len(all_subtopics)
        cycles_needed = (total_batches // batches_per_cycle) + 1
        
        print(f"📦 {self.config.qa_per_generation} Q&A per mega-batch")
        print(f"📚 {len(all_subtopics)} topic combinations")
        print(f"🔄 ~{cycles_needed} cycles needed")
        print(f"🎯 Need ~{self.config.target_dataset_size/180:.0f} Q&A/min for 30k/3h")
        print(f"🔥 Starting EXTREME generation...\n")
        
        try:
            cycle = 0
            while self.generated.get() < self.config.target_dataset_size:
                random.shuffle(all_subtopics)
                
                for topic, subtopic, difficulty in all_subtopics:
                    self.process_batch(topic, subtopic, difficulty)
                    self.print_progress()
                    
                    # Memory management
                    if self.batches.get() % self.config.clear_cache_interval == 0:
                        self.model.clear_cache()
                    
                    # Auto-save check
                    self.auto_save()
                    
                    # Checkpoint
                    if self.batches.get() % 100 == 0:
                        self.save_checkpoint()
                    
                    if self.generated.get() >= self.config.target_dataset_size:
                        break
                
                cycle += 1
                if cycle > cycles_needed + 5:
                    print("\n⚠️ Max cycles reached")
                    break
        
        except KeyboardInterrupt:
            print("\n\n⚠️ INTERRUPTED - Saving...")
        
        except Exception as e:
            print(f"\n\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            print("\n\n📥 Final flush...")
            self.writer.stop()
            self.save_checkpoint()
            self.print_final()
    
    def print_final(self):
        elapsed = time.time() - self.start_time
        hours = elapsed / 3600
        rate = self.generated.get() / (elapsed / 60) if elapsed > 0 else 0
        
        print("\n\n" + "=" * 60)
        print("🏁 GENERATION COMPLETE")
        print("=" * 60)
        print(f"📊 Generated: {self.generated.get():,} Q&A pairs")
        print(f"⏱️ Time: {hours:.2f} hours ({elapsed:.0f} seconds)")
        print(f"⚡ Actual rate: {rate:.1f} Q&A per minute")
        print(f"❌ Rejected: {self.rejected.get()}")
        print(f"🔄 Duplicates: {self.duplicates.get()}")
        print(f"📦 Batches: {self.batches.get()}")
        print(f"📁 File: {self.config.output_file}")
        
        if os.path.exists(self.config.output_file):
            size = os.path.getsize(self.config.output_file) / (1024*1024)
            print(f"💾 Size: {size:.2f} MB")
        
        # Performance analysis
        print("\n📈 PERFORMANCE ANALYSIS:")
        projected_3h = rate * 180
        print(f"   Projected 3-hour output: {projected_3h:,.0f} Q&A")
        
        if projected_3h >= 30000:
            print("   ✅ ON TRACK FOR 30K/3H TARGET!")
        elif projected_3h >= 20000:
            print("   ⚠️ Good for 20k, need more speed for 30k")
        else:
            print(f"   ❌ Need {30000/180:.0f}/min for 30k target")
        
        print("=" * 60)


# ============================================================================
# SECTION 10: UTILITIES
# ============================================================================

def validate_dataset(filepath: str):
    """Quick dataset validation."""
    if not os.path.exists(filepath):
        print("❌ File not found")
        return
    
    stats = defaultdict(int)
    total = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                qa = json.loads(line)
                total += 1
                stats[qa.get("topic", "?")] += 1
            except:
                pass
    
    print(f"\n📊 Dataset: {total:,} Q&A pairs")
    print("📚 By Topic:")
    for topic, count in sorted(stats.items(), key=lambda x: -x[1])[:10]:
        print(f"   {topic}: {count:,}")


def sample_dataset(filepath: str, n: int = 3):
    """Show sample entries."""
    if not os.path.exists(filepath):
        return
    
    print(f"\n📋 Samples:")
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            qa = json.loads(line)
            print(f"\n--- {i+1} ---")
            print(f"Q: {qa.get('question', '')[:100]}...")
            print(f"A: {qa.get('answer', '')[:150]}...")


# ============================================================================
# SECTION 11: MAIN
# ============================================================================

def download_now(filename: str = None):
    """Download the dataset file NOW."""
    if filename is None:
        filename = "financial_education_dataset_30k.jsonl"
    
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return
    
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            record_count = sum(1 for _ in f)
    except:
        record_count = "?"
    
    print(f"📁 File: {filename}")
    print(f"📊 Records: {record_count:,}" if isinstance(record_count, int) else f"📊 Records: {record_count}")
    print(f"💾 Size: {size_mb:.2f} MB")
    
    try:
        from google.colab import files
        print("📥 Starting download...")
        files.download(filename)
        print("✅ Download started!")
    except ImportError:
        print(f"\n📋 Not in Colab. File at: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"⚠️ Download error: {e}")


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║   🔥 EXTREME SPEED FINANCIAL DATASET GENERATOR 🔥        ║
    ║   Target: 30,000 Q&A in 3 hours (~167/min)               ║
    ║   🛡️ WITH EMERGENCY SAVE PROTECTION                       ║
    ╚═══════════════════════════════════════════════════════════╝
    
    📌 Key Features:
       • 25 Q&A per LLM call (mega-batches)
       • Flash Attention 2 for speed
       • Optimized prompts & parsing
       • Auto-save every 3 minutes
    
    📌 To download your data AT ANY TIME:
       • force_save_and_download()
       • download_now()
    """)
    
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU: {gpu} ({mem:.1f} GB)")
        
        # Check compute capability
        major, minor = torch.cuda.get_device_capability(0)
        print(f"   Compute Capability: {major}.{minor}")
        if major >= 8:
            print("   ✅ Supports Flash Attention 2")
        else:
            print("   ⚠️ Flash Attention may not work (need SM >= 8.0)")
    else:
        print("⚠️ NO GPU - Use Colab with GPU for speed!")
        print("   Go to: Runtime > Change runtime type > GPU")
    
    config = ExtremeSpeedConfig()
    
    print(f"\n⚙️ EXTREME Settings:")
    print(f"   Target: {config.target_dataset_size:,}")
    print(f"   Q&A per mega-batch: {config.qa_per_generation}")
    print(f"   Model: {config.model_name}")
    print(f"   Flash Attention: {config.use_flash_attention}")
    print(f"   Output: {config.output_file}")
    print(f"   Auto-save: Every {config.auto_save_interval//60} minutes")
    
    generator = ExtremeFinancialGenerator(config)
    generator.initialize()
    generator.generate()
    
    if os.path.exists(config.output_file):
        validate_dataset(config.output_file)
        sample_dataset(config.output_file)
        
        print("\n✅ Done! Dataset ready.")
        
        try:
            import google.colab
            from google.colab import files
            print("\n📥 Downloading final dataset...")
            files.download(config.output_file)
        except ImportError:
            print(f"\n📋 File saved at: {os.path.abspath(config.output_file)}")
        except Exception as e:
            print(f"⚠️ Auto-download failed: {e}")
            print("   Run download_now() to download manually")


if __name__ == "__main__":
    main()
