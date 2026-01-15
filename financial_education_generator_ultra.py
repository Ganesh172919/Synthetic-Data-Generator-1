"""
Financial Education Dataset Generator - ULTRA OPTIMIZED VERSION
================================================================
TARGET: 20,000 Q&A pairs in 3 hours (~111 Q&A/minute)

Ultra Optimizations:
1. MASSIVE batches (15 Q&A per LLM call)
2. Streamlined prompts for maximum speed
3. Fast hash-based deduplication
4. Async file I/O with large buffers
5. Minimal validation overhead
6. Optimized GPU memory management

Usage:
!python financial_education_generator_ultra.py
"""

# ============================================================================
# SECTION 1: INSTALLATION
# ============================================================================

import subprocess
import sys

def install_dependencies():
    """Install required packages."""
    packages = [
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "torch>=2.0.0",
        "tqdm",
    ]
    
    print("📦 Installing dependencies...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
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
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
# SECTION 3: ULTRA-OPTIMIZED CONFIGURATION
# ============================================================================

@dataclass
class UltraConfig:
    """Ultra-optimized configuration for 20k in 3 hours."""
    
    # Model - Mistral-7B with 4-bit quantization
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    use_quantization: bool = True
    max_new_tokens: int = 1800  # Large for big batches
    temperature: float = 0.75
    top_p: float = 0.95
    
    # ULTRA BATCH SETTINGS - KEY FOR SPEED
    target_dataset_size: int = 20000  # 20k target
    qa_per_generation: int = 15  # Generate 15 Q&A per LLM call!
    save_interval: int = 100  # Buffer 100 before writing
    
    # Output
    output_file: str = "financial_education_dataset_20k.jsonl"
    checkpoint_file: str = "ultra_checkpoint.json"
    
    # Quality (slightly relaxed for speed)
    min_answer_length: int = 50  # Reduced for speed
    
    # Memory
    clear_cache_interval: int = 50  # Clear every 50 batches


CONFIG = UltraConfig()

# ============================================================================
# SECTION 4: EXPANDED CURRICULUM
# ============================================================================

FINANCIAL_CURRICULUM = {
    "Personal Finance Basics": {
        "subtopics": ["Budgeting", "Emergency Funds", "Debt Management", "Credit Scores", 
                      "Saving Strategies", "Financial Goals", "Net Worth", "Cash Flow"],
        "difficulties": ["beginner", "intermediate"]
    },
    "Insurance & Protection": {
        "subtopics": ["Life Insurance", "Health Insurance", "Auto Insurance", 
                      "Homeowners Insurance", "Disability Insurance", "Umbrella Policies"],
        "difficulties": ["beginner", "intermediate"]
    },
    "Retirement Planning": {
        "subtopics": ["401k Plans", "IRA Accounts", "Roth vs Traditional", "Social Security",
                      "Pension Plans", "Early Retirement", "Retirement Income"],
        "difficulties": ["intermediate", "advanced"]
    },
    "Tax Planning": {
        "subtopics": ["Income Tax Basics", "Tax Deductions", "Tax Credits", "Capital Gains",
                      "Tax-Advantaged Accounts", "Estate Taxes", "Tax Filing"],
        "difficulties": ["beginner", "intermediate", "advanced"]
    },
    "Stock Market Investing": {
        "subtopics": ["Stock Basics", "How Stocks Work", "Stock Valuation", "Dividends",
                      "Growth vs Value", "Stock Picking", "Market Orders", "Blue Chips"],
        "difficulties": ["beginner", "intermediate", "advanced"]
    },
    "Bond Investing": {
        "subtopics": ["Bond Basics", "Treasury Bonds", "Corporate Bonds", "Municipal Bonds",
                      "Bond Yields", "Bond Duration", "Bond Ratings", "Bond Funds"],
        "difficulties": ["intermediate", "advanced"]
    },
    "Mutual Funds & ETFs": {
        "subtopics": ["Mutual Fund Basics", "ETF Basics", "Index Funds", "Expense Ratios",
                      "Active vs Passive", "Fund Selection", "Sector Funds"],
        "difficulties": ["beginner", "intermediate"]
    },
    "Portfolio Management": {
        "subtopics": ["Asset Allocation", "Diversification", "Rebalancing", "Risk Management",
                      "Modern Portfolio Theory", "Factor Investing", "Correlation"],
        "difficulties": ["intermediate", "advanced"]
    },
    "Advanced Investing": {
        "subtopics": ["Options Basics", "Options Strategies", "Futures", "Short Selling",
                      "Margin Trading", "Derivatives", "Hedging", "Leverage"],
        "difficulties": ["advanced"]
    },
    "Real Estate": {
        "subtopics": ["Home Buying", "Mortgages", "REITs", "Rental Properties",
                      "Real Estate Investing", "Property Taxes", "Home Equity"],
        "difficulties": ["beginner", "intermediate", "advanced"]
    },
    "Cryptocurrency": {
        "subtopics": ["Bitcoin", "Ethereum", "Blockchain", "Crypto Wallets",
                      "DeFi", "NFTs", "Staking", "Crypto Risks"],
        "difficulties": ["beginner", "intermediate", "advanced"]
    },
    "Economic Concepts": {
        "subtopics": ["Inflation", "Interest Rates", "GDP", "Monetary Policy",
                      "Fiscal Policy", "Business Cycles", "Recession", "Fed Policy"],
        "difficulties": ["intermediate", "advanced"]
    },
    "Behavioral Finance": {
        "subtopics": ["Cognitive Biases", "Loss Aversion", "Herd Mentality", 
                      "Overconfidence", "Anchoring", "Emotional Investing"],
        "difficulties": ["intermediate", "advanced"]
    },
    "Corporate Finance": {
        "subtopics": ["Financial Statements", "Ratio Analysis", "Cash Flow",
                      "Valuation", "Mergers Acquisitions", "IPOs", "Corporate Bonds"],
        "difficulties": ["intermediate", "advanced"]
    },
    "International Finance": {
        "subtopics": ["Forex Basics", "Exchange Rates", "International Investing",
                      "Emerging Markets", "Currency Risk", "ADRs"],
        "difficulties": ["intermediate", "advanced"]
    }
}

QUESTION_TYPES = ["definition", "conceptual", "comparison", "example", "application",
                  "importance", "calculation", "strategy", "risk", "misconception"]

# ============================================================================
# SECTION 5: DATA STRUCTURES
# ============================================================================

@dataclass
class QAPair:
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
    
    def __len__(self):
        with self.lock:
            return len(self.data)


class AsyncWriter:
    """High-performance async file writer."""
    
    def __init__(self, filepath: str, buffer_size: int = 100):
        self.filepath = filepath
        self.buffer_size = buffer_size
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.written_count = AtomicCounter()
        
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
    
    def _writer_loop(self):
        buffer = []
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.5)
                buffer.append(item)
                
                if len(buffer) >= self.buffer_size:
                    self._flush(buffer)
                    buffer = []
            except queue.Empty:
                if buffer:
                    self._flush(buffer)
                    buffer = []
        
        if buffer:
            self._flush(buffer)
    
    def _flush(self, buffer):
        if not buffer:
            return
        mode = 'a' if os.path.exists(self.filepath) else 'w'
        with open(self.filepath, mode, encoding='utf-8') as f:
            for item in buffer:
                f.write(item + '\n')
        self.written_count.increment(len(buffer))
    
    def write(self, qa_pair: QAPair):
        self.queue.put(qa_pair.to_jsonl())
    
    def stop(self):
        self.stop_event.set()
        self.writer_thread.join(timeout=10)
    
    def get_written(self):
        return self.written_count.get()


# ============================================================================
# SECTION 6: MODEL MANAGER
# ============================================================================

class UltraModelManager:
    """Optimized model manager."""
    
    def __init__(self, config: UltraConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        print(f"🔄 Loading: {self.config.model_name}")
        
        if self.device == "cuda":
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")
        else:
            print("⚠️ No GPU! This will be VERY slow. Use Google Colab with GPU.")
        
        # Quantization config
        quant_config = None
        if self.config.use_quantization and self.device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quant_config,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.eval()
        
        if self.device == "cuda":
            print(f"✅ Model loaded! VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        else:
            print("✅ Model loaded on CPU")
    
    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        try:
            formatted = f"[INST] {prompt} [/INST]"
            
            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            input_len = inputs["input_ids"].shape[1]
            text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            return text.strip()
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return ""
    
    def clear_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()


# ============================================================================
# SECTION 7: ULTRA Q&A GENERATOR
# ============================================================================

class UltraQAGenerator:
    """Generates many Q&A pairs per LLM call."""
    
    def __init__(self, model: UltraModelManager, config: UltraConfig):
        self.model = model
        self.config = config
    
    def generate_batch(self, topic: str, subtopic: str, difficulty: str, count: int) -> List[Dict]:
        """Generate multiple Q&A pairs in ONE LLM call."""
        
        prompt = f"""Generate {count} educational finance Q&A pairs about "{subtopic}" ({topic}).
Difficulty: {difficulty}

RULES:
- Each answer: 80-200 words
- Educational only, NO financial advice
- Clear, accurate explanations

FORMAT (follow exactly):
Q1: [question about {subtopic}]
A1: [educational answer]

Q2: [different question about {subtopic}]
A2: [educational answer]

Continue for all {count} Q&A pairs."""

        response = self.model.generate(prompt)
        return self._parse_response(response, topic, subtopic, difficulty)
    
    def _parse_response(self, response: str, topic: str, subtopic: str, difficulty: str) -> List[Dict]:
        """Parse LLM response into Q&A pairs."""
        qa_pairs = []
        
        current_q = None
        current_a_parts = []
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # New question
            if line.upper().startswith('Q') and ':' in line[:5]:
                # Save previous
                if current_q and current_a_parts:
                    answer = ' '.join(current_a_parts).strip()
                    if len(answer) >= self.config.min_answer_length and len(current_q) > 10:
                        qa_pairs.append({
                            "topic": topic,
                            "subtopic": subtopic,
                            "question": current_q,
                            "answer": answer,
                            "difficulty": difficulty,
                            "question_type": self._get_type(current_q)
                        })
                
                current_q = line.split(':', 1)[-1].strip()
                current_a_parts = []
            
            # Answer starts
            elif line.upper().startswith('A') and ':' in line[:5]:
                text = line.split(':', 1)[-1].strip()
                if text:
                    current_a_parts.append(text)
            
            # Answer continuation
            elif current_q and not line.upper().startswith('Q'):
                current_a_parts.append(line)
        
        # Don't forget the last one
        if current_q and current_a_parts:
            answer = ' '.join(current_a_parts).strip()
            if len(answer) >= self.config.min_answer_length and len(current_q) > 10:
                qa_pairs.append({
                    "topic": topic,
                    "subtopic": subtopic,
                    "question": current_q,
                    "answer": answer,
                    "difficulty": difficulty,
                    "question_type": self._get_type(current_q)
                })
        
        return qa_pairs
    
    def _get_type(self, question: str) -> str:
        """Quick question type classification."""
        q = question.lower()
        if "what is" in q or "define" in q:
            return "definition"
        elif "how" in q:
            return "conceptual"
        elif "difference" in q or "compare" in q:
            return "comparison"
        elif "example" in q:
            return "example"
        elif "why" in q:
            return "importance"
        elif "risk" in q:
            return "risk"
        elif "calculate" in q:
            return "calculation"
        else:
            return random.choice(QUESTION_TYPES)


# ============================================================================
# SECTION 8: MAIN GENERATOR
# ============================================================================

class UltraFinancialGenerator:
    """Main ultra-fast generator."""
    
    def __init__(self, config: UltraConfig):
        self.config = config
        self.model = None
        self.qa_gen = None
        self.writer = None
        self.hashes = ThreadSafeSet()
        
        self.generated = AtomicCounter()
        self.rejected = AtomicCounter()
        self.duplicates = AtomicCounter()
        self.batches = AtomicCounter()
        self.start_time = time.time()
    
    def initialize(self):
        print("=" * 60)
        print("🚀 ULTRA FINANCIAL DATASET GENERATOR")
        print(f"🎯 Target: {self.config.target_dataset_size:,} Q&A pairs in 3 hours")
        print("=" * 60)
        
        self.model = UltraModelManager(self.config)
        self.model.load_model()
        
        self.qa_gen = UltraQAGenerator(self.model, self.config)
        self.writer = AsyncWriter(self.config.output_file, self.config.save_interval)
        
        print("✅ Ready to generate!\n")
    
    def get_hash(self, text: str) -> str:
        return hashlib.md5(' '.join(text.lower().split()).encode()).hexdigest()
    
    def get_rate(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed > 0:
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
    
    def process_batch(self, topic: str, subtopic: str, topic_data: Dict):
        """Process one batch of Q&A generation."""
        difficulty = random.choice(topic_data.get("difficulties", ["intermediate"]))
        
        qa_list = self.qa_gen.generate_batch(
            topic, subtopic, difficulty,
            count=self.config.qa_per_generation
        )
        
        for qa in qa_list:
            # Deduplication
            q_hash = self.get_hash(qa["question"])
            if not self.hashes.add(q_hash):
                self.duplicates.increment()
                continue
            
            # Quick validation
            if len(qa["answer"]) < self.config.min_answer_length:
                self.rejected.increment()
                continue
            
            # Create and write
            pair = QAPair(
                id=f"ultra_{self.generated.get()}_{int(time.time()*1000)%100000}",
                **qa
            )
            self.writer.write(pair)
            self.generated.increment()
        
        self.batches.increment()
    
    def print_progress(self):
        gen = self.generated.get()
        target = self.config.target_dataset_size
        pct = (gen / target) * 100
        rate = self.get_rate()
        eta = self.get_eta()
        
        print(f"\r📊 {gen:,}/{target:,} ({pct:.1f}%) | ⚡{rate:.0f}/min | ⏱️ETA:{eta} | "
              f"❌{self.rejected.get()} | 🔄{self.duplicates.get()}", end="", flush=True)
    
    def save_checkpoint(self):
        data = {
            "generated": self.generated.get(),
            "batches": self.batches.get(),
            "time": time.time() - self.start_time
        }
        with open(self.config.checkpoint_file, 'w') as f:
            json.dump(data, f)
    
    def generate(self):
        """Main generation loop."""
        topics = list(FINANCIAL_CURRICULUM.keys())
        
        # Calculate how many batches per subtopic
        total_subtopics = sum(len(t["subtopics"]) for t in FINANCIAL_CURRICULUM.values())
        total_batches_needed = self.config.target_dataset_size // self.config.qa_per_generation + 100
        batches_per_subtopic = max(3, total_batches_needed // total_subtopics)
        
        print(f"📦 {self.config.qa_per_generation} Q&A per batch")
        print(f"📚 {total_subtopics} subtopics, ~{batches_per_subtopic} batches each")
        print(f"🔥 Starting generation...\n")
        
        try:
            rounds = 0
            while self.generated.get() < self.config.target_dataset_size:
                for topic, data in FINANCIAL_CURRICULUM.items():
                    for subtopic in data["subtopics"]:
                        for _ in range(batches_per_subtopic):
                            self.process_batch(topic, subtopic, data)
                            self.print_progress()
                            
                            # Clear cache periodically
                            if self.batches.get() % self.config.clear_cache_interval == 0:
                                self.model.clear_cache()
                            
                            # Checkpoint
                            if self.batches.get() % 50 == 0:
                                self.save_checkpoint()
                            
                            if self.generated.get() >= self.config.target_dataset_size:
                                break
                        if self.generated.get() >= self.config.target_dataset_size:
                            break
                    if self.generated.get() >= self.config.target_dataset_size:
                        break
                
                rounds += 1
                if rounds > 10:  # Safety limit
                    break
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted!")
        
        finally:
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
        print(f"⚡ Rate: {rate:.1f} Q&A per minute")
        print(f"❌ Rejected: {self.rejected.get()}")
        print(f"🔄 Duplicates: {self.duplicates.get()}")
        print(f"📦 Batches: {self.batches.get()}")
        print(f"📁 File: {self.config.output_file}")
        
        if os.path.exists(self.config.output_file):
            size = os.path.getsize(self.config.output_file) / (1024*1024)
            print(f"💾 Size: {size:.2f} MB")
        
        # Performance check
        print("\n📈 PERFORMANCE:")
        in_3hrs = rate * 180
        print(f"   At this rate, 3 hours = {in_3hrs:,.0f} Q&A")
        if self.generated.get() >= self.config.target_dataset_size:
            print("   ✅ TARGET ACHIEVED!")
        else:
            needed = self.config.target_dataset_size / 180
            print(f"   Need {needed:.1f}/min for 20k in 3h")
        
        print("=" * 60)


# ============================================================================
# SECTION 9: UTILITIES
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
# SECTION 10: MAIN
# ============================================================================

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║   ULTRA FINANCIAL DATASET GENERATOR                       ║
    ║   Target: 20,000 Q&A in 3 hours                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU: {gpu} ({mem:.1f} GB)")
    else:
        print("⚠️ NO GPU - Use Colab with GPU for speed!")
    
    config = UltraConfig()
    
    print(f"\n⚙️ Settings:")
    print(f"   Target: {config.target_dataset_size:,}")
    print(f"   Q&A per batch: {config.qa_per_generation}")
    print(f"   Model: {config.model_name}")
    
    generator = UltraFinancialGenerator(config)
    generator.initialize()
    generator.generate()
    
    if os.path.exists(config.output_file):
        validate_dataset(config.output_file)
        sample_dataset(config.output_file)
        
        print("\n✅ Done! Dataset ready.")
        
        # Colab download
        try:
            import google.colab
            from google.colab import files
            print("\n📥 Downloading...")
            files.download(config.output_file)
        except:
            pass


if __name__ == "__main__":
    main()
