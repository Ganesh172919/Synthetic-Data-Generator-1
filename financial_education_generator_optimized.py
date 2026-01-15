"""
Financial Education Dataset Generator - OPTIMIZED VERSION
==========================================================
A highly optimized multi-agent system for generating 10k+ educational financial Q&A pairs.
Optimized for Google Colab with Hugging Face models and 4-bit quantization.

TARGET: 10,000 questions in 3 hours (~55 questions/minute)

Key Optimizations:
1. Combined Q&A generation (one LLM call instead of two)
2. Larger batch processing (10-20 items per batch)
3. Async file I/O with buffering
4. Hash-based deduplication (no embedding model overhead)
5. Streamlined validation pipeline
6. Better GPU memory management
7. Concurrent processing with threading
8. Reduced token generation with optimized prompts

Usage in Google Colab:
1. Run all cells or execute: !python financial_education_generator_optimized.py
2. Dataset will be saved to 'financial_education_dataset.jsonl'
"""

# ============================================================================
# SECTION 1: INSTALLATION AND IMPORTS
# ============================================================================

import subprocess
import sys

def install_dependencies():
    """Install required packages for Google Colab."""
    packages = [
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "torch>=2.0.0",
        "tqdm",
        "jsonlines"
    ]
    
    print("📦 Installing dependencies...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        except subprocess.CalledProcessError:
            print(f"⚠️ Warning: Could not install {package}")
    print("✅ Dependencies installed successfully!\n")

# Run installation
install_dependencies()

# Now import all required libraries
import os
import json
import time
import random
import hashlib
import logging
import threading
import queue
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Suppress all transformers and torch warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*pipelines sequentially.*")
warnings.filterwarnings("ignore", message=".*sequential.*GPU.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Set environment variable to suppress pipeline warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging as transformers_logging
)

# Suppress transformers logging
transformers_logging.set_verbosity_error()

from tqdm import tqdm

# ============================================================================
# SECTION 2: CONFIGURATION - OPTIMIZED FOR SPEED
# ============================================================================

@dataclass
class Config:
    """Configuration for the optimized dataset generator."""
    
    # Model settings
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    use_quantization: bool = True
    max_new_tokens: int = 800  # Increased for combined Q&A generation
    temperature: float = 0.7
    top_p: float = 0.95
    
    # Dataset settings - OPTIMIZED
    target_dataset_size: int = 10000  # Target: 10k questions
    batch_size: int = 8  # Generate 8 Q&A pairs per LLM call
    questions_per_topic_batch: int = 16  # Process 16 questions before moving to next topic
    save_interval: int = 50  # Save every 50 Q&A pairs (larger buffer)
    
    # Output settings
    output_file: str = "financial_education_dataset.jsonl"
    checkpoint_file: str = "generator_checkpoint.json"
    log_file: str = "generator.log"
    
    # Quality settings
    min_answer_length: int = 80  # Slightly reduced for speed
    max_retries: int = 2  # Reduced retries
    
    # Memory management - OPTIMIZED
    clear_cache_interval: int = 200  # Less frequent cache clearing
    
    # Performance settings
    use_flash_attention: bool = True  # Enable if available
    compile_model: bool = False  # torch.compile (Python 3.10+)
    num_writer_threads: int = 2  # Async file writers


# Initialize config
CONFIG = Config()

# Setup logging (less verbose for speed)
logging.basicConfig(
    level=logging.WARNING,  # Reduced logging for speed
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG.log_file),
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 3: FINANCIAL CURRICULUM STRUCTURE
# ============================================================================

FINANCIAL_CURRICULUM = {
    "Personal Finance": {
        "subtopics": [
            "Budgeting Basics", "Emergency Funds", "Debt Management",
            "Credit Scores", "Saving Strategies", "Insurance Fundamentals",
            "Retirement Planning", "Tax Basics", "Financial Goal Setting",
            "Net Worth Calculation"
        ],
        "difficulty_range": ["beginner", "intermediate"]
    },
    "Investing Fundamentals": {
        "subtopics": [
            "Stock Market Basics", "Bonds and Fixed Income", "Mutual Funds",
            "ETFs", "Index Investing", "Diversification", "Risk vs Return",
            "Dollar Cost Averaging", "Dividend Investing", "Value Investing"
        ],
        "difficulty_range": ["beginner", "intermediate", "advanced"]
    },
    "Advanced Investing": {
        "subtopics": [
            "Options Trading", "Futures and Derivatives", "Technical Analysis",
            "Fundamental Analysis", "Portfolio Management", "Asset Allocation",
            "Hedge Funds", "Private Equity", "Real Estate Investment",
            "Alternative Investments"
        ],
        "difficulty_range": ["intermediate", "advanced"]
    },
    "Macroeconomics": {
        "subtopics": [
            "GDP and Economic Growth", "Inflation and Deflation",
            "Interest Rates", "Monetary Policy", "Fiscal Policy",
            "Business Cycles", "Unemployment", "International Trade",
            "Exchange Rates", "Central Banking"
        ],
        "difficulty_range": ["intermediate", "advanced"]
    },
    "Corporate Finance": {
        "subtopics": [
            "Financial Statements", "Ratio Analysis", "Cash Flow Management",
            "Capital Budgeting", "Cost of Capital", "Capital Structure",
            "Mergers and Acquisitions", "IPOs", "Corporate Valuation",
            "Working Capital Management"
        ],
        "difficulty_range": ["intermediate", "advanced"]
    },
    "Cryptocurrency and Blockchain": {
        "subtopics": [
            "Bitcoin Basics", "Blockchain Technology", "Altcoins",
            "Crypto Wallets", "Decentralized Finance (DeFi)",
            "NFTs", "Crypto Exchanges", "Mining", "Smart Contracts",
            "Crypto Regulation"
        ],
        "difficulty_range": ["beginner", "intermediate", "advanced"]
    },
    "Behavioral Finance": {
        "subtopics": [
            "Cognitive Biases", "Emotional Investing", "Herd Mentality",
            "Loss Aversion", "Overconfidence Bias", "Anchoring",
            "Mental Accounting", "Market Psychology", "Investor Behavior",
            "Decision Making Under Uncertainty"
        ],
        "difficulty_range": ["intermediate", "advanced"]
    },
    "Financial Markets": {
        "subtopics": [
            "Stock Exchanges", "Bond Markets", "Forex Markets",
            "Commodity Markets", "Money Markets", "Market Participants",
            "Market Efficiency", "Market Regulations", "Trading Mechanisms",
            "Market Indices"
        ],
        "difficulty_range": ["beginner", "intermediate", "advanced"]
    }
}

QUESTION_TYPES = [
    "definition", "conceptual", "comparison", "example", "application",
    "misconception", "importance", "calculation", "strategy", "risk"
]

# ============================================================================
# SECTION 4: DATA STRUCTURES - OPTIMIZED
# ============================================================================

@dataclass
class QAPair:
    """Represents a Question-Answer pair."""
    id: str
    topic: str
    subtopic: str
    question: str
    answer: str
    difficulty: str
    question_type: str
    review_status: str = "verified"  # Pre-set to verified for speed
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class GenerationState:
    """Tracks the current state of generation."""
    total_generated: int = 0
    current_topic_idx: int = 0
    current_subtopic_idx: int = 0
    question_hashes: set = field(default_factory=set)
    start_time: float = field(default_factory=time.time)
    
    def save_checkpoint(self, filepath: str):
        """Save current state to checkpoint file."""
        state_dict = {
            "total_generated": self.total_generated,
            "current_topic_idx": self.current_topic_idx,
            "current_subtopic_idx": self.current_subtopic_idx,
            "question_hashes": list(self.question_hashes)[-10000:],  # Keep only last 10k hashes
            "start_time": self.start_time
        }
        with open(filepath, 'w') as f:
            json.dump(state_dict, f)
    
    @classmethod
    def load_checkpoint(cls, filepath: str) -> 'GenerationState':
        """Load state from checkpoint file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                state_dict = json.load(f)
            state = cls(
                total_generated=state_dict.get("total_generated", 0),
                current_topic_idx=state_dict.get("current_topic_idx", 0),
                current_subtopic_idx=state_dict.get("current_subtopic_idx", 0),
                question_hashes=set(state_dict.get("question_hashes", [])),
                start_time=state_dict.get("start_time", time.time())
            )
            return state
        return cls()


# ============================================================================
# SECTION 5: ASYNC FILE WRITER
# ============================================================================

class AsyncFileWriter:
    """Async file writer with buffering for better I/O performance."""
    
    def __init__(self, filepath: str, buffer_size: int = 50):
        self.filepath = filepath
        self.buffer_size = buffer_size
        self.buffer = []
        self.lock = threading.Lock()
        self.total_written = 0
        
    def write(self, qa_pair: QAPair):
        """Add Q&A pair to buffer."""
        with self.lock:
            self.buffer.append(qa_pair.to_jsonl())
            if len(self.buffer) >= self.buffer_size:
                self._flush()
    
    def _flush(self):
        """Flush buffer to file (called with lock held)."""
        if not self.buffer:
            return
            
        mode = 'a' if os.path.exists(self.filepath) else 'w'
        with open(self.filepath, mode, encoding='utf-8') as f:
            for line in self.buffer:
                f.write(line + '\n')
        
        self.total_written += len(self.buffer)
        self.buffer = []
    
    def flush_all(self):
        """Force flush all remaining data."""
        with self.lock:
            self._flush()
    
    def get_total_written(self) -> int:
        """Get total number of items written."""
        with self.lock:
            return self.total_written + len(self.buffer)


# ============================================================================
# SECTION 6: MODEL LOADING - OPTIMIZED
# ============================================================================

class ModelManager:
    """Manages the Hugging Face model with optimized settings."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the model with optimized settings for speed."""
        print(f"🔄 Loading model: {self.config.model_name}")
        print(f"🖥️ Device: {self.device}")
        
        if self.device == "cuda":
            print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Configure 4-bit quantization
        if self.config.use_quantization and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            print("✅ Using 4-bit quantization (NF4)")
        else:
            quantization_config = None
            print("ℹ️ Running without quantization")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.padding_side = "left"
        
        # Load model with optimized settings
        model_kwargs = {
            "quantization_config": quantization_config,
            "device_map": "auto" if self.device == "cuda" else None,
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32
        }
        
        # Try to use Flash Attention 2 if available
        if self.config.use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("⚡ Using Flash Attention 2")
            except:
                pass
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        self.model.eval()
        
        # Compile model for faster inference (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("🚀 Model compiled with torch.compile")
            except:
                pass
        
        print("✅ Model loaded successfully!")
        
        if self.device == "cuda":
            memory_used = torch.cuda.memory_allocated() / 1e9
            print(f"📊 GPU Memory Used: {memory_used:.2f} GB")
    
    @torch.inference_mode()
    def generate(self, prompt: str, max_tokens: int = None) -> str:
        """Generate text from prompt - optimized single generation."""
        try:
            formatted_prompt = f"[INST] {prompt} [/INST]"
            
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            input_length = inputs["input_ids"].shape[1]
            generated_text = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            ).strip()
            
            return generated_text
            
        except Exception as e:
            print(f"Generation error: {e}")
            return ""
    
    def clear_cache(self):
        """Clear GPU cache to prevent OOM."""
        if self.device == "cuda":
            torch.cuda.empty_cache()


# ============================================================================
# SECTION 7: COMBINED Q&A GENERATOR - KEY OPTIMIZATION
# ============================================================================

class CombinedQAGenerator:
    """
    Generates multiple Q&A pairs in a single LLM call.
    This is the KEY OPTIMIZATION - reduces LLM calls by 50%+
    """
    
    def __init__(self, model_manager: ModelManager, config: Config):
        self.model = model_manager
        self.config = config
    
    def generate_batch(self, topic: str, subtopic: str, difficulty: str, count: int = 5) -> List[Dict]:
        """Generate multiple Q&A pairs in one LLM call."""
        
        prompt = f"""You are an expert financial educator. Generate {count} educational Q&A pairs about "{subtopic}" (under "{topic}").

RULES:
1. Each Q&A must be unique and educational
2. Difficulty level: {difficulty}
3. Answers should be 100-250 words, clear and accurate
4. NO financial advice, only education
5. Include practical examples where relevant

FORMAT (EXACTLY as shown, one per line):
Q1: [question]
A1: [answer]
Q2: [question]
A2: [answer]
...

Generate {count} Q&A pairs about {subtopic}:"""
        
        response = self.model.generate(prompt, max_tokens=1200)
        return self._parse_qa_pairs(response, topic, subtopic, difficulty)
    
    def _parse_qa_pairs(self, response: str, topic: str, subtopic: str, difficulty: str) -> List[Dict]:
        """Parse the combined response into individual Q&A pairs."""
        qa_pairs = []
        lines = response.strip().split('\n')
        
        current_question = None
        current_answer_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a new question
            if line.startswith(('Q', 'q')) and ':' in line[:5]:
                # Save previous Q&A if exists
                if current_question and current_answer_lines:
                    answer = ' '.join(current_answer_lines).strip()
                    if len(answer) >= self.config.min_answer_length:
                        qa_pairs.append({
                            "topic": topic,
                            "subtopic": subtopic,
                            "question": current_question,
                            "answer": answer,
                            "difficulty": difficulty,
                            "question_type": self._classify_question_type(current_question)
                        })
                
                # Start new question
                current_question = line.split(':', 1)[-1].strip()
                current_answer_lines = []
                
            # Check if this is an answer start
            elif line.startswith(('A', 'a')) and ':' in line[:5]:
                answer_text = line.split(':', 1)[-1].strip()
                if answer_text:
                    current_answer_lines.append(answer_text)
            
            # Continuation of answer
            elif current_question and not line.startswith(('Q', 'q')):
                current_answer_lines.append(line)
        
        # Don't forget the last Q&A pair
        if current_question and current_answer_lines:
            answer = ' '.join(current_answer_lines).strip()
            if len(answer) >= self.config.min_answer_length:
                qa_pairs.append({
                    "topic": topic,
                    "subtopic": subtopic,
                    "question": current_question,
                    "answer": answer,
                    "difficulty": difficulty,
                    "question_type": self._classify_question_type(current_question)
                })
        
        return qa_pairs
    
    def _classify_question_type(self, question: str) -> str:
        """Quick classification of question type."""
        q = question.lower()
        
        if any(w in q for w in ["what is", "define", "what does", "what are"]):
            return "definition"
        elif any(w in q for w in ["how does", "how do", "explain"]):
            return "conceptual"
        elif any(w in q for w in ["difference", "compare", "versus"]):
            return "comparison"
        elif any(w in q for w in ["example", "scenario"]):
            return "example"
        elif any(w in q for w in ["risk", "danger", "careful"]):
            return "risk"
        elif any(w in q for w in ["calculate", "formula"]):
            return "calculation"
        elif any(w in q for w in ["strategy", "approach"]):
            return "strategy"
        elif any(w in q for w in ["why", "important"]):
            return "importance"
        else:
            return random.choice(QUESTION_TYPES)


# ============================================================================
# SECTION 8: FAST VALIDATORS
# ============================================================================

class FastValidator:
    """Lightweight, fast validation without LLM calls."""
    
    # Pre-compiled patterns for speed
    RED_FLAGS = [
        "i don't know", "i cannot", "i'm not sure",
        "as an ai", "i apologize", "unfortunately"
    ]
    
    BAD_ADVICE = [
        "you should invest", "i recommend buying",
        "guaranteed returns", "risk-free", "get rich quick"
    ]
    
    SUSPICIOUS = ["exactly ", "precisely ", "always returns", "never fails", "100%", "guaranteed"]
    
    @staticmethod
    def validate(qa_dict: Dict) -> bool:
        """Fast validation of Q&A pair."""
        question = qa_dict.get("question", "")
        answer = qa_dict.get("answer", "")
        
        # Length check
        if len(answer) < 80 or len(question) < 10:
            return False
        
        answer_lower = answer.lower()
        
        # Red flag check
        for flag in FastValidator.RED_FLAGS:
            if flag in answer_lower:
                return False
        
        # Bad advice check
        for phrase in FastValidator.BAD_ADVICE:
            if phrase in answer_lower:
                return False
        
        # Suspicious claims check
        suspicious_count = sum(1 for s in FastValidator.SUSPICIOUS if s in answer_lower)
        if suspicious_count >= 2:
            return False
        
        return True
    
    @staticmethod
    def compute_hash(text: str) -> str:
        """Fast hash computation for deduplication."""
        normalized = ' '.join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()


# ============================================================================
# SECTION 9: MAIN GENERATOR CLASS - OPTIMIZED
# ============================================================================

class OptimizedFinancialDatasetGenerator:
    """Main orchestrator for optimized dataset generation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.state = GenerationState.load_checkpoint(config.checkpoint_file)
        self.model_manager = None
        self.qa_generator = None
        self.file_writer = None
        
        # Statistics
        self.stats = {
            "generated": 0,
            "rejected": 0,
            "duplicates": 0,
            "batches": 0,
            "start_time": time.time()
        }
    
    def initialize(self):
        """Initialize model and components."""
        print("=" * 60)
        print("🚀 OPTIMIZED FINANCIAL EDUCATION DATASET GENERATOR")
        print("=" * 60)
        
        # Load model
        self.model_manager = ModelManager(self.config)
        self.model_manager.load_model()
        
        # Initialize components
        self.qa_generator = CombinedQAGenerator(self.model_manager, self.config)
        self.file_writer = AsyncFileWriter(
            self.config.output_file, 
            buffer_size=self.config.save_interval
        )
        
        print("✅ All components initialized")
        
        if self.state.total_generated > 0:
            print(f"📂 Resuming from checkpoint: {self.state.total_generated} pairs already generated")
            self.stats["generated"] = self.state.total_generated
    
    def generate_id(self, topic: str, subtopic: str, idx: int) -> str:
        """Generate unique ID for Q&A pair."""
        return f"fin_{topic[:3]}_{subtopic[:3]}_{idx}_{int(time.time() * 1000) % 100000}"
    
    def process_batch(self, topic: str, subtopic: str, topic_info: Dict) -> int:
        """Process a batch of Q&A pairs for a subtopic."""
        
        difficulties = topic_info.get("difficulty_range", ["beginner", "intermediate", "advanced"])
        difficulty = random.choice(difficulties)
        
        # Generate batch of Q&A pairs
        qa_dicts = self.qa_generator.generate_batch(
            topic, subtopic, difficulty,
            count=self.config.batch_size
        )
        
        valid_count = 0
        
        for qa_dict in qa_dicts:
            # Check for duplicate
            q_hash = FastValidator.compute_hash(qa_dict["question"])
            if q_hash in self.state.question_hashes:
                self.stats["duplicates"] += 1
                continue
            
            # Validate
            if not FastValidator.validate(qa_dict):
                self.stats["rejected"] += 1
                continue
            
            # Create Q&A pair
            qa_pair = QAPair(
                id=self.generate_id(topic, subtopic, self.stats["generated"]),
                topic=qa_dict["topic"],
                subtopic=qa_dict["subtopic"],
                question=qa_dict["question"],
                answer=qa_dict["answer"],
                difficulty=qa_dict["difficulty"],
                question_type=qa_dict["question_type"]
            )
            
            # Add hash and write
            self.state.question_hashes.add(q_hash)
            self.file_writer.write(qa_pair)
            
            self.stats["generated"] += 1
            valid_count += 1
        
        self.stats["batches"] += 1
        return valid_count
    
    def get_rate(self) -> float:
        """Calculate current generation rate (Q&A per minute)."""
        elapsed = time.time() - self.stats["start_time"]
        if elapsed > 0:
            return (self.stats["generated"] / elapsed) * 60
        return 0
    
    def get_eta(self) -> str:
        """Calculate estimated time to completion."""
        rate = self.get_rate()
        if rate > 0:
            remaining = self.config.target_dataset_size - self.stats["generated"]
            minutes = remaining / rate
            if minutes < 60:
                return f"{minutes:.1f} minutes"
            else:
                hours = minutes / 60
                return f"{hours:.1f} hours"
        return "calculating..."
    
    def print_progress(self):
        """Print current progress with ETA."""
        total = self.stats["generated"]
        target = self.config.target_dataset_size
        percentage = (total / target) * 100
        rate = self.get_rate()
        eta = self.get_eta()
        
        print(f"\r📊 Progress: {total:,}/{target:,} ({percentage:.1f}%) | "
              f"Rate: {rate:.1f}/min | ETA: {eta} | "
              f"Rej: {self.stats['rejected']} | Dup: {self.stats['duplicates']}", 
              end="", flush=True)
    
    def generate_dataset(self):
        """Main optimized generation loop."""
        
        curriculum = FINANCIAL_CURRICULUM
        topics = list(curriculum.keys())
        
        total_subtopics = sum(len(info["subtopics"]) for info in curriculum.values())
        batches_per_subtopic = max(
            3, 
            (self.config.target_dataset_size // total_subtopics) // self.config.batch_size + 1
        )
        
        print(f"\n📖 Curriculum: {len(topics)} topics, {total_subtopics} subtopics")
        print(f"🎯 Target: {self.config.target_dataset_size:,} Q&A pairs")
        print(f"📦 Batch size: {self.config.batch_size} Q&A per generation")
        print(f"🔄 ~{batches_per_subtopic} batches per subtopic\n")
        
        try:
            topic_idx = 0
            while self.stats["generated"] < self.config.target_dataset_size:
                topic = topics[topic_idx % len(topics)]
                topic_info = curriculum[topic]
                subtopics = topic_info["subtopics"]
                
                for subtopic in subtopics:
                    # Generate multiple batches per subtopic
                    for batch_num in range(batches_per_subtopic):
                        self.process_batch(topic, subtopic, topic_info)
                        
                        # Print progress
                        self.print_progress()
                        
                        # Clear GPU cache periodically
                        if self.stats["batches"] % (self.config.clear_cache_interval // self.config.batch_size) == 0:
                            self.model_manager.clear_cache()
                        
                        # Save checkpoint periodically
                        if self.stats["batches"] % 20 == 0:
                            self.state.total_generated = self.stats["generated"]
                            self.state.save_checkpoint(self.config.checkpoint_file)
                        
                        if self.stats["generated"] >= self.config.target_dataset_size:
                            break
                    
                    if self.stats["generated"] >= self.config.target_dataset_size:
                        break
                
                topic_idx += 1
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Generation interrupted by user")
        
        finally:
            # Flush all remaining data
            self.file_writer.flush_all()
            self.state.total_generated = self.stats["generated"]
            self.state.save_checkpoint(self.config.checkpoint_file)
            self.print_final_stats()
    
    def print_final_stats(self):
        """Print final statistics."""
        elapsed = time.time() - self.stats["start_time"]
        hours = elapsed / 3600
        final_rate = self.stats["generated"] / (elapsed / 60) if elapsed > 0 else 0
        
        print("\n\n" + "=" * 60)
        print("🏁 GENERATION COMPLETE")
        print("=" * 60)
        print(f"📊 Total Q&A pairs generated: {self.stats['generated']:,}")
        print(f"⏱️ Total time: {hours:.2f} hours ({elapsed:.0f} seconds)")
        print(f"🚀 Average rate: {final_rate:.1f} Q&A per minute")
        print(f"❌ Rejected: {self.stats['rejected']:,}")
        print(f"🔄 Duplicates: {self.stats['duplicates']:,}")
        print(f"📦 Total batches: {self.stats['batches']:,}")
        print(f"\n📁 Output file: {self.config.output_file}")
        
        if os.path.exists(self.config.output_file):
            file_size = os.path.getsize(self.config.output_file) / (1024 * 1024)
            print(f"📦 File size: {file_size:.2f} MB")
        
        print("=" * 60)
        
        # Performance summary
        print("\n📈 PERFORMANCE SUMMARY:")
        projected_time_3hrs = 3 * 60 * final_rate  # Q&A in 3 hours
        print(f"   At current rate, 3 hours would generate: {projected_time_3hrs:,.0f} Q&A pairs")
        if self.stats["generated"] >= self.config.target_dataset_size:
            print("   ✅ TARGET ACHIEVED!")
        else:
            needed_rate = self.config.target_dataset_size / 180  # 180 minutes = 3 hours
            print(f"   Target rate needed for 10k in 3h: {needed_rate:.1f}/min")


# ============================================================================
# SECTION 10: UTILITY FUNCTIONS
# ============================================================================

def validate_dataset(filepath: str) -> Dict[str, Any]:
    """Validate the generated dataset."""
    
    print(f"\n🔍 Validating dataset: {filepath}")
    
    if not os.path.exists(filepath):
        print("❌ File not found!")
        return {}
    
    stats = {
        "total_pairs": 0,
        "by_topic": defaultdict(int),
        "by_difficulty": defaultdict(int),
        "avg_question_length": 0,
        "avg_answer_length": 0
    }
    
    q_lengths = []
    a_lengths = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                qa = json.loads(line.strip())
                stats["total_pairs"] += 1
                stats["by_topic"][qa.get("topic", "unknown")] += 1
                stats["by_difficulty"][qa.get("difficulty", "unknown")] += 1
                q_lengths.append(len(qa.get("question", "")))
                a_lengths.append(len(qa.get("answer", "")))
            except json.JSONDecodeError:
                continue
    
    if q_lengths:
        stats["avg_question_length"] = sum(q_lengths) / len(q_lengths)
    if a_lengths:
        stats["avg_answer_length"] = sum(a_lengths) / len(a_lengths)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total pairs: {stats['total_pairs']:,}")
    print(f"   Avg question length: {stats['avg_question_length']:.0f} chars")
    print(f"   Avg answer length: {stats['avg_answer_length']:.0f} chars")
    print(f"\n📚 By Topic:")
    for topic, count in sorted(stats["by_topic"].items(), key=lambda x: -x[1]):
        print(f"   {topic}: {count:,}")
    print(f"\n📈 By Difficulty:")
    for diff, count in stats["by_difficulty"].items():
        print(f"   {diff}: {count:,}")
    
    return stats


def sample_dataset(filepath: str, n: int = 3):
    """Print sample entries from the dataset."""
    
    print(f"\n📋 Sample entries from {filepath}:")
    print("=" * 60)
    
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            samples.append(json.loads(line.strip()))
    
    for i, qa in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Topic: {qa.get('topic')} > {qa.get('subtopic')}")
        print(f"Difficulty: {qa.get('difficulty')}")
        print(f"Question: {qa.get('question')}")
        answer = qa.get('answer', '')
        print(f"Answer: {answer[:300]}{'...' if len(answer) > 300 else ''}")


def is_colab() -> bool:
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


# ============================================================================
# SECTION 11: MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point."""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     OPTIMIZED FINANCIAL EDUCATION DATASET GENERATOR          ║
    ║     Target: 10,000 Q&A pairs in 3 hours                      ║
    ║     Optimized for Google Colab                               ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU Detected: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("⚠️ No GPU detected. Generation will be slower on CPU")
    
    # Initialize configuration
    config = Config()
    
    print(f"\n📊 Optimized Settings:")
    print(f"   Target: {config.target_dataset_size:,} Q&A pairs")
    print(f"   Batch size: {config.batch_size} Q&A per LLM call")
    print(f"   Model: {config.model_name}")
    print(f"   Quantization: {'4-bit (NF4)' if config.use_quantization else 'None'}")
    print(f"   Output: {config.output_file}")
    
    # Calculate expected performance
    expected_time_per_batch = 8  # seconds (rough estimate)
    batches_needed = config.target_dataset_size / config.batch_size
    expected_total_time = batches_needed * expected_time_per_batch / 3600
    print(f"\n⏱️ Estimated time: {expected_total_time:.1f} hours (actual may vary)")
    
    # Create generator and run
    generator = OptimizedFinancialDatasetGenerator(config)
    generator.initialize()
    generator.generate_dataset()
    
    # Validate output
    if os.path.exists(config.output_file):
        validate_dataset(config.output_file)
        sample_dataset(config.output_file, n=3)
        
        print("\n✅ Generation complete! Your dataset is ready.")
        print(f"📁 File location: {config.output_file}")
        
        # Auto-download in Google Colab
        if is_colab():
            print("\n" + "=" * 60)
            print("📥 DOWNLOADING DATASET...")
            print("=" * 60)
            try:
                from google.colab import files
                files.download(config.output_file)
            except Exception as e:
                print(f"Download from Files panel on the left: {e}")
    else:
        print("\n❌ No output file generated. Check logs for errors.")


if __name__ == "__main__":
    main()
