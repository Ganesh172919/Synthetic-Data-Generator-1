"""
Universal Synthetic Dataset Generator
======================================
A highly optimized, flexible synthetic data generator that takes ANY user prompt
and generates complete datasets for any use case.

Features:
✅ Universal prompt-based generation (any domain/use case)
✅ Optimized batch processing for speed
✅ Thread-safe async file writing
✅ Deduplication with hash-based filtering
✅ Progress tracking with auto-save
✅ Multiple output formats (JSONL, CSV, JSON)
✅ Configurable dataset size and quality settings
✅ Support for local models (HuggingFace) and APIs (OpenAI, etc.)

Usage:
    python universal_dataset_generator.py
    
    # Or programmatically:
    from universal_dataset_generator import UniversalGenerator
    generator = UniversalGenerator()
    generator.run()
"""

# ============================================================================
# SECTION 1: INSTALLATION & IMPORTS
# ============================================================================

import subprocess
import sys

def install_dependencies():
    """Install required packages quietly."""
    packages = [
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "torch>=2.0.0",
        "tqdm",
        "openai",
    ]
    print("📦 Installing dependencies...")
    for pkg in packages:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q"] + pkg.split(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception:
            pass
    print("✅ Dependencies ready!\n")

# Auto-install on import
install_dependencies()

import os
import json
import csv
import time
import random
import hashlib
import threading
import queue
import signal
import atexit
import warnings
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum, auto

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        logging as transformers_logging
    )
    transformers_logging.set_verbosity_error()
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    torch = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


# ============================================================================
# SECTION 2: CONFIGURATION
# ============================================================================

class ModelProvider(Enum):
    """Supported model providers."""
    HUGGINGFACE = auto()
    OPENAI = auto()
    MOCK = auto()  # For testing


@dataclass
class GeneratorConfig:
    """Configuration for the dataset generator."""
    
    # Generation settings
    target_size: int = 1000
    items_per_batch: int = 10
    
    # Model settings
    provider: ModelProvider = ModelProvider.HUGGINGFACE
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    openai_model: str = "gpt-3.5-turbo"
    use_quantization: bool = True
    max_new_tokens: int = 2000
    temperature: float = 0.8
    
    # Output settings
    output_file: str = "generated_dataset"
    output_format: str = "jsonl"  # jsonl, json, csv
    checkpoint_file: str = "generator_checkpoint.json"
    
    # Quality settings
    min_content_length: int = 50
    enable_deduplication: bool = True
    
    # Performance settings
    save_interval: int = 100
    auto_save_seconds: int = 180
    clear_cache_interval: int = 20


# ============================================================================
# SECTION 3: DATA STRUCTURES
# ============================================================================

@dataclass
class DataItem:
    """A single data item in the dataset."""
    id: str
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            **self.content,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class AtomicCounter:
    """Thread-safe counter."""
    
    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        with self._lock:
            self._value += amount
            return self._value
    
    def get(self) -> int:
        with self._lock:
            return self._value
    
    def set(self, value: int):
        with self._lock:
            self._value = value


class ThreadSafeSet:
    """Thread-safe set for deduplication."""
    
    def __init__(self):
        self._data = set()
        self._lock = threading.Lock()
    
    def add(self, item: str) -> bool:
        """Add item, return True if new, False if duplicate."""
        with self._lock:
            if item in self._data:
                return False
            self._data.add(item)
            return True
    
    def add_batch(self, items: List[str]) -> List[bool]:
        """Add multiple items, return list of results."""
        with self._lock:
            results = []
            for item in items:
                is_new = item not in self._data
                if is_new:
                    self._data.add(item)
                results.append(is_new)
            return results
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


# ============================================================================
# SECTION 4: ASYNC FILE WRITER
# ============================================================================

class AsyncFileWriter:
    """High-performance async file writer with buffering."""
    
    def __init__(self, filepath: str, output_format: str = "jsonl", buffer_size: int = 50):
        self.filepath = filepath
        self.format = output_format
        self.buffer_size = buffer_size
        self._queue = queue.Queue(maxsize=5000)
        self._stop_event = threading.Event()
        self._written = AtomicCounter()
        self._all_items = []  # For JSON/CSV formats
        
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
    
    def _writer_loop(self):
        buffer = []
        last_flush = time.time()
        
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.1)
                buffer.append(item)
                
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
    
    def _flush(self, buffer: List[DataItem]):
        if not buffer:
            return
        
        try:
            if self.format == "jsonl":
                mode = 'a' if os.path.exists(self.filepath) else 'w'
                with open(self.filepath, mode, encoding='utf-8', buffering=65536) as f:
                    for item in buffer:
                        f.write(item.to_json() + '\n')
            else:
                # For JSON/CSV, accumulate items
                self._all_items.extend([item.to_dict() for item in buffer])
            
            self._written.increment(len(buffer))
        except Exception as e:
            print(f"\n⚠️ Write error: {e}")
    
    def write(self, item: DataItem):
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            time.sleep(0.05)
            self._queue.put(item)
    
    def write_batch(self, items: List[DataItem]):
        for item in items:
            self.write(item)
    
    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=10)
        
        # Write accumulated items for JSON/CSV
        if self.format == "json" and self._all_items:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self._all_items, f, indent=2, ensure_ascii=False)
        elif self.format == "csv" and self._all_items:
            self._write_csv()
    
    def _write_csv(self):
        if not self._all_items:
            return
        
        # Flatten nested dicts for CSV
        flat_items = []
        for item in self._all_items:
            flat = {}
            for k, v in item.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        flat[f"{k}_{k2}"] = v2
                else:
                    flat[k] = v
            flat_items.append(flat)
        
        fieldnames = list(flat_items[0].keys())
        with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_items)
    
    def get_written_count(self) -> int:
        return self._written.get()


# ============================================================================
# SECTION 5: MODEL BACKENDS
# ============================================================================

class BaseModelBackend(ABC):
    """Abstract base class for model backends."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    def load(self):
        pass
    
    def clear_cache(self):
        pass


class HuggingFaceBackend(BaseModelBackend):
    """HuggingFace transformers backend with optimizations."""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if HF_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()
    
    def load(self):
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace transformers not available")
        
        print(f"🔄 Loading model: {self.config.model_name}")
        
        if self.device == "cuda":
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")
        else:
            print("⚠️ Running on CPU - generation will be slower")
        
        # Quantization config for efficiency
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
            self.config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Load model
        model_kwargs = {
            "device_map": "auto" if self.device == "cuda" else None,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if quant_config:
            model_kwargs["quantization_config"] = quant_config
        
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
    
    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        with self._lock:
            try:
                formatted = f"[INST] {prompt} [/INST]"
                
                inputs = self.tokenizer(
                    formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=False
                ).to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    repetition_penalty=1.1
                )
                
                input_len = inputs["input_ids"].shape[1]
                return self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
            
            except Exception as e:
                print(f"\n❌ Generation error: {e}")
                return ""
    
    def clear_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class OpenAIBackend(BaseModelBackend):
    """OpenAI API backend."""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.client = None
    
    def load(self):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available")
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        print(f"✅ OpenAI client initialized with model: {self.config.openai_model}")
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"\n❌ OpenAI error: {e}")
            return ""


class MockBackend(BaseModelBackend):
    """Mock backend for testing without actual model."""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
    
    def load(self):
        print("✅ Mock backend loaded (for testing)")
    
    def generate(self, prompt: str) -> str:
        # Generate dummy responses for testing
        items = []
        for i in range(self.config.items_per_batch):
            items.append(f"Q{i+1}: Sample question about the topic?")
            items.append(f"A{i+1}: This is a comprehensive answer with detailed explanation. " * 5)
            items.append("")
        return "\n".join(items)


def get_backend(config: GeneratorConfig) -> BaseModelBackend:
    """Factory function to get appropriate backend."""
    if config.provider == ModelProvider.HUGGINGFACE:
        return HuggingFaceBackend(config)
    elif config.provider == ModelProvider.OPENAI:
        return OpenAIBackend(config)
    else:
        return MockBackend(config)


# ============================================================================
# SECTION 6: PROMPT TEMPLATES
# ============================================================================

class PromptBuilder:
    """Builds generation prompts based on user specifications."""
    
    @staticmethod
    def build_qa_prompt(user_prompt: str, count: int, context: Optional[str] = None) -> str:
        """Build a Q&A generation prompt."""
        prompt = f"""Generate {count} high-quality question-answer pairs based on the following specification:

TOPIC/DOMAIN: {user_prompt}

{f'ADDITIONAL CONTEXT: {context}' if context else ''}

Requirements:
- Each answer should be 60-200 words, educational and accurate
- Include variety in question types (what, why, how, when, compare, explain, etc.)
- Ensure each Q&A is unique and valuable

Format your response EXACTLY as follows:
Q1: [question here]
A1: [detailed answer here]

Q2: [question here]
A2: [detailed answer here]

...continue up to Q{count}"""
        return prompt
    
    @staticmethod
    def build_text_prompt(user_prompt: str, count: int, format_spec: str = "paragraph") -> str:
        """Build a text generation prompt."""
        prompt = f"""Generate {count} unique text samples based on this specification:

SPECIFICATION: {user_prompt}
FORMAT: {format_spec}

Generate each sample clearly separated by "---SAMPLE---" marker.
Each sample should be complete and self-contained.

Begin:"""
        return prompt
    
    @staticmethod
    def build_structured_prompt(user_prompt: str, count: int, fields: List[str]) -> str:
        """Build a structured data generation prompt."""
        fields_str = ", ".join(fields)
        prompt = f"""Generate {count} structured data entries based on this specification:

SPECIFICATION: {user_prompt}
REQUIRED FIELDS: {fields_str}

Format each entry as valid JSON with the specified fields.
Separate entries with "---ENTRY---" marker.

Example format:
{{"field1": "value1", "field2": "value2", ...}}
---ENTRY---
{{"field1": "value1", "field2": "value2", ...}}

Begin generating {count} entries:"""
        return prompt


# ============================================================================
# SECTION 7: RESPONSE PARSERS
# ============================================================================

class ResponseParser:
    """Parses model responses into structured data."""
    
    @staticmethod
    def parse_qa_response(response: str, min_length: int = 50) -> List[Dict]:
        """Parse Q&A formatted response."""
        qa_pairs = []
        current_q = None
        current_a_lines = []
        
        for line in response.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            
            upper = stripped.upper()
            
            # Detect question (Q1:, Q2:, Question 1:, etc.)
            if upper.startswith('Q') and ':' in stripped[:10]:
                if current_q and current_a_lines:
                    answer = ' '.join(current_a_lines).strip()
                    if len(answer) >= min_length and len(current_q) > 10:
                        qa_pairs.append({
                            "question": current_q,
                            "answer": answer
                        })
                
                colon_idx = stripped.find(':')
                current_q = stripped[colon_idx+1:].strip() if colon_idx != -1 else stripped[2:].strip()
                current_a_lines = []
            
            # Detect answer (A1:, A2:, Answer 1:, etc.)
            elif upper.startswith('A') and ':' in stripped[:10]:
                colon_idx = stripped.find(':')
                text = stripped[colon_idx+1:].strip() if colon_idx != -1 else stripped[2:].strip()
                if text:
                    current_a_lines.append(text)
            
            # Continuation of answer
            elif current_q and current_a_lines:
                current_a_lines.append(stripped)
        
        # Don't forget last Q&A
        if current_q and current_a_lines:
            answer = ' '.join(current_a_lines).strip()
            if len(answer) >= min_length and len(current_q) > 10:
                qa_pairs.append({
                    "question": current_q,
                    "answer": answer
                })
        
        return qa_pairs
    
    @staticmethod
    def parse_text_response(response: str, separator: str = "---SAMPLE---") -> List[Dict]:
        """Parse text samples separated by marker."""
        samples = response.split(separator)
        return [{"text": s.strip()} for s in samples if s.strip()]
    
    @staticmethod
    def parse_json_response(response: str, separator: str = "---ENTRY---") -> List[Dict]:
        """Parse JSON entries separated by marker."""
        entries = []
        parts = response.split(separator)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Find JSON object in the text
            try:
                start = part.find('{')
                end = part.rfind('}') + 1
                if start != -1 and end > start:
                    json_str = part[start:end]
                    entries.append(json.loads(json_str))
            except json.JSONDecodeError:
                continue
        
        return entries


# ============================================================================
# SECTION 8: MAIN GENERATOR CLASS
# ============================================================================

# Global references for emergency save
_global_generator = None
_global_writer = None
_emergency_save_done = False


def emergency_save():
    """Emergency save handler."""
    global _emergency_save_done
    
    if _emergency_save_done:
        return
    
    print("\n\n🚨 EMERGENCY SAVE TRIGGERED...")
    
    if _global_writer:
        try:
            _global_writer.stop()
            print(f"✅ Data saved: {_global_writer.get_written_count()} items")
        except Exception:
            pass
    
    if _global_generator:
        try:
            _global_generator._save_checkpoint()
            print("✅ Checkpoint saved!")
        except Exception:
            pass
    
    _emergency_save_done = True


class UniversalGenerator:
    """Universal synthetic dataset generator."""
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.backend: Optional[BaseModelBackend] = None
        self.writer: Optional[AsyncFileWriter] = None
        self.hashes = ThreadSafeSet()
        
        self.generated = AtomicCounter()
        self.duplicates = AtomicCounter()
        self.errors = AtomicCounter()
        self.start_time = None
        self.last_save_time = None
        
        # User prompt and parsing settings
        self.user_prompt: str = ""
        self.parse_mode: str = "qa"  # qa, text, json
        self.extra_fields: List[str] = []
    
    def _setup_handlers(self):
        """Setup emergency save handlers."""
        global _global_generator, _global_writer
        _global_generator = self
        _global_writer = self.writer
        
        atexit.register(emergency_save)
        
        try:
            signal.signal(signal.SIGINT, lambda s, f: emergency_save())
            signal.signal(signal.SIGTERM, lambda s, f: emergency_save())
        except Exception:
            pass
    
    def _get_output_path(self) -> str:
        """Get full output file path."""
        ext = {"jsonl": ".jsonl", "json": ".json", "csv": ".csv"}.get(
            self.config.output_format, ".jsonl"
        )
        return f"{self.config.output_file}{ext}"
    
    def _hash_content(self, content: Dict) -> str:
        """Generate hash for content deduplication."""
        text = json.dumps(content, sort_keys=True)
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def _save_checkpoint(self):
        """Save current progress checkpoint."""
        checkpoint = {
            "generated": self.generated.get(),
            "duplicates": self.duplicates.get(),
            "hash_count": len(self.hashes),
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(self.config.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
        except Exception:
            pass
    
    def _load_checkpoint(self) -> int:
        """Load checkpoint and return number of existing items."""
        if not os.path.exists(self.config.checkpoint_file):
            return 0
        
        try:
            with open(self.config.checkpoint_file, 'r') as f:
                data = json.load(f)
            
            # Load existing hashes from output file
            output_path = self._get_output_path()
            if os.path.exists(output_path) and self.config.output_format == "jsonl":
                with open(output_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = json.loads(line.strip())
                            content_hash = self._hash_content(item)
                            self.hashes.add(content_hash)
                        except Exception:
                            pass
            
            count = data.get("generated", 0)
            print(f"📂 Resuming from checkpoint: {count} items")
            return count
        except Exception:
            return 0
    
    def _generate_batch(self, context: Optional[str] = None) -> List[DataItem]:
        """Generate a batch of data items."""
        # Build appropriate prompt
        if self.parse_mode == "qa":
            prompt = PromptBuilder.build_qa_prompt(
                self.user_prompt, 
                self.config.items_per_batch,
                context
            )
        elif self.parse_mode == "text":
            prompt = PromptBuilder.build_text_prompt(
                self.user_prompt,
                self.config.items_per_batch
            )
        else:  # json
            prompt = PromptBuilder.build_structured_prompt(
                self.user_prompt,
                self.config.items_per_batch,
                self.extra_fields
            )
        
        # Generate response
        response = self.backend.generate(prompt)
        
        if not response:
            self.errors.increment()
            return []
        
        # Parse response
        if self.parse_mode == "qa":
            parsed = ResponseParser.parse_qa_response(response, self.config.min_content_length)
        elif self.parse_mode == "text":
            parsed = ResponseParser.parse_text_response(response)
        else:
            parsed = ResponseParser.parse_json_response(response)
        
        # Create data items with deduplication
        items = []
        for content in parsed:
            content_hash = self._hash_content(content)
            
            if self.config.enable_deduplication:
                if not self.hashes.add(content_hash):
                    self.duplicates.increment()
                    continue
            
            item = DataItem(
                id=f"item_{content_hash}_{random.randint(1000, 9999)}",
                content=content,
                metadata={
                    "source_prompt": self.user_prompt[:100],
                    "parse_mode": self.parse_mode
                }
            )
            items.append(item)
        
        return items
    
    def _print_progress(self, batch_num: int):
        """Print generation progress."""
        generated = self.generated.get()
        elapsed = time.time() - self.start_time
        rate = generated / elapsed if elapsed > 0 else 0
        eta_seconds = (self.config.target_size - generated) / rate if rate > 0 else 0
        
        progress_pct = (generated / self.config.target_size) * 100
        bar_filled = int(progress_pct / 5)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        
        eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s" if eta_seconds < 3600 else f"{eta_seconds / 3600:.1f}h"
        
        print(f"\r📊 [{bar}] {progress_pct:.1f}% | {generated:,}/{self.config.target_size:,} | "
              f"Rate: {rate:.1f}/s | ETA: {eta_str} | Dups: {self.duplicates.get()}", end="", flush=True)
    
    def run(self, 
            user_prompt: Optional[str] = None,
            parse_mode: str = "qa",
            extra_fields: Optional[List[str]] = None):
        """
        Run the dataset generation.
        
        Args:
            user_prompt: Description of what data to generate
            parse_mode: 'qa' for Q&A, 'text' for paragraphs, 'json' for structured
            extra_fields: For json mode, list of field names to generate
        """
        # Get user prompt if not provided
        if user_prompt is None:
            print("=" * 60)
            print("🚀 UNIVERSAL SYNTHETIC DATASET GENERATOR")
            print("=" * 60)
            print("\nEnter your data generation prompt.")
            print("Examples:")
            print("  - 'Generate educational Q&A about machine learning'")
            print("  - 'Create customer service conversations for a bank'")
            print("  - 'Generate product descriptions for electronics'")
            print()
            user_prompt = input("📝 Enter your prompt: ").strip()
            
            if not user_prompt:
                print("❌ No prompt provided. Exiting.")
                return
            
            # Get additional settings
            print("\n📊 Configuration:")
            
            # Target size
            size_input = input(f"   Target dataset size [{self.config.target_size}]: ").strip()
            if size_input.isdigit():
                self.config.target_size = int(size_input)
            
            # Parse mode
            mode_input = input("   Output format - qa/text/json [qa]: ").strip().lower()
            if mode_input in ["qa", "text", "json"]:
                parse_mode = mode_input
            
            # Output format
            fmt_input = input("   File format - jsonl/json/csv [jsonl]: ").strip().lower()
            if fmt_input in ["jsonl", "json", "csv"]:
                self.config.output_format = fmt_input
            
            # Output filename
            name_input = input(f"   Output filename [{self.config.output_file}]: ").strip()
            if name_input:
                self.config.output_file = name_input
            
            # Extra fields for JSON mode
            if parse_mode == "json":
                fields_input = input("   Fields to generate (comma-separated): ").strip()
                if fields_input:
                    extra_fields = [f.strip() for f in fields_input.split(",")]
        
        self.user_prompt = user_prompt
        self.parse_mode = parse_mode
        self.extra_fields = extra_fields or []
        
        # Initialize
        print("\n" + "=" * 60)
        print("🔧 INITIALIZING...")
        print("=" * 60)
        
        # Load backend
        self.backend = get_backend(self.config)
        self.backend.load()
        
        # Setup writer
        output_path = self._get_output_path()
        self.writer = AsyncFileWriter(output_path, self.config.output_format)
        
        # Setup handlers
        self._setup_handlers()
        
        # Load checkpoint
        existing_count = self._load_checkpoint()
        self.generated.set(existing_count)
        
        remaining = self.config.target_size - existing_count
        if remaining <= 0:
            print(f"✅ Dataset already complete! {existing_count} items exist.")
            return
        
        print(f"\n📊 Generation Plan:")
        print(f"   • Prompt: {user_prompt[:50]}...")
        print(f"   • Mode: {parse_mode}")
        print(f"   • Target: {self.config.target_size:,} items")
        print(f"   • Remaining: {remaining:,} items")
        print(f"   • Batch size: {self.config.items_per_batch}")
        print(f"   • Output: {output_path}")
        print()
        
        # Start generation
        self.start_time = time.time()
        self.last_save_time = time.time()
        batch_num = 0
        
        print("🚀 Starting generation...\n")
        
        try:
            while self.generated.get() < self.config.target_size:
                # Generate batch
                items = self._generate_batch()
                
                if items:
                    self.writer.write_batch(items)
                    self.generated.increment(len(items))
                
                batch_num += 1
                
                # Print progress
                self._print_progress(batch_num)
                
                # Auto-save checkpoint
                if time.time() - self.last_save_time > self.config.auto_save_seconds:
                    self._save_checkpoint()
                    self.last_save_time = time.time()
                
                # Clear cache periodically
                if batch_num % self.config.clear_cache_interval == 0:
                    self.backend.clear_cache()
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
        finally:
            # Cleanup
            print("\n\n" + "=" * 60)
            print("📦 FINALIZING...")
            print("=" * 60)
            
            if self.writer:
                self.writer.stop()
            
            self._save_checkpoint()
            
            # Final stats
            elapsed = time.time() - self.start_time
            final_count = self.generated.get()
            
            print(f"\n✅ GENERATION COMPLETE!")
            print(f"   • Total items: {final_count:,}")
            print(f"   • Duplicates skipped: {self.duplicates.get():,}")
            print(f"   • Time elapsed: {elapsed/60:.1f} minutes")
            print(f"   • Average rate: {final_count/elapsed:.1f} items/second")
            print(f"   • Output file: {output_path}")
            
            # File size
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"   • File size: {size_mb:.2f} MB")
            
            print("\n💡 Dataset saved successfully!")


# ============================================================================
# SECTION 9: MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for command-line usage."""
    
    # Allow configuration via command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Universal Synthetic Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python universal_dataset_generator.py
  python universal_dataset_generator.py --prompt "Generate Q&A about Python programming" --size 500
  python universal_dataset_generator.py --prompt "Customer support dialogues" --mode text --format json
        """
    )
    
    parser.add_argument("--prompt", "-p", type=str, help="Generation prompt")
    parser.add_argument("--size", "-s", type=int, default=1000, help="Target dataset size")
    parser.add_argument("--mode", "-m", type=str, choices=["qa", "text", "json"], default="qa",
                       help="Parse mode: qa, text, or json")
    parser.add_argument("--format", "-f", type=str, choices=["jsonl", "json", "csv"], default="jsonl",
                       help="Output file format")
    parser.add_argument("--output", "-o", type=str, default="generated_dataset",
                       help="Output filename (without extension)")
    parser.add_argument("--batch", "-b", type=int, default=10, help="Items per batch")
    parser.add_argument("--provider", type=str, choices=["huggingface", "openai", "mock"],
                       default="huggingface", help="Model provider")
    parser.add_argument("--model", type=str, help="Model name (for HuggingFace)")
    parser.add_argument("--fields", type=str, help="Comma-separated fields for JSON mode")
    
    args = parser.parse_args()
    
    # Create config
    config = GeneratorConfig(
        target_size=args.size,
        items_per_batch=args.batch,
        output_file=args.output,
        output_format=args.format,
        provider=ModelProvider[args.provider.upper()] if args.provider else ModelProvider.HUGGINGFACE
    )
    
    if args.model:
        config.model_name = args.model
    
    # Parse extra fields
    extra_fields = [f.strip() for f in args.fields.split(",")] if args.fields else None
    
    # Run generator
    generator = UniversalGenerator(config)
    generator.run(
        user_prompt=args.prompt,
        parse_mode=args.mode,
        extra_fields=extra_fields
    )


if __name__ == "__main__":
    main()
