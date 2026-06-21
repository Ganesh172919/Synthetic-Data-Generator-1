"""
Universal Synthetic Dataset Generator
======================================
A highly optimized, flexible synthetic data generator that takes ANY user prompt
and generates complete datasets for any use case.

Features:
??? Universal prompt-based generation (any domain/use case)
??? Optimized batch processing for speed
??? Thread-safe async file writing
??? Deduplication with hash-based filtering
??? Progress tracking with auto-save
??? Multiple output formats (JSONL, CSV, JSON)
??? Configurable dataset size and quality settings
??? Support for local models (HuggingFace) and APIs (OpenAI, etc.)

Reality-aligned notes (important for learners):

- This file is a *standalone generator* that writes datasets to disk (JSONL/JSON/CSV).
  It is not automatically wired into the web demo backend in `website/server/`.
- The goal is to make end-to-end generation robust for long runs:
  buffering + checkpoints + emergency-save handlers exist because large jobs fail.

Model providers (choose in `GeneratorConfig.provider`):

- HUGGINGFACE: Runs a local model via Transformers (GPU recommended).
- OPENAI: Uses OpenAI API via `OPENAI_API_KEY` environment variable.
- MOCK: Generates deterministic dummy output to test the pipeline without a model.

Parse modes (choose in `run(..., parse_mode=...)`):

- qa: Question/answer pairs (expects "Q1:" / "A1:" style output).
- text: Free-form samples separated by a marker (default: "---SAMPLE---").
- json: Structured JSON objects separated by a marker (default: "---ENTRY---").

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
    """
    Install required packages quietly.

    Educational note:
    - This ???auto-install on import??? pattern is convenient for Colab notebooks and quick demos.
    - In production or controlled environments, you usually want a pinned `requirements.txt`
      and explicit installs (auto-install can be slow and surprising).
    - Errors are swallowed on purpose here so the script can still run in "MOCK" mode even
      if heavyweight dependencies fail to install.
    """
    packages = [
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "torch>=2.0.0",
        "tqdm",
        "openai",
    ]
    print("???? Installing dependencies...")
    for pkg in packages:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q"] + pkg.split(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception:
            pass
    print("??? Dependencies ready!\n")

def _running_in_colab() -> bool:
    """Detect Google Colab runtime."""
    try:
        import os as _os
        return "google.colab" in sys.modules or bool(_os.environ.get("COLAB_GPU"))
    except Exception:
        return False


def _should_auto_install() -> bool:
    """
    Auto-install policy:
    - Enabled in Colab.
    - Enabled when `SYNTHGEN_AUTO_INSTALL=1`.
    - Enabled when CLI flag `--auto-install` is present.
    """
    try:
        import os as _os
        if _os.environ.get("SYNTHGEN_AUTO_INSTALL") == "1":
            return True
    except Exception:
        pass
    return "--auto-install" in sys.argv or _running_in_colab()


if _should_auto_install():
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
    ANTHROPIC = auto()
    GOOGLE = auto()
    OLLAMA = auto()
    AZURE_OPENAI = auto()
    GROQ = auto()
    TOGETHER = auto()
    CUSTOM = auto()
    AWS_BEDROCK = auto()
    REPLICATE = auto()


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
    """
    High-performance async file writer with buffering.

    All formats are now stream-friendly:
    - JSONL: append one JSON object per line (fastest, most memory-efficient).
    - CSV: write header on first flush, then append rows incrementally.
    - JSON: stream as JSON Lines to a temp file, then wrap into a JSON array on stop.
      For very large datasets, consider using JSONL instead.
    """

    def __init__(self, filepath: str, output_format: str = "jsonl", buffer_size: int = 50):
        self.filepath = filepath
        self.format = output_format
        self.buffer_size = buffer_size
        self._queue = queue.Queue(maxsize=5000)
        self._stop_event = threading.Event()
        self._written = AtomicCounter()

        # CSV state
        self._csv_headers_written = False
        self._csv_fieldnames = None

        # JSON state: track count for final assembly
        self._json_count = 0
        self._json_temp_path = filepath + '.tmp' if output_format == 'json' else None

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

            elif self.format == "csv":
                # Stream CSV: write header on first batch, then append rows
                dicts = [item.to_dict() for item in buffer]
                flat_dicts = [self._flatten(d) for d in dicts]

                if not self._csv_headers_written:
                    self._csv_fieldnames = list(flat_dicts[0].keys())
                    with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
                        writer.writeheader()
                        writer.writerows(flat_dicts)
                    self._csv_headers_written = True
                else:
                    with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames, extrasaction='ignore')
                        writer.writerows(flat_dicts)

            elif self.format == "json":
                # Stream JSON: write each item as a JSON line to temp file
                # We'll assemble the final JSON array on stop()
                mode = 'a' if os.path.exists(self._json_temp_path) else 'w'
                with open(self._json_temp_path, mode, encoding='utf-8', buffering=65536) as f:
                    for item in buffer:
                        f.write(json.dumps(item.to_dict(), ensure_ascii=False) + '\n')
                        self._json_count += 1

            self._written.increment(len(buffer))
        except Exception as e:
            print(f"\n?????? Write error: {e}")

    @staticmethod
    def _flatten(d: dict) -> dict:
        """Flatten nested dicts for CSV output."""
        flat = {}
        for k, v in d.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat[f"{k}_{k2}"] = v2
            else:
                flat[k] = v
        return flat

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

        # For JSON format: assemble temp lines into a proper JSON array
        if self.format == "json" and self._json_temp_path and os.path.exists(self._json_temp_path):
            try:
                with open(self._json_temp_path, 'r', encoding='utf-8') as tmp:
                    with open(self.filepath, 'w', encoding='utf-8') as out:
                        out.write('[\n')
                        first = True
                        for line in tmp:
                            line = line.strip()
                            if not line:
                                continue
                            if not first:
                                out.write(',\n')
                            out.write(line)
                            first = False
                        out.write('\n]')
                os.remove(self._json_temp_path)
            except Exception as e:
                print(f"\n?????? JSON assembly error: {e}")
                # Fallback: keep the temp file as-is
                if os.path.exists(self._json_temp_path):
                    os.rename(self._json_temp_path, self.filepath)

    def get_written_count(self) -> int:
        return self._written.get()


# ============================================================================
# SECTION 5: MODEL BACKENDS
# ============================================================================

class BaseModelBackend(ABC):
    """
    Abstract base class for model backends.

    Educational note:
    Backends are responsible for:
    - loading any required model/client resources (`load`)
    - generating a raw text response for a given prompt (`generate`)
    - optionally clearing caches between batches (`clear_cache`)

    The rest of the pipeline (parsing, validation, dedup, writing) is provider-agnostic.
    """
    
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
        
        print(f"???? Loading model: {self.config.model_name}")
        
        if self.device == "cuda":
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"???? GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")
        else:
            print("?????? Running on CPU - generation will be slower")
        
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
        
        print(f"??? Model loaded successfully!")
    
    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        with self._lock:
            try:
                # Many instruct-tuned models (including Mistral Instruct) use an [INST] wrapper.
                # If you swap models, you may need to adjust this formatting to match the model's chat template.
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
                print(f"\n??? Generation error: {e}")
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
        print(f"??? OpenAI client initialized with model: {self.config.openai_model}")
    
    def generate(self, prompt: str) -> str:
        try:
            # Educational note:
            # OpenAI token limits are model-dependent. If you see API errors about context length
            # or max tokens, reduce `GeneratorConfig.max_new_tokens` and/or shorten prompts,
            # or switch to a model with a larger context window.
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"\n??? OpenAI error: {e}")
            return ""


class MockBackend(BaseModelBackend):
    """Mock backend for testing without actual model."""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self._call_count = 0
    
    def load(self):
        print("??? Mock backend loaded (for testing)")
    
    def generate(self, prompt: str) -> str:
        # Generate deterministic-but-unique responses for testing.
        self._call_count += 1
        batch_tag = int(time.time() * 1000) + self._call_count
        items = []
        for i in range(self.config.items_per_batch):
            item_id = f"{batch_tag}_{i+1}"
            items.append(f"Q{i+1}: Sample question {item_id} about the topic?")
            items.append(
                f"A{i+1}: This is a comprehensive answer {item_id} with detailed explanation. " * 3
            )
            items.append("")
        return "\n".join(items)


class ProviderBackend(BaseModelBackend):
    """
    Backend that wraps the new provider abstraction layer.
    Used for providers beyond HuggingFace/OpenAI/Mock (Anthropic, Google, Ollama, etc.).
    """

    def __init__(self, config: GeneratorConfig, provider_name: str):
        self.config = config
        self.provider_name = provider_name
        self._provider = None

    def load(self):
        try:
            from providers.factory import get_provider
            from providers.base import GenerationRequest
            self._provider = get_provider(self.provider_name)
            self._GenerationRequest = GenerationRequest
            print(f"??? Provider '{self.provider_name}' loaded ({self._provider.name})")
        except ImportError:
            raise ImportError(
                f"Provider abstraction layer not available. "
                f"Ensure Pre-Work/providers/ directory exists."
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load provider '{self.provider_name}': {exc}")

    def generate(self, prompt: str) -> str:
        try:
            request = self._GenerationRequest(
                prompt=prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                model=self.config.model_name if self.provider_name != 'openai' else self.config.openai_model,
            )
            response = self._provider.generate(request)
            return response.text
        except Exception as e:
            print(f"\n??? {self.provider_name} error: {e}")
            return ""

    def clear_cache(self):
        pass


def get_backend(config: GeneratorConfig) -> BaseModelBackend:
    """
    Factory function to get appropriate backend.

    Educational note:
    A factory keeps provider selection in one place, which makes it easier to add new
    providers later (e.g., Anthropic, Azure OpenAI, local llama.cpp, etc.).
    """
    # Legacy backends (direct implementation, no provider abstraction)
    if config.provider == ModelProvider.HUGGINGFACE:
        return HuggingFaceBackend(config)
    elif config.provider == ModelProvider.OPENAI:
        return OpenAIBackend(config)
    elif config.provider == ModelProvider.MOCK:
        return MockBackend(config)

    # New providers go through the provider abstraction layer
    provider_name_map = {
        ModelProvider.ANTHROPIC: 'anthropic',
        ModelProvider.GOOGLE: 'google',
        ModelProvider.OLLAMA: 'ollama',
        ModelProvider.AZURE_OPENAI: 'azure_openai',
        ModelProvider.GROQ: 'groq',
        ModelProvider.TOGETHER: 'together',
        ModelProvider.CUSTOM: 'custom',
        ModelProvider.AWS_BEDROCK: 'aws_bedrock',
        ModelProvider.REPLICATE: 'replicate',
    }
    provider_name = provider_name_map.get(config.provider)
    if provider_name:
        return ProviderBackend(config, provider_name)

    # Fallback
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
        # Educational note:
        # Strict formatting instructions improve parseability, but models can still drift.
        # If parsing fails often, reduce temperature and tighten the format contract further.
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
        """
        Parse Q&A formatted response.

        Expected model output is a repeated pattern like:
        - Q1: ...
        - A1: ...

        Edge cases to be aware of:
        - Models sometimes emit "Question 1:" instead of "Q1:" (this parser is tolerant-ish).
        - Answers can span multiple lines; we treat subsequent non-Q/non-A lines as continuation.
        - This is a lightweight parser: it aims for speed, not perfect robustness.
        """
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
        """
        Parse text samples separated by marker.

        Educational note:
        Separators are a cheap but effective way to split model output into records.
        If the model forgets separators, lower temperature and add more explicit examples.
        """
        samples = response.split(separator)
        return [{"text": s.strip()} for s in samples if s.strip()]
    
    @staticmethod
    def parse_json_response(response: str, separator: str = "---ENTRY---") -> List[Dict]:
        """
        Parse JSON entries separated by marker.

        Implementation notes:
        - We search for the first '{' and last '}' in each entry chunk and attempt `json.loads`.
        - This tolerates extra prose around the JSON but can fail on trailing commas or invalid JSON.

        If you need high reliability, consider:
        - prompting the model to emit strict JSON only
        - adding a JSON repair step
        - using a streaming JSON parser or a schema validator (pydantic/jsonschema)
        """
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
    
    print("\n\n???? EMERGENCY SAVE TRIGGERED...")
    
    if _global_writer:
        try:
            _global_writer.stop()
            print(f"??? Data saved: {_global_writer.get_written_count()} items")
        except Exception:
            pass
    
    if _global_generator:
        try:
            _global_generator._save_checkpoint()
            print("??? Checkpoint saved!")
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
        """
        Generate hash for content deduplication.

        Educational note:
        - Hash dedup is fast and prevents exact duplicates.
        - It does not catch paraphrases / semantic duplicates.
        - We hash sorted JSON to avoid key-order differences changing the hash.
        """
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
            checkpoint_dir = os.path.dirname(self.config.checkpoint_file)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
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
            print(f"???? Resuming from checkpoint: {count} items")
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
    
    def _build_progress_payload(self, status: str = "running") -> Dict[str, Any]:
        """Build a structured progress payload for callbacks and worker integrations."""
        generated = self.generated.get()
        elapsed = max(0.0, time.time() - self.start_time) if self.start_time else 0.0
        rate = (generated / elapsed) if elapsed > 0 else 0.0
        remaining = max(0, self.config.target_size - generated)
        eta_seconds = int(remaining / rate) if rate > 0 else None

        return {
            "status": status,
            "generated_count": generated,
            "target_count": self.config.target_size,
            "duplicates_count": self.duplicates.get(),
            "invalid_count": self.errors.get(),
            "rate_items_per_sec": rate,
            "eta_seconds": eta_seconds,
            "elapsed_seconds": int(elapsed),
            "output_path": self._get_output_path(),
            "checkpoint_file": self.config.checkpoint_file,
        }

    def _print_progress(self, batch_num: int):
        """Print generation progress."""
        payload = self._build_progress_payload("running")
        generated = payload["generated_count"]
        rate = payload["rate_items_per_sec"]
        eta_seconds = payload["eta_seconds"]

        progress_pct = (generated / self.config.target_size) * 100
        bar_filled = int(progress_pct / 5)
        bar = "#" * bar_filled + "-" * (20 - bar_filled)

        if eta_seconds is None:
            eta_str = "n/a"
        elif eta_seconds < 3600:
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
        else:
            eta_str = f"{eta_seconds / 3600:.1f}h"

        print(
            f"\r[Progress] [{bar}] {progress_pct:.1f}% | {generated:,}/{self.config.target_size:,} | "
            f"Rate: {rate:.1f}/s | ETA: {eta_str} | Dups: {payload['duplicates_count']} | Batch: {batch_num}",
            end="",
            flush=True
        )

    def run(
        self,
        user_prompt: Optional[str] = None,
        parse_mode: str = "qa",
        extra_fields: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        non_interactive: bool = False
    ) -> Dict[str, Any]:
        """
        Run the dataset generation.

        Args:
            user_prompt: Description of what data to generate.
            parse_mode: 'qa' for Q&A, 'text' for paragraphs, 'json' for structured objects.
            extra_fields: For json mode, list of field names to generate.
            progress_callback: Optional callback called with structured progress dicts.
            should_stop: Optional callback checked at least once per batch.
            non_interactive: If True, never prompt via input().
        """
        error_message = None
        final_status = "failed"

        def emit_progress(status: str):
            if not progress_callback:
                return
            try:
                progress_callback(self._build_progress_payload(status))
            except Exception:
                pass

        if user_prompt is None:
            if non_interactive:
                error_message = "Prompt is required in non-interactive mode."
                return {
                    "status": "failed",
                    "generated_count": self.generated.get(),
                    "duplicates_count": self.duplicates.get(),
                    "invalid_count": self.errors.get(),
                    "error_message": error_message,
                }

            print("=" * 60)
            print("UNIVERSAL SYNTHETIC DATASET GENERATOR")
            print("=" * 60)
            print("\nEnter your data generation prompt.")
            print("Examples:")
            print("  - 'Generate educational Q&A about machine learning'")
            print("  - 'Create customer service conversations for a bank'")
            print("  - 'Generate product descriptions for electronics'")
            print()
            user_prompt = input("Enter your prompt: ").strip()

            if not user_prompt:
                error_message = "No prompt provided."
                return {
                    "status": "failed",
                    "generated_count": self.generated.get(),
                    "duplicates_count": self.duplicates.get(),
                    "invalid_count": self.errors.get(),
                    "error_message": error_message,
                }

            print("\nConfiguration:")

            size_input = input(f"   Target dataset size [{self.config.target_size}]: ").strip()
            if size_input.isdigit():
                self.config.target_size = int(size_input)

            mode_input = input("   Output mode - qa/text/json [qa]: ").strip().lower()
            if mode_input in ["qa", "text", "json"]:
                parse_mode = mode_input

            fmt_input = input("   File format - jsonl/json/csv [jsonl]: ").strip().lower()
            if fmt_input in ["jsonl", "json", "csv"]:
                self.config.output_format = fmt_input

            name_input = input(f"   Output filename [{self.config.output_file}]: ").strip()
            if name_input:
                self.config.output_file = name_input

            if parse_mode == "json":
                fields_input = input("   Fields to generate (comma-separated): ").strip()
                if fields_input:
                    extra_fields = [f.strip() for f in fields_input.split(",")]

        self.user_prompt = user_prompt or ""
        self.parse_mode = parse_mode
        self.extra_fields = extra_fields or []

        print("\n" + "=" * 60)
        print("INITIALIZING...")
        print("=" * 60)

        self.backend = get_backend(self.config)
        self.backend.load()

        output_path = self._get_output_path()
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.writer = AsyncFileWriter(output_path, self.config.output_format)

        self._setup_handlers()

        existing_count = self._load_checkpoint()
        self.generated.set(existing_count)

        remaining = self.config.target_size - existing_count
        if remaining <= 0:
            final_status = "completed"
            emit_progress(final_status)
            print(f"Dataset already complete! {existing_count} items exist.")
            summary = self._build_progress_payload(final_status)
            summary["error_message"] = None
            return summary

        print("\nGeneration Plan:")
        print(f"   - Prompt: {self.user_prompt[:50]}...")
        print(f"   - Mode: {parse_mode}")
        print(f"   - Target: {self.config.target_size:,} items")
        print(f"   - Remaining: {remaining:,} items")
        print(f"   - Batch size: {self.config.items_per_batch}")
        print(f"   - Output: {output_path}")
        print()

        self.start_time = time.time()
        self.last_save_time = time.time()
        next_save_at_count = self.generated.get() + max(1, self.config.save_interval)
        batch_num = 0
        stop_requested = False

        emit_progress("running")
        print("Starting generation...\n")

        try:
            while self.generated.get() < self.config.target_size:
                if should_stop and should_stop():
                    stop_requested = True
                    print("\n\nStop requested. Finalizing...")
                    break

                items = self._generate_batch()

                if items:
                    remaining_slots = max(0, self.config.target_size - self.generated.get())
                    if remaining_slots < len(items):
                        items = items[:remaining_slots]
                    self.writer.write_batch(items)
                    self.generated.increment(len(items))

                batch_num += 1
                self._print_progress(batch_num)
                emit_progress("running")

                save_due_to_time = time.time() - self.last_save_time > self.config.auto_save_seconds
                save_due_to_items = self.generated.get() >= next_save_at_count
                if save_due_to_time or save_due_to_items:
                    self._save_checkpoint()
                    self.last_save_time = time.time()
                    if save_due_to_items:
                        next_save_at_count = self.generated.get() + max(1, self.config.save_interval)

                if batch_num % self.config.clear_cache_interval == 0:
                    self.backend.clear_cache()

            if stop_requested:
                final_status = "stopped"
            elif self.generated.get() >= self.config.target_size:
                final_status = "completed"
            else:
                final_status = "stopped"

        except KeyboardInterrupt:
            final_status = "stopped"
            print("\n\nInterrupted by user")
        except Exception as e:
            final_status = "failed"
            error_message = str(e)
            print(f"\n\nError: {e}")
        finally:
            print("\n\n" + "=" * 60)
            print("FINALIZING...")
            print("=" * 60)

            if self.writer:
                self.writer.stop()

            self._save_checkpoint()

            elapsed = max(0.0001, time.time() - self.start_time) if self.start_time else 0.0001
            final_count = self.generated.get()

            print(f"\nStatus: {final_status}")
            print(f"   - Total items: {final_count:,}")
            print(f"   - Duplicates skipped: {self.duplicates.get():,}")
            print(f"   - Time elapsed: {elapsed/60:.1f} minutes")
            print(f"   - Average rate: {final_count/elapsed:.1f} items/second")
            print(f"   - Output file: {output_path}")

            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"   - File size: {size_mb:.2f} MB")

            if error_message:
                print(f"   - Error: {error_message}")

            emit_progress(final_status)

            summary = self._build_progress_payload(final_status)
            summary["error_message"] = error_message
            return summary


# ============================================================================
# SECTION 9: MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for command-line usage."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Universal Synthetic Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python universal_dataset_generator.py
  python universal_dataset_generator.py --prompt "Generate Q&A about Python programming" --size 500
  python universal_dataset_generator.py --config job.json
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
    parser.add_argument("--checkpoint", type=str, default="generator_checkpoint.json",
                       help="Checkpoint file path")
    parser.add_argument("--batch", "-b", type=int, default=10, help="Items per batch")
    parser.add_argument("--provider", type=str,
                       choices=["huggingface", "openai", "mock", "anthropic", "google",
                                "ollama", "azure_openai", "groq", "together", "custom",
                                "aws_bedrock", "replicate"],
                       default="huggingface", help="Model provider")
    parser.add_argument("--model", type=str, help="Model name (for HuggingFace)")
    parser.add_argument("--fields", type=str, help="Comma-separated fields for JSON mode")
    parser.add_argument("--config", type=str, help="Path to JSON config file for non-interactive execution")
    parser.add_argument("--auto-install", action="store_true", help="Enable runtime dependency installation")

    args = parser.parse_args()

    config_json = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            parsed = json.load(f)
            if not isinstance(parsed, dict):
                raise ValueError("--config file must contain a JSON object")
            config_json = parsed

    provider_name = str(config_json.get("provider", args.provider)).lower()
    try:
        provider = ModelProvider[provider_name.upper()]
    except KeyError:
        raise ValueError(f"Unsupported provider in config: {provider_name}")

    target_size = int(config_json.get("targetCount", config_json.get("size", args.size)))
    items_per_batch = int(config_json.get("batchSize", config_json.get("batch", args.batch)))
    output_format = str(config_json.get("outputFormat", config_json.get("format", args.format))).lower()

    output_path = config_json.get("outputPath", config_json.get("output", args.output))
    output_base = str(output_path)
    output_root, output_ext = os.path.splitext(output_base)
    if output_ext.lower() in [".jsonl", ".json", ".csv"]:
        if "outputFormat" not in config_json and "format" not in config_json:
            output_format = output_ext.lower().lstrip(".")
        output_base = output_root

    checkpoint_file = str(
        config_json.get("checkpointPath", config_json.get("checkpointFile", args.checkpoint))
    )

    config = GeneratorConfig(
        target_size=target_size,
        items_per_batch=items_per_batch,
        output_file=output_base,
        output_format=output_format,
        checkpoint_file=checkpoint_file,
        provider=provider,
    )

    if args.model:
        config.model_name = args.model
    if config_json.get("model"):
        config.model_name = str(config_json["model"])
    if config_json.get("modelName"):
        config.model_name = str(config_json["modelName"])
    if config_json.get("openaiModel"):
        config.openai_model = str(config_json["openaiModel"])
    if config_json.get("saveInterval") is not None:
        config.save_interval = max(1, int(config_json["saveInterval"]))
    if config_json.get("autoSaveSeconds") is not None:
        config.auto_save_seconds = max(1, int(config_json["autoSaveSeconds"]))

    parse_mode = str(config_json.get("parseMode", args.mode)).lower()

    prompt = config_json.get("prompt", args.prompt)
    if not prompt:
        domain_description = str(config_json.get("domainDescription", "")).strip()
        topics = config_json.get("topics", [])
        if isinstance(topics, list):
            topic_values = [str(t).strip() for t in topics if str(t).strip()]
        else:
            topic_values = []

        prompt_parts = []
        if domain_description:
            prompt_parts.append(domain_description)
        if topic_values:
            prompt_parts.append(f"Topics: {', '.join(topic_values)}")
        prompt = "\n".join(prompt_parts).strip()

    if args.fields:
        extra_fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    elif isinstance(config_json.get("extraFields"), list):
        extra_fields = [str(f).strip() for f in config_json["extraFields"] if str(f).strip()]
    elif isinstance(config_json.get("fields"), list):
        extra_fields = [str(f).strip() for f in config_json["fields"] if str(f).strip()]
    else:
        extra_fields = None

    generator = UniversalGenerator(config)
    result = generator.run(
        user_prompt=prompt,
        parse_mode=parse_mode,
        extra_fields=extra_fields,
        non_interactive=bool(args.config)
    )

    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()


