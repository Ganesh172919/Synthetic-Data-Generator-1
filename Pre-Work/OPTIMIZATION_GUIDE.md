# Financial Dataset Generator - Optimization Guide

## Target: 10,000 Q&A pairs in 3 hours

This guide explains the optimizations made to achieve the target generation speed.

---

## Key Performance Optimizations

### 1. Combined Q&A Generation (50% Reduction in LLM Calls)

**Before:** 2 LLM calls per Q&A pair

- Call 1: Generate question
- Call 2: Generate answer

**After:** 1 LLM call generates 8 Q&A pairs

- Single prompt generates multiple complete Q&A pairs
- Reduces overhead from tokenization, model inference, and I/O

**Impact:** ~3-4x faster generation

### 2. Larger Batch Size

**Before:** 5 questions per batch, processed one-by-one

**After:** 8 Q&A pairs generated in single LLM call

**Configuration:**

```python
batch_size: int = 8  # Q&A pairs per LLM call
```

### 3. Async File I/O with Buffering

**Before:** Write each Q&A immediately to disk

**After:** Buffer 50 Q&A pairs, write in batch

```python
save_interval: int = 50  # Buffer size
```

**Impact:** Reduces I/O overhead by ~90%

### 4. Fast Hash-Based Deduplication

**Before:** Loads sentence-transformers embedding model (~400MB)

**After:** MD5 hash comparison

```python
@staticmethod
def compute_hash(text: str) -> str:
    normalized = ' '.join(text.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()
```

**Impact:**

- No embedding model loading time
- ~100x faster deduplication
- Lower memory usage

### 5. Streamlined Validation

**Before:** Multiple validation passes with LLM review

**After:** Fast pattern matching without LLM calls

```python
class FastValidator:
    RED_FLAGS = ["i don't know", "as an ai", ...]
    BAD_ADVICE = ["you should invest", "guaranteed returns", ...]

    @staticmethod
    def validate(qa_dict: Dict) -> bool:
        # Simple pattern matching - no LLM calls
```

**Impact:** ~50x faster validation

### 6. Reduced Logging Overhead

**Before:** INFO level logging to console and file

**After:** WARNING level logging to file only

```python
level=logging.WARNING,  # Reduced for speed
handlers=[logging.FileHandler(CONFIG.log_file)]  # No console
```

### 7. GPU Memory Management

**Before:** Clear cache every 100 generations

**After:** Clear cache every 200 generations

```python
clear_cache_interval: int = 200
```

---

## Performance Calculation

### Target Analysis

- **Goal:** 10,000 Q&A in 3 hours (180 minutes)
- **Required Rate:** ~55.5 Q&A per minute
- **Per Q&A Time Budget:** ~1.08 seconds

### With Optimizations

- **Batch Size:** 8 Q&A per LLM call
- **LLM Calls Needed:** ~1,250 calls
- **Time per LLM Call:** ~8-12 seconds (typical for Mistral-7B)
- **Effective Rate:** ~40-60 Q&A per minute

---

## Usage Instructions

### Google Colab (Recommended)

```python
# Upload the optimized file to Colab, then run:
!python financial_education_generator_optimized.py
```

### Local Machine with GPU

```bash
# Install dependencies first
pip install transformers accelerate bitsandbytes torch tqdm

# Run the generator
python financial_education_generator_optimized.py
```

---

## Configuration Tuning

### For Faster Speed (Lower Quality)

```python
@dataclass
class Config:
    batch_size: int = 10          # Increase batch size
    min_answer_length: int = 50   # Accept shorter answers
    max_new_tokens: int = 600     # Generate less per call
    temperature: float = 0.8      # More creative, possibly faster
```

### For Higher Quality (Slower)

```python
@dataclass
class Config:
    batch_size: int = 5           # Smaller batches
    min_answer_length: int = 120  # Require longer answers
    max_new_tokens: int = 1000    # Allow more detailed responses
    temperature: float = 0.6      # More focused responses
```

### Memory Constrained (8GB VRAM)

```python
@dataclass
class Config:
    batch_size: int = 4           # Smaller batches
    clear_cache_interval: int = 100  # More frequent clearing
    use_quantization: bool = True    # Always use 4-bit
```

---

## Expected Performance by Hardware

| Hardware             | Expected Rate  | Time for 10k |
| -------------------- | -------------- | ------------ |
| T4 GPU (Colab Free)  | 40-50 Q&A/min  | ~3-4 hours   |
| A100 GPU (Colab Pro) | 80-120 Q&A/min | ~1.5-2 hours |
| RTX 3090             | 60-80 Q&A/min  | ~2-3 hours   |
| CPU Only             | 5-10 Q&A/min   | ~15-30 hours |

---

## Troubleshooting

### Out of Memory Errors

1. Reduce `batch_size` to 4
2. Reduce `max_new_tokens` to 500
3. Increase `clear_cache_interval` to 50

### Slow Generation

1. Ensure GPU is being used (`torch.cuda.is_available()`)
2. Check if model is fully loaded
3. Verify quantization is enabled

### Low Quality Outputs

1. Increase `min_answer_length` to 100
2. Reduce `batch_size` to 5
3. Lower `temperature` to 0.6

### Duplicate Questions

1. Run is working correctly - duplicates are detected and skipped
2. Check `stats["duplicates"]` counter

---

## Monitoring Progress

The generator shows real-time progress:

```
📊 Progress: 5,234/10,000 (52.3%) | Rate: 48.2/min | ETA: 1.6 hours | Rej: 23 | Dup: 12
```

- **Progress:** Current count vs target
- **Rate:** Q&A pairs generated per minute
- **ETA:** Estimated time to completion
- **Rej:** Rejected pairs (failed validation)
- **Dup:** Duplicate pairs skipped

---

## Files Generated

1. `financial_education_dataset.jsonl` - Main dataset
2. `generator_checkpoint.json` - Progress checkpoint (for resuming)
3. `generator.log` - Error logs

---

## Resuming Interrupted Generation

The generator automatically saves checkpoints. If interrupted:

```bash
# Just run again - it will resume from last checkpoint
python financial_education_generator_optimized.py
```

---

## Sample Output Format

```json
{
  "id": "fin_Per_Bud_1234_56789",
  "topic": "Personal Finance",
  "subtopic": "Budgeting Basics",
  "question": "What is the 50/30/20 rule in budgeting?",
  "answer": "The 50/30/20 rule is a simple budgeting framework...",
  "difficulty": "beginner",
  "question_type": "definition",
  "review_status": "verified",
  "created_at": "2026-01-15T10:30:00.000000"
}
```

## Educational Notes (Added)

### What this file is for

This guide explains *why* the “ultra/optimized” generator variants can be dramatically faster than naive LLM prompting loops. It’s a performance playbook you can reuse for other domains.

### Reality-aligned updates (paths, current behavior)

The optimization concepts here apply to both:

- `Pre-Work/financial_education_generator_ultra.py` (finance-specific, very aggressive)
- `Pre-Work/universal_dataset_generator.py` (domain-agnostic, more general)

The web backend (`website/server/index.js`) does not run these generators today; it simulates job progress.

### Why these optimizations work (intuition + mechanics)

1. **Batching amortizes fixed costs**
   - Each model call has overhead (tokenization, scheduling, Python glue, I/O).
   - Generating N items per call spreads that overhead across more outputs.

2. **Async/buffered I/O reduces stalls**
   - Writing every item synchronously creates many small writes.
   - Buffering lets the generator stay compute-bound instead of I/O-bound.

3. **Simple dedup beats semantic dedup (for speed)**
   - Hash-based deduplication is extremely fast.
   - It does not catch paraphrase-level duplicates, but it prevents exact/repeated outputs cheaply.

4. **Quantization increases throughput**
   - 4-bit quantization reduces VRAM and can increase effective throughput on limited hardware.

### When NOT to use these settings

The fastest settings are not always the best:

- If your model frequently drifts from the output format, mega-batches can produce many unusable items at once.
- If quality matters more than speed, you may prefer smaller batches and more conservative sampling.
- Hash dedup can still allow “semantic duplicates” that hurt dataset diversity.

### Benchmarking guidance (practical)

Track these metrics for each run:

- **items/min** (or Q&A/min) — primary throughput metric
- **acceptance rate** — accepted items / total generated items
- **dup rate** — duplicates skipped / total attempts
- **VRAM headroom** — peak VRAM usage and “OOM margin”

Recommendation: record parameters (batch size, max tokens, temperature) alongside results so the benchmark is reproducible.

### Hardware-tier tuning recipes

These are starting points. Adjust based on failure modes.

#### T4 (Colab Free)
- Keep batches moderate
- Keep token limits conservative
- Expect slower warm-up (model download + install)

#### RTX 3090/4090
- Increase batch size gradually until you see OOM or format drift
- Use quantization if VRAM is tight; otherwise compare full precision vs quantized

#### CPU-only
- Reduce expectations; focus on correctness and small dataset sizes
- Consider using the `MOCK` provider for pipeline testing

### Learning notes

- High throughput generation is usually about **reducing per-item overhead**, not “making the model smarter”.
- Many performance tricks become reliability tricks: buffering + checkpoints exist because long runs fail.

### Next steps / exercises

1. Choose one knob (batch size, max tokens, temperature) and run a small sweep; chart throughput vs quality.
2. Design a lightweight validator for your domain (red flags, forbidden phrases) and measure how it affects acceptance rate.
