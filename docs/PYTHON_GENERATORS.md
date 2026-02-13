# Python Generators — Deep Dive (Universal + Finance Ultra)

This repo contains two generator scripts in `Pre-Work/`:

- `universal_dataset_generator.py` — flexible, prompt-driven generation for many domains
- `financial_education_generator_ultra.py` — a speed-optimized finance Q&A generator

## Universal generator (`universal_dataset_generator.py`)

### What problem it solves

You want to create a dataset from a natural-language prompt, with:

- batching for throughput
- deduplication
- periodic checkpoint saves
- multiple output formats (JSONL/JSON/CSV)
- multiple “providers” (local model, API, or mock)

### Key concepts

#### Provider abstraction

The generator selects a backend based on `GeneratorConfig.provider`:

- `HUGGINGFACE`: local model via Transformers
- `OPENAI`: API model via `OPENAI_API_KEY`
- `MOCK`: deterministic fake output for testing flows

#### Parse modes

- `qa`: question/answer style
- `text`: paragraphs or single-field samples
- `json`: structured records with user-specified fields

#### Throughput vs quality

Important knobs:

- `items_per_batch`: higher = fewer model calls, but more parsing work per call
- `max_new_tokens`: higher = more content per call, but slower per call
- `temperature`: higher = more diversity, higher risk of format drift

#### Async buffered file writer

`AsyncFileWriter` buffers items and writes in a background thread. This:

- reduces time spent blocking on disk I/O
- makes generation smoother
- requires careful flushing on shutdown (handled in `stop()` and emergency handlers)

#### Deduplication

The generator hashes normalized JSON content to avoid duplicates. Tradeoffs:

- fast and simple
- can miss “semantic duplicates” that differ by punctuation
- can treat reordered JSON keys differently if not normalized (this repo uses `sort_keys=True`)

### Extension recipe (new provider)

To add a new provider:

1. Add a new `ModelProvider` enum value
2. Implement a `BaseModelBackend` subclass
3. Update backend factory selection (`get_backend(...)`)
4. Document configuration + required secrets/env vars

## Finance ultra generator (`financial_education_generator_ultra.py`)

### What problem it solves

Generate a large finance Q&A dataset quickly (e.g., 30k items) while surviving:

- Colab disconnects
- long-running sessions
- GPU memory pressure

### Key concepts

#### Mega-batching

Instead of generating a single Q&A per model call, prompts ask for **many** Q&As.
This amortizes:

- tokenization overhead
- model forward pass setup
- Python-side parsing and validation

#### Emergency save/download

The script registers `atexit` and signal handlers so you can:

- interrupt safely (Ctrl+C)
- flush buffered output
- trigger download in Colab (when available)

#### Flash Attention dependency risk

`flash-attn` can fail to install depending on:

- CUDA version
- compiler toolchain
- platform (Windows is often problematic)

The docs and comments in the script explain what happens when it fails and how
to fall back.

## Practical guidance

- Prefer JSONL when generating large datasets.
- Track provenance in `metadata` (prompt, provider/model, sampling params).
- Add validators and red-flag filters for safety in sensitive domains.

See also:
- `docs/DATASET_SCHEMA.md`
- `docs/TROUBLESHOOTING.md`
- `docs/SECURITY_AND_SAFETY.md`

