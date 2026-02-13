# Dataset Schema & Formats (JSONL / JSON / CSV)

This repo produces (or simulates) datasets in multiple formats. This doc explains:

- the most common schema patterns used in this repo
- practical differences between JSONL/JSON/CSV
- pitfalls when training ML/LLM systems with synthetic text

## Quick definitions

- **JSONL** (JSON Lines): one JSON object per line. Best for streaming and large datasets.
- **JSON**: one large JSON array/object. Convenient for small datasets; less convenient for huge files.
- **CSV**: tabular. Easy to open in spreadsheets; weaker for nested metadata.

## “Q&A dataset” shape (common pattern)

Many examples in this repo use a Q&A structure:

```json
{
  "id": "item_abc123_0001",
  "topic": "Investing",
  "question": "What is dollar-cost averaging?",
  "answer": "Dollar-cost averaging is an investing strategy where...",
  "difficulty": "beginner",
  "metadata": { "source_prompt": "..." },
  "created_at": "2026-01-29T00:00:00.000Z"
}
```

### Field notes
- `id`: stable identifier; useful for dedup and traceability.
- `topic`: optional but helps with stratified sampling.
- `difficulty`: good for curriculum-style training; treat as a label.
- `metadata`: store provenance (prompt, model/provider, parameters).

## Web demo server format (reality-aligned)

The web backend in `website/server/index.js` returns a **mock dataset** for downloads.
It uses fields like:

- `id`, `topic`, `question`, `answer`, `difficulty`, `created_at`

This is useful for UI demos and learning the API flow, but it is not “real generation”.

## Python universal generator output (reality-aligned)

`Pre-Work/universal_dataset_generator.py` writes a structure shaped like:

- `id`
- one or more “content fields” depending on the parse mode
- `metadata` (source prompt, parse mode)
- `created_at`

### Parse modes
- `qa`: expects Q&A-like structured content
- `text`: paragraph-like samples
- `json`: arbitrary structured fields (you specify field names)

## Format examples

### JSONL example

```json
{"id":"item_a1b2c3_1234","question":"...","answer":"...","metadata":{"source_prompt":"...","parse_mode":"qa"},"created_at":"2026-01-29T00:00:00"}
{"id":"item_d4e5f6_5678","question":"...","answer":"...","metadata":{"source_prompt":"...","parse_mode":"qa"},"created_at":"2026-01-29T00:00:01"}
```

### CSV example

CSV works best for flat structures:

```csv
id,question,answer,created_at
item_a1b2c3_1234,"What is ...?","...",2026-01-29T00:00:00
```

If you need nested metadata, prefer JSONL/JSON.

## Practical pitfalls (synthetic data)

1. **PII and sensitive info**
   - Synthetic data can still include real names/addresses if prompted poorly.
2. **Harmful “advice”**
   - Finance/health/legal outputs can look authoritative but be wrong.
3. **Mode collapse**
   - Too low temperature or too repetitive prompts create near-duplicates.
4. **Train/test leakage**
   - If you include real documents in prompts, you can leak copyrighted content.

See `docs/SECURITY_AND_SAFETY.md` for a deeper discussion.

