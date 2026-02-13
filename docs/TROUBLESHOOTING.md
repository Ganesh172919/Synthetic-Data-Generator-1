# Troubleshooting — Common Issues and Fixes

This is a practical “things go wrong” guide for the repo.

## Web platform issues

### UI loads but API calls fail

Symptoms:
- Templates page shows fallback templates
- Dashboard starts “demo mode”
- Network errors in browser console

Checks:
1. Confirm server is running: `http://localhost:3001/api/health`
2. Confirm Vite proxy is configured: `website/client/vite.config.js`
3. Confirm ports aren’t in use (3001 and 5173)

### CORS confusion

In dev, you generally should **not** need CORS if:
- UI uses `/api/...`
- Vite proxies to `http://localhost:3001`

If you call the API directly from the browser (absolute URL), then CORS matters.

## Node/npm issues

### “react-router-dom not found” / missing dependencies

Run installs in the correct folders:
- `cd website/server && npm install`
- `cd website/client && npm install`

There is no `website/package.json` in this repo, so `npm install` from `website/`
won’t install both projects.

## Python / GPU issues

### Torch installation is huge or fails

The generator scripts attempt to auto-install dependencies. On some systems:
- installing `torch` via pip can be slow and platform-specific
- CUDA-enabled builds may require extra index URLs

If you are on Windows or CPU-only, you may prefer:
- running on Google Colab (GPU preconfigured)
- or installing a CPU-only PyTorch build intentionally

### Out of memory (CUDA OOM)

Mitigations:
- reduce `batch_size` / `items_per_batch`
- reduce `max_new_tokens`
- clear cache more often (`clear_cache_interval`)
- use quantization if available

### flash-attn build failures

Common causes:
- unsupported CUDA/toolchain
- platform incompatibilities (often Windows)

Fallback:
- disable flash attention and/or remove the dependency
- use smaller models and reduce token limits

## Dataset quality issues

### Too many duplicates

Try:
- increase temperature slightly
- broaden topics
- add more variation in prompting
- add additional fields (difficulty, subtopic) to encourage diversity

### Format drift (bad JSON, missing fields)

Try:
- lower temperature
- use stricter output formatting instructions
- add lightweight parsing/repair steps (documented in the Python generator comments)

