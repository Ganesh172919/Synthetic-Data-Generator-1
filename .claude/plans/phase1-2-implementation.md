# Phase 1 + 2 Implementation Plan: Foundation & Multi-Provider LLM Engine

## Overview

Build on the existing Express + SQLite + React codebase. Fix stability issues first, then add the multi-provider abstraction layer. All changes maintain backward compatibility with the existing job lifecycle.

---

## Part A: Foundation & Stability (Phase 1)

### A1. Fix stale documentation
**Files:** `Readme.md`, `docs/ARCHITECTURE.md`
- Rewrite project structure to reflect actual layout (`website/server/`, `website/client/`, `worker/`, `Pre-Work/`)
- Remove references to non-existent files (`test-integration.js`, `generator_runner.py`, `server/`)

### A2. Entry point hardening
**Files:** `website/server/index.js`
- Add uncaught exception / unhandled rejection handlers
- Add `.catch()` on `start()` call

### A3. Config improvements
**Files:** `website/server/src/config.js`
- Add `authMode` validation (must be `'none'` or `'api_key'`)
- Add `logLevel` config (currently inline in server.js)
- Add `maxConcurrentJobs` config for worker
- Freeze the config object after creation

### A4. Database migrations system
**Files:** `website/server/src/migrations.js` (new), `website/server/src/db.js`
- Create a simple migration runner: `migrations` version table + numbered SQL functions
- Convert existing `CREATE TABLE IF NOT EXISTS` to migration v1
- Add indexes for `domains` table (`name`, `created_at`)
- Add `DELETE /api/domains/:id` and `PUT /api/domains/:id` support

### A5. Server error handling & cleanup
**Files:** `website/server/src/server.js`
- Add `asyncHandler` wrapper for async routes (Express 4 doesn't catch promise rejections)
- Add request validation for `POST /api/domains` (max length on name)
- Add domain CRUD: `DELETE /api/domains/:id`, `PUT /api/domains/:id`
- Expand `validProviders` list for new providers (Phase 2)
- Expand `validParseModes` for new modes
- Add `language` parameter to generate endpoint
- Remove unused `uuid` dependency from `package.json`

### A6. Worker multi-concurrency support
**Files:** `worker/main.py`
- Remove the `MAX_CONCURRENT_JOBS=1` hardcode warning
- Implement `asyncio`-based concurrent job execution with semaphore
- Add configurable concurrency via `MAX_CONCURRENT_JOBS` env var
- Add per-provider retry logic with exponential backoff

### A7. Test suite
**Files:** `website/server/src/__tests__/config.test.js` (new), `website/server/src/__tests__/db.test.js` (new), `website/server/src/__tests__/server.test.js` (new)
- Unit tests for `config.js` (env parsing, defaults, clamping, validation)
- Unit tests for `db.js` (init, CRUD helpers, migration runner)
- Integration tests for API routes (health, templates, generate, jobs CRUD, domains CRUD, downloads, preview, metrics, SSE)
- Update `package.json` dev script to use `--watch`

### A8. Multi-language support
**Files:** `website/server/src/server.js`, `website/server/src/templates.js`
- Add `language` field to generate config (default: `'en'`)
- Store language in job config
- Pass language to worker via config_json

---

## Part B: Multi-Provider LLM Engine (Phase 2)

### B1. Provider abstraction layer
**Files:** `Pre-Work/providers/__init__.py` (new), `Pre-Work/providers/base.py` (new), `Pre-Work/providers/factory.py` (new)

Create a `BaseProvider` ABC with unified interface:
```python
class BaseProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, config: GenerationRequest) -> str: ...
    @abstractmethod
    def health_check(self) -> ProviderHealth: ...
    @abstractmethod
    def get_models(self) -> list[str]: ...
```

Provider factory with registry pattern for dynamic provider lookup.

### B2. Provider implementations
**Files:** `Pre-Work/providers/mock.py`, `Pre-Work/providers/openai_provider.py`, `Pre-Work/providers/huggingface.py`, `Pre-Work/providers/anthropic.py`, `Pre-Work/providers/google.py`, `Pre-Work/providers/ollama.py`, `Pre-Work/providers/azure_openai.py`, `Pre-Work/providers/groq.py`, `Pre-Work/providers/together.py`, `Pre-Work/providers/custom.py`

- Refactor existing mock/openai/huggingface logic from `universal_dataset_generator.py` into provider classes
- Each provider: `generate()`, `health_check()`, `get_models()`
- Provider-specific retry logic (rate limits, token limits)
- Custom endpoint support for any OpenAI-compatible API (vLLM, text-generation-inference, llama.cpp)

### B3. Integrate providers into generator
**Files:** `Pre-Work/universal_dataset_generator.py`
- Replace inline provider logic with factory-based provider lookup
- Keep backward compatibility with existing `ModelProvider` enum
- Add `language` parameter to generation prompts

### B4. Integrate providers into worker
**Files:** `worker/main.py`
- Use provider factory instead of directly importing generator module's provider enum
- Add provider health check before job execution
- Add automatic retry with backoff on provider errors

### B5. API & config updates for providers
**Files:** `website/server/src/server.js`, `website/server/src/config.js`, `website/server/src/templates.js`
- Expand `validProviders` to include all new providers
- Add provider config endpoint (`GET /api/providers`) listing available providers and their models
- Add provider health check endpoint (`GET /api/providers/:name/health`)
- Add `language` field to templates
- Expand templates to 15+ domains

### B6. Client updates
**Files:** `website/client/src/services/api.js`, `website/client/src/pages/Dashboard.jsx`
- Add provider selection with model dropdown in generate form
- Add language selector
- Add provider status indicators
- Add new API methods: `getProviders()`, `getProviderHealth()`, `deleteDomain()`, `updateDomain()`

---

## Implementation Order

1. **A2** → **A3** → **A4** → **A5** (server foundation)
2. **A7** (tests — verify foundation works)
3. **A6** (worker concurrency)
4. **A1** (docs)
5. **A8** (language support)
6. **B1** → **B2** (provider abstraction)
7. **B3** → **B4** (integrate into generator + worker)
8. **B5** → **B6** (API + client updates)

## Key Files Modified

| File | Changes |
|------|---------|
| `website/server/index.js` | Error handlers |
| `website/server/package.json` | Remove uuid, add nodemon dev dep, add test scripts |
| `website/server/src/config.js` | Add logLevel, maxConcurrentJobs, authMode validation, freeze |
| `website/server/src/db.js` | Integrate migration runner |
| `website/server/src/migrations.js` | **New** — migration system |
| `website/server/src/server.js` | asyncHandler, domain CRUD, expanded providers/modes, language |
| `website/server/src/templates.js` | Expand to 15+ templates, add language field |
| `website/server/src/__tests__/*.test.js` | **New** — test suite |
| `worker/main.py` | Multi-concurrency, provider factory integration |
| `Pre-Work/providers/*.py` | **New** — provider abstraction + implementations |
| `Pre-Work/universal_dataset_generator.py` | Use provider factory, add language |
| `website/client/src/services/api.js` | New API methods |
| `website/client/src/pages/Dashboard.jsx` | Provider/language selectors |
| `Readme.md` | Fix stale docs |
| `docs/ARCHITECTURE.md` | Fix stale docs |

## Risks & Mitigations

- **Risk:** Changing `universal_dataset_generator.py` breaks worker. **Mitigation:** Keep `ModelProvider` enum as backward-compatible wrapper around new factory.
- **Risk:** New providers need API keys that users may not have. **Mitigation:** Mock provider always works; other providers gracefully report "not configured".
- **Risk:** Test suite needs better-sqlite3 which requires native compilation. **Mitigation:** Tests use in-memory SQLite (`:memory:`) when possible.
