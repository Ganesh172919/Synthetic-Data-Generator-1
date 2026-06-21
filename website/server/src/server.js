const fs = require('fs');
const path = require('path');
const readline = require('readline');
const crypto = require('crypto');
const express = require('express');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const pino = require('pino');
const pinoHttp = require('pino-http');

const { config } = require('./config');
const { templates } = require('./templates');
const {
  initDatabase,
  safeJsonParse,
  nowIso,
  toApiJob,
  toApiDomain,
  insertJobEvent,
  deleteArtifactsDirectory,
} = require('./db');

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const validDomains = [
  'financial', 'healthcare', 'legal', 'technology', 'science', 'education',
  'customer_support', 'ecommerce', 'realestate', 'gaming', 'marketing',
  'hr', 'news', 'cybersecurity', 'travel', 'food', 'custom',
];

const validOutputFormats = ['jsonl', 'csv', 'json'];

const validProviders = [
  'auto', 'mock', 'openai', 'huggingface', 'anthropic', 'google',
  'ollama', 'azure_openai', 'groq', 'together', 'custom',
  'aws_bedrock', 'replicate',
];

const validParseModes = [
  'qa', 'text', 'json', 'instruction', 'conversation',
  'classification', 'ner', 'summarization', 'translation', 'code', 'reasoning',
];

const terminalStatuses = ['completed', 'failed', 'stopped'];

const validLanguages = [
  'en', 'es', 'fr', 'de', 'it', 'pt', 'zh', 'ja', 'ko', 'hi',
  'ar', 'ru', 'nl', 'pl', 'tr', 'vi', 'th', 'sv', 'da', 'fi',
];

// ---------------------------------------------------------------------------
// Logger & DB
// ---------------------------------------------------------------------------

const logger = pino({
  level: config.logLevel,
});

const db = initDatabase(config);

// ---------------------------------------------------------------------------
// SaaS Platform: Extended schema & services
// ---------------------------------------------------------------------------
const { applyExtendedSchema, migrateJobsTable } = require('./services/schemaExtensions');
const { UserService } = require('./services/userService');
const { SubscriptionService } = require('./services/subscriptionService');
const { UsageService } = require('./services/usageService');
const { ApiKeyService } = require('./services/apiKeyService');
const { FeatureFlagService } = require('./services/featureFlagService');
const { PluginService } = require('./services/pluginService');
const { AnalyticsService } = require('./services/analyticsService');
const { cache } = require('./services/cacheService');
const { createAuthMiddleware } = require('./middleware/auth');
const { createUsageTracker, createJobCreationTracker, createDownloadTracker } = require('./middleware/usage');
const { errorHandler } = require('./utils/errors');
const { createAuthRoutes } = require('./routes/authRoutes');
const { createAdminRoutes } = require('./routes/adminRoutes');
const { createBillingRoutes } = require('./routes/billingRoutes');
const { createApiKeyRoutes } = require('./routes/apiKeyRoutes');
const { createPluginRoutes } = require('./routes/pluginRoutes');
const { createAnalyticsRoutes } = require('./routes/analyticsRoutes');

if (config.enableSaaS) {
  applyExtendedSchema(db);
  migrateJobsTable(db);
}

const userService = config.enableSaaS ? new UserService(db) : null;
const subscriptionService = config.enableSaaS ? new SubscriptionService(db) : null;
const usageService = config.enableSaaS ? new UsageService(db) : null;
const apiKeyService = config.enableSaaS ? new ApiKeyService(db) : null;
const featureFlagService = config.enableSaaS ? new FeatureFlagService(db) : null;
const pluginService = config.enableSaaS ? new PluginService(db) : null;
const analyticsService = config.enableSaaS ? new AnalyticsService(db) : null;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const idJob = () => `gen_${crypto.randomUUID().replace(/-/g, '').slice(0, 8)}`;
const idDomain = () => `domain_${crypto.randomUUID().replace(/-/g, '').slice(0, 8)}`;

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const parseArrayOfStrings = (value) => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => (typeof item === 'string' ? item.trim() : ''))
    .filter(Boolean);
};

const toIsoMs = (iso) => (iso ? new Date(iso).getTime() : null);

const isLocalRequest = (req) => {
  const raw = (req.ip || req.socket?.remoteAddress || '').replace('::ffff:', '');
  return raw === '127.0.0.1' || raw === '::1' || raw.startsWith('127.');
};

/**
 * Wraps an async route handler so thrown/rejected errors reach Express error handler.
 * Express 4 does not catch promise rejections in route handlers.
 */
const asyncHandler = (fn) => (req, res, next) => {
  Promise.resolve(fn(req, res, next)).catch(next);
};

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

const extractApiKey = (req) => {
  const xApiKey = req.get('x-api-key');
  if (xApiKey) {
    return xApiKey.trim();
  }
  const auth = req.get('authorization');
  if (auth && auth.toLowerCase().startsWith('bearer ')) {
    return auth.slice(7).trim();
  }
  return null;
};

const authMiddleware = (req, res, next) => {
  if (!req.path.startsWith('/api')) {
    next();
    return;
  }

  if (config.authMode !== 'api_key') {
    next();
    return;
  }

  const key = extractApiKey(req);
  if (key && config.apiKeys.includes(key)) {
    next();
    return;
  }

  if (isLocalRequest(req)) {
    next();
    return;
  }

  res.status(401).json({ error: 'Unauthorized' });
};

// ---------------------------------------------------------------------------
// Rate Limiters
// ---------------------------------------------------------------------------

const rateKeyGenerator = (req) => rateLimit.ipKeyGenerator(req.ip);

const generalLimiter = rateLimit({
  windowMs: config.generalRateLimitWindowMs,
  max: config.generalRateLimitMax,
  keyGenerator: rateKeyGenerator,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Rate limit exceeded' },
});

const generateLimiter = rateLimit({
  windowMs: config.generateRateLimitWindowMs,
  max: config.generateRateLimitMax,
  keyGenerator: rateKeyGenerator,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Generation rate limit exceeded' },
});

const downloadLimiter = rateLimit({
  windowMs: config.downloadRateLimitWindowMs,
  max: config.downloadRateLimitMax,
  keyGenerator: rateKeyGenerator,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Download rate limit exceeded' },
});

// ---------------------------------------------------------------------------
// Prepared Statements & Queries
// ---------------------------------------------------------------------------

const getJobRow = db.prepare('SELECT * FROM jobs WHERE id = ?');
const getDomainRow = db.prepare('SELECT * FROM domains WHERE id = ?');

const createDomainPrompt = (configJson, fallbackDomain) => {
  const cfg = safeJsonParse(configJson, {});
  const topicList = Array.isArray(cfg.topics)
    ? cfg.topics
        .map((topic) => {
          if (typeof topic === 'string') {
            return topic.trim();
          }
          if (topic && typeof topic.name === 'string') {
            return topic.name.trim();
          }
          return '';
        })
        .filter(Boolean)
    : [];

  const parts = [];
  if (cfg.description) {
    parts.push(cfg.description);
  }
  if (topicList.length > 0) {
    parts.push(`Topics: ${topicList.join(', ')}`);
  }
  if (parts.length === 0) {
    parts.push(`Generate high-quality ${fallbackDomain} synthetic data samples.`);
  }

  return parts.join('\n');
};

const resolveOutputPath = (jobRow, requestedFormat) => {
  const outputsBase = path.resolve(config.outputsDir);
  const outputDirPath = path.resolve(path.dirname(outputsBase), jobRow.output_dir);

  if (!outputDirPath.startsWith(outputsBase)) {
    throw new Error('Invalid output directory');
  }

  const filename = jobRow.output_file || `dataset.${requestedFormat}`;
  if (!filename.endsWith(`.${requestedFormat}`)) {
    throw new Error('Format mismatch');
  }

  const filePath = path.resolve(outputDirPath, filename);
  if (!filePath.startsWith(outputDirPath)) {
    throw new Error('Invalid output file path');
  }

  return filePath;
};

const parseCsvLine = (line) => {
  const values = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    const next = line[i + 1];

    if (ch === '"') {
      if (inQuotes && next === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (ch === ',' && !inQuotes) {
      values.push(current);
      current = '';
      continue;
    }

    current += ch;
  }

  values.push(current);
  return values;
};

const readPreview = async (filePath, format, limit) => {
  const records = [];

  if (format === 'json') {
    const raw = await fs.promises.readFile(filePath, 'utf-8');
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parsed.slice(0, limit);
    }
    return [];
  }

  if (format === 'jsonl') {
    const stream = fs.createReadStream(filePath, { encoding: 'utf-8' });
    const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

    for await (const line of rl) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      try {
        records.push(JSON.parse(trimmed));
      } catch {
        records.push({ raw: trimmed });
      }
      if (records.length >= limit) {
        rl.close();
        break;
      }
    }

    return records;
  }

  if (format === 'csv') {
    const stream = fs.createReadStream(filePath, { encoding: 'utf-8' });
    const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

    let headers = null;
    for await (const line of rl) {
      if (!line.trim()) {
        continue;
      }
      if (!headers) {
        headers = parseCsvLine(line);
        continue;
      }
      const row = parseCsvLine(line);
      const item = {};
      headers.forEach((header, index) => {
        item[header] = row[index] ?? '';
      });
      records.push(item);
      if (records.length >= limit) {
        rl.close();
        break;
      }
    }

    return records;
  }

  return records;
};

const listJobsQuery = (status, limit, offset) => {
  if (status) {
    return db
      .prepare(
        'SELECT * FROM jobs WHERE status = ? ORDER BY datetime(created_at) DESC LIMIT ? OFFSET ?'
      )
      .all(status, limit, offset);
  }

  return db
    .prepare('SELECT * FROM jobs ORDER BY datetime(created_at) DESC LIMIT ? OFFSET ?')
    .all(limit, offset);
};

const countJobs = (status) => {
  if (status) {
    return db.prepare('SELECT COUNT(*) AS count FROM jobs WHERE status = ?').get(status).count;
  }
  return db.prepare('SELECT COUNT(*) AS count FROM jobs').get().count;
};

const cleanupOldJobs = () => {
  const threshold = new Date(Date.now() - config.jobRetentionDays * 24 * 60 * 60 * 1000).toISOString();
  const rows = db
    .prepare(
      `SELECT id, output_dir
       FROM jobs
       WHERE status IN ('completed', 'failed', 'stopped')
         AND COALESCE(completed_at, updated_at) < ?`
    )
    .all(threshold);

  if (rows.length === 0) {
    return 0;
  }

  const removeTxn = db.transaction((jobsToDelete) => {
    const deleteEventsStmt = db.prepare('DELETE FROM job_events WHERE job_id = ?');
    const deleteJobStmt = db.prepare('DELETE FROM jobs WHERE id = ?');

    for (const row of jobsToDelete) {
      deleteEventsStmt.run(row.id);
      deleteJobStmt.run(row.id);
    }
  });

  for (const row of rows) {
    try {
      deleteArtifactsDirectory(config.outputsDir, row.id, row.output_dir);
    } catch (error) {
      logger.warn({ err: error, jobId: row.id }, 'Artifact cleanup failed');
    }
  }

  removeTxn(rows);
  return rows.length;
};

// ---------------------------------------------------------------------------
// Provider helpers (for /api/providers endpoints)
// ---------------------------------------------------------------------------

const providerConfigs = {
  auto: { name: 'Auto (Smart Routing)', description: 'Automatically selects the best provider based on task complexity and availability', models: ['auto'], requiresKey: false },
  mock: { name: 'Mock', description: 'Deterministic test provider (no API key needed)', models: ['mock'], requiresKey: false },
  openai: { name: 'OpenAI', description: 'GPT-4o, GPT-4o-mini, GPT-3.5-turbo', models: ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'], requiresKey: true, envVar: 'OPENAI_API_KEY' },
  huggingface: { name: 'HuggingFace', description: 'Local models via Transformers (GPU recommended)', models: ['mistralai/Mistral-7B-Instruct-v0.2', 'meta-llama/Llama-2-7b-chat-hf'], requiresKey: false },
  anthropic: { name: 'Anthropic', description: 'Claude 4 family', models: ['claude-sonnet-4-20250514', 'claude-haiku-4-5-20251001'], requiresKey: true, envVar: 'ANTHROPIC_API_KEY' },
  google: { name: 'Google', description: 'Gemini 2.5 family', models: ['gemini-2.5-pro', 'gemini-2.5-flash'], requiresKey: true, envVar: 'GOOGLE_API_KEY' },
  ollama: { name: 'Ollama', description: 'Local models via Ollama server', models: ['llama3', 'mistral', 'codellama'], requiresKey: false, envVar: 'OLLAMA_HOST' },
  azure_openai: { name: 'Azure OpenAI', description: 'Enterprise OpenAI via Azure', models: ['gpt-4o', 'gpt-4o-mini'], requiresKey: true, envVar: 'AZURE_OPENAI_API_KEY' },
  groq: { name: 'Groq', description: 'Ultra-fast inference', models: ['llama-3.1-70b-versatile', 'mixtral-8x7b-32768'], requiresKey: true, envVar: 'GROQ_API_KEY' },
  together: { name: 'Together.ai', description: 'Open model hosting', models: ['meta-llama/Llama-3-70b-chat-hf', 'mistralai/Mixtral-8x7B-Instruct-v0.1'], requiresKey: true, envVar: 'TOGETHER_API_KEY' },
  custom: { name: 'Custom Endpoint', description: 'Any OpenAI-compatible API (vLLM, TGI, llama.cpp)', models: ['custom'], requiresKey: false, envVar: 'CUSTOM_API_BASE' },
  aws_bedrock: { name: 'AWS Bedrock', description: 'Multi-model access via AWS Bedrock (Claude, Llama, Mistral)', models: ['anthropic.claude-3-5-sonnet-20241022-v2:0', 'meta.llama3-1-70b-instruct-v1:0'], requiresKey: true, envVar: 'AWS_REGION' },
  replicate: { name: 'Replicate', description: 'Open-source model hosting in the cloud', models: ['meta/llama-2-70b-chat', 'mistralai/mixtral-8x7b-instruct-v0.1'], requiresKey: true, envVar: 'REPLICATE_API_TOKEN' },
};

// ---------------------------------------------------------------------------
// Express Application
// ---------------------------------------------------------------------------

const createApp = () => {
  const app = express();

  // Middleware
  app.use(
    pinoHttp({
      logger,
      genReqId: (req, res) => {
        const existing = req.headers['x-request-id'];
        if (existing && typeof existing === 'string') {
          return existing;
        }
        const generated = crypto.randomUUID();
        res.setHeader('x-request-id', generated);
        return generated;
      },
    })
  );

  app.use(cors());
  app.use(express.json({ limit: config.maxBodySize }));
  app.use(authMiddleware);
  app.use('/api', generalLimiter);

  // -------------------------------------------------------------------------
  // SaaS Platform: auth, usage tracking, and extended routes
  // -------------------------------------------------------------------------

  if (config.enableSaaS) {
    const saasAuth = createAuthMiddleware({
      jwtSecret: config.jwtSecret,
      apiKeyService,
      legacyApiKeys: config.apiKeys,
      authMode: config.authMode,
    });
    app.use(saasAuth);

    if (config.enableUsageTracking) {
      app.use('/api', createUsageTracker(usageService));
    }

    app.use('/api/auth', createAuthRoutes({
      userService,
      subscriptionService,
      jwtSecret: config.jwtSecret,
      jwtExpiresIn: config.jwtExpiresIn,
    }));

    app.use('/api/admin', createAdminRoutes({
      userService,
      subscriptionService,
      analyticsService,
      featureFlagService,
      pluginService,
      usageService,
    }));

    app.use('/api/billing', createBillingRoutes({
      subscriptionService,
      usageService,
      analyticsService,
    }));

    app.use('/api/api-keys', createApiKeyRoutes({
      apiKeyService,
      subscriptionService,
      analyticsService,
    }));

    app.use('/api/plugins', createPluginRoutes({
      pluginService,
      featureFlagService,
    }));

    app.use('/api/analytics', createAnalyticsRoutes({
      analyticsService,
      usageService,
    }));

    // Subscription tiers (public)
    app.get('/api/tiers', (_req, res) => {
      res.json({ tiers: subscriptionService.getAllTiers() });
    });
  }

  // -------------------------------------------------------------------------
  // Health & Info
  // -------------------------------------------------------------------------

  app.get('/api/health', (_req, res) => {
    res.json({
      status: 'ok',
      timestamp: nowIso(),
      sqlitePath: config.sqlitePath,
      outputsDir: config.outputsDir,
      version: '2.0.0',
      saas: config.enableSaaS,
      cacheStats: cache.getStats(),
    });
  });

  // -------------------------------------------------------------------------
  // Templates
  // -------------------------------------------------------------------------

  app.get('/api/templates', (_req, res) => {
    res.json({ templates });
  });

  app.get('/api/templates/:id', (req, res) => {
    const template = templates.find((item) => item.id === req.params.id);
    if (!template) {
      res.status(404).json({ error: 'Template not found' });
      return;
    }
    res.json(template);
  });

  // -------------------------------------------------------------------------
  // Providers
  // -------------------------------------------------------------------------

  app.get('/api/providers', (_req, res) => {
    const providers = Object.entries(providerConfigs).map(([key, cfg]) => {
      let configured = !cfg.requiresKey;
      if (cfg.requiresKey && cfg.envVar) {
        configured = Boolean(process.env[cfg.envVar]);
      }
      return {
        id: key,
        name: cfg.name,
        description: cfg.description,
        models: cfg.models,
        requiresKey: cfg.requiresKey,
        configured,
      };
    });
    res.json({ providers });
  });

  app.get('/api/providers/:name', (req, res) => {
    const name = req.params.name.toLowerCase();
    const cfg = providerConfigs[name];
    if (!cfg) {
      res.status(404).json({ error: 'Provider not found' });
      return;
    }

    let configured = !cfg.requiresKey;
    if (cfg.requiresKey && cfg.envVar) {
      configured = Boolean(process.env[cfg.envVar]);
    }

    res.json({
      id: name,
      name: cfg.name,
      description: cfg.description,
      models: cfg.models,
      requiresKey: cfg.requiresKey,
      configured,
    });
  });

  app.get('/api/providers/:name/health', (req, res) => {
    const name = req.params.name.toLowerCase();
    const cfg = providerConfigs[name];
    if (!cfg) {
      res.status(404).json({ error: 'Provider not found' });
      return;
    }

    let status = 'unconfigured';
    let configured = false;

    if (!cfg.requiresKey) {
      status = 'available';
      configured = true;
    } else if (cfg.envVar && process.env[cfg.envVar]) {
      configured = true;
      status = 'configured'; // Actual health check would require calling the provider API
    }

    res.json({
      provider: name,
      status,
      configured,
      timestamp: nowIso(),
    });
  });

  // -------------------------------------------------------------------------
  // Generate
  // -------------------------------------------------------------------------

  app.post('/api/generate', generateLimiter, (req, res) => {
    const body = req.body || {};
    const now = nowIso();

    const domain = typeof body.domain === 'string' ? body.domain.trim().toLowerCase() : '';
    const outputFormat = typeof body.outputFormat === 'string' ? body.outputFormat.trim().toLowerCase() : 'jsonl';
    const provider = typeof body.provider === 'string' ? body.provider.trim().toLowerCase() : config.defaultProvider;
    const parseMode = typeof body.parseMode === 'string' ? body.parseMode.trim().toLowerCase() : 'qa';
    const language = typeof body.language === 'string' ? body.language.trim().toLowerCase() : 'en';
    const targetCountInput = Number.parseInt(body.targetCount ?? '1000', 10);
    const batchSizeInput = Number.parseInt(body.batchSize ?? '25', 10);
    const targetCount = Number.isNaN(targetCountInput) ? 1000 : targetCountInput;
    const batchSize = Number.isNaN(batchSizeInput) ? 25 : batchSizeInput;

    if (!domain || !validDomains.includes(domain)) {
      res.status(400).json({ error: `Invalid domain. Must be one of: ${validDomains.join(', ')}` });
      return;
    }

    if (targetCount < config.targetCountMin || targetCount > config.targetCountMax) {
      res.status(400).json({
        error: `Target count must be between ${config.targetCountMin} and ${config.targetCountMax}`,
      });
      return;
    }

    if (batchSize < config.batchSizeMin || batchSize > config.batchSizeMax) {
      res.status(400).json({
        error: `Batch size must be between ${config.batchSizeMin} and ${config.batchSizeMax}`,
      });
      return;
    }

    if (!validOutputFormats.includes(outputFormat)) {
      res.status(400).json({ error: `Invalid output format. Must be one of: ${validOutputFormats.join(', ')}` });
      return;
    }

    if (!validProviders.includes(provider)) {
      res.status(400).json({ error: `Invalid provider. Must be one of: ${validProviders.join(', ')}` });
      return;
    }

    if (!validParseModes.includes(parseMode)) {
      res.status(400).json({ error: `Invalid parseMode. Must be one of: ${validParseModes.join(', ')}` });
      return;
    }

    if (!validLanguages.includes(language)) {
      res.status(400).json({ error: `Invalid language. Must be one of: ${validLanguages.join(', ')}` });
      return;
    }

    // Resolve 'auto' provider to a real one based on configured providers
    let resolvedProvider = provider;
    let autoReason = '';
    if (provider === 'auto') {
      const configuredProviders = Object.entries(providerConfigs)
        .filter(([key, cfg]) => key !== 'auto' && key !== 'mock' && (cfg.requiresKey ? Boolean(process.env[cfg.envVar]) : true))
        .map(([key]) => key);

      if (configuredProviders.length === 0) {
        resolvedProvider = 'mock';
        autoReason = 'No API providers configured, using mock';
      } else {
        // Simple heuristic: prefer fast/cheap for small jobs, quality for large
        const pref = targetCount > 10000 ? 'speed' : targetCount < 500 ? 'quality' : 'balanced';
        const tierOrder = {
          speed: ['groq', 'openai', 'anthropic', 'google', 'together', 'ollama', 'huggingface', 'azure_openai', 'aws_bedrock', 'replicate', 'custom'],
          quality: ['anthropic', 'openai', 'google', 'aws_bedrock', 'azure_openai', 'groq', 'together', 'ollama', 'huggingface', 'replicate', 'custom'],
          balanced: ['openai', 'anthropic', 'google', 'groq', 'azure_openai', 'together', 'ollama', 'huggingface', 'aws_bedrock', 'replicate', 'custom'],
        };
        const order = tierOrder[pref] || tierOrder.balanced;
        resolvedProvider = order.find((p) => configuredProviders.includes(p)) || configuredProviders[0];
        autoReason = `Auto-selected ${resolvedProvider} (preference=${pref}, ${configuredProviders.length} providers available)`;
      }
    }

    const domainId = typeof body.domainId === 'string' ? body.domainId.trim() : '';
    const explicitPrompt = typeof body.prompt === 'string' ? body.prompt.trim() : '';

    let domainRow = null;
    if (domainId) {
      domainRow = getDomainRow.get(domainId);
      if (!domainRow) {
        res.status(404).json({ error: 'Domain not found' });
        return;
      }
    }

    const normalizedTopics = parseArrayOfStrings(body.topics);
    const normalizedExtraFields = parseArrayOfStrings(body.extraFields);

    const prompt = explicitPrompt || (domainRow ? createDomainPrompt(domainRow.config_json, domain) : '');
    const domainDescription = typeof body.domainDescription === 'string' ? body.domainDescription.trim() : '';

    const normalizedConfig = {
      domain,
      targetCount,
      batchSize,
      outputFormat,
      provider: resolvedProvider,
      requestedProvider: provider !== resolvedProvider ? provider : undefined,
      autoReason: autoReason || undefined,
      parseMode,
      language,
      prompt,
      domainId: domainId || null,
      domainDescription,
      topics: normalizedTopics,
      extraFields: normalizedExtraFields,
      modelName: typeof body.modelName === 'string' ? body.modelName.trim() : undefined,
    };

    const jobId = idJob();
    const outputDir = `outputs/${jobId}`;

    db.prepare(
      `INSERT INTO jobs (
        id, status, domain, config_json, prompt, provider, parse_mode, output_format,
        target_count, batch_size, language, output_dir, created_at, updated_at
      ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).run(
      jobId,
      domain,
      JSON.stringify(normalizedConfig),
      prompt || null,
      resolvedProvider,
      parseMode,
      outputFormat,
      targetCount,
      batchSize,
      language,
      outputDir,
      now,
      now
    );

    insertJobEvent(db, jobId, 'status', {
      status: 'queued',
      message: 'Job queued',
    });

    res.json({
      jobId,
      status: 'queued',
      createdAt: now,
      config: normalizedConfig,
    });
  });

  // -------------------------------------------------------------------------
  // Jobs
  // -------------------------------------------------------------------------

  app.get('/api/jobs', (req, res) => {
    const page = Math.max(1, Number.parseInt(req.query.page ?? '1', 10) || 1);
    const limit = clamp(Number.parseInt(req.query.limit ?? '20', 10) || 20, 1, 100);
    const status = typeof req.query.status === 'string' ? req.query.status.trim().toLowerCase() : '';

    if (status && !['queued', 'running', ...terminalStatuses].includes(status)) {
      res.status(400).json({ error: 'Invalid status filter' });
      return;
    }

    const offset = (page - 1) * limit;
    const rows = listJobsQuery(status || null, limit, offset);
    const total = countJobs(status || null);

    res.json({
      jobs: rows.map(toApiJob),
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit),
    });
  });

  app.get('/api/jobs/:jobId', (req, res) => {
    const row = getJobRow.get(req.params.jobId);
    if (!row) {
      res.status(404).json({ error: 'Job not found' });
      return;
    }

    res.json(toApiJob(row));
  });

  app.get('/api/jobs/:jobId/events', (req, res) => {
    const jobId = req.params.jobId;
    const row = getJobRow.get(jobId);

    if (!row) {
      res.status(404).json({ error: 'Job not found' });
      return;
    }

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders?.();

    let cursor = Number.parseInt(req.query.sinceId ?? req.get('last-event-id') ?? '0', 10) || 0;

    const sendPending = () => {
      const events = db
        .prepare('SELECT id, job_id, ts, type, payload_json FROM job_events WHERE job_id = ? AND id > ? ORDER BY id ASC')
        .all(jobId, cursor);

      for (const event of events) {
        cursor = event.id;
        const payload = safeJsonParse(event.payload_json, {});
        res.write(`id: ${event.id}\n`);
        res.write(`data: ${JSON.stringify({
          eventId: event.id,
          jobId: event.job_id,
          type: event.type,
          ts: event.ts,
          payload,
        })}\n\n`);
      }
    };

    sendPending();

    const pollHandle = setInterval(sendPending, config.ssePollMs);
    const heartbeatHandle = setInterval(() => {
      res.write(': heartbeat\n\n');
    }, config.sseHeartbeatMs);

    req.on('close', () => {
      clearInterval(pollHandle);
      clearInterval(heartbeatHandle);
    });
  });

  app.post('/api/jobs/:jobId/stop', (req, res) => {
    const row = getJobRow.get(req.params.jobId);
    if (!row) {
      res.status(404).json({ error: 'Job not found' });
      return;
    }

    const now = nowIso();

    if (row.status === 'queued') {
      db.prepare(
        `UPDATE jobs
         SET status = 'stopped', stop_requested = 1, completed_at = ?, updated_at = ?
         WHERE id = ?`
      ).run(now, now, row.id);

      insertJobEvent(db, row.id, 'status', {
        status: 'stopped',
        message: 'Queued job stopped before execution',
      });
    } else if (row.status === 'running') {
      db.prepare('UPDATE jobs SET stop_requested = 1, updated_at = ? WHERE id = ?').run(now, row.id);
      insertJobEvent(db, row.id, 'status', {
        status: 'running',
        message: 'Stop requested',
        stopRequested: true,
      });
    } else {
      res.status(400).json({ error: 'Job is not stoppable in current state' });
      return;
    }

    const updated = getJobRow.get(row.id);
    res.json({
      message: 'Stop requested',
      job: toApiJob(updated),
    });
  });

  app.post('/api/jobs/:jobId/retry', (req, res) => {
    const row = getJobRow.get(req.params.jobId);
    if (!row) {
      res.status(404).json({ error: 'Job not found' });
      return;
    }

    if (!['failed', 'stopped'].includes(row.status)) {
      res.status(400).json({ error: 'Only failed or stopped jobs can be retried' });
      return;
    }

    const now = nowIso();

    deleteArtifactsDirectory(config.outputsDir, row.id, row.output_dir);

    db.prepare(
      `UPDATE jobs SET
        status = 'queued',
        generated_count = 0,
        duplicates_count = 0,
        invalid_count = 0,
        rate_items_per_sec = 0,
        eta_seconds = NULL,
        stop_requested = 0,
        error_message = NULL,
        output_file = NULL,
        checkpoint_file = NULL,
        started_at = NULL,
        completed_at = NULL,
        updated_at = ?
      WHERE id = ?`
    ).run(now, row.id);

    insertJobEvent(db, row.id, 'status', {
      status: 'queued',
      message: 'Job re-queued',
    });

    res.json({
      message: 'Job queued for retry',
      job: toApiJob(getJobRow.get(row.id)),
    });
  });

  app.delete('/api/jobs/:jobId', (req, res) => {
    const row = getJobRow.get(req.params.jobId);
    if (!row) {
      res.status(404).json({ error: 'Job not found' });
      return;
    }

    if (['queued', 'running'].includes(row.status)) {
      res.status(400).json({ error: 'Stop the job before deleting it' });
      return;
    }

    deleteArtifactsDirectory(config.outputsDir, row.id, row.output_dir);

    const deleteTxn = db.transaction(() => {
      db.prepare('DELETE FROM job_events WHERE job_id = ?').run(row.id);
      db.prepare('DELETE FROM jobs WHERE id = ?').run(row.id);
    });

    deleteTxn();

    res.json({ message: 'Job deleted successfully' });
  });

  // -------------------------------------------------------------------------
  // Downloads & Preview
  // -------------------------------------------------------------------------

  app.get('/api/downloads/:jobId/:format', downloadLimiter, (req, res) => {
    const jobId = req.params.jobId;
    const format = String(req.params.format || '').toLowerCase();

    if (!validOutputFormats.includes(format)) {
      res.status(400).json({ error: `Invalid output format. Must be one of: ${validOutputFormats.join(', ')}` });
      return;
    }

    const row = getJobRow.get(jobId);
    if (!row) {
      res.status(404).json({ error: 'Job not found' });
      return;
    }

    if (!['completed', 'stopped'].includes(row.status)) {
      res.status(400).json({ error: 'Job is not ready for download' });
      return;
    }

    let outputPath;
    try {
      outputPath = resolveOutputPath(row, format);
    } catch {
      res.status(400).json({ error: 'Invalid output path' });
      return;
    }

    if (!fs.existsSync(outputPath)) {
      res.status(404).json({ error: 'Output artifact not found' });
      return;
    }

    const contentTypes = {
      jsonl: 'application/x-ndjson',
      csv: 'text/csv',
      json: 'application/json',
    };

    res.setHeader('Content-Type', contentTypes[format] || 'application/octet-stream');
    res.setHeader('Content-Disposition', `attachment; filename=\"${jobId}.${format}\"`);

    const stream = fs.createReadStream(outputPath);
    stream.on('error', (error) => {
      req.log.error({ err: error, jobId }, 'Failed to stream output file');
      if (!res.headersSent) {
        res.status(500).json({ error: 'Failed to stream output file' });
      } else {
        res.end();
      }
    });
    stream.pipe(res);
  });

  app.get('/api/jobs/:jobId/preview', asyncHandler(async (req, res) => {
    const row = getJobRow.get(req.params.jobId);
    if (!row) {
      res.status(404).json({ error: 'Job not found' });
      return;
    }

    const limit = clamp(Number.parseInt(req.query.limit ?? '20', 10) || 20, 1, 200);
    const format = row.output_format;

    let outputPath;
    try {
      outputPath = resolveOutputPath(row, format);
    } catch {
      res.status(400).json({ error: 'Invalid output path' });
      return;
    }

    if (!fs.existsSync(outputPath)) {
      res.status(404).json({ error: 'Output artifact not found' });
      return;
    }

    const records = await readPreview(outputPath, format, limit);
    res.json({
      jobId: row.id,
      format,
      limit,
      records,
    });
  }));

  // -------------------------------------------------------------------------
  // Domains CRUD
  // -------------------------------------------------------------------------

  app.post('/api/domains', (req, res) => {
    const body = req.body || {};
    const now = nowIso();

    if (typeof body.name !== 'string' || body.name.trim().length < 1) {
      res.status(400).json({ error: 'Domain name is required' });
      return;
    }

    const name = body.name.trim();
    if (name.length > 200) {
      res.status(400).json({ error: 'Domain name must be 200 characters or fewer' });
      return;
    }

    const topics = Array.isArray(body.topics) ? body.topics : [];
    if (topics.length === 0) {
      res.status(400).json({ error: 'At least one topic is required' });
      return;
    }

    const id = idDomain();
    const configJson = JSON.stringify(body);

    db.prepare(
      'INSERT INTO domains (id, name, config_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?)'
    ).run(id, name, configJson, now, now);

    res.json({ id, message: 'Domain configuration saved successfully' });
  });

  app.get('/api/domains', (_req, res) => {
    const rows = db.prepare('SELECT * FROM domains ORDER BY datetime(created_at) DESC').all();
    res.json({ domains: rows.map(toApiDomain) });
  });

  app.get('/api/domains/:id', (req, res) => {
    const row = getDomainRow.get(req.params.id);
    if (!row) {
      res.status(404).json({ error: 'Domain not found' });
      return;
    }
    res.json(toApiDomain(row));
  });

  app.put('/api/domains/:id', (req, res) => {
    const row = getDomainRow.get(req.params.id);
    if (!row) {
      res.status(404).json({ error: 'Domain not found' });
      return;
    }

    const body = req.body || {};
    const now = nowIso();

    if (typeof body.name === 'string') {
      const name = body.name.trim();
      if (name.length < 1) {
        res.status(400).json({ error: 'Domain name cannot be empty' });
        return;
      }
      if (name.length > 200) {
        res.status(400).json({ error: 'Domain name must be 200 characters or fewer' });
        return;
      }
    }

    const updatedConfig = { ...safeJsonParse(row.config_json, {}), ...body };
    const name = typeof body.name === 'string' ? body.name.trim() : row.name;

    db.prepare('UPDATE domains SET name = ?, config_json = ?, updated_at = ? WHERE id = ?').run(
      name,
      JSON.stringify(updatedConfig),
      now,
      req.params.id
    );

    const updated = getDomainRow.get(req.params.id);
    res.json(toApiDomain(updated));
  });

  app.delete('/api/domains/:id', (req, res) => {
    const row = getDomainRow.get(req.params.id);
    if (!row) {
      res.status(404).json({ error: 'Domain not found' });
      return;
    }

    db.prepare('DELETE FROM domains WHERE id = ?').run(req.params.id);
    res.json({ message: 'Domain deleted successfully' });
  });

  // -------------------------------------------------------------------------
  // Metrics
  // -------------------------------------------------------------------------

  app.get('/api/metrics', (_req, res) => {
    const statusCounts = db
      .prepare('SELECT status, COUNT(*) AS count FROM jobs GROUP BY status')
      .all()
      .reduce(
        (acc, row) => {
          acc[row.status] = row.count;
          return acc;
        },
        {
          queued: 0,
          running: 0,
          completed: 0,
          failed: 0,
          stopped: 0,
        }
      );

    const queueWaitRows = db
      .prepare('SELECT created_at, started_at FROM jobs WHERE started_at IS NOT NULL')
      .all();

    const avgQueueWaitSeconds =
      queueWaitRows.length > 0
        ? queueWaitRows
            .map((row) => (toIsoMs(row.started_at) - toIsoMs(row.created_at)) / 1000)
            .reduce((a, b) => a + b, 0) / queueWaitRows.length
        : 0;

    const throughputRows = db
      .prepare(
        `SELECT rate_items_per_sec
         FROM jobs
         WHERE status = 'completed' AND rate_items_per_sec > 0
         ORDER BY datetime(completed_at) DESC
         LIMIT 50`
      )
      .all();

    const avgItemsPerSec =
      throughputRows.length > 0
        ? throughputRows.reduce((acc, row) => acc + Number(row.rate_items_per_sec || 0), 0) / throughputRows.length
        : 0;

    res.json({
      jobs: statusCounts,
      averageQueueWaitSeconds: Number(avgQueueWaitSeconds.toFixed(2)),
      throughput: {
        sampleSize: throughputRows.length,
        averageItemsPerSec: Number(avgItemsPerSec.toFixed(4)),
      },
      timestamp: nowIso(),
    });
  });

  // -------------------------------------------------------------------------
  // Error Handler
  // -------------------------------------------------------------------------

  app.use(errorHandler);

  return app;
};

// ---------------------------------------------------------------------------
// Server Lifecycle
// ---------------------------------------------------------------------------

const start = () => {
  return new Promise((resolve, reject) => {
    try {
      const app = createApp();

      const cleanedCount = cleanupOldJobs();
      if (cleanedCount > 0) {
        logger.info({ cleanedCount }, 'Old jobs cleaned on startup');
      }

      const cleanupHandle = setInterval(() => {
        try {
          const removed = cleanupOldJobs();
          if (removed > 0) {
            logger.info({ removed }, 'Retention cleanup removed jobs');
          }
        } catch (error) {
          logger.error({ err: error }, 'Retention cleanup failed');
        }
      }, config.cleanupIntervalMs);

      const server = app.listen(config.port, () => {
        logger.info({ port: config.port }, 'API server listening');
        resolve({ app, server, db });
      });

      server.on('error', (error) => {
        reject(error);
      });

      const shutdown = (signalName) => {
        logger.info({ signal: signalName }, 'Shutting down API server');
        clearInterval(cleanupHandle);

        server.close(() => {
          try {
            db.close();
          } catch {
            // ignore
          }
          process.exit(0);
        });
      };

      process.on('SIGINT', () => shutdown('SIGINT'));
      process.on('SIGTERM', () => shutdown('SIGTERM'));
    } catch (error) {
      reject(error);
    }
  });
};

module.exports = {
  createApp,
  start,
  db,
  logger,
  userService,
  subscriptionService,
  usageService,
  apiKeyService,
  featureFlagService,
  pluginService,
  analyticsService,
  cache,
};
