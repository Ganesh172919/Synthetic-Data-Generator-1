const path = require('path');

const parseIntEnv = (name, fallback, min = null, max = null) => {
  const raw = process.env[name];
  const value = Number.parseInt(raw ?? '', 10);
  if (Number.isNaN(value)) {
    return fallback;
  }
  if (min !== null && value < min) {
    return min;
  }
  if (max !== null && value > max) {
    return max;
  }
  return value;
};

const parseCsvEnv = (value) => {
  if (!value) {
    return [];
  }
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
};

const defaultDataDir = path.resolve(__dirname, '..', 'data');
const dataDir = process.env.DATA_DIR || defaultDataDir;

const config = {
  port: parseIntEnv('PORT', 3001, 1),
  nodeEnv: process.env.NODE_ENV || 'development',

  dataDir,
  sqlitePath: process.env.SQLITE_PATH || path.join(dataDir, 'synthgen.sqlite'),
  outputsDir: process.env.OUTPUTS_DIR || path.join(dataDir, 'outputs'),

  maxBodySize: process.env.MAX_BODY_SIZE || '50kb',

  authMode: (process.env.AUTH_MODE || 'none').toLowerCase(),
  apiKeys: parseCsvEnv(process.env.API_KEYS),

  targetCountMin: parseIntEnv('TARGET_COUNT_MIN', 100, 1),
  targetCountMax: parseIntEnv('TARGET_COUNT_MAX', 100000, 1),
  batchSizeMin: parseIntEnv('BATCH_SIZE_MIN', 1, 1),
  batchSizeMax: parseIntEnv('BATCH_SIZE_MAX', 50, 1),

  generalRateLimitWindowMs: parseIntEnv('RATE_LIMIT_WINDOW_MS', 15 * 60 * 1000, 1000),
  generalRateLimitMax: parseIntEnv('RATE_LIMIT_MAX', 200, 1),
  generateRateLimitWindowMs: parseIntEnv('GENERATE_RATE_LIMIT_WINDOW_MS', 60 * 60 * 1000, 1000),
  generateRateLimitMax: parseIntEnv('GENERATE_RATE_LIMIT_MAX', 20, 1),
  downloadRateLimitWindowMs: parseIntEnv('DOWNLOAD_RATE_LIMIT_WINDOW_MS', 15 * 60 * 1000, 1000),
  downloadRateLimitMax: parseIntEnv('DOWNLOAD_RATE_LIMIT_MAX', 100, 1),

  jobRetentionDays: parseIntEnv('JOB_RETENTION_DAYS', 7, 1),
  cleanupIntervalMs: parseIntEnv('CLEANUP_INTERVAL_MS', 12 * 60 * 60 * 1000, 60 * 1000),

  ssePollMs: parseIntEnv('SSE_POLL_MS', 1500, 250),
  sseHeartbeatMs: parseIntEnv('SSE_HEARTBEAT_MS', 15000, 1000),

  defaultProvider: (process.env.DEFAULT_PROVIDER || 'mock').toLowerCase(),
};

module.exports = {
  config,
};
