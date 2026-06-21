const { describe, it, beforeEach, afterEach } = require('node:test');
const assert = require('node:assert/strict');

describe('config', () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    // Clear relevant env vars before each test
    for (const key of Object.keys(process.env)) {
      if (
        key.startsWith('PORT') ||
        key.startsWith('AUTH_') ||
        key.startsWith('TARGET_') ||
        key.startsWith('BATCH_') ||
        key.startsWith('RATE_') ||
        key.startsWith('GENERATE_') ||
        key.startsWith('DOWNLOAD_') ||
        key.startsWith('SSE_') ||
        key.startsWith('LOG_') ||
        key.startsWith('MAX_') ||
        key === 'NODE_ENV' ||
        key === 'DATA_DIR' ||
        key === 'SQLITE_PATH' ||
        key === 'OUTPUTS_DIR' ||
        key === 'DEFAULT_PROVIDER' ||
        key === 'JOB_RETENTION_DAYS' ||
        key === 'CLEANUP_INTERVAL_MS' ||
        key === 'MAX_BODY_SIZE'
      ) {
        delete process.env[key];
      }
    }
  });

  afterEach(() => {
    // Restore original env
    for (const key of Object.keys(process.env)) {
      if (!(key in originalEnv)) {
        delete process.env[key];
      }
    }
    Object.assign(process.env, originalEnv);
  });

  // We need to re-require config.js to pick up env changes.
  // Since Node caches modules, we use delete require.cache trick.
  function loadConfig(overrides = {}) {
    // Clear the module cache for config.js and its dependencies
    const configPath = require.resolve('../config');
    delete require.cache[configPath];

    for (const [key, value] of Object.entries(overrides)) {
      process.env[key] = String(value);
    }

    return require('../config').config;
  }

  it('should return default values when no env vars are set', () => {
    const config = loadConfig();
    assert.equal(config.port, 3001);
    assert.equal(config.nodeEnv, 'development');
    assert.equal(config.logLevel, 'info');
    assert.equal(config.authMode, 'none');
    assert.deepEqual(config.apiKeys, []);
    assert.equal(config.targetCountMin, 100);
    assert.equal(config.targetCountMax, 100000);
    assert.equal(config.batchSizeMin, 1);
    assert.equal(config.batchSizeMax, 50);
    assert.equal(config.defaultProvider, 'mock');
    assert.equal(config.maxConcurrentJobs, 1);
    assert.equal(config.jobRetentionDays, 7);
  });

  it('should parse PORT from env and clamp to minimum', () => {
    const config = loadConfig({ PORT: '8080' });
    assert.equal(config.port, 8080);
  });

  it('should clamp PORT below minimum to 1', () => {
    const config = loadConfig({ PORT: '-5' });
    assert.equal(config.port, 1);
  });

  it('should use fallback for invalid PORT', () => {
    const config = loadConfig({ PORT: 'abc' });
    assert.equal(config.port, 3001);
  });

  it('should parse AUTH_MODE and validate it', () => {
    const config1 = loadConfig({ AUTH_MODE: 'api_key' });
    assert.equal(config1.authMode, 'api_key');

    const config2 = loadConfig({ AUTH_MODE: 'INVALID' });
    assert.equal(config2.authMode, 'none');
  });

  it('should parse CSV API_KEYS', () => {
    const config = loadConfig({ API_KEYS: 'key1, key2 ,key3' });
    assert.deepEqual(config.apiKeys, ['key1', 'key2', 'key3']);
  });

  it('should parse LOG_LEVEL from env', () => {
    const config = loadConfig({ LOG_LEVEL: 'debug' });
    assert.equal(config.logLevel, 'debug');
  });

  it('should parse MAX_CONCURRENT_JOBS with clamping', () => {
    const config1 = loadConfig({ MAX_CONCURRENT_JOBS: '4' });
    assert.equal(config1.maxConcurrentJobs, 4);

    const config2 = loadConfig({ MAX_CONCURRENT_JOBS: '0' });
    assert.equal(config2.maxConcurrentJobs, 1);

    const config3 = loadConfig({ MAX_CONCURRENT_JOBS: '20' });
    assert.equal(config3.maxConcurrentJobs, 16);
  });

  it('should parse rate limit settings', () => {
    const config = loadConfig({
      RATE_LIMIT_MAX: '500',
      GENERATE_RATE_LIMIT_MAX: '50',
      DOWNLOAD_RATE_LIMIT_MAX: '200',
    });
    assert.equal(config.generalRateLimitMax, 500);
    assert.equal(config.generateRateLimitMax, 50);
    assert.equal(config.downloadRateLimitMax, 200);
  });

  it('should be frozen (immutable)', () => {
    const config = loadConfig();
    assert.equal(Object.isFrozen(config), true);
  });
});
