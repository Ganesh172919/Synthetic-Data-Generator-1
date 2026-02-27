'use strict';

const { describe, it, before, after } = require('node:test');
const assert = require('node:assert/strict');

const BASE_URL = 'http://localhost:13099';
let server, db, token, apiKeyRaw;

const json = (r) => r.json();
const post = (path, body, headers = {}) =>
  fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify(body),
  });
const get = (path, headers = {}) =>
  fetch(`${BASE_URL}${path}`, { headers });

describe('SaaS Platform Integration Tests', () => {
  before(async () => {
    const fs = require('fs');
    const dbPath = require('../src/config').config.sqlitePath;
    try { fs.unlinkSync(dbPath); } catch {}
    try { fs.unlinkSync(dbPath + '-wal'); } catch {}
    try { fs.unlinkSync(dbPath + '-shm'); } catch {}

    const { createApp, db: appDb } = require('../src/server');
    db = appDb;
    const app = createApp();
    await new Promise((resolve) => {
      server = app.listen(13099, resolve);
    });
  });

  after(async () => {
    await new Promise((resolve) => server.close(resolve));
    try { db.close(); } catch {}
  });

  // ── Health ──

  it('GET /api/health returns ok with SaaS enabled', async () => {
    const res = await get('/api/health');
    assert.equal(res.status, 200);
    const body = await json(res);
    assert.equal(body.status, 'ok');
    assert.equal(body.saas, true);
    assert.ok(body.cacheStats);
  });

  // ── Auth: Register ──

  it('POST /api/auth/register creates a user', async () => {
    const res = await post('/api/auth/register', {
      email: 'alice@test.com',
      username: 'alice',
      password: 'securepass8',
    });
    assert.equal(res.status, 201);
    const body = await json(res);
    assert.equal(body.user.email, 'alice@test.com');
    assert.equal(body.user.username, 'alice');
    assert.equal(body.user.tier, 'free');
    assert.equal(body.user.role, 'user');
    assert.ok(body.token);
    token = body.token;
  });

  it('POST /api/auth/register rejects duplicate email', async () => {
    const res = await post('/api/auth/register', {
      email: 'alice@test.com',
      username: 'alice2',
      password: 'securepass8',
    });
    assert.equal(res.status, 409);
  });

  it('POST /api/auth/register rejects short password', async () => {
    const res = await post('/api/auth/register', {
      email: 'short@test.com',
      username: 'shortpw',
      password: '12',
    });
    assert.equal(res.status, 400);
  });

  // ── Auth: Login ──

  it('POST /api/auth/login succeeds with correct credentials', async () => {
    const res = await post('/api/auth/login', {
      email: 'alice@test.com',
      password: 'securepass8',
    });
    assert.equal(res.status, 200);
    const body = await json(res);
    assert.ok(body.token);
    assert.equal(body.user.email, 'alice@test.com');
  });

  it('POST /api/auth/login fails with wrong password', async () => {
    const res = await post('/api/auth/login', {
      email: 'alice@test.com',
      password: 'wrongpassword',
    });
    assert.equal(res.status, 401);
  });

  // ── Auth: Profile ──

  it('GET /api/auth/profile returns user data', async () => {
    const res = await get('/api/auth/profile', { Authorization: `Bearer ${token}` });
    assert.equal(res.status, 200);
    const body = await json(res);
    assert.equal(body.user.email, 'alice@test.com');
    assert.ok(body.subscription);
  });

  it('PUT /api/auth/profile updates display name', async () => {
    const res = await fetch(`${BASE_URL}/api/auth/profile`, {
      method: 'PUT',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ displayName: 'Alice Smith' }),
    });
    assert.equal(res.status, 200);
    const body = await json(res);
    assert.equal(body.user.displayName, 'Alice Smith');
  });

  // ── Auth: Password change ──

  it('POST /api/auth/change-password works', async () => {
    const res = await post('/api/auth/change-password', {
      currentPassword: 'securepass8',
      newPassword: 'newsecure9',
    }, { Authorization: `Bearer ${token}` });
    assert.equal(res.status, 200);

    const loginRes = await post('/api/auth/login', {
      email: 'alice@test.com',
      password: 'newsecure9',
    });
    assert.equal(loginRes.status, 200);
    const body = await json(loginRes);
    token = body.token;
  });

  // ── Tiers ──

  it('GET /api/tiers returns tier definitions', async () => {
    const res = await get('/api/tiers');
    assert.equal(res.status, 200);
    const body = await json(res);
    assert.ok(Array.isArray(body.tiers));
    assert.equal(body.tiers.length, 3);
    const ids = body.tiers.map((t) => t.id);
    assert.deepEqual(ids, ['free', 'pro', 'enterprise']);
  });

  // ── Billing ──

  it('GET /api/billing/subscription returns subscription', async () => {
    const res = await get('/api/billing/subscription', { Authorization: `Bearer ${token}` });
    assert.equal(res.status, 200);
    const body = await json(res);
    assert.equal(body.subscription.tier, 'free');
  });

  it('GET /api/billing/usage returns usage summary', async () => {
    const res = await get('/api/billing/usage', { Authorization: `Bearer ${token}` });
    assert.equal(res.status, 200);
    const body = await json(res);
    assert.equal(body.tier, 'free');
    assert.ok(body.limits);
  });

  it('GET /api/billing/quota-check returns quota status', async () => {
    const res = await get('/api/billing/quota-check', { Authorization: `Bearer ${token}` });
    assert.equal(res.status, 200);
    const body = await json(res);
    assert.equal(body.canCreate, true);
  });

  // ── API Keys ──

  it('POST /api/api-keys creates a key', async () => {
    const res = await post('/api/api-keys', { name: 'test-key' }, { Authorization: `Bearer ${token}` });
    assert.equal(res.status, 201);
    const body = await json(res);
    assert.ok(body.key.rawKey);
    assert.ok(body.key.rawKey.startsWith('sg_live_'));
    apiKeyRaw = body.key.rawKey;
  });

  it('GET /api/api-keys lists keys', async () => {
    const res = await get('/api/api-keys', { Authorization: `Bearer ${token}` });
    assert.equal(res.status, 200);
    const body = await json(res);
    assert.ok(Array.isArray(body.keys));
    assert.equal(body.keys.length, 1);
  });

  it('API key can authenticate', async () => {
    const res = await get('/api/auth/profile', { 'x-api-key': apiKeyRaw });
    assert.equal(res.status, 200);
    const body = await json(res);
    assert.equal(body.user.email, 'alice@test.com');
  });

  // ── Plugins ──

  it('GET /api/plugins lists enabled plugins', async () => {
    const res = await get('/api/plugins');
    assert.equal(res.status, 200);
    const body = await json(res);
    assert.ok(Array.isArray(body.plugins));
  });

  // ── Existing endpoints preserved ──

  it('GET /api/templates still works', async () => {
    const res = await get('/api/templates');
    assert.equal(res.status, 200);
    const body = await json(res);
    assert.ok(Array.isArray(body.templates));
    assert.ok(body.templates.length > 0);
  });

  it('GET /api/jobs still works', async () => {
    const res = await get('/api/jobs');
    assert.equal(res.status, 200);
    const body = await json(res);
    assert.ok(typeof body.total === 'number');
  });

  it('GET /api/metrics still works', async () => {
    const res = await get('/api/metrics');
    assert.equal(res.status, 200);
    const body = await json(res);
    assert.ok(body.jobs);
    assert.ok(body.throughput);
  });

  it('POST /api/generate still works', async () => {
    const res = await post('/api/generate', {
      domain: 'financial',
      targetCount: 100,
      batchSize: 10,
      provider: 'mock',
    });
    assert.equal(res.status, 200);
    const body = await json(res);
    assert.ok(body.jobId);
    assert.equal(body.status, 'queued');
  });
});
