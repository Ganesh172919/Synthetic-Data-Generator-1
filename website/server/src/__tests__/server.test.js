const { describe, it, beforeEach, afterEach } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const os = require('os');

describe('server API routes', () => {
  let tmpDir;
  let app;
  let db;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'synthgen-api-test-'));

    // Override config before loading server
    process.env.DATA_DIR = tmpDir;
    process.env.SQLITE_PATH = path.join(tmpDir, 'test.sqlite');
    process.env.OUTPUTS_DIR = path.join(tmpDir, 'outputs');
    process.env.AUTH_MODE = 'none';
    process.env.LOG_LEVEL = 'silent';

    // Clear module cache so config reloads with new env
    const configPath = require.resolve('../config');
    delete require.cache[configPath];
    const dbPath = require.resolve('../db');
    delete require.cache[dbPath];
    const serverPath = require.resolve('../server');
    delete require.cache[serverPath];
    const templatesPath = require.resolve('../templates');
    delete require.cache[templatesPath];
    const migrationsPath = require.resolve('../migrations');
    delete require.cache[migrationsPath];

    const serverModule = require('../server');
    app = serverModule.createApp();
    db = serverModule.db;
  });

  afterEach(() => {
    if (db) {
      try { db.close(); } catch {}
    }
    fs.rmSync(tmpDir, { recursive: true, force: true });
    delete process.env.DATA_DIR;
    delete process.env.SQLITE_PATH;
    delete process.env.OUTPUTS_DIR;
    delete process.env.AUTH_MODE;
    delete process.env.LOG_LEVEL;
  });

  // Helper: make HTTP request to the Express app without starting a server
  async function request(method, urlPath, body = null) {
    const http = require('http');
    return new Promise((resolve, reject) => {
      const server = http.createServer(app);
      server.listen(0, async () => {
        const { port } = server.address();
        try {
          const options = {
            hostname: '127.0.0.1',
            port,
            path: urlPath,
            method,
            headers: { 'Content-Type': 'application/json' },
          };

          const res = await new Promise((resResolve, resReject) => {
            const req = http.request(options, (res) => {
              let data = '';
              res.on('data', (chunk) => (data += chunk));
              res.on('end', () => {
                resResolve({
                  status: res.statusCode,
                  headers: res.headers,
                  body: data,
                  json: () => JSON.parse(data),
                });
              });
            });
            req.on('error', resReject);
            if (body) {
              req.write(JSON.stringify(body));
            }
            req.end();
          });

          resolve(res);
        } catch (err) {
          reject(err);
        } finally {
          server.close();
        }
      });
    });
  }

  // -----------------------------------------------------------------------
  // Health
  // -----------------------------------------------------------------------

  describe('GET /api/health', () => {
    it('should return 200 with status ok', async () => {
      const res = await request('GET', '/api/health');
      assert.equal(res.status, 200);
      const data = res.json();
      assert.equal(data.status, 'ok');
      assert.ok(data.timestamp);
      assert.equal(data.version, '2.0.0');
    });
  });

  // -----------------------------------------------------------------------
  // Templates
  // -----------------------------------------------------------------------

  describe('GET /api/templates', () => {
    it('should return array of templates', async () => {
      const res = await request('GET', '/api/templates');
      assert.equal(res.status, 200);
      const data = res.json();
      assert.ok(Array.isArray(data.templates));
      assert.ok(data.templates.length >= 6);
    });
  });

  describe('GET /api/templates/:id', () => {
    it('should return a specific template', async () => {
      const res = await request('GET', '/api/templates/fin-education');
      assert.equal(res.status, 200);
      const data = res.json();
      assert.equal(data.id, 'fin-education');
      assert.equal(data.category, 'financial');
    });

    it('should return 404 for unknown template', async () => {
      const res = await request('GET', '/api/templates/nonexistent');
      assert.equal(res.status, 404);
    });
  });

  // -----------------------------------------------------------------------
  // Providers
  // -----------------------------------------------------------------------

  describe('GET /api/providers', () => {
    it('should return list of providers', async () => {
      const res = await request('GET', '/api/providers');
      assert.equal(res.status, 200);
      const data = res.json();
      assert.ok(Array.isArray(data.providers));
      assert.ok(data.providers.length >= 5);

      const mockProvider = data.providers.find((p) => p.id === 'mock');
      assert.ok(mockProvider);
      assert.equal(mockProvider.configured, true);
    });
  });

  describe('GET /api/providers/:name', () => {
    it('should return provider details', async () => {
      const res = await request('GET', '/api/providers/mock');
      assert.equal(res.status, 200);
      const data = res.json();
      assert.equal(data.id, 'mock');
      assert.equal(data.configured, true);
    });

    it('should return 404 for unknown provider', async () => {
      const res = await request('GET', '/api/providers/nonexistent');
      assert.equal(res.status, 404);
    });
  });

  describe('GET /api/providers/:name/health', () => {
    it('should return health status for mock provider', async () => {
      const res = await request('GET', '/api/providers/mock/health');
      assert.equal(res.status, 200);
      const data = res.json();
      assert.equal(data.provider, 'mock');
      assert.equal(data.status, 'available');
    });
  });

  // -----------------------------------------------------------------------
  // Generate
  // -----------------------------------------------------------------------

  describe('POST /api/generate', () => {
    it('should create a job with valid config', async () => {
      const res = await request('POST', '/api/generate', {
        domain: 'financial',
        targetCount: 100,
        batchSize: 10,
        outputFormat: 'jsonl',
        provider: 'mock',
        parseMode: 'qa',
      });
      assert.equal(res.status, 200);
      const data = res.json();
      assert.ok(data.jobId);
      assert.equal(data.status, 'queued');
      assert.ok(data.config);
    });

    it('should reject invalid domain', async () => {
      const res = await request('POST', '/api/generate', {
        domain: 'nonexistent',
      });
      assert.equal(res.status, 400);
      const data = res.json();
      assert.ok(data.error.includes('Invalid domain'));
    });

    it('should reject invalid provider', async () => {
      const res = await request('POST', '/api/generate', {
        domain: 'financial',
        provider: 'nonexistent',
      });
      assert.equal(res.status, 400);
      const data = res.json();
      assert.ok(data.error.includes('Invalid provider'));
    });

    it('should reject invalid parse mode', async () => {
      const res = await request('POST', '/api/generate', {
        domain: 'financial',
        parseMode: 'nonexistent',
      });
      assert.equal(res.status, 400);
      const data = res.json();
      assert.ok(data.error.includes('Invalid parseMode'));
    });

    it('should accept language parameter', async () => {
      const res = await request('POST', '/api/generate', {
        domain: 'financial',
        targetCount: 100,
        provider: 'mock',
        language: 'es',
      });
      assert.equal(res.status, 200);
      const data = res.json();
      assert.equal(data.config.language, 'es');
    });

    it('should reject invalid language', async () => {
      const res = await request('POST', '/api/generate', {
        domain: 'financial',
        language: 'xx',
      });
      assert.equal(res.status, 400);
      const data = res.json();
      assert.ok(data.error.includes('Invalid language'));
    });
  });

  // -----------------------------------------------------------------------
  // Jobs
  // -----------------------------------------------------------------------

  describe('GET /api/jobs', () => {
    it('should return empty list initially', async () => {
      const res = await request('GET', '/api/jobs');
      assert.equal(res.status, 200);
      const data = res.json();
      assert.ok(Array.isArray(data.jobs));
      assert.equal(data.total, 0);
    });

    it('should list created jobs', async () => {
      await request('POST', '/api/generate', {
        domain: 'financial',
        targetCount: 100,
        provider: 'mock',
      });

      const res = await request('GET', '/api/jobs');
      const data = res.json();
      assert.equal(data.total, 1);
      assert.equal(data.jobs[0].domain, 'financial');
    });
  });

  describe('GET /api/jobs/:jobId', () => {
    it('should return a specific job', async () => {
      const create = await request('POST', '/api/generate', {
        domain: 'financial',
        targetCount: 100,
        provider: 'mock',
      });
      const { jobId } = create.json();

      const res = await request('GET', `/api/jobs/${jobId}`);
      assert.equal(res.status, 200);
      const data = res.json();
      assert.equal(data.id, jobId);
    });

    it('should return 404 for unknown job', async () => {
      const res = await request('GET', '/api/jobs/gen_nonexist');
      assert.equal(res.status, 404);
    });
  });

  describe('POST /api/jobs/:jobId/stop', () => {
    it('should stop a queued job', async () => {
      const create = await request('POST', '/api/generate', {
        domain: 'financial',
        targetCount: 100,
        provider: 'mock',
      });
      const { jobId } = create.json();

      const res = await request('POST', `/api/jobs/${jobId}/stop`);
      assert.equal(res.status, 200);
      const data = res.json();
      assert.equal(data.job.status, 'stopped');
    });
  });

  describe('DELETE /api/jobs/:jobId', () => {
    it('should delete a terminal job', async () => {
      const create = await request('POST', '/api/generate', {
        domain: 'financial',
        targetCount: 100,
        provider: 'mock',
      });
      const { jobId } = create.json();

      // Stop it first
      await request('POST', `/api/jobs/${jobId}/stop`);

      // Now delete
      const res = await request('DELETE', `/api/jobs/${jobId}`);
      assert.equal(res.status, 200);

      // Verify it's gone
      const getRes = await request('GET', `/api/jobs/${jobId}`);
      assert.equal(getRes.status, 404);
    });
  });

  // -----------------------------------------------------------------------
  // Domains CRUD
  // -----------------------------------------------------------------------

  describe('POST /api/domains', () => {
    it('should create a domain', async () => {
      const res = await request('POST', '/api/domains', {
        name: 'Test Domain',
        topics: ['Topic A', 'Topic B'],
        description: 'A test domain',
      });
      assert.equal(res.status, 200);
      const data = res.json();
      assert.ok(data.id);
      assert.equal(data.message, 'Domain configuration saved successfully');
    });

    it('should reject empty name', async () => {
      const res = await request('POST', '/api/domains', {
        name: '',
        topics: ['A'],
      });
      assert.equal(res.status, 400);
    });

    it('should reject missing topics', async () => {
      const res = await request('POST', '/api/domains', {
        name: 'Test',
        topics: [],
      });
      assert.equal(res.status, 400);
    });

    it('should reject name over 200 chars', async () => {
      const res = await request('POST', '/api/domains', {
        name: 'A'.repeat(201),
        topics: ['A'],
      });
      assert.equal(res.status, 400);
    });
  });

  describe('GET /api/domains', () => {
    it('should list domains', async () => {
      await request('POST', '/api/domains', {
        name: 'Domain A',
        topics: ['A'],
      });

      const res = await request('GET', '/api/domains');
      assert.equal(res.status, 200);
      const data = res.json();
      assert.ok(Array.isArray(data.domains));
      assert.ok(data.domains.length >= 1);
    });
  });

  describe('PUT /api/domains/:id', () => {
    it('should update a domain', async () => {
      const create = await request('POST', '/api/domains', {
        name: 'Original',
        topics: ['A'],
      });
      const { id } = create.json();

      const res = await request('PUT', `/api/domains/${id}`, {
        name: 'Updated',
        topics: ['A', 'B'],
      });
      assert.equal(res.status, 200);
      const data = res.json();
      assert.equal(data.name, 'Updated');
    });
  });

  describe('DELETE /api/domains/:id', () => {
    it('should delete a domain', async () => {
      const create = await request('POST', '/api/domains', {
        name: 'To Delete',
        topics: ['A'],
      });
      const { id } = create.json();

      const res = await request('DELETE', `/api/domains/${id}`);
      assert.equal(res.status, 200);

      const getRes = await request('GET', `/api/domains/${id}`);
      assert.equal(getRes.status, 404);
    });
  });

  // -----------------------------------------------------------------------
  // Metrics
  // -----------------------------------------------------------------------

  describe('GET /api/metrics', () => {
    it('should return aggregate metrics', async () => {
      const res = await request('GET', '/api/metrics');
      assert.equal(res.status, 200);
      const data = res.json();
      assert.ok(data.jobs);
      assert.ok('queued' in data.jobs);
      assert.ok('running' in data.jobs);
      assert.ok('completed' in data.jobs);
      assert.ok(typeof data.averageQueueWaitSeconds === 'number');
      assert.ok(data.throughput);
      assert.ok(data.timestamp);
    });
  });

  // -----------------------------------------------------------------------
  // SSE Events
  // -----------------------------------------------------------------------

  describe('GET /api/jobs/:jobId/events', () => {
    // Helper: open an SSE connection and collect events for a limited time
    function sseRequest(urlPath, timeoutMs = 2000) {
      const http = require('http');
      return new Promise((resolve, reject) => {
        const server = http.createServer(app);
        server.listen(0, async () => {
          const { port } = server.address();
          try {
            const req = http.request(
              {
                hostname: '127.0.0.1',
                port,
                path: urlPath,
                method: 'GET',
                headers: { Accept: 'text/event-stream' },
              },
              (res) => {
                let data = '';
                const events = [];

                res.on('data', (chunk) => {
                  data += chunk.toString();
                });

                const timer = setTimeout(() => {
                  req.destroy();
                  server.close();
                  // Parse SSE events from accumulated data
                  const blocks = data.split('\n\n').filter(Boolean);
                  for (const block of blocks) {
                    const lines = block.split('\n');
                    let eventId = null;
                    let eventData = null;
                    for (const line of lines) {
                      if (line.startsWith('id: ')) {
                        eventId = line.slice(4).trim();
                      } else if (line.startsWith('data: ')) {
                        try {
                          eventData = JSON.parse(line.slice(6));
                        } catch {
                          eventData = line.slice(6);
                        }
                      } else if (line.startsWith(':')) {
                        // Comment (heartbeat)
                        events.push({ type: 'comment', data: line.slice(1).trim() });
                      }
                    }
                    if (eventId && eventData) {
                      events.push({ type: 'event', id: eventId, data: eventData });
                    }
                  }
                  resolve({ status: res.statusCode, headers: res.headers, events });
                }, timeoutMs);

                res.on('error', (err) => {
                  clearTimeout(timer);
                  server.close();
                  reject(err);
                });
              }
            );
            req.on('error', (err) => {
              server.close();
              reject(err);
            });
            req.end();
          } catch (err) {
            server.close();
            reject(err);
          }
        });
      });
    }

    it('should return 404 for non-existent job', async () => {
      const res = await request('GET', '/api/jobs/gen_nonexist/events');
      assert.equal(res.status, 404);
    });

    it('should establish SSE connection with correct headers', async () => {
      const create = await request('POST', '/api/generate', {
        domain: 'financial',
        targetCount: 100,
        provider: 'mock',
      });
      const { jobId } = create.json();

      const result = await sseRequest(`/api/jobs/${jobId}/events`, 1500);
      assert.equal(result.status, 200);
      assert.equal(result.headers['content-type'], 'text/event-stream');
      assert.equal(result.headers['cache-control'], 'no-cache');
      assert.equal(result.headers['connection'], 'keep-alive');
    });

    it('should deliver existing events on connect', async () => {
      const create = await request('POST', '/api/generate', {
        domain: 'financial',
        targetCount: 100,
        provider: 'mock',
      });
      const { jobId } = create.json();

      // The generate endpoint inserts a 'status' event with status=queued
      const result = await sseRequest(`/api/jobs/${jobId}/events`, 1500);
      const statusEvents = result.events.filter(
        (e) => e.type === 'event' && e.data && e.data.type === 'status'
      );
      assert.ok(statusEvents.length >= 1, 'Should have at least one status event');
      assert.equal(statusEvents[0].data.payload.status, 'queued');
    });

    it('should send heartbeat comments', async () => {
      const create = await request('POST', '/api/generate', {
        domain: 'financial',
        targetCount: 100,
        provider: 'mock',
      });
      const { jobId } = create.json();

      // Use a longer timeout to ensure heartbeat arrives (default heartbeat is 15s,
      // but we set SSE_HEARTBEAT_MS env var for testing)
      process.env.SSE_HEARTBEAT_MS = '500';
      // Need to reload config to pick up env change
      const configPath = require.resolve('../config');
      delete require.cache[configPath];
      const serverPath2 = require.resolve('../server');
      delete require.cache[serverPath2];
      const serverModule2 = require('../server');
      const app2 = serverModule2.createApp();
      const db2 = serverModule2.db;

      // Create a job in the new db
      const create2 = await new Promise((resolve, reject) => {
        const http = require('http');
        const server = http.createServer(app2);
        server.listen(0, async () => {
          const { port } = server.address();
          try {
            const req = http.request(
              {
                hostname: '127.0.0.1',
                port,
                path: '/api/generate',
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
              },
              (res) => {
                let data = '';
                res.on('data', (chunk) => (data += chunk));
                res.on('end', () => {
                  server.close();
                  resolve(JSON.parse(data));
                });
              }
            );
            req.write(
              JSON.stringify({
                domain: 'financial',
                targetCount: 100,
                provider: 'mock',
              })
            );
            req.end();
          } catch (err) {
            server.close();
            reject(err);
          }
        });
      });

      // SSE request against the new app with fast heartbeat
      const result = await new Promise((resolve, reject) => {
        const http = require('http');
        const server = http.createServer(app2);
        server.listen(0, async () => {
          const { port } = server.address();
          try {
            const req = http.request(
              {
                hostname: '127.0.0.1',
                port,
                path: `/api/jobs/${create2.jobId}/events`,
                method: 'GET',
                headers: { Accept: 'text/event-stream' },
              },
              (res) => {
                let data = '';
                res.on('data', (chunk) => (data += chunk.toString()));
                const timer = setTimeout(() => {
                  req.destroy();
                  server.close();
                  resolve(data);
                }, 2000);
                res.on('error', () => {
                  clearTimeout(timer);
                  server.close();
                });
              }
            );
            req.on('error', reject);
            req.end();
          } catch (err) {
            server.close();
            reject(err);
          }
        });
      });

      // Check for heartbeat comments (lines starting with ':')
      assert.ok(result.includes(': heartbeat'), 'Should contain heartbeat comments');

      // Cleanup
      try { db2.close(); } catch {}
      delete process.env.SSE_HEARTBEAT_MS;
    });
  });
});
