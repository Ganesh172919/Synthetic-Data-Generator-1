const { describe, it, beforeEach, afterEach } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const os = require('os');

// We need to test initDatabase which depends on config. We'll create a minimal config.
describe('db', () => {
  let tmpDir;
  let db;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'synthgen-test-'));
    const config = {
      dataDir: tmpDir,
      outputsDir: path.join(tmpDir, 'outputs'),
      sqlitePath: path.join(tmpDir, 'test.sqlite'),
      logLevel: 'silent',
    };
    const { initDatabase } = require('../db');
    db = initDatabase(config);
  });

  afterEach(() => {
    if (db) {
      try { db.close(); } catch {}
    }
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  describe('initDatabase', () => {
    it('should create data and outputs directories', () => {
      assert.ok(fs.existsSync(tmpDir));
      assert.ok(fs.existsSync(path.join(tmpDir, 'outputs')));
    });

    it('should open database in WAL mode', () => {
      const mode = db.pragma('journal_mode', { simple: true });
      assert.equal(mode, 'wal');
    });

    it('should have foreign keys enabled', () => {
      const fk = db.pragma('foreign_keys', { simple: true });
      assert.equal(fk, 1);
    });
  });

  describe('safeJsonParse', () => {
    const { safeJsonParse } = require('../db');

    it('should parse valid JSON', () => {
      const result = safeJsonParse('{"a": 1}');
      assert.deepEqual(result, { a: 1 });
    });

    it('should return fallback for invalid JSON', () => {
      const result = safeJsonParse('not json', { default: true });
      assert.deepEqual(result, { default: true });
    });

    it('should return fallback for null', () => {
      const result = safeJsonParse(null, []);
      assert.deepEqual(result, []);
    });

    it('should return null as default fallback', () => {
      const result = safeJsonParse(undefined);
      assert.equal(result, null);
    });
  });

  describe('toApiJob', () => {
    const { toApiJob } = require('../db');

    it('should return null for falsy input', () => {
      assert.equal(toApiJob(null), null);
      assert.equal(toApiJob(undefined), null);
    });

    it('should map a raw row to API format', () => {
      const row = {
        id: 'gen_abc12345',
        status: 'running',
        domain: 'financial',
        config_json: '{"domain":"financial"}',
        provider: 'mock',
        parse_mode: 'qa',
        output_format: 'jsonl',
        target_count: 1000,
        batch_size: 25,
        generated_count: 500,
        duplicates_count: 10,
        invalid_count: 5,
        rate_items_per_sec: 2.5,
        eta_seconds: 200,
        stop_requested: 0,
        error_message: null,
        output_dir: 'outputs/gen_abc12345',
        output_file: 'dataset.jsonl',
        checkpoint_file: 'checkpoint.json',
        language: 'en',
        created_at: '2024-01-01T00:00:00.000Z',
        started_at: '2024-01-01T00:01:00.000Z',
        completed_at: null,
        updated_at: '2024-01-01T00:05:00.000Z',
      };

      const api = toApiJob(row);

      assert.equal(api.id, 'gen_abc12345');
      assert.equal(api.jobId, 'gen_abc12345');
      assert.equal(api.status, 'running');
      assert.equal(api.domain, 'financial');
      assert.equal(api.provider, 'mock');
      assert.equal(api.parseMode, 'qa');
      assert.equal(api.outputFormat, 'jsonl');
      assert.equal(api.targetCount, 1000);
      assert.equal(api.generatedCount, 500);
      assert.equal(api.generated, 500);
      assert.equal(api.progress, 50);
      assert.equal(api.language, 'en');
      assert.equal(api.stopRequested, false);
      assert.equal(api.errorMessage, null);
      assert.deepEqual(api.config, { domain: 'financial' });
    });

    it('should compute 100% progress correctly', () => {
      const row = {
        id: 'gen_test',
        status: 'completed',
        domain: 'test',
        config_json: '{}',
        provider: 'mock',
        parse_mode: 'qa',
        output_format: 'jsonl',
        target_count: 100,
        batch_size: 10,
        generated_count: 100,
        duplicates_count: 0,
        invalid_count: 0,
        rate_items_per_sec: 1,
        eta_seconds: 0,
        stop_requested: 0,
        error_message: null,
        output_dir: 'outputs/test',
        output_file: null,
        checkpoint_file: null,
        language: 'en',
        created_at: '2024-01-01T00:00:00.000Z',
        started_at: null,
        completed_at: null,
        updated_at: '2024-01-01T00:00:00.000Z',
      };

      const api = toApiJob(row);
      assert.equal(api.progress, 100);
    });
  });

  describe('toApiDomain', () => {
    const { toApiDomain } = require('../db');

    it('should return null for falsy input', () => {
      assert.equal(toApiDomain(null), null);
    });

    it('should map domain row to API format', () => {
      const row = {
        id: 'domain_abc',
        name: 'Test Domain',
        config_json: '{"description":"test desc","topics":["A","B"]}',
        created_at: '2024-01-01T00:00:00.000Z',
        updated_at: '2024-01-01T00:00:00.000Z',
      };

      const api = toApiDomain(row);
      assert.equal(api.id, 'domain_abc');
      assert.equal(api.name, 'Test Domain');
      assert.equal(api.config.description, 'test desc');
      assert.deepEqual(api.config.topics, ['A', 'B']);
    });
  });

  describe('insertJobEvent', () => {
    const { insertJobEvent, nowMs } = require('../db');

    it('should insert an event into job_events', () => {
      // First insert a job
      db.prepare(
        `INSERT INTO jobs (id, status, domain, config_json, provider, parse_mode, output_format,
         target_count, batch_size, output_dir, created_at, updated_at)
         VALUES (?, 'running', 'test', '{}', 'mock', 'qa', 'jsonl', 100, 10, 'outputs/test', '2024-01-01', '2024-01-01')`
      ).run('gen_test_event');

      insertJobEvent(db, 'gen_test_event', 'progress', { count: 50 });

      const events = db
        .prepare('SELECT * FROM job_events WHERE job_id = ?')
        .all('gen_test_event');

      assert.equal(events.length, 1);
      assert.equal(events[0].type, 'progress');
      assert.equal(events[0].job_id, 'gen_test_event');
      const payload = JSON.parse(events[0].payload_json);
      assert.equal(payload.count, 50);
    });
  });

  describe('deleteArtifactsDirectory', () => {
    const { deleteArtifactsDirectory } = require('../db');

    it('should delete an existing directory', () => {
      const dir = path.join(tmpDir, 'outputs', 'gen_del_test');
      fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(path.join(dir, 'test.txt'), 'data');

      const result = deleteArtifactsDirectory(path.join(tmpDir, 'outputs'), 'gen_del_test');
      assert.equal(result, true);
      assert.ok(!fs.existsSync(dir));
    });

    it('should return true for non-existing directory (no-op)', () => {
      const result = deleteArtifactsDirectory(path.join(tmpDir, 'outputs'), 'gen_nonexist');
      assert.equal(result, true);
    });

    it('should reject path traversal', () => {
      const result = deleteArtifactsDirectory(path.join(tmpDir, 'outputs'), '../../etc');
      assert.equal(result, false);
    });
  });
});
