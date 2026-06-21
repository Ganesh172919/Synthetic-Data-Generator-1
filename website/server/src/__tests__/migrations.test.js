const { describe, it, beforeEach } = require('node:test');
const assert = require('node:assert/strict');
const Database = require('better-sqlite3');
const { runMigrations, migrations } = require('../migrations');

describe('migrations', () => {
  let db;

  beforeEach(() => {
    db = new Database(':memory:');
  });

  it('should create _migrations tracking table', () => {
    runMigrations(db);
    const tables = db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='_migrations'")
      .all();
    assert.equal(tables.length, 1);
  });

  it('should apply all migrations on first run', () => {
    const result = runMigrations(db);
    assert.equal(result.applied, migrations.length);
    assert.equal(result.current, migrations.length);
  });

  it('should not re-apply migrations on second run', () => {
    runMigrations(db);
    const result = runMigrations(db);
    assert.equal(result.applied, 0);
  });

  it('should create jobs table with expected columns', () => {
    runMigrations(db);
    const columns = db.prepare('PRAGMA table_info(jobs)').all();
    const columnNames = columns.map((c) => c.name);

    assert.ok(columnNames.includes('id'));
    assert.ok(columnNames.includes('status'));
    assert.ok(columnNames.includes('domain'));
    assert.ok(columnNames.includes('provider'));
    assert.ok(columnNames.includes('language'));
    assert.ok(columnNames.includes('target_count'));
    assert.ok(columnNames.includes('generated_count'));
    assert.ok(columnNames.includes('output_dir'));
  });

  it('should create job_events table', () => {
    runMigrations(db);
    const tables = db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='job_events'")
      .all();
    assert.equal(tables.length, 1);
  });

  it('should create domains table', () => {
    runMigrations(db);
    const tables = db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='domains'")
      .all();
    assert.equal(tables.length, 1);
  });

  it('should create domain indexes', () => {
    runMigrations(db);
    const indexes = db
      .prepare("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_domains_%'")
      .all()
      .map((r) => r.name);

    assert.ok(indexes.includes('idx_domains_name'));
    assert.ok(indexes.includes('idx_domains_created_at'));
  });

  it('should add language column to jobs table', () => {
    runMigrations(db);
    const columns = db.prepare('PRAGMA table_info(jobs)').all();
    const langCol = columns.find((c) => c.name === 'language');
    assert.ok(langCol, 'language column should exist');
    assert.equal(langCol.dflt_value, "'en'");
  });

  it('should record migration names in _migrations table', () => {
    runMigrations(db);
    const applied = db.prepare('SELECT id, name FROM _migrations ORDER BY id').all();
    assert.ok(applied.length > 0);
    assert.equal(applied[0].name, 'initial_schema');
  });
});
