const fs = require('fs');
const path = require('path');
const Database = require('better-sqlite3');

const schemaSql = `
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  domain TEXT NOT NULL,
  config_json TEXT NOT NULL,
  prompt TEXT,
  provider TEXT NOT NULL,
  parse_mode TEXT NOT NULL,
  output_format TEXT NOT NULL,
  target_count INTEGER NOT NULL,
  batch_size INTEGER NOT NULL,
  generated_count INTEGER NOT NULL DEFAULT 0,
  duplicates_count INTEGER NOT NULL DEFAULT 0,
  invalid_count INTEGER NOT NULL DEFAULT 0,
  rate_items_per_sec REAL NOT NULL DEFAULT 0,
  eta_seconds INTEGER,
  stop_requested INTEGER NOT NULL DEFAULT 0,
  error_message TEXT,
  output_dir TEXT NOT NULL,
  output_file TEXT,
  checkpoint_file TEXT,
  created_at TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at);

CREATE TABLE IF NOT EXISTS job_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  ts INTEGER NOT NULL,
  type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_job_events_job_id_id ON job_events(job_id, id);

CREATE TABLE IF NOT EXISTS domains (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  config_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
`;

const initDatabase = (config) => {
  fs.mkdirSync(config.dataDir, { recursive: true });
  fs.mkdirSync(config.outputsDir, { recursive: true });

  const db = new Database(config.sqlitePath);
  db.pragma('journal_mode = WAL');
  db.pragma('synchronous = NORMAL');
  db.pragma('foreign_keys = ON');
  db.pragma('busy_timeout = 5000');
  db.exec(schemaSql);

  return db;
};

const safeJsonParse = (value, fallback = null) => {
  if (value === null || value === undefined) {
    return fallback;
  }
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
};

const nowIso = () => new Date().toISOString();
const nowMs = () => Date.now();

const toApiJob = (row) => {
  if (!row) {
    return null;
  }

  const config = safeJsonParse(row.config_json, {});
  const generatedCount = Number(row.generated_count || 0);
  const targetCount = Number(row.target_count || 0);
  const progressPct = targetCount > 0 ? (generatedCount / targetCount) * 100 : 0;

  return {
    id: row.id,
    jobId: row.id,
    status: row.status,
    domain: row.domain,
    provider: row.provider,
    parseMode: row.parse_mode,
    outputFormat: row.output_format,
    targetCount,
    batchSize: Number(row.batch_size || 0),
    generatedCount,
    generated: generatedCount,
    duplicatesCount: Number(row.duplicates_count || 0),
    invalidCount: Number(row.invalid_count || 0),
    rateItemsPerSec: Number(row.rate_items_per_sec || 0),
    etaSeconds: row.eta_seconds === null ? null : Number(row.eta_seconds),
    stopRequested: Boolean(row.stop_requested),
    errorMessage: row.error_message || null,
    outputDir: row.output_dir,
    outputFile: row.output_file || null,
    checkpointFile: row.checkpoint_file || null,
    createdAt: row.created_at,
    startedAt: row.started_at,
    completedAt: row.completed_at,
    updatedAt: row.updated_at,
    progress: Math.max(0, Math.min(100, progressPct)),
    config,
  };
};

const toApiDomain = (row) => {
  if (!row) {
    return null;
  }
  const cfg = safeJsonParse(row.config_json, {});
  return {
    id: row.id,
    name: row.name,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    ...cfg,
  };
};

const insertJobEvent = (db, jobId, type, payload) => {
  const payloadJson = JSON.stringify(payload || {});
  db.prepare(
    'INSERT INTO job_events (job_id, ts, type, payload_json) VALUES (?, ?, ?, ?)'
  ).run(jobId, nowMs(), type, payloadJson);
};

const deleteArtifactsDirectory = (outputsDir, jobId, outputDir = null) => {
  const base = path.resolve(outputsDir);
  const candidate = outputDir
    ? path.resolve(path.dirname(base), outputDir)
    : path.resolve(base, jobId);

  if (!candidate.startsWith(base)) {
    return false;
  }

  if (fs.existsSync(candidate)) {
    fs.rmSync(candidate, { recursive: true, force: true });
  }

  return true;
};

module.exports = {
  initDatabase,
  safeJsonParse,
  nowIso,
  nowMs,
  toApiJob,
  toApiDomain,
  insertJobEvent,
  deleteArtifactsDirectory,
};
