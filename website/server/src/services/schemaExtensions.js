'use strict';

/**
 * Extended database schema for SaaS platform features.
 * This module adds users, API keys, subscriptions, usage tracking,
 * feature flags, and plugin tables alongside the existing schema.
 *
 * Call `applyExtendedSchema(db)` after `initDatabase()` to add
 * the new tables without modifying the existing db.js.
 */

const extendedSchemaSql = `
-- Users table for authentication and profile management
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  email TEXT NOT NULL UNIQUE,
  username TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  display_name TEXT,
  role TEXT NOT NULL DEFAULT 'user',
  tier TEXT NOT NULL DEFAULT 'free',
  is_active INTEGER NOT NULL DEFAULT 1,
  email_verified INTEGER NOT NULL DEFAULT 0,
  last_login_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_tier ON users(tier);

-- API keys for programmatic access
CREATE TABLE IF NOT EXISTS api_keys (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  name TEXT NOT NULL,
  key_hash TEXT NOT NULL UNIQUE,
  key_prefix TEXT NOT NULL,
  scopes TEXT NOT NULL DEFAULT '["read","write"]',
  is_active INTEGER NOT NULL DEFAULT 1,
  last_used_at TEXT,
  expires_at TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);

-- Subscription management
CREATE TABLE IF NOT EXISTS subscriptions (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL UNIQUE,
  tier TEXT NOT NULL DEFAULT 'free',
  status TEXT NOT NULL DEFAULT 'active',
  started_at TEXT NOT NULL,
  expires_at TEXT,
  cancelled_at TEXT,
  payment_provider TEXT,
  external_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_subscriptions_user_id ON subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions(status);

-- Usage records for metering and billing
CREATE TABLE IF NOT EXISTS usage_records (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  resource_type TEXT NOT NULL,
  quantity INTEGER NOT NULL DEFAULT 1,
  metadata_json TEXT,
  recorded_at TEXT NOT NULL,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_usage_records_user_id ON usage_records(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_records_resource_type ON usage_records(resource_type);
CREATE INDEX IF NOT EXISTS idx_usage_records_recorded_at ON usage_records(recorded_at);

-- Feature flags for dynamic feature toggling
CREATE TABLE IF NOT EXISTS feature_flags (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  description TEXT,
  is_enabled INTEGER NOT NULL DEFAULT 0,
  allowed_tiers TEXT NOT NULL DEFAULT '["free","pro","enterprise"]',
  allowed_roles TEXT NOT NULL DEFAULT '["user","admin"]',
  config_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feature_flags_name ON feature_flags(name);

-- Plugins registry
CREATE TABLE IF NOT EXISTS plugins (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  version TEXT NOT NULL,
  description TEXT,
  author TEXT,
  entry_point TEXT NOT NULL,
  is_enabled INTEGER NOT NULL DEFAULT 0,
  config_json TEXT,
  installed_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_plugins_name ON plugins(name);

-- Audit log for security-sensitive actions
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT,
  action TEXT NOT NULL,
  resource_type TEXT,
  resource_id TEXT,
  details_json TEXT,
  ip_address TEXT,
  recorded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_log_recorded_at ON audit_log(recorded_at);
`;

const defaultFeatureFlags = [
  { id: 'ff_custom_domains', name: 'custom_domains', description: 'Allow custom domain configurations', tiers: '["pro","enterprise"]' },
  { id: 'ff_csv_export', name: 'csv_export', description: 'Export datasets as CSV', tiers: '["free","pro","enterprise"]' },
  { id: 'ff_json_export', name: 'json_export', description: 'Export datasets as JSON', tiers: '["free","pro","enterprise"]' },
  { id: 'ff_api_access', name: 'api_access', description: 'Programmatic API access', tiers: '["pro","enterprise"]' },
  { id: 'ff_priority_queue', name: 'priority_queue', description: 'Priority job queue placement', tiers: '["enterprise"]' },
  { id: 'ff_bulk_generation', name: 'bulk_generation', description: 'Generate datasets > 10k rows', tiers: '["pro","enterprise"]' },
  { id: 'ff_advanced_analytics', name: 'advanced_analytics', description: 'Advanced analytics dashboard', tiers: '["pro","enterprise"]' },
  { id: 'ff_plugin_marketplace', name: 'plugin_marketplace', description: 'Access to plugin marketplace', tiers: '["pro","enterprise"]' },
  { id: 'ff_team_management', name: 'team_management', description: 'Team and organization features', tiers: '["enterprise"]' },
  { id: 'ff_sla_support', name: 'sla_support', description: 'SLA-backed support', tiers: '["enterprise"]' },
];

const applyExtendedSchema = (db) => {
  db.exec(extendedSchemaSql);

  const insertFlag = db.prepare(`
    INSERT OR IGNORE INTO feature_flags (id, name, description, is_enabled, allowed_tiers, created_at, updated_at)
    VALUES (?, ?, ?, 1, ?, datetime('now'), datetime('now'))
  `);

  const seedFlags = db.transaction(() => {
    for (const flag of defaultFeatureFlags) {
      insertFlag.run(flag.id, flag.name, flag.description, flag.tiers);
    }
  });

  seedFlags();
};

/**
 * Add user_id column to existing jobs table if not present.
 * Safe to call multiple times (uses IF NOT EXISTS via pragma check).
 */
const migrateJobsTable = (db) => {
  const columns = db.prepare("PRAGMA table_info(jobs)").all();
  const hasUserId = columns.some((col) => col.name === 'user_id');
  if (!hasUserId) {
    db.exec('ALTER TABLE jobs ADD COLUMN user_id TEXT');
    db.exec('CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id)');
  }

  const hasPriority = columns.some((col) => col.name === 'priority');
  if (!hasPriority) {
    db.exec('ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 0');
    db.exec('CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority)');
  }
};

module.exports = {
  applyExtendedSchema,
  migrateJobsTable,
};
