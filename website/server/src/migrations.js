/**
 * Database Migrations System
 *
 * Simple numbered migration runner for SQLite.
 * Tracks applied migrations in a `_migrations` table.
 * Each migration is a function that receives the db instance.
 */

const migrations = [
  {
    id: 1,
    name: 'initial_schema',
    up: (db) => {
      db.exec(`
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
      `);
    },
  },
  {
    id: 2,
    name: 'domain_indexes_and_language',
    up: (db) => {
      db.exec(`
        CREATE INDEX IF NOT EXISTS idx_domains_name ON domains(name);
        CREATE INDEX IF NOT EXISTS idx_domains_created_at ON domains(created_at);
      `);

      // Add language column to jobs if it doesn't exist
      const columns = db.prepare("PRAGMA table_info(jobs)").all();
      const hasLanguage = columns.some((col) => col.name === 'language');
      if (!hasLanguage) {
        db.exec(`ALTER TABLE jobs ADD COLUMN language TEXT NOT NULL DEFAULT 'en'`);
      }
    },
  },
  {
    id: 3,
    name: 'retry_count_and_dead_letter',
    up: (db) => {
      const columns = db.prepare("PRAGMA table_info(jobs)").all();
      const hasRetryCount = columns.some((col) => col.name === 'retry_count');
      if (!hasRetryCount) {
        db.exec(`ALTER TABLE jobs ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0`);
      }
    },
  },
];

/**
 * Run all pending migrations.
 * @param {import('better-sqlite3').Database} db
 * @returns {{ applied: number, current: number }}
 */
function runMigrations(db) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS _migrations (
      id INTEGER PRIMARY KEY,
      name TEXT NOT NULL,
      applied_at TEXT NOT NULL
    );
  `);

  const applied = db.prepare('SELECT id FROM _migrations ORDER BY id').all();
  const appliedIds = new Set(applied.map((row) => row.id));

  let count = 0;
  for (const migration of migrations) {
    if (appliedIds.has(migration.id)) {
      continue;
    }

    const now = new Date().toISOString();
    db.transaction(() => {
      migration.up(db);
      db.prepare('INSERT INTO _migrations (id, name, applied_at) VALUES (?, ?, ?)').run(
        migration.id,
        migration.name,
        now
      );
    })();

    count += 1;
  }

  return { applied: count, current: migrations.length };
}

module.exports = { runMigrations, migrations };
