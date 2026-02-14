import importlib.util
import json
import os
import sqlite3
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "website" / "server" / "data"
DATA_DIR = Path(os.environ.get("DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()
SQLITE_PATH = Path(os.environ.get("SQLITE_PATH", str(DATA_DIR / "synthgen.sqlite"))).resolve()
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", str(DATA_DIR / "outputs"))).resolve()

POLL_INTERVAL_MS = int(os.environ.get("POLL_INTERVAL_MS", "1000"))
PROGRESS_UPDATE_INTERVAL_MS = int(os.environ.get("PROGRESS_UPDATE_INTERVAL_MS", "2000"))
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "1"))

VALID_STATUSES = {"queued", "running", "completed", "failed", "stopped"}
TERMINAL_STATUSES = {"completed", "failed", "stopped"}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time() % 1 * 1000):03d}Z"


def now_ms() -> int:
    return int(time.time() * 1000)


def log(level: str, message: str, **fields: Any) -> None:
    payload = {
        "ts": now_iso(),
        "level": level,
        "service": "synthgen-worker",
        "message": message,
    }
    if fields:
        payload.update(fields)
    print(json.dumps(payload), flush=True)


def open_conn() -> sqlite3.Connection:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
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
          payload_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_job_events_job_id_id ON job_events(job_id, id);

        CREATE TABLE IF NOT EXISTS domains (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          config_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        """
    )


def add_event(conn: sqlite3.Connection, job_id: str, event_type: str, payload: Dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO job_events (job_id, ts, type, payload_json) VALUES (?, ?, ?, ?)",
        (job_id, now_ms(), event_type, json.dumps(payload)),
    )


def load_job(conn: sqlite3.Connection, job_id: str) -> Optional[sqlite3.Row]:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return row


def claim_next_job(conn: sqlite3.Connection) -> Optional[str]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT id FROM jobs WHERE status = 'queued' ORDER BY datetime(created_at) ASC LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None

        now = now_iso()
        updated = conn.execute(
            """
            UPDATE jobs
            SET status = 'running', started_at = ?, updated_at = ?, stop_requested = 0
            WHERE id = ? AND status = 'queued'
            """,
            (now, now, row["id"]),
        ).rowcount
        conn.execute("COMMIT")
        if updated != 1:
            return None
        return str(row["id"])
    except Exception:
        conn.execute("ROLLBACK")
        raise


def is_stop_requested(conn: sqlite3.Connection, job_id: str) -> bool:
    row = conn.execute("SELECT stop_requested FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return True
    return bool(row["stop_requested"])


def update_progress(conn: sqlite3.Connection, job_id: str, payload: Dict[str, Any]) -> None:
    conn.execute(
        """
        UPDATE jobs
        SET generated_count = ?,
            duplicates_count = ?,
            invalid_count = ?,
            rate_items_per_sec = ?,
            eta_seconds = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            int(payload.get("generated_count", 0)),
            int(payload.get("duplicates_count", 0)),
            int(payload.get("invalid_count", 0)),
            float(payload.get("rate_items_per_sec", 0.0)),
            payload.get("eta_seconds"),
            now_iso(),
            job_id,
        ),
    )


def set_job_terminal(
    conn: sqlite3.Connection,
    job_id: str,
    status: str,
    progress_payload: Dict[str, Any],
    error_message: Optional[str] = None,
) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}")

    now = now_iso()
    conn.execute(
        """
        UPDATE jobs
        SET status = ?,
            generated_count = ?,
            duplicates_count = ?,
            invalid_count = ?,
            rate_items_per_sec = ?,
            eta_seconds = ?,
            error_message = ?,
            completed_at = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            status,
            int(progress_payload.get("generated_count", 0)),
            int(progress_payload.get("duplicates_count", 0)),
            int(progress_payload.get("invalid_count", 0)),
            float(progress_payload.get("rate_items_per_sec", 0.0)),
            progress_payload.get("eta_seconds"),
            error_message,
            now,
            now,
            job_id,
        ),
    )

    add_event(
        conn,
        job_id,
        "status",
        {
            "status": status,
            "error": error_message,
            "generated_count": int(progress_payload.get("generated_count", 0)),
            "target_count": int(progress_payload.get("target_count", 0)),
        },
    )


def _load_generator_module():
    module_path = ROOT_DIR / "Pre-Work" / "universal_dataset_generator.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Generator module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("universal_dataset_generator", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load universal generator module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN_MODULE = None


def get_generator_module():
    global GEN_MODULE
    if GEN_MODULE is None:
        GEN_MODULE = _load_generator_module()
    return GEN_MODULE


def build_prompt(job_row: sqlite3.Row, config_json: Dict[str, Any]) -> str:
    prompt = str(config_json.get("prompt") or "").strip()
    if prompt:
        return prompt

    fallback_parts = []
    domain_description = str(config_json.get("domainDescription") or "").strip()
    if domain_description:
        fallback_parts.append(domain_description)

    topics = config_json.get("topics")
    if isinstance(topics, list):
        topic_values = [str(item).strip() for item in topics if str(item).strip()]
        if topic_values:
            fallback_parts.append("Topics: " + ", ".join(topic_values))

    if fallback_parts:
        return "\n".join(fallback_parts)

    return f"Generate high-quality synthetic data for the {job_row['domain']} domain."


def run_job(conn: sqlite3.Connection, job_id: str) -> None:
    job = load_job(conn, job_id)
    if job is None:
        log("warn", "Claimed job no longer exists", job_id=job_id)
        return

    final_progress: Dict[str, Any] = {
        "generated_count": int(job["generated_count"]),
        "target_count": int(job["target_count"]),
        "duplicates_count": int(job["duplicates_count"]),
        "invalid_count": int(job["invalid_count"]),
        "rate_items_per_sec": float(job["rate_items_per_sec"]),
        "eta_seconds": job["eta_seconds"],
    }
    result: Dict[str, Any] = dict(final_progress)
    result_status = "failed"
    result_error: Optional[str] = None

    try:
        config_json = json.loads(job["config_json"])
        output_format = str(job["output_format"]).lower()
        output_dir = OUTPUTS_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        output_base = output_dir / "dataset"
        output_file = f"dataset.{output_format}"
        checkpoint_file = "checkpoint.json"
        output_rel_dir = f"outputs/{job_id}"

        conn.execute(
            """
            UPDATE jobs
            SET output_dir = ?, output_file = ?, checkpoint_file = ?, updated_at = ?
            WHERE id = ?
            """,
            (output_rel_dir, output_file, checkpoint_file, now_iso(), job_id),
        )

        resumed = (output_dir / checkpoint_file).exists() and (output_dir / output_file).exists()
        add_event(
            conn,
            job_id,
            "status",
            {
                "status": "running",
                "message": "Worker started job",
                "resume_mode": "resume" if resumed else "fresh",
            },
        )

        mod = get_generator_module()
        provider_name = str(job["provider"]).strip().lower()

        try:
            provider_enum = mod.ModelProvider[provider_name.upper()]
        except KeyError as exc:
            raise ValueError(f"Unsupported provider: {provider_name}") from exc

        generator_config = mod.GeneratorConfig(
            target_size=int(job["target_count"]),
            items_per_batch=int(job["batch_size"]),
            provider=provider_enum,
            output_file=str(output_base),
            output_format=output_format,
            checkpoint_file=str(output_dir / checkpoint_file),
        )

        # Mock provider must generate unique-enough progress quickly in CI/dev.
        if provider_name == "mock":
            generator_config.enable_deduplication = False

        model_name = config_json.get("modelName") or config_json.get("model")
        if model_name:
            generator_config.model_name = str(model_name)
        if config_json.get("openaiModel"):
            generator_config.openai_model = str(config_json["openaiModel"])
        if config_json.get("saveInterval") is not None:
            generator_config.save_interval = max(1, int(config_json["saveInterval"]))
        generator_config.auto_save_seconds = min(
            int(config_json.get("autoSaveSeconds", generator_config.auto_save_seconds)),
            2,
        )

        prompt = build_prompt(job, config_json)
        parse_mode = str(job["parse_mode"]).lower()
        extra_fields = config_json.get("extraFields")
        if not isinstance(extra_fields, list):
            extra_fields = []
        extra_fields = [str(item).strip() for item in extra_fields if str(item).strip()]

        generator = mod.UniversalGenerator(generator_config)
        last_progress_emit = 0.0

        def on_progress(payload: Dict[str, Any]) -> None:
            nonlocal last_progress_emit, final_progress
            final_progress = payload
            now = time.monotonic()
            should_emit = (now - last_progress_emit) * 1000 >= PROGRESS_UPDATE_INTERVAL_MS
            if payload.get("status") != "running":
                should_emit = True
            if not should_emit:
                return

            update_progress(conn, job_id, payload)
            add_event(conn, job_id, "progress", payload)
            last_progress_emit = now

        def should_stop() -> bool:
            return is_stop_requested(conn, job_id)

        result = generator.run(
            user_prompt=prompt,
            parse_mode=parse_mode,
            extra_fields=extra_fields,
            progress_callback=on_progress,
            should_stop=should_stop,
            non_interactive=True,
        )

        result_status = str(result.get("status", "failed")).lower()
        if result_status not in {"completed", "stopped", "failed"}:
            result_status = "failed"
        result_error = result.get("error_message")
    except Exception as exc:
        result_status = "failed"
        result_error = f"{type(exc).__name__}: {exc}"
        result = dict(final_progress)
        result["error_message"] = result_error
        log("error", "Job execution error", job_id=job_id, error=result_error, traceback=traceback.format_exc())

    update_progress(conn, job_id, result)
    current = load_job(conn, job_id)
    if current is not None and current["status"] not in TERMINAL_STATUSES:
        set_job_terminal(conn, job_id, result_status, result, result_error)

    log(
        "info",
        "Job finished",
        job_id=job_id,
        status=result_status,
        generated_count=int(result.get("generated_count", 0)),
        target_count=int(result.get("target_count", 0)),
    )


def recover_stale_running_jobs(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT id FROM jobs WHERE status = 'running'").fetchall()
    if not rows:
        return

    now = now_iso()
    for row in rows:
        job_id = row["id"]
        conn.execute(
            """
            UPDATE jobs
            SET status = 'queued',
                started_at = NULL,
                stop_requested = 0,
                updated_at = ?
            WHERE id = ?
            """,
            (now, job_id),
        )
        add_event(
            conn,
            job_id,
            "status",
            {
                "status": "queued",
                "message": "Recovered queued job after worker restart",
            },
        )
    log("warn", "Recovered stale running jobs", count=len(rows))


def worker_loop() -> int:
    if MAX_CONCURRENT_JOBS != 1:
        log(
            "warn",
            "Current worker implementation is single-concurrency; forcing MAX_CONCURRENT_JOBS=1",
            configured_max=MAX_CONCURRENT_JOBS,
        )

    conn = open_conn()
    ensure_schema(conn)
    recover_stale_running_jobs(conn)
    log("info", "Worker started", sqlite_path=str(SQLITE_PATH), outputs_dir=str(OUTPUTS_DIR))

    try:
        while True:
            try:
                job_id = claim_next_job(conn)
                if not job_id:
                    time.sleep(POLL_INTERVAL_MS / 1000.0)
                    continue

                log("info", "Claimed job", job_id=job_id)
                run_job(conn, job_id)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                log("error", "Worker loop error", error=str(exc), traceback=traceback.format_exc())
                time.sleep(max(0.2, POLL_INTERVAL_MS / 1000.0))
    except KeyboardInterrupt:
        log("info", "Worker interrupted, exiting")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(worker_loop())
