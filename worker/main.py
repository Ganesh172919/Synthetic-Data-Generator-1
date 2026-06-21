import importlib.util
import json
import os
import signal
import sqlite3
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "website" / "server" / "data"
DATA_DIR = Path(os.environ.get("DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()
SQLITE_PATH = Path(os.environ.get("SQLITE_PATH", str(DATA_DIR / "synthgen.sqlite"))).resolve()
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", str(DATA_DIR / "outputs"))).resolve()

POLL_INTERVAL_MS = int(os.environ.get("POLL_INTERVAL_MS", "1000"))
PROGRESS_UPDATE_INTERVAL_MS = int(os.environ.get("PROGRESS_UPDATE_INTERVAL_MS", "2000"))
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "1"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
DEAD_LETTER_THRESHOLD = int(os.environ.get("DEAD_LETTER_THRESHOLD", "5"))

VALID_STATUSES = {"queued", "running", "completed", "failed", "stopped", "dead_letter"}
TERMINAL_STATUSES = {"completed", "failed", "stopped", "dead_letter"}

# Global shutdown flag — set by SIGTERM/SIGINT handler
_shutdown_event = threading.Event()


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

    # Ensure language column exists (migration v2)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if "language" not in columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN language TEXT NOT NULL DEFAULT 'en'")

    # Ensure retry_count column exists (migration v3)
    if "retry_count" not in columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0")


def setup_signal_handlers() -> None:
    """Register SIGTERM and SIGINT handlers for graceful shutdown."""
    def _handle_signal(signum, frame):
        sig_name = signal.Signals(signum).name
        log("info", f"Received {sig_name}, initiating graceful shutdown...")
        _shutdown_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)


def is_shutting_down() -> bool:
    """Check if a shutdown signal has been received."""
    return _shutdown_event.is_set()


def increment_retry_count(conn: sqlite3.Connection, job_id: str) -> int:
    """Increment the retry_count for a job and return the new count."""
    conn.execute(
        "UPDATE jobs SET retry_count = retry_count + 1, updated_at = ? WHERE id = ?",
        (now_iso(), job_id),
    )
    row = conn.execute("SELECT retry_count FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return int(row["retry_count"]) if row else 0


def mark_dead_letter(conn: sqlite3.Connection, job_id: str, error_message: str) -> None:
    """Move a job to dead_letter status after exceeding retry threshold."""
    now = now_iso()
    conn.execute(
        """
        UPDATE jobs
        SET status = 'dead_letter',
            error_message = ?,
            completed_at = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (f"Dead letter after exceeding retry threshold: {error_message}", now, now, job_id),
    )
    add_event(conn, job_id, "status", {
        "status": "dead_letter",
        "message": f"Job moved to dead letter queue after {DEAD_LETTER_THRESHOLD} failures",
        "error": error_message,
    })
    log("warn", "Job moved to dead letter queue", job_id=job_id, error=error_message)


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


def claim_next_jobs(conn: sqlite3.Connection, count: int) -> List[str]:
    """Claim up to `count` queued jobs in a single transaction."""
    claimed = []
    conn.execute("BEGIN IMMEDIATE")
    try:
        rows = conn.execute(
            "SELECT id FROM jobs WHERE status = 'queued' ORDER BY datetime(created_at) ASC LIMIT ?",
            (count,),
        ).fetchall()
        if not rows:
            conn.execute("COMMIT")
            return []

        now = now_iso()
        for row in rows:
            updated = conn.execute(
                """
                UPDATE jobs
                SET status = 'running', started_at = ?, updated_at = ?, stop_requested = 0
                WHERE id = ? AND status = 'queued'
                """,
                (now, now, row["id"]),
            ).rowcount
            if updated == 1:
                claimed.append(str(row["id"]))

        conn.execute("COMMIT")
        return claimed
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


def _add_language_instruction(prompt: str, language: str) -> str:
    """Add language instruction to prompt if not English."""
    if language == "en":
        return prompt
    lang_names = {
        "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
        "pt": "Portuguese", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
        "hi": "Hindi", "ar": "Arabic", "ru": "Russian", "nl": "Dutch",
        "pl": "Polish", "tr": "Turkish", "vi": "Vietnamese", "th": "Thai",
        "sv": "Swedish", "da": "Danish", "fi": "Finnish",
    }
    lang_name = lang_names.get(language, language)
    return f"{prompt}\n\nIMPORTANT: Generate all content in {lang_name} ({language})."


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

        # Add language instruction
        language = str(job["language"] if "language" in job.keys() else config_json.get("language", "en")).lower()
        prompt = _add_language_instruction(prompt, language)

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
            return is_stop_requested(conn, job_id) or is_shutting_down()

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
        # Retry logic for failed jobs
        if result_status == "failed" and result_error and not is_shutting_down():
            retry_count = increment_retry_count(conn, job_id)
            if retry_count >= DEAD_LETTER_THRESHOLD:
                mark_dead_letter(conn, job_id, result_error)
            elif retry_count <= MAX_RETRIES:
                # Re-queue for retry with backoff delay
                backoff_seconds = min(2 ** retry_count, 60)
                log("info", "Re-queuing failed job for retry",
                    job_id=job_id, retry_count=retry_count, backoff_seconds=backoff_seconds)
                time.sleep(backoff_seconds)
                now = now_iso()
                conn.execute(
                    """UPDATE jobs SET status = 'queued', started_at = NULL,
                       stop_requested = 0, error_message = NULL, updated_at = ?
                       WHERE id = ?""",
                    (now, job_id),
                )
                add_event(conn, job_id, "status", {
                    "status": "queued",
                    "message": f"Auto-retry #{retry_count} after {backoff_seconds}s backoff",
                })
            else:
                set_job_terminal(conn, job_id, result_status, result, result_error)
        else:
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


def _run_single_job_wrapper(job_id: str) -> None:
    """Run a single job in its own connection (for thread pool)."""
    conn = open_conn()
    try:
        run_job(conn, job_id)
    except Exception as exc:
        log("error", "Job wrapper error", job_id=job_id, error=str(exc), traceback=traceback.format_exc())
    finally:
        conn.close()


def worker_loop() -> int:
    setup_signal_handlers()

    concurrency = max(1, MAX_CONCURRENT_JOBS)
    log("info", "Worker starting",
        max_concurrent_jobs=concurrency,
        max_retries=MAX_RETRIES,
        dead_letter_threshold=DEAD_LETTER_THRESHOLD,
        sqlite_path=str(SQLITE_PATH),
        outputs_dir=str(OUTPUTS_DIR))

    conn = open_conn()
    ensure_schema(conn)
    recover_stale_running_jobs(conn)
    conn.close()

    if concurrency == 1:
        # Single-threaded mode (original behavior, no thread pool overhead)
        return _worker_loop_single()
    else:
        # Multi-threaded mode with thread pool
        return _worker_loop_concurrent(concurrency)


def _worker_loop_single() -> int:
    conn = open_conn()
    try:
        while not is_shutting_down():
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
    finally:
        if is_shutting_down():
            log("info", "Shutdown signal received, closing gracefully")
        conn.close()
    return 0


def _worker_loop_concurrent(concurrency: int) -> int:
    active_futures = {}
    claim_conn = open_conn()

    try:
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="job") as executor:
            while not is_shutting_down():
                try:
                    # Clean up completed futures
                    done_ids = []
                    for jid, future in active_futures.items():
                        if future.done():
                            done_ids.append(jid)
                            exc = future.exception()
                            if exc:
                                log("error", "Job thread error", job_id=jid, error=str(exc))
                    for jid in done_ids:
                        del active_futures[jid]

                    # Claim jobs to fill available slots
                    available = concurrency - len(active_futures)
                    if available > 0:
                        job_ids = claim_next_jobs(claim_conn, available)
                        for jid in job_ids:
                            log("info", "Claimed job", job_id=jid, slot=len(active_futures) + 1)
                            active_futures[jid] = executor.submit(_run_single_job_wrapper, jid)

                    if not active_futures:
                        time.sleep(POLL_INTERVAL_MS / 1000.0)
                    else:
                        time.sleep(max(0.1, POLL_INTERVAL_MS / 2000.0))

                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    log("error", "Worker loop error", error=str(exc), traceback=traceback.format_exc())
                    time.sleep(max(0.2, POLL_INTERVAL_MS / 1000.0))
    except KeyboardInterrupt:
        log("info", "Worker interrupted, waiting for active jobs to finish...")
    finally:
        if is_shutting_down() and active_futures:
            log("info", "Shutdown requested, waiting for active jobs to complete...",
                active_jobs=len(active_futures))
        # ThreadPoolExecutor.__exit__ will wait for futures
        claim_conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(worker_loop())
