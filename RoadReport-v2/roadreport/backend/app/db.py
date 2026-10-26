"""SQLite database layer for Pothole & Road Damage Reporter."""

from contextlib import contextmanager
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional, Union
import uuid

logger = logging.getLogger(__name__)

# Configurable database path via ROADREPORT_DB environment variable
DEFAULT_DB_FILENAME = "roadreport.db"


def get_db_path() -> Path:
    """Return the resolved Path to the SQLite database file."""
    db_env = os.getenv("ROADREPORT_DB", DEFAULT_DB_FILENAME)
    db_path = Path(db_env).resolve()
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Context manager yielding a SQLite connection with Row row_factory and foreign keys enabled.
    
    Commits on successful block execution, rolls back on exception, and ensures connection is closed.
    """
    conn = sqlite3.connect(str(get_db_path()), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("Database transaction rolled back due to error")
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Initialize database tables and indexes if they do not exist."""
    with get_db() as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")

        # Jobs table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                video_key TEXT NOT NULL,
                location_text TEXT,
                lat REAL,
                lon REAL,
                created_at TEXT NOT NULL,
                frames_total INTEGER,
                frames_used INTEGER,
                frames_done INTEGER DEFAULT 0,
                summary_json TEXT,
                error_message TEXT
            );
        """)

        # Findings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS findings (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                type TEXT NOT NULL,
                severity_label TEXT NOT NULL,
                severity_score REAL NOT NULL,
                confidence REAL NOT NULL,
                status TEXT DEFAULT 'confirmed',
                frame_index INTEGER,
                timestamp_s REAL,
                bbox_json TEXT,
                image_key TEXT,
                crop_key TEXT,
                measurements_json TEXT,
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
            );
        """)

        # Agent steps table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                finding_ref TEXT,
                step_no INTEGER NOT NULL,
                action TEXT NOT NULL,
                reason TEXT NOT NULL,
                confidence_before REAL,
                confidence_after REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
            );
        """)

        # Indexes for query performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_findings_job_id ON findings(job_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_steps_job_id ON agent_steps(job_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_steps_finding_ref ON agent_steps(finding_ref);")

    logger.info("Database initialized at %s", get_db_path())


def create_job(
    video_key: str,
    location_text: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    job_id: Optional[str] = None,
    status: str = "uploaded",
    frames_total: Optional[int] = None,
    frames_used: Optional[int] = None,
    frames_done: int = 0,
    summary_json: Optional[str] = None,
    error_message: Optional[str] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new job record in the database."""
    assigned_id = job_id or uuid.uuid4().hex
    timestamp = created_at or datetime.now(timezone.utc).isoformat()

    query = """
        INSERT INTO jobs (
            id, status, video_key, location_text, lat, lon,
            created_at, frames_total, frames_used, frames_done,
            summary_json, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_db() as conn:
        conn.execute(
            query,
            (
                assigned_id,
                status,
                video_key,
                location_text,
                lat,
                lon,
                timestamp,
                frames_total,
                frames_used,
                frames_done,
                summary_json,
                error_message,
            ),
        )

    job = get_job(assigned_id)
    if job is None:
        raise RuntimeError(f"Failed to fetch job immediately after creation: {assigned_id}")
    return job


def update_job(job_id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Update fields of an existing job record."""
    allowed_columns = {
        "status",
        "video_key",
        "location_text",
        "lat",
        "lon",
        "frames_total",
        "frames_used",
        "frames_done",
        "summary_json",
        "error_message",
    }
    updates = {k: v for k, v in kwargs.items() if k in allowed_columns}
    if not updates:
        return get_job(job_id)

    set_clause = ", ".join([f"{col} = ?" for col in updates.keys()])
    values = list(updates.values())
    values.append(job_id)

    query = f"UPDATE jobs SET {set_clause} WHERE id = ?"
    with get_db() as conn:
        cursor = conn.execute(query, values)
        if cursor.rowcount == 0:
            return None

    return get_job(job_id)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single job by its ID."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve list of recent jobs ordered by created_at descending."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def create_finding(
    job_id: Union[str, Dict[str, Any]],
    type: Optional[str] = None,
    severity_label: Optional[str] = None,
    severity_score: Optional[float] = None,
    confidence: Optional[float] = None,
    finding_id: Optional[str] = None,
    status: str = "confirmed",
    frame_index: Optional[int] = None,
    timestamp_s: Optional[float] = None,
    bbox_json: Optional[str] = None,
    image_key: Optional[str] = None,
    crop_key: Optional[str] = None,
    measurements_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new road damage finding record in the database."""
    if isinstance(job_id, dict):
        d = job_id
        assigned_id = d.get("id") or d.get("finding_id") or uuid.uuid4().hex
        real_job_id = d.get("job_id", "")
        f_type = d.get("type", "pothole")
        sev_label = d.get("severity_label", "medium")
        sev_score = float(d.get("severity_score", 0.5))
        conf = float(d.get("confidence", 0.8))
        f_status = d.get("status", "confirmed")
        f_idx = d.get("frame_index")
        t_s = d.get("timestamp_s")
        bb_json = d.get("bbox_json") or (json.dumps(d["bbox"]) if "bbox" in d else None)
        img_key = d.get("image_key")
        crp_key = d.get("crop_key")
        meas_json = d.get("measurements_json") or (json.dumps(d["measurements"]) if "measurements" in d else None)
    else:
        assigned_id = finding_id or uuid.uuid4().hex
        real_job_id = job_id
        f_type = type or "pothole"
        sev_label = severity_label or "medium"
        sev_score = float(severity_score if severity_score is not None else 0.5)
        conf = float(confidence if confidence is not None else 0.8)
        f_status = status
        f_idx = frame_index
        t_s = timestamp_s
        bb_json = bbox_json
        img_key = image_key
        crp_key = crop_key
        meas_json = measurements_json

    query = """
        INSERT INTO findings (
            id, job_id, type, severity_label, severity_score, confidence,
            status, frame_index, timestamp_s, bbox_json, image_key,
            crop_key, measurements_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_db() as conn:
        conn.execute(
            query,
            (
                assigned_id,
                real_job_id,
                f_type,
                sev_label,
                sev_score,
                conf,
                f_status,
                f_idx,
                t_s,
                bb_json,
                img_key,
                crp_key,
                meas_json,
            ),
        )
    finding = get_finding(assigned_id)
    if finding is None:
        raise RuntimeError(f"Failed to fetch finding immediately after creation: {assigned_id}")
    return finding


def get_finding(finding_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single finding by its ID."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM findings WHERE id = ?", (finding_id,)).fetchone()
        return dict(row) if row else None


def update_finding(finding_id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Update fields of an existing finding record."""
    allowed_columns = {
        "job_id",
        "type",
        "severity_label",
        "severity_score",
        "confidence",
        "status",
        "frame_index",
        "timestamp_s",
        "bbox_json",
        "image_key",
        "crop_key",
        "measurements_json",
    }
    updates = {k: v for k, v in kwargs.items() if k in allowed_columns}
    if not updates:
        return get_finding(finding_id)

    set_clause = ", ".join([f"{col} = ?" for col in updates.keys()])
    values = list(updates.values())
    values.append(finding_id)

    query = f"UPDATE findings SET {set_clause} WHERE id = ?"
    with get_db() as conn:
        cursor = conn.execute(query, values)
        if cursor.rowcount == 0:
            return None

    return get_finding(finding_id)


def get_findings(job_id: str) -> List[Dict[str, Any]]:
    """Retrieve all findings associated with a given job ID."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM findings WHERE job_id = ? ORDER BY frame_index ASC, id ASC",
            (job_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def add_agent_step(
    job_id: str,
    step_no: int,
    action: str,
    reason: str,
    finding_ref: Optional[str] = None,
    confidence_before: Optional[float] = None,
    confidence_after: Optional[float] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Record an agent verification or processing step in the database."""
    timestamp = created_at or datetime.now(timezone.utc).isoformat()

    query = """
        INSERT INTO agent_steps (
            job_id, finding_ref, step_no, action, reason,
            confidence_before, confidence_after, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_db() as conn:
        cursor = conn.execute(
            query,
            (
                job_id,
                finding_ref,
                step_no,
                action,
                reason,
                confidence_before,
                confidence_after,
                timestamp,
            ),
        )
        step_id = cursor.lastrowid

        row = conn.execute("SELECT * FROM agent_steps WHERE id = ?", (step_id,)).fetchone()
        if not row:
            raise RuntimeError(f"Failed to fetch newly created agent step: {step_id}")
        return dict(row)


def get_agent_steps(job_id: str, finding_ref: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve agent execution steps for a job, optionally filtered by finding reference."""
    with get_db() as conn:
        if finding_ref is not None:
            rows = conn.execute(
                """
                SELECT * FROM agent_steps
                WHERE job_id = ? AND finding_ref = ?
                ORDER BY step_no ASC, id ASC
                """,
                (job_id, finding_ref),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM agent_steps
                WHERE job_id = ?
                ORDER BY step_no ASC, id ASC
                """,
                (job_id,),
            ).fetchall()
        return [dict(row) for row in rows]


# Initialize the database upon module import
init_db()
