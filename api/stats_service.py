"""Service layer for detection-event persistence and historical statistics.

All DB access goes through db_client which connects via DATABASE_URL as the
postgres superuser – this bypasses Supabase RLS completely, so no 42501 errors.

Write path
----------
log_detection_event() / log_multi_detection_events() are fire-and-forget;
call them from FastAPI BackgroundTasks (they run in a thread executor).

Read path
---------
All get_*() helpers are sync. Wrap them with asyncio.to_thread() when
calling from async route handlers.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import db_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _serialize(row: dict) -> dict:
    """Convert any non-JSON-native values (datetime, Decimal, …) to strings."""
    out = {}
    for k, v in row.items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


def _rows(cur) -> list[dict[str, Any]]:
    return [_serialize(dict(r)) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

_UPSERT_SQL = """
    INSERT INTO detection_events
        (image_hash, recognizer, endpoint, is_flagged, label, score,
         all_predictions, scan_count, times_flagged)
    VALUES (%s, %s, %s, %s, %s, %s, %s, 1, %s)
    ON CONFLICT (image_hash, recognizer) WHERE image_hash IS NOT NULL
    DO UPDATE SET
        scan_count      = detection_events.scan_count + 1,
        times_flagged   = detection_events.times_flagged + EXCLUDED.times_flagged,
        is_flagged      = EXCLUDED.is_flagged,
        label           = EXCLUDED.label,
        score           = EXCLUDED.score,
        all_predictions = EXCLUDED.all_predictions,
        endpoint        = EXCLUDED.endpoint
"""


def _upsert_row(cur, *, image_hash, recognizer, endpoint, is_flagged, label, score, all_predictions):
    from psycopg2.extras import Json
    cur.execute(
        _UPSERT_SQL,
        (
            image_hash,
            recognizer,
            endpoint,
            is_flagged,
            label,
            round(float(score), 6),
            Json(all_predictions),
            1 if is_flagged else 0,   # times_flagged delta
        ),
    )


def log_detection_event(
    *,
    recognizer: str,
    endpoint: str,
    image_hash: str | None = None,
    is_flagged: bool,
    label: str,
    score: float,
    all_predictions: list[dict[str, Any]],
) -> None:
    """Persist / upsert a single detection result. Safe to call from a background task."""
    try:
        with db_client.cursor() as cur:
            if cur is None:
                return
            _upsert_row(
                cur,
                image_hash=image_hash,
                recognizer=recognizer,
                endpoint=endpoint,
                is_flagged=is_flagged,
                label=label,
                score=score,
                all_predictions=all_predictions,
            )
    except Exception:
        logger.exception(
            "Failed to log detection event (recognizer=%s, is_flagged=%s)",
            recognizer, is_flagged,
        )


def log_multi_detection_events(
    *,
    endpoint: str,
    image_hash: str | None = None,
    results: dict[str, Any],
) -> None:
    """Upsert detection events produced by the /detect/image endpoint."""
    if not results:
        return
    try:
        with db_client.cursor() as cur:
            if cur is None:
                return
            for recognizer, data in results.items():
                _upsert_row(
                    cur,
                    image_hash=image_hash,
                    recognizer=recognizer,
                    endpoint=endpoint,
                    is_flagged=data.is_target,
                    label=data.label,
                    score=data.score,
                    all_predictions=data.all_predictions,
                )
    except Exception:
        logger.exception("Failed to batch-log %d detection events.", len(results))


def flush_batch(items: list[dict]) -> None:
    """Write a list of queued detection events in a single transaction."""
    if not items:
        return
    try:
        with db_client.cursor() as cur:
            if cur is None:
                return
            for item in items:
                _upsert_row(cur, **item)
    except Exception:
        logger.exception("Failed to flush batch of %d detection events", len(items))


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def get_recognizer_overview() -> list[dict[str, Any]]:
    """All-time statistics aggregated per recognizer."""
    try:
        with db_client.cursor() as cur:
            if cur is None:
                return []
            cur.execute("SELECT * FROM recognizer_overview")
            return _rows(cur)
    except Exception:
        logger.exception("Failed to query recognizer_overview.")
        return []


def get_daily_summary(
    *,
    recognizer: str | None = None,
    days: int = 30,
) -> list[dict[str, Any]]:
    """Daily breakdown, optionally filtered by recognizer and time window."""
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    try:
        with db_client.cursor() as cur:
            if cur is None:
                return []
            if recognizer:
                cur.execute(
                    """
                    SELECT * FROM daily_detection_summary
                    WHERE day >= %s AND recognizer = %s
                    ORDER BY day DESC
                    """,
                    (cutoff, recognizer),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM daily_detection_summary
                    WHERE day >= %s
                    ORDER BY day DESC
                    """,
                    (cutoff,),
                )
            return _rows(cur)
    except Exception:
        logger.exception("Failed to query daily_detection_summary.")
        return []


def get_recent_hourly_activity() -> list[dict[str, Any]]:
    """Hourly scan and flag counts for the last 7 days."""
    try:
        with db_client.cursor() as cur:
            if cur is None:
                return []
            cur.execute("SELECT * FROM recent_hourly_activity")
            return _rows(cur)
    except Exception:
        logger.exception("Failed to query recent_hourly_activity.")
        return []


def get_score_distribution(
    recognizer: str | None = None,
) -> list[dict[str, Any]]:
    """Score distribution in 10 equal-width buckets, optionally per recognizer."""
    try:
        with db_client.cursor() as cur:
            if cur is None:
                return []
            if recognizer:
                cur.execute(
                    "SELECT * FROM score_distribution WHERE recognizer = %s ORDER BY bucket",
                    (recognizer,),
                )
            else:
                cur.execute("SELECT * FROM score_distribution ORDER BY recognizer, bucket")
            return _rows(cur)
    except Exception:
        logger.exception("Failed to query score_distribution.")
        return []


def get_high_risk_events(limit: int = 50) -> list[dict[str, Any]]:
    """Events where multiple recognizers flagged the same image simultaneously."""
    try:
        with db_client.cursor() as cur:
            if cur is None:
                return []
            cur.execute("SELECT * FROM high_risk_events LIMIT %s", (limit,))
            return _rows(cur)
    except Exception:
        logger.exception("Failed to query high_risk_events.")
        return []


def get_events(
    *,
    recognizer: str | None = None,
    flagged_only: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Paginated raw event log, most-recent first."""
    conditions: list[str] = []
    params: list[Any] = []

    if recognizer:
        conditions.append("recognizer = %s")
        params.append(recognizer)
    if flagged_only:
        conditions.append("is_flagged = TRUE")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.extend([limit, offset])

    try:
        with db_client.cursor() as cur:
            if cur is None:
                return []
            cur.execute(
                f"""
                SELECT id::text, created_at, recognizer, endpoint, is_flagged, label, score,
                       image_hash, scan_count, times_flagged
                FROM detection_events
                {where}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                params,
            )
            return _rows(cur)
    except Exception:
        logger.exception("Failed to query detection_events.")
        return []
