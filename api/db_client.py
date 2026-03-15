"""PostgreSQL connection pool backed by DATABASE_URL.

Connects directly as the postgres superuser, which bypasses Supabase's
Row-Level Security entirely – correct for a trusted server-side process.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from threading import Lock
from typing import Generator

import env_loader

logger = logging.getLogger(__name__)

_pool = None
_lock = Lock()


def _get_pool():
    global _pool
    if _pool is not None:
        return _pool

    with _lock:
        if _pool is not None:
            return _pool

        env_loader.load_environment()
        url = os.getenv("DATABASE_URL", "").strip()

        if not url:
            logger.warning(
                "DATABASE_URL not configured – DB operations unavailable."
            )
            return None

        try:
            import psycopg2.pool
            _pool = psycopg2.pool.ThreadedConnectionPool(minconn=2, maxconn=20, dsn=url)
            logger.info("PostgreSQL connection pool ready.")
            return _pool
        except Exception:
            logger.exception("Failed to create PostgreSQL connection pool.")
            return None


def get_connection():
    pool = _get_pool()
    if pool is None:
        return None
    return pool.getconn()


def release_connection(conn) -> None:
    pool = _get_pool()
    if pool is not None and conn is not None:
        pool.putconn(conn)


@contextmanager
def cursor() -> Generator:
    """Yield a RealDictCursor, or None if the DB is unavailable.

    Commits on clean exit, rolls back on exception, always releases the
    connection back to the pool.
    """
    conn = get_connection()
    if conn is None:
        yield None
        return

    cur = None
    try:
        import psycopg2.extras
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        if cur is not None:
            cur.close()
        release_connection(conn)


def reset_pool() -> None:
    """Close all pool connections and force re-creation on next use (testing)."""
    global _pool
    with _lock:
        if _pool is not None:
            try:
                _pool.closeall()
            except Exception:
                pass
        _pool = None
