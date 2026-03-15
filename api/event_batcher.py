"""Background event batcher – buffers detection events and flushes to DB periodically.

Instead of acquiring a DB connection per request, detection endpoints call
enqueue() (non-blocking) and a single background thread drains the queue every
FLUSH_INTERVAL seconds using one connection for the whole batch.
"""
from __future__ import annotations

import logging
import queue
import threading
import time

import stats_service

logger = logging.getLogger(__name__)

FLUSH_INTERVAL = 2.0  # seconds between flushes

_queue: queue.Queue[dict] = queue.Queue()
_thread: threading.Thread | None = None
_running = False


def enqueue(event: dict) -> None:
    """Add a detection event dict to the buffer (non-blocking, thread-safe)."""
    _queue.put_nowait(event)


def _drain() -> list[dict]:
    items: list[dict] = []
    while True:
        try:
            items.append(_queue.get_nowait())
        except queue.Empty:
            break
    return items


def _flush() -> None:
    items = _drain()
    if items:
        stats_service.flush_batch(items)


def _worker() -> None:
    while _running:
        time.sleep(FLUSH_INTERVAL)
        try:
            _flush()
        except Exception:
            logger.exception("Event batcher flush failed")
    # Final drain after stop() sets _running = False
    try:
        _flush()
    except Exception:
        logger.exception("Event batcher final flush failed")


def start() -> None:
    global _thread, _running
    _running = True
    _thread = threading.Thread(target=_worker, daemon=True, name="event-batcher")
    _thread.start()
    logger.info("Event batcher started (flush_interval=%.1fs)", FLUSH_INTERVAL)


def stop() -> None:
    global _running
    _running = False
    if _thread:
        _thread.join(timeout=10)
    logger.info("Event batcher stopped")
