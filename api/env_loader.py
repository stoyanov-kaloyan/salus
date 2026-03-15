"""Utilities for loading environment variables from .env files."""
from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency at import time
    load_dotenv = None

_lock = Lock()
_loaded = False
_loaded_path: str | None = None


def load_environment() -> str | None:
    """Load env vars from the first matching .env file and return its path.

    Search order:
    1) Current working directory
    2) Directory containing this module (proxy-poc)
    3) Parent of this module directory (workspace root)
    """
    global _loaded, _loaded_path

    with _lock:
        if _loaded:
            return _loaded_path

        _loaded = True

        if load_dotenv is None:
            logger.warning(
                "python-dotenv is not installed; .env files will not be auto-loaded."
            )
            return None

        module_dir = Path(__file__).resolve().parent
        candidates = [
            Path.cwd() / ".env",
            module_dir / ".env",
            module_dir.parent / ".env",
        ]

        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            if resolved.is_file():
                load_dotenv(dotenv_path=resolved, override=False)
                _loaded_path = str(resolved)
                logger.info("Loaded environment variables from %s", resolved)
                return _loaded_path

        logger.warning(
            "No .env file found in expected locations: %s",
            ", ".join(str(path) for path in candidates),
        )
        return None
