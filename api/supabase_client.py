"""Supabase client singleton.

The client is initialised lazily on first use. If SUPABASE_URL or
SUPABASE_KEY are absent the helper returns None and all stats
operations become silent no-ops, so the API still works without a DB.
"""
from __future__ import annotations

import base64
import json
import logging
import os

import env_loader

logger = logging.getLogger(__name__)

_client = None


def _extract_jwt_role(token: str) -> str | None:
    """Best-effort extraction of role from legacy JWT-style Supabase API keys."""
    parts = token.split(".")
    if len(parts) != 3:
        return None

    payload = parts[1]
    padding = "=" * ((4 - (len(payload) % 4)) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload + padding)
        obj = json.loads(raw.decode("utf-8"))
        role = obj.get("role")
        if isinstance(role, str) and role:
            return role
        return None
    except Exception:
        return None


def _is_non_privileged_key(key: str) -> bool:
    """Return True when the key cannot bypass RLS for server-side writes."""
    if key.startswith("sb_publishable_"):
        return True

    role = _extract_jwt_role(key)
    if role in {"anon", "authenticated"}:
        return True

    return False


def get_client():
    """Return a lazy-initialised Supabase Client, or None if unconfigured."""
    global _client
    if _client is not None:
        return _client

    # Load .env once before reading variables.
    env_loader.load_environment()

    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()

    if not url or not key:
        logger.warning(
            "SUPABASE_URL or SUPABASE_KEY "
            "not configured - detection events will not be persisted."
        )
        return None

    if _is_non_privileged_key(key):
        logger.error(
            "Configured Supabase key is publishable/anon and cannot write to RLS "
            "protected tables. Configure SUPABASE_KEY with the service-role "
            "secret key."
        )
        return None

    if key.startswith("sb_secret_"):
        logger.info("Supabase service key format detected (sb_secret_*).")

    if key.startswith("eyJ"):
        role = _extract_jwt_role(key)
        if role and role != "service_role":
            logger.error(
                "Configured JWT-style Supabase key role is '%s'; expected "
                "'service_role' for server-side writes.",
                role,
            )
            return None

    try:
        from supabase import create_client
        _client = create_client(url, key)
        logger.info("Supabase client ready (project: %s)", url)
        return _client
    except Exception:
        logger.exception("Failed to initialise Supabase client.")
        return None


def reset_client() -> None:
    """Force re-initialisation on next use (useful for testing)."""
    global _client
    _client = None
