"""API key authentication.

Keys are configured via the API_KEYS env var (comma-separated). When the list
is empty, auth is disabled (development only). Keys are hashed before being
stored on jobs so they can be queried per-tenant without exposing the raw key.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from fastapi import Depends, Header

from .errors import UnauthorizedError
from .settings import Settings, get_settings


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


async def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> Optional[str]:
    """Validate the X-API-Key header. Returns the SHA-256 of the key, or None if auth is disabled."""
    keys = settings.api_key_set
    if not keys:
        return None
    if not x_api_key or x_api_key not in keys:
        raise UnauthorizedError("Invalid or missing API key")
    return _hash_key(x_api_key)
