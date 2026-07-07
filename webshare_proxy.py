"""Helpers for Webshare rotating residential proxy configuration."""

from __future__ import annotations

import os
from typing import Dict, Optional
from urllib.parse import quote


def webshare_rotating_proxy_url() -> str:
    username = str(os.getenv("WEBSHARE_PROXY_USERNAME") or "").strip()
    password = str(os.getenv("WEBSHARE_PROXY_PASSWORD") or "").strip()
    if not username or not password:
        return ""

    proxy_username = username if username.endswith("-rotate") else f"{username}-rotate"
    return f"http://{quote(proxy_username, safe='')}:{quote(password, safe='')}@p.webshare.io:80"


def webshare_rotating_proxies() -> Optional[Dict[str, str]]:
    proxy_url = webshare_rotating_proxy_url()
    if not proxy_url:
        return None
    return {"http": proxy_url, "https": proxy_url}


def should_retry_with_webshare(status_code: object) -> bool:
    try:
        status = int(status_code)
    except (TypeError, ValueError):
        return False
    return status in {403, 429}
