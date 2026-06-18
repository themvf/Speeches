#!/usr/bin/env python3
"""Apify-backed Bloomberg article connector."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests


APIFY_API_BASE = "https://api.apify.com/v2"
DEFAULT_ACTOR_ID = "xtracto/bloomberg-news-article-scraper"


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _walk_values(value: Any) -> List[Any]:
    value = _maybe_json(value)
    out = [value]
    if isinstance(value, dict):
        for child in value.values():
            out.extend(_walk_values(child))
    elif isinstance(value, list):
        for child in value:
            out.extend(_walk_values(child))
    return out


def _find_key_value(value: Any, keys: List[str]) -> Any:
    normalized_keys = {key.lower().replace("_", "").replace("-", "") for key in keys}
    value = _maybe_json(value)
    if isinstance(value, dict):
        for key, child in value.items():
            key_norm = str(key).lower().replace("_", "").replace("-", "")
            if key_norm in normalized_keys:
                return _maybe_json(child)
        for child in value.values():
            found = _find_key_value(child, keys)
            if found not in (None, "", [], {}):
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_key_value(child, keys)
            if found not in (None, "", [], {}):
                return found
    return None


def _first_text(item: Dict[str, Any], keys: List[str], *, deep: bool = True) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return _clean(value)
        if isinstance(value, (int, float)):
            return str(value)
    if deep:
        value = _find_key_value(item, keys)
        if isinstance(value, str) and value.strip():
            return _clean(value)
        if isinstance(value, (int, float)):
            return str(value)
    return ""


def _coerce_list_text(value: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, dict):
        raw_items = list(value.values())
    elif isinstance(value, str) and value.strip():
        raw_items = [part.strip() for part in re.split(r"[,;|]+", value)]
    else:
        raw_items = []
    for raw in raw_items:
        if isinstance(raw, dict):
            candidate = _clean(raw.get("name") or raw.get("title") or raw.get("text") or raw.get("value") or raw.get("label"))
        else:
            candidate = _clean(raw)
        key_norm = candidate.lower()
        if candidate and key_norm not in seen:
            seen.add(key_norm)
            out.append(candidate)
    return out


def _first_list_text(item: Dict[str, Any], keys: List[str]) -> List[str]:
    for key in keys:
        values = _coerce_list_text(item.get(key))
        if values:
            return values
    value = _find_key_value(item, keys)
    return _coerce_list_text(value)


def _request_url(item: Dict[str, Any]) -> str:
    for key in ["url", "canonicalUrl", "articleUrl", "pageUrl", "link", "sourceUrl", "article_url"]:
        value = _first_text(item, [key])
        if value:
            return value
    request = item.get("request")
    if isinstance(request, dict):
        value = _first_text(request, ["loadedUrl", "url"])
        if value:
            return value
    return ""


def _cap_int_field(payload: Dict[str, Any], key: str, limit: int) -> None:
    raw = payload.get(key)
    try:
        current = int(raw) if raw is not None and str(raw).strip() else None
    except (TypeError, ValueError):
        current = None
    if current is None or current <= 0 or current > limit:
        payload[key] = limit


def _body_text(item: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in [
        "articleText",
        "article_text",
        "fullText",
        "full_text",
        "text",
        "content",
        "body",
        "markdown",
        "articleBody",
        "article_body",
        "bodyText",
        "body_text",
        "paragraphs",
        "description",
        "summary",
        "contentText",
        "content_text",
    ]:
        value = item.get(key)
        if value in (None, "", [], {}):
            value = _find_key_value(item, [key])
        if isinstance(value, str) and value.strip():
            cleaned = value.strip()
            if cleaned not in parts:
                parts.append(cleaned)
        elif isinstance(value, list):
            text = "\n".join(_clean(part) for part in value if _clean(part))
            if text and text not in parts:
                parts.append(text)
    return "\n\n".join(parts).strip()


def _sanitize_debug(value: Any, depth: int = 0) -> Any:
    value = _maybe_json(value)
    if depth > 3:
        return "[truncated]"
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, child in list(value.items())[:40]:
            key_text = str(key)
            if re.search(r"token|secret|password|proxy|cookie|auth|credential", key_text, re.I):
                out[key_text] = "[redacted]"
            else:
                out[key_text] = _sanitize_debug(child, depth + 1)
        return out
    if isinstance(value, list):
        return [_sanitize_debug(child, depth + 1) for child in value[:5]]
    if isinstance(value, str):
        return value[:500] + ("..." if len(value) > 500 else "")
    return value


class ApifyBloombergNewsScraper:
    """Runs the configured Apify actor and normalizes dataset items."""

    def __init__(
        self,
        *,
        api_token: Optional[str] = None,
        actor_id: Optional[str] = None,
        proxy_url: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        self.api_token = _clean(api_token or os.getenv("APIFY_TOKEN", ""))
        self.actor_id = _clean(actor_id or os.getenv("APIFY_BLOOMBERG_ACTOR_ID", "") or DEFAULT_ACTOR_ID)
        self.proxy_url = _clean(proxy_url or os.getenv("APIFY_PROXY_URL", ""))
        self.timeout_seconds = int(timeout_seconds or os.getenv("APIFY_TIMEOUT_SECONDS", "900") or "900")
        self.last_discovery_debug: Dict[str, Any] = {}

    @property
    def actor_api_id(self) -> str:
        return self.actor_id.replace("/", "~")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    def _build_input(self, base_url: str, max_pages: int) -> Dict[str, Any]:
        raw = os.getenv("APIFY_BLOOMBERG_INPUT_JSON", "").strip()
        if raw:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise RuntimeError("APIFY_BLOOMBERG_INPUT_JSON must be a JSON object.")
        else:
            payload = {}

        target_url = _clean(base_url)
        if target_url:
            payload.setdefault("startUrls", [{"url": target_url}])
            payload.setdefault("urls", [target_url])

        if max_pages > 0:
            _cap_int_field(payload, "maxItems", max_pages)
            _cap_int_field(payload, "maxPages", max_pages)
            _cap_int_field(payload, "limit", max_pages)

        if self.proxy_url:
            payload.setdefault(
                "proxyConfiguration",
                {
                    "useApifyProxy": False,
                    "proxyUrls": [self.proxy_url],
                },
            )

        return payload

    def _request_json(self, method: str, url: str, **kwargs: Any) -> Dict[str, Any]:
        response = requests.request(method, url, headers=self._headers(), timeout=60, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f"Apify API returned {response.status_code}: {response.text[:500]}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Apify API returned an unexpected response.")
        return payload

    def _start_run(self, actor_input: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{APIFY_API_BASE}/actors/{self.actor_api_id}/runs"
        payload = self._request_json("POST", url, json=actor_input)
        data = payload.get("data")
        if not isinstance(data, dict) or not data.get("id"):
            raise RuntimeError("Apify run response did not include a run id.")
        return data

    def _wait_for_run(self, run_id: str) -> Dict[str, Any]:
        deadline = time.time() + self.timeout_seconds
        terminal = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}
        last: Dict[str, Any] = {}
        while time.time() < deadline:
            payload = self._request_json("GET", f"{APIFY_API_BASE}/actor-runs/{run_id}")
            data = payload.get("data")
            if isinstance(data, dict):
                last = data
                status = str(data.get("status") or "")
                if status in terminal:
                    if status != "SUCCEEDED":
                        raise RuntimeError(f"Apify actor run ended with status {status}.")
                    return data
            time.sleep(5)
        raise RuntimeError(f"Timed out waiting for Apify actor run {run_id}. Last status: {last.get('status', 'unknown')}")

    def _fetch_dataset_items(self, dataset_id: str) -> List[Dict[str, Any]]:
        response = requests.get(
            f"{APIFY_API_BASE}/datasets/{dataset_id}/items",
            headers=self._headers(),
            params={"clean": "true"},
            timeout=120,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Apify dataset fetch returned {response.status_code}: {response.text[:500]}")
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError("Apify dataset endpoint returned an unexpected response.")
        return [item for item in payload if isinstance(item, dict)]

    def discover_documents(self, *, base_url: str, max_pages: int) -> List[Dict[str, Any]]:
        if not self.api_token:
            raise RuntimeError("APIFY_TOKEN is not configured.")

        actor_input = self._build_input(base_url, max_pages)
        run = self._start_run(actor_input)
        completed = self._wait_for_run(str(run["id"]))
        dataset_id = _clean(completed.get("defaultDatasetId") or run.get("defaultDatasetId"))
        if not dataset_id:
            raise RuntimeError("Apify actor run did not return a default dataset id.")

        raw_items = self._fetch_dataset_items(dataset_id)
        docs = [self._normalize_item(item, idx) for idx, item in enumerate(raw_items, 1)]
        docs = [doc for doc in docs if doc.get("url") or doc.get("title")]
        sample = raw_items[0] if raw_items else {}
        self.last_discovery_debug = {
            "actor_id": self.actor_id,
            "run_id": str(run.get("id", "")),
            "dataset_id": dataset_id,
            "raw_item_count": len(raw_items),
            "normalized_count": len(docs),
            "used_custom_input": bool(os.getenv("APIFY_BLOOMBERG_INPUT_JSON", "").strip()),
            "used_custom_proxy": bool(self.proxy_url),
            "sample_item_keys": sorted(str(key) for key in sample.keys()) if isinstance(sample, dict) else [],
            "sample_item_preview": _sanitize_debug(sample) if isinstance(sample, dict) else {},
            "sample_normalized_preview": _sanitize_debug(docs[0]) if docs else {},
        }
        return docs

    def _normalize_item(self, item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        title = _first_text(item, ["title", "headline", "name", "articleTitle", "article_title"])
        url = _request_url(item)
        text = _body_text(item)
        date = _first_text(
            item,
            [
                "publishedAt",
                "published_at",
                "publishedDate",
                "published_date",
                "datePublished",
                "date_published",
                "date",
                "publishDate",
                "publish_date",
                "time",
                "timestamp",
            ],
        )
        authors = _first_list_text(item, ["authors", "author", "byline", "bylines"])
        keywords = _first_list_text(item, ["keywords", "tags", "topics", "categories", "section"])
        summary = _first_text(item, ["summary", "description", "dek", "subheadline", "subtitle"])
        return {
            "url": url,
            "title": title or f"Bloomberg article {idx}",
            "date": date,
            "authors": authors,
            "keywords": keywords,
            "summary": summary,
            "full_text": text,
            "source": _first_text(item, ["source", "siteName", "publisher"]) or "Bloomberg",
            "raw_item": item,
        }
