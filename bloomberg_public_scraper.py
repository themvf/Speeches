#!/usr/bin/env python3
"""Public Bloomberg news connector.

Discovers Bloomberg articles from public RSS feeds and extracts publicly
available article text. If Bloomberg only exposes metadata or a short summary,
the connector returns that summary so downstream workflows can still track the
item without bypassing access controls.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


BLOOMBERG_DEFAULT_FEEDS = [
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://feeds.bloomberg.com/business/news.rss",
    "https://feeds.bloomberg.com/economics/news.rss",
    "https://feeds.bloomberg.com/industries/news.rss",
    "https://feeds.bloomberg.com/technology/news.rss",
    "https://feeds.bloomberg.com/wealth/news.rss",
    "https://feeds.bloomberg.com/politics/news.rss",
    "https://feeds.bloomberg.com/crypto/news.rss",
    "https://feeds.bloomberg.com/green/news.rss",
]

_SECTION_FEED_BY_SLUG = {
    "markets": "https://feeds.bloomberg.com/markets/news.rss",
    "business": "https://feeds.bloomberg.com/business/news.rss",
    "economics": "https://feeds.bloomberg.com/economics/news.rss",
    "industries": "https://feeds.bloomberg.com/industries/news.rss",
    "technology": "https://feeds.bloomberg.com/technology/news.rss",
    "wealth": "https://feeds.bloomberg.com/wealth/news.rss",
    "politics": "https://feeds.bloomberg.com/politics/news.rss",
    "crypto": "https://feeds.bloomberg.com/crypto/news.rss",
    "green": "https://feeds.bloomberg.com/green/news.rss",
}


def _normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_multiline(value: Any) -> str:
    lines = [_normalize_space(line) for line in str(value or "").splitlines()]
    return "\n\n".join(line for line in lines if line).strip()


def _xml_local_name(tag: str) -> str:
    raw = str(tag or "")
    return raw.rsplit("}", 1)[-1] if "}" in raw else raw


def _strip_html(value: Any) -> str:
    try:
        return _normalize_space(BeautifulSoup(str(value or ""), "html.parser").get_text(" ", strip=True))
    except Exception:
        return _normalize_space(re.sub(r"<[^>]+>", " ", str(value or "")))


def _parse_date_text(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).replace(tzinfo=None)
    except Exception:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=None)
        except ValueError:
            continue
    try:
        parsed = parsedate_to_datetime(text)
        if parsed is not None:
            return parsed.replace(tzinfo=None)
    except Exception:
        pass
    return None


def _date_to_display(value: Any) -> str:
    parsed = _parse_date_text(value)
    return parsed.strftime("%B %d, %Y") if parsed is not None else str(value or "").strip()


def _url_key(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path.rstrip('/') or '/'}"


def _is_feed_url(url: str) -> bool:
    parsed = urlparse(str(url or "").strip())
    path = parsed.path.lower()
    return parsed.netloc.lower() == "feeds.bloomberg.com" or path.endswith((".rss", ".xml", "/feed"))


def _is_bloomberg_article_url(url: str) -> bool:
    parsed = urlparse(str(url or "").strip())
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host != "bloomberg.com":
        return False
    path = parsed.path.lower()
    if not path or path == "/":
        return False
    return not any(part in path for part in ("/feeds/", "/authors/", "/podcasts/", "/videos/"))


def _section_feed_from_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    parts = [part.lower() for part in parsed.path.split("/") if part]
    for part in parts:
        if part in _SECTION_FEED_BY_SLUG:
            return _SECTION_FEED_BY_SLUG[part]
    return ""


def _looks_like_access_limited(html_text: str, status_code: int = 200, final_url: str = "") -> bool:
    blob = " ".join([str(final_url or ""), str(html_text or "")]).lower()
    if status_code in (401, 403):
        return True
    markers = (
        "subscribe to continue",
        "subscribe to read",
        "sign in to continue",
        "sign in to read",
        "for subscribers",
        "are you a robot",
        "captcha",
        "please enable cookies",
        "access denied",
        "__cf_bm",
        "cf-chl-",
    )
    return any(marker in blob for marker in markers)


def _meta_content(soup: BeautifulSoup, attrs: Dict[str, str]) -> str:
    node = soup.find("meta", attrs=attrs)
    return _normalize_space(node.get("content", "")) if node and node.get("content") else ""


def _best_article_text(soup: BeautifulSoup) -> str:
    selectors = [
        "article",
        '[data-component="ArticleBody"]',
        '[data-testid="article-body"]',
        '[itemprop="articleBody"]',
        "main",
    ]
    best = ""
    best_words = 0
    for selector in selectors:
        for node in soup.select(selector):
            paragraphs = [
                _normalize_space(child.get_text(" ", strip=True))
                for child in node.select("p, li")
                if len(_normalize_space(child.get_text(" ", strip=True)).split()) >= 5
            ]
            text = "\n\n".join(paragraphs) if paragraphs else _clean_multiline(node.get_text("\n"))
            words = len(text.split())
            if words > best_words:
                best = text
                best_words = words
    return best if best_words >= 25 else ""


class BloombergPublicNewsScraper:
    def __init__(self, min_delay_seconds: float = 0.5) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self.min_delay_seconds = max(0.0, float(min_delay_seconds))
        self._last_request_ts = 0.0
        self.last_discovery_debug: Dict[str, Any] = {}

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_delay_seconds:
            time.sleep(self.min_delay_seconds - elapsed)
        self._last_request_ts = time.time()

    def _fetch(self, url: str, timeout: int = 45) -> requests.Response:
        target = str(url or "").strip()
        if not target:
            raise ValueError("URL is required")
        self._rate_limit()
        response = self.session.get(target, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        return response

    def _feed_urls_for_base(self, base_url: str) -> List[str]:
        target = str(base_url or "").strip()
        if not target:
            return list(BLOOMBERG_DEFAULT_FEEDS)
        if _is_feed_url(target):
            return [target]
        section_feed = _section_feed_from_url(target)
        if section_feed:
            return [section_feed]
        return []

    def _parse_feed(self, feed_url: str, max_items: int) -> List[Dict[str, Any]]:
        resp = self._fetch(feed_url, timeout=45)
        root = ET.fromstring(str(resp.text or "").lstrip("\ufeff").strip())
        results: List[Dict[str, Any]] = []

        for item in root.iter():
            if _xml_local_name(item.tag).lower() != "item":
                continue

            entry: Dict[str, Any] = {"feed_url": feed_url}
            categories: List[str] = []
            for child in item:
                local = _xml_local_name(child.tag)
                text = _normalize_space(child.text or "")
                if local == "title" and not entry.get("title"):
                    entry["title"] = text
                elif local == "link" and not entry.get("url"):
                    entry["url"] = text or _normalize_space(child.get("href", ""))
                elif local in {"pubDate", "published", "updated"} and not entry.get("date"):
                    entry["date"] = _date_to_display(text)
                elif local in {"description", "summary"} and not entry.get("summary"):
                    entry["summary"] = _strip_html(child.text or "")
                elif local in {"creator", "author"} and not entry.get("author"):
                    entry["author"] = text
                elif local == "category" and text:
                    categories.append(text)
                elif local == "guid" and not entry.get("guid"):
                    entry["guid"] = text

            url = _normalize_space(entry.get("url", ""))
            title = _normalize_space(entry.get("title", ""))
            if not url or not title:
                continue
            entry["keywords"] = categories
            entry["source"] = "Bloomberg"
            results.append(entry)
            if len(results) >= max_items:
                break

        return results

    def discover_documents(self, *, base_url: str = "", max_pages: int = 25) -> List[Dict[str, Any]]:
        limit = max(1, int(max_pages or 25))
        target = str(base_url or "").strip()
        debug: Dict[str, Any] = {
            "mode": "public_bloomberg",
            "base_url": target,
            "feed_urls": [],
            "errors": [],
        }

        if target and _is_bloomberg_article_url(target):
            doc = self.extract_document(target)
            data = doc.get("data", {}) if isinstance(doc.get("data", {}), dict) else {}
            result = {
                "url": target,
                "title": data.get("title", "") or target,
                "date": data.get("date", ""),
                "authors": [data.get("author", "")] if data.get("author") else [],
                "keywords": [],
                "summary": data.get("summary", ""),
                "full_text": data.get("full_text", ""),
                "source": "Bloomberg",
                "extraction_mode": data.get("extraction_mode", ""),
                "access_limited": data.get("access_limited", False),
            }
            self.last_discovery_debug = {**debug, "article_url": target, "items_found": 1}
            return [result]

        feed_urls = self._feed_urls_for_base(target)
        debug["feed_urls"] = feed_urls
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        per_feed_limit = limit
        for feed_url in feed_urls:
            try:
                for item in self._parse_feed(feed_url, per_feed_limit):
                    key = _url_key(str(item.get("url", "")))
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    out.append(item)
                    if len(out) >= limit:
                        break
            except Exception as exc:
                debug["errors"].append(f"{feed_url}: {exc}")
            if len(out) >= limit:
                break

        debug["items_found"] = len(out)
        self.last_discovery_debug = debug
        return out

    def extract_document(
        self,
        url: str,
        fallback_title: str = "",
        fallback_date: str = "",
        fallback_summary: str = "",
        fallback_author: str = "",
    ) -> Dict[str, Any]:
        target = str(url or "").strip()
        if not target:
            return {"success": False, "error": "No URL provided", "data": {}}

        title = _normalize_space(fallback_title)
        date_text = _normalize_space(fallback_date)
        summary = _normalize_space(fallback_summary)
        author = _normalize_space(fallback_author)
        full_text = ""
        extraction_mode = "metadata_fallback"
        access_limited = False
        status_code = 0

        try:
            resp = self._fetch(target, timeout=45)
            status_code = resp.status_code
            html_text = resp.text
            access_limited = _looks_like_access_limited(html_text, status_code, resp.url)
            soup = BeautifulSoup(html_text, "html.parser")

            title = title or _meta_content(soup, {"property": "og:title"}) or _meta_content(soup, {"name": "twitter:title"})
            date_text = (
                date_text
                or _meta_content(soup, {"property": "article:published_time"})
                or _meta_content(soup, {"name": "pubdate"})
                or _meta_content(soup, {"name": "date"})
            )
            summary = (
                summary
                or _meta_content(soup, {"property": "og:description"})
                or _meta_content(soup, {"name": "description"})
                or _meta_content(soup, {"name": "twitter:description"})
            )
            author = author or _meta_content(soup, {"name": "author"}) or _meta_content(soup, {"property": "article:author"})

            if not access_limited:
                candidate = _best_article_text(soup)
                if candidate:
                    full_text = candidate
                    extraction_mode = "html_body"
        except Exception:
            access_limited = True

        if not full_text and summary:
            full_text = summary
            extraction_mode = "summary_fallback"

        if not full_text and title:
            full_text = title
            extraction_mode = "title_fallback"

        if not full_text:
            return {"success": False, "error": "No usable public text extracted", "data": {}}

        return {
            "success": True,
            "data": {
                "url": target,
                "title": title,
                "date": _date_to_display(date_text),
                "full_text": full_text,
                "summary": summary,
                "author": author,
                "source_name": "Bloomberg",
                "source_format": "html",
                "extraction_mode": extraction_mode,
                "access_limited": access_limited,
                "status_code": status_code,
                "final_url": target,
            },
        }
