#!/usr/bin/env python3
"""
WSJ / Dow Jones RSS scraper (feeds.content.dowjones.io).

Discovers articles from any Dow Jones RSS feed URL and extracts article
metadata. Full text is fetched when possible; falls back to the RSS
description when the article is paywalled or inaccessible.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


WSJ_DEFAULT_FEED_URL = "https://feeds.content.dowjones.io/public/rss/WSJcomUSBusinessNews"

WSJ_RSS_FEEDS: Dict[str, Dict[str, Any]] = {
    "wsj_us_business": {
        "label": "WSJ US Business News",
        "feed_url": "https://feeds.content.dowjones.io/public/rss/WSJcomUSBusinessNews",
        "tags_csv": "wsj,business,financial-news",
    },
    "wsj_markets": {
        "label": "WSJ Markets",
        "feed_url": "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
        "tags_csv": "wsj,markets,financial-news",
    },
    "wsj_opinion": {
        "label": "WSJ Opinion",
        "feed_url": "https://feeds.content.dowjones.io/public/rss/RSSOpinion",
        "tags_csv": "wsj,opinion,editorial",
    },
    "mw_top_stories": {
        "label": "MarketWatch Top Stories",
        "feed_url": "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
        "tags_csv": "marketwatch,markets,financial-news",
    },
    "wsj_custom": {
        "label": "Custom Feed URL",
        "feed_url": WSJ_DEFAULT_FEED_URL,
        "tags_csv": "wsj,financial-news",
    },
}


def _normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clean_multiline(text: Any) -> str:
    lines = [_normalize_space(line) for line in str(text or "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _xml_local_name(tag: str) -> str:
    raw = str(tag or "")
    return raw.rsplit("}", 1)[-1] if "}" in raw else raw


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
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt)
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
    return parsed.strftime("%B %d, %Y") if parsed else str(value or "").strip()


def _strip_html(text: str) -> str:
    try:
        return BeautifulSoup(str(text or ""), "html.parser").get_text(" ").strip()
    except Exception:
        return _normalize_space(re.sub(r"<[^>]+>", " ", str(text or "")))


_BOILERPLATE_PATTERNS = (
    r"^(updated|published)\s*(on)?\s*[:\-]?\s*.+$",
    r"^by\s+[a-z][a-z .'-]{1,60}$",
    r"^[|:\-•]+$",
)


def _is_boilerplate(line: str) -> bool:
    text = _normalize_space(line).lower()
    if not text or text in {"by", "updated", "published", "read more", "advertisement"}:
        return True
    return any(re.fullmatch(p, text) for p in _BOILERPLATE_PATTERNS)


def _best_article_text(soup: BeautifulSoup) -> str:
    selectors = [
        "article",
        '[itemprop="articleBody"]',
        "div.article-content",
        "div.wsj-article-body",
        "div.article__body",
        "div.articleBody",
        "main",
    ]
    best_text = ""
    best_words = 0
    for selector in selectors:
        for node in soup.select(selector):
            paragraphs = [
                _normalize_space(p.get_text(" ", strip=True))
                for p in node.select("p, li")
                if len(_normalize_space(p.get_text(" ", strip=True)).split()) >= 5
            ]
            candidate = "\n\n".join(paragraphs) if paragraphs else _clean_multiline(node.get_text("\n"))
            words = len(candidate.split())
            if words > best_words:
                best_text = candidate
                best_words = words
    if best_words >= 80:
        return best_text
    body = soup.body or soup
    lines = [
        _normalize_space(line)
        for line in body.get_text("\n").splitlines()
        if _normalize_space(line) and not _is_boilerplate(_normalize_space(line))
    ]
    return "\n\n".join(lines)


def _looks_like_paywall(html_text: str, status_code: int = 200) -> bool:
    if status_code in (401, 403):
        return True
    blob = str(html_text or "").lower()
    markers = (
        "subscribe to continue reading",
        "subscribe now",
        "this content is for subscribers",
        "already a subscriber",
        "sign in to read",
        "log in to read",
        "premium content",
        "wsj.com/subscribe",
    )
    return any(m in blob for m in markers)


class WSJRssScraper:
    def __init__(self, min_delay_seconds: float = 0.5):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self.min_delay_seconds = max(0.0, float(min_delay_seconds))
        self._last_request_ts = 0.0
        self.last_discovery_debug: Dict[str, Any] = {}

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_delay_seconds:
            time.sleep(self.min_delay_seconds - elapsed)
        self._last_request_ts = time.time()

    def _fetch(self, url: str, timeout: int = 30) -> requests.Response:
        self._rate_limit()
        resp = self.session.get(str(url or "").strip(), timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        return resp

    def discover_documents(
        self,
        feed_url: str = WSJ_DEFAULT_FEED_URL,
        max_items: int = 50,
    ) -> List[Dict[str, Any]]:
        feed_url = str(feed_url or WSJ_DEFAULT_FEED_URL).strip()
        debug: Dict[str, Any] = {"feed_url": feed_url, "items_found": 0, "errors": []}

        try:
            resp = self._fetch(feed_url)
            raw_xml = resp.text
        except Exception as e:
            debug["errors"].append(f"Feed fetch failed: {e}")
            self.last_discovery_debug = debug
            return []

        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError as e:
            debug["errors"].append(f"XML parse error: {e}")
            self.last_discovery_debug = debug
            return []

        items = root.findall(".//item")
        if not items:
            items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

        results: List[Dict[str, Any]] = []
        for item in items[:max_items]:
            entry: Dict[str, str] = {}
            for child in item:
                local = _xml_local_name(child.tag)
                text = _normalize_space(child.text or "")
                if local == "title" and not entry.get("title"):
                    entry["title"] = text
                elif local == "link" and not entry.get("url"):
                    entry["url"] = text or _normalize_space(child.get("href", ""))
                elif local == "pubDate" and not entry.get("date"):
                    entry["date"] = _date_to_display(text)
                elif local in ("description", "summary") and not entry.get("description"):
                    entry["description"] = _strip_html(text or child.text or "")
                elif local == "guid" and not entry.get("guid"):
                    entry["guid"] = text
                elif local == "creator" and not entry.get("author"):
                    entry["author"] = text
                elif local in ("name", "author") and not entry.get("author"):
                    entry["author"] = text

            url = str(entry.get("url", "") or "").strip()
            title = str(entry.get("title", "") or "").strip()
            if not url or not title:
                continue

            results.append({
                "source_url": url,
                "title": title,
                "date": str(entry.get("date", "") or "").strip(),
                "description": str(entry.get("description", "") or "").strip(),
                "author": str(entry.get("author", "") or "").strip(),
                "guid": str(entry.get("guid", "") or "").strip(),
            })

        debug["items_found"] = len(results)
        self.last_discovery_debug = debug
        return results

    def extract_document(
        self,
        url: str,
        fallback_title: str = "",
        fallback_date: str = "",
        fallback_description: str = "",
        fallback_author: str = "",
    ) -> Dict[str, Any]:
        url = str(url or "").strip()
        if not url:
            return {"success": False, "error": "No URL provided", "data": {}}

        full_text = ""
        extraction_mode = "rss_description"
        status_code = 0

        try:
            resp = self._fetch(url, timeout=30)
            status_code = resp.status_code
            html_text = resp.text

            if _looks_like_paywall(html_text, status_code):
                extraction_mode = "rss_description"
            else:
                soup = BeautifulSoup(html_text, "html.parser")
                candidate = _best_article_text(soup)
                if len(candidate.split()) >= 50:
                    full_text = candidate
                    extraction_mode = "html_body"

                if not full_text:
                    for meta_attrs in [
                        {"name": "description"},
                        {"property": "og:description"},
                        {"name": "twitter:description"},
                    ]:
                        tag = soup.find("meta", attrs=meta_attrs)
                        if tag and tag.get("content"):
                            full_text = _normalize_space(tag["content"])
                            extraction_mode = "meta_description"
                            break

                if not full_text:
                    for meta_attrs in [{"property": "og:title"}, {"name": "title"}]:
                        tag = soup.find("meta", attrs=meta_attrs)
                        if tag and tag.get("content"):
                            fallback_title = fallback_title or _normalize_space(tag["content"])
                            break

                for meta_attrs in [
                    {"name": "author"},
                    {"property": "article:author"},
                    {"name": "byl"},
                ]:
                    tag = soup.find("meta", attrs=meta_attrs)
                    if tag and tag.get("content"):
                        fallback_author = fallback_author or _normalize_space(tag["content"])
                        break

        except Exception as e:
            extraction_mode = "rss_description"

        if not full_text and fallback_description:
            full_text = str(fallback_description or "").strip()

        if not full_text:
            return {"success": False, "error": "No usable text extracted", "data": {}}

        parsed = urlparse(url)
        source_name = parsed.netloc.lstrip("www.").split(".")[0].capitalize()

        return {
            "success": True,
            "data": {
                "url": url,
                "title": fallback_title,
                "date": fallback_date,
                "full_text": full_text,
                "author": fallback_author,
                "source_name": source_name,
                "source_format": "html",
                "extraction_mode": extraction_mode,
                "status_code": status_code,
            },
        }
