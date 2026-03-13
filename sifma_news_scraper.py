#!/usr/bin/env python3
"""
SIFMA news scraper.

This connector targets the public SIFMA news surface at https://www.sifma.org/news
and extracts the latest visible cards from the rendered listing page.

SIFMA is fronted by a Vercel checkpoint. When the site serves the checkpoint
instead of content, discovery/extraction raises a clear error and stores debug
details so the UI can surface the block reason instead of silently failing.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag


SIFMA_HOME_URL = "https://www.sifma.org"
SIFMA_NEWS_URL = f"{SIFMA_HOME_URL}/news"

_SIFMA_LISTING_PATHS = {
    "/news",
    "/news/blog",
    "/news/press-releases",
    "/news/speeches",
    "/news/podcasts",
}

_SIFMA_DETAIL_PREFIXES = (
    "/news/blog/",
    "/news/press-releases/",
    "/news/speeches/",
    "/news/podcasts/",
)


def _normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clean_multiline(text: Any) -> str:
    lines: List[str] = []
    for raw in str(text or "").splitlines():
        line = _normalize_space(raw)
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _parse_date_text(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is not None:
            return parsed.replace(tzinfo=None)
        return parsed
    except ValueError:
        pass
    text = (
        text.replace("Jan.", "Jan")
        .replace("Feb.", "Feb")
        .replace("Mar.", "Mar")
        .replace("Apr.", "Apr")
        .replace("Jun.", "Jun")
        .replace("Jul.", "Jul")
        .replace("Aug.", "Aug")
        .replace("Sep.", "Sep")
        .replace("Sept.", "Sep")
        .replace("Oct.", "Oct")
        .replace("Nov.", "Nov")
        .replace("Dec.", "Dec")
    )
    for fmt in (
        "%b %d, %Y",
        "%B %d, %Y",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
    ):
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
    if parsed is None:
        return str(value or "").strip()
    return parsed.strftime("%B %d, %Y")


def _extract_first_date(text: Any) -> str:
    blob = str(text or "")
    patterns = [
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|May|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)"
        r"\s+\d{1,2},\s+\d{4})",
        r"(\b\d{4}-\d{2}-\d{2}\b)",
    ]
    for pattern in patterns:
        match = re.search(pattern, blob, flags=re.IGNORECASE)
        if match:
            return _date_to_display(match.group(1))
    return ""


def _url_without_query(url: Any) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _url_key(url: Any) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}"


def _title_from_url(url: Any, fallback: str = "SIFMA News Item") -> str:
    path = urlparse(str(url or "")).path
    slug = path.rstrip("/").rsplit("/", 1)[-1].strip()
    if not slug:
        return fallback
    words = [part for part in re.split(r"[-_]+", slug) if part]
    if not words:
        return fallback
    return " ".join(words[:20]).strip().title() or fallback


def _is_sifma_checkpoint_html(html: Any, status_code: int = 0) -> bool:
    text = str(html or "")
    if "Vercel Security Checkpoint" in text:
        return True
    if status_code == 403 and "Please enable JavaScript and cookies" in text:
        return True
    return False


def _is_sifma_detail_url(url: Any) -> bool:
    raw = str(url or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw)
    if parsed.netloc and "sifma.org" not in parsed.netloc.lower():
        return False
    path = (parsed.path or "").rstrip("/")
    if not path:
        return False
    lower_path = path.lower()
    if lower_path in _SIFMA_LISTING_PATHS:
        return False
    return any(lower_path.startswith(prefix.rstrip("/").lower()) for prefix in _SIFMA_DETAIL_PREFIXES)


def _infer_sifma_doc_type(title: Any, category: Any = "", url: Any = "", text: Any = "") -> str:
    category_blob = _normalize_space(category).lower()
    title_blob = " ".join(value for value in [_normalize_space(title), _normalize_space(url)] if value).lower()
    text_blob = _normalize_space(text).lower()
    blob = " ".join(value for value in [category_blob, title_blob, text_blob] if value)
    if "press release" in blob or "/news/press-releases/" in title_blob:
        return "Press Release"
    if "speech" in blob or "/news/speeches/" in title_blob:
        return "Speech"
    if "podcast" in blob or "/news/podcasts/" in title_blob:
        return "Podcast"
    if "pennsylvania + wall" in blob or "/news/blog/" in title_blob or "blog" in category_blob:
        return "Blog Post"
    return "News Item"


def _best_listing_container(anchor: Tag) -> Tag:
    candidate: Tag = anchor
    for parent in anchor.parents:
        if not isinstance(parent, Tag):
            continue
        if parent.name not in {"div", "article", "li", "section"}:
            continue
        detail_urls = set()
        for link in parent.select("a[href]"):
            href = _normalize_space(link.get("href", ""))
            full_url = _url_without_query(urljoin(SIFMA_HOME_URL, href))
            if _is_sifma_detail_url(full_url):
                detail_urls.add(_url_key(full_url))
        if 1 <= len(detail_urls) <= 2:
            text = _normalize_space(parent.get_text(" ", strip=True))
            if len(text) >= 20:
                candidate = parent
                break
    return candidate


def _extract_listing_fields(row: Tag, anchor: Tag, detail_url: str) -> Dict[str, str]:
    heading = row.select_one("h1, h2, h3, h4")
    title = (
        _normalize_space(anchor.get("aria-label", ""))
        or _normalize_space(heading.get_text(" ", strip=True) if isinstance(heading, Tag) else "")
        or _normalize_space(anchor.get_text(" ", strip=True))
        or _title_from_url(detail_url)
    )

    row_text = _clean_multiline(row.get_text("\n", strip=True))
    lines = [_normalize_space(line) for line in row_text.splitlines() if _normalize_space(line)]

    date_value = ""
    time_node = row.find("time")
    if isinstance(time_node, Tag):
        date_value = _date_to_display(time_node.get("datetime") or time_node.get_text(" ", strip=True))
    if not date_value:
        for line in lines[:8]:
            parsed = _parse_date_text(line)
            if parsed is not None:
                date_value = parsed.strftime("%B %d, %Y")
                break
    if not date_value:
        date_value = _extract_first_date(row_text)

    category = ""
    for line in lines[:8]:
        lower = line.lower()
        if line in {title, date_value}:
            continue
        if _parse_date_text(line) is not None:
            continue
        if lower in {"featured posts", "news", "subscribe"}:
            continue
        if len(line) > 80:
            continue
        category = line
        break

    topics = []
    seen_topics = set()
    for link in row.select("a[href*='/issues/'], a[href*='/resources/']"):
        topic = _normalize_space(link.get_text(" ", strip=True))
        if not topic or topic in {title, category} or topic in seen_topics:
            continue
        seen_topics.add(topic)
        topics.append(topic)

    return {
        "title": title,
        "date": date_value,
        "category": category,
        "doc_type": _infer_sifma_doc_type(title, category, detail_url, row_text),
        "topics": ", ".join(topics),
    }


def _find_next_page_url(soup: BeautifulSoup, current_url: str) -> str:
    selectors = [
        "a[rel='next']",
        "a[aria-label*='next' i]",
        "a[title*='next' i]",
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        if isinstance(node, Tag):
            href = _normalize_space(node.get("href", ""))
            if not href:
                continue
            next_url = _url_without_query(urljoin(current_url, href))
            if next_url and _url_key(next_url) != _url_key(current_url):
                return next_url

    for node in soup.select("a[href]"):
        if not isinstance(node, Tag):
            continue
        text = _normalize_space(node.get_text(" ", strip=True)).lower()
        if text not in {"next", "next page", "older", "older posts", "more"}:
            continue
        href = _normalize_space(node.get("href", ""))
        if not href:
            continue
        next_url = _url_without_query(urljoin(current_url, href))
        if next_url and _url_key(next_url) != _url_key(current_url):
            return next_url
    return ""


def _content_score(node: Optional[Tag]) -> int:
    if not isinstance(node, Tag):
        return -1
    return len(_normalize_space(node.get_text(" ", strip=True)).split())


class SIFMANewsScraper:
    def __init__(self, min_delay_seconds: float = 0.8):
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

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_delay_seconds:
            time.sleep(self.min_delay_seconds - elapsed)
        self._last_request_ts = time.time()

    def _fetch_html(self, url: str, timeout: int = 60) -> requests.Response:
        target = str(url or "").strip()
        if not target:
            raise ValueError("URL is required")
        self._rate_limit()
        response = self.session.get(target, timeout=timeout, allow_redirects=True)
        return response

    @staticmethod
    def _raise_for_checkpoint(response: requests.Response, context: str):
        html = str(getattr(response, "text", "") or "")
        status_code = int(getattr(response, "status_code", 0) or 0)
        if _is_sifma_checkpoint_html(html, status_code):
            raise RuntimeError(
                f"SIFMA blocked automated access for {context} with the Vercel Security Checkpoint "
                f"(HTTP {status_code or 'unknown'})."
            )
        response.raise_for_status()

    def _discover_from_listing_page(self, page_url: str) -> Tuple[List[Dict[str, str]], str, Dict[str, Any]]:
        response = self._fetch_html(page_url, timeout=60)
        page_debug: Dict[str, Any] = {
            "page_url": page_url,
            "final_url": str(getattr(response, "url", page_url) or page_url),
            "status_code": int(getattr(response, "status_code", 0) or 0),
            "checkpoint_blocked": False,
        }
        try:
            self._raise_for_checkpoint(response, "listing discovery")
        except Exception:
            page_debug["checkpoint_blocked"] = _is_sifma_checkpoint_html(response.text, page_debug["status_code"])
            raise

        soup = BeautifulSoup(response.text, "html.parser")
        out: List[Dict[str, str]] = []
        seen = set()

        for anchor in soup.select("a[href]"):
            if not isinstance(anchor, Tag):
                continue
            href = _normalize_space(anchor.get("href", ""))
            detail_url = _url_without_query(urljoin(SIFMA_HOME_URL, href))
            if not _is_sifma_detail_url(detail_url):
                continue
            key = _url_key(detail_url)
            if not key or key in seen:
                continue

            container = _best_listing_container(anchor)
            fields = _extract_listing_fields(container, anchor, detail_url)
            seen.add(key)
            out.append(
                {
                    "url": detail_url,
                    "title": fields.get("title", "") or _title_from_url(detail_url),
                    "date": fields.get("date", ""),
                    "category": fields.get("category", ""),
                    "doc_type": fields.get("doc_type", ""),
                    "topics": fields.get("topics", ""),
                    "source_format": "html",
                    "listing_page": str(page_url or ""),
                }
            )

        page_debug["returned_items"] = len(out)
        next_page = _find_next_page_url(soup, str(getattr(response, "url", page_url) or page_url))
        page_debug["next_page_url"] = next_page
        return out, next_page, page_debug

    def discover_documents(self, base_url: str = SIFMA_NEWS_URL, max_pages: int = 1) -> List[Dict[str, str]]:
        start_url = str(base_url or "").strip() or SIFMA_NEWS_URL
        max_pages = max(1, int(max_pages or 1))

        out: List[Dict[str, str]] = []
        seen = set()
        current_url = start_url
        pages_scanned = 0
        debug: Dict[str, Any] = {
            "base_url": start_url,
            "max_pages_requested": max_pages,
            "pages": [],
            "listing_added": 0,
            "total_unique": 0,
            "checkpoint_blocked": False,
            "stop_reason": "",
        }

        while current_url and pages_scanned < max_pages:
            try:
                discovered, next_page, page_debug = self._discover_from_listing_page(current_url)
                page_debug["page"] = pages_scanned + 1
                page_debug["unique_added"] = 0
                for item in discovered:
                    key = _url_key(item.get("url", ""))
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    out.append(item)
                    page_debug["unique_added"] += 1
                debug["pages"].append(page_debug)
                current_url = next_page
                pages_scanned += 1
                if not next_page:
                    debug["stop_reason"] = "pagination_exhausted"
                    break
            except Exception as exc:
                page_debug = {
                    "page": pages_scanned + 1,
                    "page_url": current_url,
                    "returned_items": 0,
                    "unique_added": 0,
                    "next_page_url": "",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "checkpoint_blocked": "Vercel Security Checkpoint" in str(exc),
                }
                debug["pages"].append(page_debug)
                debug["checkpoint_blocked"] = bool(page_debug["checkpoint_blocked"])
                debug["stop_reason"] = "checkpoint_blocked" if debug["checkpoint_blocked"] else "error"
                self.last_discovery_debug = debug
                raise

        if not debug["stop_reason"]:
            debug["stop_reason"] = "completed"

        def _sort_key(item: Dict[str, str]):
            return _parse_date_text(item.get("date", "")) or datetime.min

        out.sort(key=_sort_key, reverse=True)
        debug["listing_added"] = int(len(out))
        debug["total_unique"] = int(len(out))
        self.last_discovery_debug = debug
        return out

    def extract_document(
        self,
        url: str,
        fallback_title: str = "",
        fallback_date: str = "",
        fallback_category: str = "",
        fallback_doc_type: str = "",
    ) -> Dict[str, Any]:
        response = self._fetch_html(url, timeout=75)
        self._raise_for_checkpoint(response, "detail extraction")
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup.find_all(["script", "style", "noscript", "svg"]):
            tag.decompose()

        title = ""
        for selector in ("main h1", "article h1", "h1", "meta[property='og:title']"):
            node = soup.select_one(selector)
            if not isinstance(node, Tag):
                continue
            if node.name == "meta":
                title = _normalize_space(node.get("content", ""))
            else:
                title = _normalize_space(node.get_text(" ", strip=True))
            if title:
                break
        if not title:
            title = _normalize_space(fallback_title) or _title_from_url(url)

        page_text = _clean_multiline(soup.get_text("\n", strip=True))
        date_text = ""
        for selector in ("time", "[datetime]", "meta[property='article:published_time']"):
            node = soup.select_one(selector)
            if not isinstance(node, Tag):
                continue
            if node.name == "meta":
                date_text = _normalize_space(node.get("content", ""))
            else:
                date_text = _normalize_space(node.get("datetime") or node.get_text(" ", strip=True))
            if _parse_date_text(date_text) is not None:
                break
        if not date_text:
            date_text = _extract_first_date(page_text)
        date_value = _date_to_display(date_text or fallback_date)

        category = ""
        for selector in (
            "main [class*='text-label']",
            "main [class*='eyebrow']",
            "article [class*='text-label']",
            "article [class*='eyebrow']",
        ):
            for node in soup.select(selector):
                if not isinstance(node, Tag):
                    continue
                value = _normalize_space(node.get_text(" ", strip=True))
                if not value or value == title or value == date_value:
                    continue
                if len(value) > 80:
                    continue
                if _parse_date_text(value) is not None:
                    continue
                category = value
                break
            if category:
                break
        if not category:
            category = _normalize_space(fallback_category)

        candidates: List[Tag] = []
        seen_candidates = set()
        for selector in ("main article", "article", "main", "div.rich-text", "body"):
            for node in soup.select(selector):
                if not isinstance(node, Tag):
                    continue
                ident = id(node)
                if ident in seen_candidates:
                    continue
                seen_candidates.add(ident)
                candidates.append(node)
        best_node = max(candidates, key=_content_score) if candidates else soup.body or soup
        content_soup = BeautifulSoup(str(best_node), "html.parser")
        content_node = content_soup.find()

        if isinstance(content_node, Tag):
            for selector in (
                "nav",
                "header",
                "footer",
                "aside",
                "form",
                "button",
                "svg",
                "[aria-label*='newsletter' i]",
                "[class*='newsletter']",
                "[class*='subscribe']",
                "[class*='share']",
                "[class*='social']",
            ):
                for node in content_node.select(selector):
                    node.decompose()

        full_text = _clean_multiline(content_node.get_text("\n")) if isinstance(content_node, Tag) else ""
        if not full_text:
            full_text = page_text

        doc_type = _normalize_space(fallback_doc_type) or _infer_sifma_doc_type(title, category, url, full_text[:1200])
        return {
            "success": True,
            "data": {
                "url": _url_without_query(str(getattr(response, "url", url) or url)),
                "title": title,
                "date": date_value,
                "category": category,
                "doc_type": doc_type,
                "full_text": full_text,
                "word_count": len(full_text.split()),
                "source_format": "html",
            },
        }
