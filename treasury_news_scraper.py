#!/usr/bin/env python3
"""
U.S. Department of the Treasury news scraper.

Supports discovery and extraction for:
- Treasury featured stories
- Treasury press releases
- Treasury statements and remarks
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


TREASURY_FEATURED_STORIES_URL = "https://home.treasury.gov/news/featured-stories"
TREASURY_PRESS_RELEASES_URL = "https://home.treasury.gov/news/press-releases"
TREASURY_STATEMENTS_REMARKS_URL = "https://home.treasury.gov/news/press-releases/statements-remarks"

_SOURCE_CONFIG: Dict[str, Dict[str, Any]] = {
    "treasury_featured_story": {
        "default_url": TREASURY_FEATURED_STORIES_URL,
        "allowed_prefixes": ["/news/featured-stories/"],
        "listing_paths": ["/news/featured-stories"],
        "fallback_title": "Treasury Featured Story",
        "default_doc_type": "Featured Story",
    },
    "treasury_press_release": {
        "default_url": TREASURY_PRESS_RELEASES_URL,
        "allowed_prefixes": ["/news/press-releases/"],
        "listing_paths": ["/news/press-releases", "/news/press-releases/statements-remarks"],
        "fallback_title": "Treasury Press Release",
        "default_doc_type": "Press Release",
    },
    "treasury_statement_remark": {
        "default_url": TREASURY_STATEMENTS_REMARKS_URL,
        "allowed_prefixes": ["/news/press-releases/"],
        "listing_paths": ["/news/press-releases/statements-remarks"],
        "fallback_title": "Treasury Statement or Remark",
        "default_doc_type": "Statement",
    },
}


def _normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clean_multiline(text: Any) -> str:
    lines = []
    for raw in str(text or "").splitlines():
        line = _normalize_space(raw)
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _parse_date_text(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
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
        "%A, %B %d, %Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y-%m-%d",
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
        r"(\b\d{1,2}/\d{1,2}/\d{4}\b)",
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


def _title_from_url(url: Any, fallback: str) -> str:
    path = urlparse(str(url or "")).path
    slug = path.rstrip("/").rsplit("/", 1)[-1].strip()
    if not slug or slug.lower() in {"index.htm", "index.html"}:
        return fallback
    cleaned = re.sub(r"[-_]+", " ", slug).strip()
    if not cleaned:
        return fallback
    return " ".join(word.capitalize() for word in cleaned.split()[:24]).strip() or fallback


def _infer_treasury_doc_type(source_key: str, title: Any, url: Any = "", text: Any = "") -> str:
    if source_key == "treasury_featured_story":
        return "Featured Story"

    blob = " ".join(
        value for value in [_normalize_space(title), _normalize_space(url), _normalize_space(text)] if value
    ).lower()
    if "testimony" in blob:
        return "Testimony"
    if "remarks" in blob or "speech" in blob or "prepared remarks" in blob or "keynote" in blob:
        return "Remarks"
    if "readout" in blob:
        return "Readout"
    if "statement" in blob:
        return "Statement"
    if source_key == "treasury_statement_remark":
        return "Statement"
    return "Press Release"


def _speaker_from_title(title: Any) -> str:
    text = _normalize_space(title)
    if not text:
        return ""
    patterns = [
        r"^(?:remarks|statement|testimony|readout)\s+(?:by|from|of)\s+(.+?)(?::|,| on | at )",
        r"^(?:prepared remarks by)\s+(.+?)(?::|,| on | at )",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        speaker = _normalize_space(match.group(1))
        if speaker and len(speaker) <= 120:
            return speaker
    return ""


def _looks_like_speaker(text: Any) -> bool:
    value = _normalize_space(text)
    lower = value.lower()
    if not value or len(value) > 140:
        return False
    if lower.startswith(("page ", "next", "search", "filter", "featured stories", "press releases", "statements and remarks")):
        return False
    return bool(
        re.search(
            r"\b(secretary|deputy secretary|under secretary|assistant secretary|treasurer|chief|director)\b",
            value,
            flags=re.IGNORECASE,
        )
        or re.fullmatch(r"[A-Z][A-Za-z.\-']+(?:\s+[A-Z][A-Za-z.\-']+){1,10}", value)
    )


def _is_treasury_detail_url(url: Any, source_key: str) -> bool:
    raw = str(url or "").strip()
    if not raw:
        return False
    cfg = _SOURCE_CONFIG.get(source_key)
    if not cfg:
        return False
    parsed = urlparse(raw)
    if parsed.netloc and "home.treasury.gov" not in parsed.netloc.lower():
        return False
    path = (parsed.path or "").rstrip("/")
    if not path:
        return False
    lower_path = path.lower()
    allowed_prefixes = [str(item).rstrip("/").lower() for item in cfg.get("allowed_prefixes", [])]
    if not any(lower_path.startswith(prefix) for prefix in allowed_prefixes):
        return False
    listing_paths = [str(item).rstrip("/").lower() for item in cfg.get("listing_paths", [])]
    if lower_path in listing_paths or lower_path.endswith("/index.htm") or lower_path.endswith("/index.html"):
        return False
    return True


def _best_listing_container(anchor: Tag) -> Tag:
    candidate: Tag = anchor
    for parent in anchor.parents:
        if not isinstance(parent, Tag):
            continue
        if parent.name not in {"div", "li", "article", "tr", "section"}:
            continue
        anchors = [a for a in parent.select("a[href]") if isinstance(a, Tag)]
        if 1 <= len(anchors) <= 6:
            text = _normalize_space(parent.get_text(" ", strip=True))
            if len(text) >= max(20, len(_normalize_space(anchor.get_text(" ", strip=True))) + 8):
                candidate = parent
                break
    return candidate


def _extract_listing_fields(row: Tag, anchor: Tag, source_key: str) -> Dict[str, str]:
    title = _normalize_space(anchor.get_text(" ", strip=True))
    row_text = _clean_multiline(row.get_text("\n", strip=True))
    lines = [_normalize_space(line) for line in row_text.splitlines() if _normalize_space(line)]

    date_value = ""
    time_node = row.find("time")
    if time_node is not None:
        date_value = _date_to_display(time_node.get("datetime") or time_node.get_text(" ", strip=True))
    if not date_value:
        for line in lines[:8]:
            parsed = _parse_date_text(line)
            if parsed is not None:
                date_value = parsed.strftime("%B %d, %Y")
                break
    if not date_value:
        date_value = _extract_first_date(row_text)

    speaker = ""
    if source_key == "treasury_statement_remark":
        for line in lines:
            if line == title or line == date_value:
                continue
            if _looks_like_speaker(line):
                speaker = line
                break

    doc_type = _infer_treasury_doc_type(source_key, title, text=row_text)
    return {
        "title": title,
        "date": date_value,
        "speaker": speaker,
        "doc_type": doc_type,
    }


def _find_next_page_url(soup: BeautifulSoup, current_url: str) -> str:
    selectors = [
        "a[rel='next']",
        "li.pager__item--next a[href]",
        "a[title*='Go to next page']",
        "a[aria-label*='next page' i]",
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        if isinstance(node, Tag):
            href = _normalize_space(node.get("href", ""))
            if href:
                next_url = _url_without_query(urljoin(current_url, href))
                if next_url and _url_key(next_url) != _url_key(current_url):
                    return next_url

    for node in soup.select("a[href]"):
        if not isinstance(node, Tag):
            continue
        text = _normalize_space(node.get_text(" ", strip=True)).lower()
        aria = _normalize_space(node.get("aria-label", "")).lower()
        title = _normalize_space(node.get("title", "")).lower()
        if "next page" not in text and "next" not in text and "next page" not in aria and "next page" not in title:
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


class TreasuryNewsScraper:
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
        response.raise_for_status()
        return response

    def _discover_from_listing_page(self, page_url: str, source_key: str) -> Tuple[List[Dict[str, str]], str]:
        response = self._fetch_html(page_url, timeout=60)
        soup = BeautifulSoup(response.text, "html.parser")
        rows = soup.select("div.views-row, article, li.views-row, li, div.news-item, div.story-item, tr")

        out: List[Dict[str, str]] = []
        seen = set()
        for row in rows:
            if not isinstance(row, Tag):
                continue
            detail_anchor: Optional[Tag] = None
            for anchor in row.select("a[href]"):
                if not isinstance(anchor, Tag):
                    continue
                href = _normalize_space(anchor.get("href", ""))
                full_url = _url_without_query(urljoin("https://home.treasury.gov", href))
                if _is_treasury_detail_url(full_url, source_key):
                    detail_anchor = anchor
                    break
            if detail_anchor is None:
                continue

            detail_url = _url_without_query(urljoin("https://home.treasury.gov", detail_anchor.get("href", "")))
            key = _url_key(detail_url)
            if not key or key in seen:
                continue

            container = _best_listing_container(detail_anchor)
            fields = _extract_listing_fields(container, detail_anchor, source_key)
            if source_key == "treasury_press_release" and fields.get("doc_type", "") != "Press Release":
                continue
            seen.add(key)
            out.append(
                {
                    "url": detail_url,
                    "title": fields.get("title", "") or _title_from_url(detail_url, _SOURCE_CONFIG[source_key]["fallback_title"]),
                    "date": fields.get("date", ""),
                    "speaker": fields.get("speaker", ""),
                    "doc_type": fields.get("doc_type", ""),
                    "source_format": "html",
                    "listing_page": str(page_url or ""),
                }
            )

        if not out:
            for anchor in soup.select("a[href]"):
                if not isinstance(anchor, Tag):
                    continue
                href = _normalize_space(anchor.get("href", ""))
                detail_url = _url_without_query(urljoin("https://home.treasury.gov", href))
                if not _is_treasury_detail_url(detail_url, source_key):
                    continue
                key = _url_key(detail_url)
                if not key or key in seen:
                    continue
                title = _normalize_space(anchor.get_text(" ", strip=True))
                doc_type = _infer_treasury_doc_type(source_key, title, detail_url)
                if source_key == "treasury_press_release" and doc_type != "Press Release":
                    continue
                seen.add(key)
                out.append(
                    {
                        "url": detail_url,
                        "title": title or _title_from_url(detail_url, _SOURCE_CONFIG[source_key]["fallback_title"]),
                        "date": "",
                        "speaker": "",
                        "doc_type": doc_type,
                        "source_format": "html",
                        "listing_page": str(page_url or ""),
                    }
                )

        return out, _find_next_page_url(soup, str(getattr(response, "url", page_url) or page_url))

    def discover_documents(self, source_key: str, base_url: str = "", max_pages: int = 3) -> List[Dict[str, str]]:
        cfg = _SOURCE_CONFIG.get(str(source_key or "").strip())
        if not cfg:
            raise ValueError(f"Unsupported Treasury source_key: {source_key}")

        start_url = str(base_url or "").strip() or cfg["default_url"]
        max_pages = max(1, int(max_pages or 1))

        out: List[Dict[str, str]] = []
        seen = set()
        current_url = start_url
        pages_scanned = 0
        debug: Dict[str, Any] = {
            "source_key": str(source_key or ""),
            "base_url": start_url,
            "max_pages_requested": max_pages,
            "pages": [],
            "listing_added": 0,
            "total_unique": 0,
            "stop_reason": "",
        }

        while current_url and pages_scanned < max_pages:
            page_debug: Dict[str, Any] = {
                "page": pages_scanned + 1,
                "page_url": current_url,
                "returned_items": 0,
                "unique_added": 0,
                "next_page_url": "",
                "error_type": "",
                "error_status": 0,
                "error_message": "",
            }
            try:
                discovered, next_page = self._discover_from_listing_page(current_url, source_key)
                page_debug["returned_items"] = len(discovered)
                page_debug["next_page_url"] = next_page
                for item in discovered:
                    key = _url_key(item.get("url", ""))
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    out.append(item)
                    page_debug["unique_added"] += 1
                current_url = next_page
                pages_scanned += 1
                debug["pages"].append(page_debug)
                if not next_page:
                    debug["stop_reason"] = "pagination_exhausted"
                    break
            except requests.HTTPError as exc:
                page_debug["error_type"] = "HTTPError"
                page_debug["error_status"] = int(getattr(getattr(exc, "response", None), "status_code", 0) or 0)
                page_debug["error_message"] = str(exc)
                debug["pages"].append(page_debug)
                debug["stop_reason"] = f"http_error_{page_debug['error_status']}"
                self.last_discovery_debug = debug
                raise
            except Exception as exc:
                page_debug["error_type"] = type(exc).__name__
                page_debug["error_message"] = str(exc)
                debug["pages"].append(page_debug)
                debug["stop_reason"] = "error"
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
        fallback_speaker: str = "",
        fallback_doc_type: str = "",
    ) -> Dict[str, Any]:
        response = self._fetch_html(url, timeout=75)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup.find_all(["script", "style", "noscript", "svg"]):
            tag.decompose()

        canonical_url = ""
        canonical_link = soup.find("link", rel="canonical")
        if isinstance(canonical_link, Tag):
            canonical_url = _normalize_space(canonical_link.get("href", ""))
        og_url = soup.find("meta", attrs={"property": "og:url"})
        if not canonical_url and isinstance(og_url, Tag):
            canonical_url = _normalize_space(og_url.get("content", ""))
        final_url = canonical_url or _url_without_query(str(getattr(response, "url", url) or url))

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
            title = _normalize_space(fallback_title) or _title_from_url(final_url, "Treasury Document")

        date_text = ""
        for selector in ("time", "[datetime]", ".published-date", ".date", "p.date"):
            node = soup.select_one(selector)
            if not isinstance(node, Tag):
                continue
            date_text = _normalize_space(node.get("datetime") or node.get_text(" ", strip=True))
            if _parse_date_text(date_text) is not None:
                break
        if not date_text:
            date_text = _extract_first_date(soup.get_text("\n", strip=True))
        date_value = _date_to_display(date_text or fallback_date)

        candidates: List[Tag] = []
        seen_candidates = set()
        for selector in (
            "main article",
            "article",
            "main",
            "div.treasury-content",
            "div.field--name-body",
            "div.node__content",
            "div.region-content",
            "body",
        ):
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
        for selector in (
            "nav",
            "header",
            "footer",
            "aside",
            "form",
            ".breadcrumb",
            ".pager",
            ".pagination",
            ".share",
            ".social-share",
            ".social-links",
            ".region-sidebar-first",
            ".region-sidebar-second",
            ".menu",
            ".related-links",
            ".view-filters",
        ):
            for node in content_node.select(selector):
                node.decompose()

        full_text = _clean_multiline(content_node.get_text("\n")) if isinstance(content_node, Tag) else ""
        if not full_text:
            full_text = _clean_multiline(soup.get_text("\n"))

        speaker = ""
        for selector in (".byline", "p.byline", ".field--name-field-speaker", ".field--name-field-author"):
            node = soup.select_one(selector)
            if not isinstance(node, Tag):
                continue
            speaker = _normalize_space(node.get_text(" ", strip=True))
            if speaker:
                break
        if not speaker:
            speaker = _speaker_from_title(title)
        if not speaker:
            speaker = _normalize_space(fallback_speaker)

        doc_type = _normalize_space(fallback_doc_type) or _infer_treasury_doc_type(
            "treasury_statement_remark" if "/statements-remarks" in final_url.lower() else "treasury_press_release",
            title,
            final_url,
            full_text[:1200],
        )

        return {
            "success": True,
            "data": {
                "url": final_url,
                "title": title,
                "date": date_value,
                "speaker": speaker,
                "doc_type": doc_type,
                "full_text": full_text,
                "word_count": len(full_text.split()),
                "source_format": "html",
            },
        }
