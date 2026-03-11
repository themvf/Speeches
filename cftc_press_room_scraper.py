#!/usr/bin/env python3
"""
CFTC Press Room scraper.

Supports discovery and extraction for:
- CFTC press releases
- CFTC public statements and remarks
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


CFTC_PRESS_RELEASES_URL = "https://www.cftc.gov/PressRoom/PressReleases"
CFTC_SPEECHES_TESTIMONY_URL = "https://www.cftc.gov/PressRoom/SpeechesTestimony/index.htm"

_SOURCE_CONFIG: Dict[str, Dict[str, str]] = {
    "cftc_press_release": {
        "default_url": CFTC_PRESS_RELEASES_URL,
        "path_prefix": "/PressRoom/PressReleases/",
        "listing_path": "/PressRoom/PressReleases",
        "fallback_title": "CFTC Press Release",
    },
    "cftc_public_statement_remark": {
        "default_url": CFTC_SPEECHES_TESTIMONY_URL,
        "path_prefix": "/PressRoom/SpeechesTestimony/",
        "listing_path": "/PressRoom/SpeechesTestimony",
        "fallback_title": "CFTC Public Statement or Remark",
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


def _is_cftc_detail_url(url: Any, source_key: str) -> bool:
    raw = str(url or "").strip()
    if not raw:
        return False
    cfg = _SOURCE_CONFIG.get(source_key)
    if not cfg:
        return False
    parsed = urlparse(raw)
    if parsed.netloc and "cftc.gov" not in parsed.netloc.lower():
        return False
    path = (parsed.path or "").rstrip("/")
    if not path:
        return False
    prefix = str(cfg.get("path_prefix", "") or "").rstrip("/")
    listing_path = str(cfg.get("listing_path", "") or "").rstrip("/")
    lower_path = path.lower()
    if not lower_path.startswith(prefix.lower()):
        return False
    if lower_path in {listing_path.lower(), f"{listing_path.lower()}/index.htm", f"{listing_path.lower()}/index.html"}:
        return False
    return True


def _title_from_url(url: Any, fallback: str) -> str:
    path = urlparse(str(url or "")).path
    slug = path.rstrip("/").rsplit("/", 1)[-1].strip()
    if not slug or slug.lower() in {"index.htm", "index.html"}:
        return fallback
    cleaned = re.sub(r"[-_]+", " ", slug).strip()
    if not cleaned:
        return fallback
    if re.fullmatch(r"\d{4}-\d{2}", cleaned):
        return fallback
    return " ".join(word.capitalize() for word in cleaned.split()[:20]).strip() or fallback


def _infer_public_statement_doc_type(title: Any, url: Any = "", text: Any = "") -> str:
    blob = " ".join(
        value for value in [_normalize_space(title), _normalize_space(url), _normalize_space(text)] if value
    ).lower()
    if not blob:
        return "Statement"
    if "testimony" in blob:
        return "Testimony"
    if "remarks" in blob or "speech" in blob or "keynote" in blob or "prepared for delivery" in blob:
        return "Remarks"
    if (
        "statement" in blob
        or "op-ed" in blob
        or "opening statement" in blob
        or "concurring statement" in blob
        or "dissenting statement" in blob
    ):
        return "Statement"
    return "Statement"


def _speaker_from_title(title: Any) -> str:
    text = _normalize_space(title)
    if not text:
        return ""
    patterns = [
        r"^(?:remarks|speech|testimony|statement|opening statement|concurring statement|dissenting statement)\s+(?:of\s+)?(.+?)(?::|,| regarding | on | at )",
        r"^(?:cftc\s+)?(.+?)(?::|,)\s*(?:op-ed|chairman|commissioner|director)",
        r"^(.+?)\s*:\s*(?:op-ed|remarks|statement)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        speaker = _normalize_space(match.group(1))
        if speaker and len(speaker) <= 120:
            return speaker
    return ""


def _looks_like_listing_speaker(text: Any) -> bool:
    value = _normalize_space(text)
    lower = value.lower()
    if not value:
        return False
    if len(value) > 120:
        return False
    if lower in {
        "press releases",
        "public statements & remarks",
        "pagination",
        "date press releases",
        "date public statements & remarks",
    }:
        return False
    if lower.startswith(("page ", "next page", "last page", "keyword", "type", "by year")):
        return False
    return bool(
        re.search(r"\b(chairman|commissioner|director|acting chairman|chief|office)\b", value, flags=re.IGNORECASE)
        or re.fullmatch(r"[A-Z][A-Za-z.\-']+(?:\s+[A-Z][A-Za-z.\-']+){1,8}", value)
    )


def _best_listing_container(anchor: Tag) -> Tag:
    candidate: Tag = anchor
    for parent in anchor.parents:
        if not isinstance(parent, Tag):
            continue
        if parent.name not in {"div", "li", "article", "tr", "section"}:
            continue
        anchors = [a for a in parent.select("a[href]") if isinstance(a, Tag)]
        if 1 <= len(anchors) <= 4:
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
        for line in lines[:6]:
            parsed = _parse_date_text(line)
            if parsed is not None:
                date_value = parsed.strftime("%B %d, %Y")
                break
    if not date_value:
        date_value = _extract_first_date(row_text)

    speaker = ""
    if source_key == "cftc_public_statement_remark":
        for line in lines:
            if line == title or line == date_value:
                continue
            if re.fullmatch(r"\d{4}-\d{2}", line):
                continue
            if _looks_like_listing_speaker(line):
                speaker = line
                break

    doc_type = "Press Release" if source_key == "cftc_press_release" else _infer_public_statement_doc_type(title, text=row_text)
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
        "li.pager-next a[href]",
        "a[title*='Go to next page']",
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
    text = _normalize_space(node.get_text(" ", strip=True))
    return len(text.split())


class CFTCPressRoomScraper:
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
        rows = soup.select("div.views-row, article, li.views-row, tr")

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
                full_url = _url_without_query(urljoin("https://www.cftc.gov", href))
                if _is_cftc_detail_url(full_url, source_key):
                    detail_anchor = anchor
                    break
            if detail_anchor is None:
                continue

            detail_url = _url_without_query(urljoin("https://www.cftc.gov", detail_anchor.get("href", "")))
            key = _url_key(detail_url)
            if not key or key in seen:
                continue
            seen.add(key)

            container = _best_listing_container(detail_anchor)
            fields = _extract_listing_fields(container, detail_anchor, source_key)
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
                detail_url = _url_without_query(urljoin("https://www.cftc.gov", href))
                if not _is_cftc_detail_url(detail_url, source_key):
                    continue
                key = _url_key(detail_url)
                if not key or key in seen:
                    continue
                seen.add(key)
                title = _normalize_space(anchor.get_text(" ", strip=True))
                out.append(
                    {
                        "url": detail_url,
                        "title": title or _title_from_url(detail_url, _SOURCE_CONFIG[source_key]["fallback_title"]),
                        "date": "",
                        "speaker": "",
                        "doc_type": "Press Release" if source_key == "cftc_press_release" else _infer_public_statement_doc_type(title, detail_url),
                        "source_format": "html",
                        "listing_page": str(page_url or ""),
                    }
                )

        return out, _find_next_page_url(soup, str(getattr(response, "url", page_url) or page_url))

    def discover_documents(
        self,
        source_key: str,
        base_url: str = "",
        max_pages: int = 3,
    ) -> List[Dict[str, str]]:
        cfg = _SOURCE_CONFIG.get(str(source_key or "").strip())
        if not cfg:
            raise ValueError(f"Unsupported CFTC source_key: {source_key}")

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
            title = _normalize_space(fallback_title) or _title_from_url(final_url, "CFTC Document")

        date_text = ""
        for selector in ("time", "[datetime]", "div.field--name-field-date", "span.date-display-single", "p.date"):
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
            "div.field--name-body",
            "div.node__content",
            "div.layout-content",
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
            ".views-exposed-form",
            ".related-links",
        ):
            for node in content_node.select(selector):
                node.decompose()

        full_text = _clean_multiline(content_node.get_text("\n")) if isinstance(content_node, Tag) else ""
        if not full_text:
            full_text = _clean_multiline(soup.get_text("\n"))

        speaker = ""
        for selector in (
            ".field--name-field-speaker",
            ".field--name-field-author",
            ".field--name-field-person",
            ".byline",
            "p.byline",
        ):
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

        location = ""
        h1_node = soup.select_one("main h1, article h1, h1")
        if isinstance(h1_node, Tag):
            next_heading = h1_node.find_next(["h2", "h3", "p"])
            if isinstance(next_heading, Tag):
                candidate = _normalize_space(next_heading.get_text(" ", strip=True))
                if candidate and candidate.lower() not in {"public statements & remarks", title.lower()} and len(candidate) <= 180:
                    location = candidate

        doc_type = _normalize_space(fallback_doc_type)
        if not doc_type:
            if "/pressroom/pressreleases/" in final_url.lower():
                doc_type = "Press Release"
            else:
                doc_type = _infer_public_statement_doc_type(title, final_url, full_text[:1200])

        return {
            "success": True,
            "data": {
                "url": final_url,
                "title": title,
                "date": date_value,
                "speaker": speaker,
                "location": location,
                "doc_type": doc_type,
                "full_text": full_text,
                "word_count": len(full_text.split()),
                "source_format": "html",
            },
        }
