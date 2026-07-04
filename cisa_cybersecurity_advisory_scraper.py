#!/usr/bin/env python3
"""
CISA cybersecurity alerts and advisories scraper.

Discovery uses CISA's official cybersecurity advisories RSS feed plus the
public Cybersecurity Alerts & Advisories listing page. Extraction pulls full
text from CISA detail pages such as cybersecurity advisories, ICS advisories,
medical advisories, and KEV catalog alerts.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag


CISA_HOME_URL = "https://www.cisa.gov"
CISA_CYBERSECURITY_ADVISORIES_URL = f"{CISA_HOME_URL}/news-events/cybersecurity-advisories"
CISA_CYBERSECURITY_ADVISORIES_RSS_URL = f"{CISA_HOME_URL}/cybersecurity-advisories/all.xml"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36 PolicyResearchHub/1.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

DETAIL_PATH_PREFIXES = (
    "/news-events/cybersecurity-advisories/",
    "/news-events/ics-advisories/",
    "/news-events/ics-medical-advisories/",
    "/news-events/alerts/",
)

MONTH_RE = (
    r"January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?"
)


def _normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_multiline(value: Any) -> str:
    lines: List[str] = []
    for raw in str(value or "").splitlines():
        line = _normalize_space(raw)
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _parse_date_text(value: Any) -> Optional[datetime]:
    text = _normalize_space(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        return parsed.replace(tzinfo=None)
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
        return _normalize_space(value)
    return parsed.strftime("%B %d, %Y")


def _extract_first_date(value: Any) -> str:
    blob = str(value or "")
    patterns = [
        rf"((?:{MONTH_RE})\s+\d{{1,2}},\s+\d{{4}})",
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
    raw = _url_without_query(url)
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}"


def _xml_local_name(tag: Any) -> str:
    raw = str(tag or "")
    if "}" in raw:
        return raw.rsplit("}", 1)[-1]
    return raw


def _xml_child_text(parent: ET.Element, local_name: str) -> str:
    target = str(local_name or "").strip().lower()
    for child in list(parent):
        if _xml_local_name(child.tag).lower() != target:
            continue
        text = _normalize_space(child.text or "")
        if text:
            return text
    return ""


def _is_cisa_detail_url(url: Any) -> bool:
    raw = str(url or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw)
    if parsed.netloc and parsed.netloc.lower() != "www.cisa.gov":
        return False
    path = (parsed.path or "").rstrip("/")
    if not path:
        return False
    lower_path = path.lower()
    if lower_path in {"/news-events/cybersecurity-advisories", "/news-events/alerts"}:
        return False
    return any(lower_path.startswith(prefix.rstrip("/")) for prefix in DETAIL_PATH_PREFIXES)


def _infer_doc_type(url: Any, title: Any = "", text: Any = "") -> str:
    path = urlparse(str(url or "")).path.lower()
    blob = " ".join([_normalize_space(title), _normalize_space(text)]).lower()
    if "/ics-medical-advisories/" in path:
        return "ICS Medical Advisory"
    if "/ics-advisories/" in path:
        return "ICS Advisory"
    if "/cybersecurity-advisories/" in path:
        return "Cybersecurity Advisory"
    if "/alerts/" in path:
        if "known exploited vulnerabilit" in blob or "kev catalog" in blob:
            return "KEV Alert"
        return "Alert"
    return "CISA Advisory"


def _extract_alert_code(url: Any, text: Any = "") -> str:
    path_slug = urlparse(str(url or "")).path.rstrip("/").rsplit("/", 1)[-1].upper()
    if re.fullmatch(r"(?:(?:AA-?\d{2})|(?:ICSA|ICSMA)-\d{2})-\d{3}[A-Z0-9-]*", path_slug):
        return path_slug
    blob = _normalize_space(text)
    match = re.search(
        r"\b((?:(?:AA-?\d{2})|(?:ICSA|ICSMA)-\d{2})-\d{3}[A-Z0-9-]*)\b",
        blob,
        flags=re.IGNORECASE,
    )
    return match.group(1).upper() if match else ""


def _clean_title(value: Any) -> str:
    title = _normalize_space(value)
    title = re.sub(r"\s*\|\s*CISA\s*$", "", title, flags=re.IGNORECASE).strip()
    return title


def _build_page_url(base_url: str, page: int) -> str:
    parsed = urlparse(str(base_url or CISA_CYBERSECURITY_ADVISORIES_URL))
    pairs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() != "page"]
    if page > 0:
        pairs.append(("page", str(page)))
    query = urlencode(pairs, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, query, parsed.fragment))


def _find_next_page_url(soup: BeautifulSoup, current_url: str) -> str:
    selectors = [
        "a[rel='next']",
        "a.c-pager__link--next[href]",
        "a[aria-label*='next page' i]",
        "a[title*='next page' i]",
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        if not isinstance(node, Tag):
            continue
        href = _normalize_space(node.get("href", ""))
        if not href:
            continue
        next_url = urljoin(current_url, href)
        if next_url and _url_key(next_url) != _url_key(current_url):
            return next_url
    return ""


class CISACybersecurityAdvisoryScraper:
    def __init__(self, min_delay_seconds: float = 0.35):
        self.session = requests.Session()
        self.session.headers.update(REQUEST_HEADERS)
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

    def _discover_from_rss(self, rss_url: str = CISA_CYBERSECURITY_ADVISORIES_RSS_URL) -> List[Dict[str, str]]:
        response = self._fetch(rss_url, timeout=45)
        root = ET.fromstring(response.content)
        items = [node for node in root.iter() if _xml_local_name(node.tag).lower() == "item"]
        out: List[Dict[str, str]] = []
        seen = set()
        for item in items:
            link = _url_without_query(_xml_child_text(item, "link"))
            if not _is_cisa_detail_url(link):
                continue
            key = _url_key(link)
            if not key or key in seen:
                continue
            seen.add(key)
            title = _xml_child_text(item, "title") or "CISA Advisory"
            date_text = _date_to_display(_xml_child_text(item, "pubDate"))
            description = _xml_child_text(item, "description")
            out.append(
                {
                    "url": link,
                    "title": title,
                    "date": date_text,
                    "summary": description,
                    "doc_type": _infer_doc_type(link, title, description),
                    "alert_code": _extract_alert_code(link, title),
                    "source_format": "xml_rss",
                    "listing_page": rss_url,
                }
            )
        return out

    def _discover_from_listing_page(self, page_url: str) -> Tuple[List[Dict[str, str]], str]:
        response = self._fetch(page_url, timeout=45)
        soup = BeautifulSoup(response.text, "html.parser")
        rows = [row for row in soup.select("article") if isinstance(row, Tag)]
        out: List[Dict[str, str]] = []
        seen = set()
        for row in rows:
            link = row.select_one("a[href]")
            if not isinstance(link, Tag):
                continue
            detail_url = _url_without_query(urljoin(CISA_HOME_URL, str(link.get("href", "") or "")))
            if not _is_cisa_detail_url(detail_url):
                continue
            key = _url_key(detail_url)
            if not key or key in seen:
                continue
            seen.add(key)

            row_text = _clean_multiline(row.get_text("\n", strip=True))
            title = _normalize_space(link.get_text(" ", strip=True)) or "CISA Advisory"
            time_node = row.find("time")
            date_text = _date_to_display(time_node.get("datetime") or time_node.get_text(" ", strip=True)) if isinstance(time_node, Tag) else ""
            if not date_text:
                date_text = _extract_first_date(row_text)
            out.append(
                {
                    "url": detail_url,
                    "title": title,
                    "date": date_text,
                    "summary": row_text,
                    "doc_type": _infer_doc_type(detail_url, title, row_text),
                    "alert_code": _extract_alert_code(detail_url, row_text),
                    "source_format": "html",
                    "listing_page": str(page_url or ""),
                }
            )
        return out, _find_next_page_url(soup, str(getattr(response, "url", page_url) or page_url))

    def discover_documents(
        self,
        base_url: str = CISA_CYBERSECURITY_ADVISORIES_URL,
        max_pages: int = 2,
        include_rss: bool = True,
    ) -> List[Dict[str, str]]:
        max_pages = max(1, int(max_pages or 1))
        base_url = str(base_url or "").strip() or CISA_CYBERSECURITY_ADVISORIES_URL
        out: List[Dict[str, str]] = []
        seen = set()
        debug: Dict[str, Any] = {
            "base_url": base_url,
            "max_pages_requested": max_pages,
            "include_rss": bool(include_rss),
            "rss_added": 0,
            "rss_error": "",
            "listing_pages": [],
            "listing_added": 0,
            "total_unique": 0,
        }

        def add_items(items: Iterable[Dict[str, str]], counter_key: str) -> None:
            added = 0
            for item in items:
                key = _url_key(item.get("url", ""))
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(item)
                added += 1
            debug[counter_key] = int(debug.get(counter_key, 0) or 0) + added

        if include_rss:
            try:
                add_items(self._discover_from_rss(), "rss_added")
            except Exception as exc:
                debug["rss_error"] = str(exc)

        current_url = base_url
        for page in range(max_pages):
            page_debug: Dict[str, Any] = {
                "page": page + 1,
                "page_url": current_url,
                "returned_items": 0,
                "unique_added": 0,
                "next_page_url": "",
                "error": "",
            }
            try:
                discovered, next_page = self._discover_from_listing_page(current_url)
                page_debug["returned_items"] = len(discovered)
                before = len(out)
                add_items(discovered, "listing_added")
                page_debug["unique_added"] = len(out) - before
                page_debug["next_page_url"] = next_page
                debug["listing_pages"].append(page_debug)
                if not next_page:
                    break
                current_url = next_page
            except Exception as exc:
                page_debug["error"] = str(exc)
                debug["listing_pages"].append(page_debug)
                break

        out.sort(key=lambda item: _parse_date_text(item.get("date", "")) or datetime.min, reverse=True)
        debug["total_unique"] = len(out)
        self.last_discovery_debug = debug
        return out

    def _extract_title(self, soup: BeautifulSoup, fallback_title: str = "") -> str:
        for selector in ("main h1", "article h1", "h1", "meta[property='og:title']", "title"):
            node = soup.select_one(selector)
            if not isinstance(node, Tag):
                continue
            raw = node.get("content", "") if node.name == "meta" else node.get_text(" ", strip=True)
            title = _clean_title(raw)
            if title:
                return title
        return _clean_title(fallback_title) or "CISA Advisory"

    def _extract_date(self, soup: BeautifulSoup, fallback_date: str = "") -> str:
        for selector in (".c-field--name-field-release-date time", "time"):
            node = soup.select_one(selector)
            if isinstance(node, Tag):
                date_text = _date_to_display(node.get("datetime", "") or node.get_text(" ", strip=True))
                if date_text:
                    return date_text
        if fallback_date:
            return _date_to_display(fallback_date)
        return _extract_first_date(soup.get_text(" ", strip=True)[:2500])

    def _extract_body(self, soup: BeautifulSoup) -> str:
        for tag in soup(["script", "style", "noscript", "svg", "form", "iframe"]):
            tag.decompose()
        for selector in [
            "nav",
            "header",
            "footer",
            ".breadcrumb",
            ".c-share",
            ".social",
            ".sidebar",
            ".skip-link",
            ".c-pager",
        ]:
            for tag in soup.select(selector):
                tag.decompose()

        candidates: List[Tuple[int, str]] = []
        for selector in (".l-page-section", ".usa-prose", "main", "article", "body"):
            for node in soup.select(selector):
                if not isinstance(node, Tag):
                    continue
                text = _clean_multiline(node.get_text("\n", strip=True))
                word_count = len(text.split())
                if word_count:
                    candidates.append((word_count, text))
        if not candidates:
            return _clean_multiline(soup.get_text("\n", strip=True))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1].strip()

    def extract_document(
        self,
        entry: Any,
        fallback_title: str = "",
        fallback_date: str = "",
        fallback_doc_type: str = "",
    ) -> Dict[str, Any]:
        if isinstance(entry, dict):
            url = str(entry.get("url", "") or "").strip()
            fallback_title = str(entry.get("title", "") or fallback_title or "").strip()
            fallback_date = str(entry.get("date", "") or fallback_date or "").strip()
            fallback_doc_type = str(entry.get("doc_type", "") or fallback_doc_type or "").strip()
        else:
            url = str(entry or "").strip()

        if not url:
            return {"success": False, "error": "No CISA advisory URL supplied."}

        try:
            response = self._fetch(url, timeout=60)
        except Exception as exc:
            return {"success": False, "error": str(exc)}

        soup = BeautifulSoup(response.text, "html.parser")
        canonical_link = soup.find("link", rel="canonical")
        canonical_url = _normalize_space(canonical_link.get("href", "")) if isinstance(canonical_link, Tag) else ""
        final_url = canonical_url or _url_without_query(str(getattr(response, "url", url) or url))
        title = self._extract_title(soup, fallback_title)
        date_text = self._extract_date(soup, fallback_date)
        full_text = self._extract_body(soup)
        doc_type = fallback_doc_type or _infer_doc_type(final_url, title, full_text)
        alert_code = _extract_alert_code(final_url, full_text)
        if len(full_text.split()) < 20:
            return {"success": False, "error": "CISA advisory body extraction was too short."}

        return {
            "success": True,
            "data": {
                "url": final_url,
                "title": title,
                "date": date_text,
                "summary": " ".join(full_text.split()[:80]),
                "doc_type": doc_type,
                "alert_code": alert_code,
                "full_text": full_text,
                "word_count": len(full_text.split()),
                "source_format": "html",
                "extraction_mode": "cisa_html",
            },
        }
