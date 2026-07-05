#!/usr/bin/env python3
"""
Direct Senate committee site scraper.

This connector discovers current committee news and press-release pages from
official Senate committee sites, then extracts the detail-page body text for
storage in the corpus. It intentionally limits discovery to known committee
news URL patterns so navigation, photo galleries, contact pages, and archive
chrome are not ingested as documents.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag


SENATE_COMMITTEE_DEFAULT_URL = "https://www.senate.gov/committees/"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36 PolicyResearchHub/1.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

GENERIC_LINK_TEXT = {
    "",
    "continue reading",
    "read more",
    "more",
    "view all",
    "view all majority press",
    "view all minority press",
    "press releases",
    "majority press releases",
    "minority press releases",
    "latest news",
    "majority news",
    "minority news",
    "newsroom",
}

MONTH_RE = (
    r"January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?"
)
DATE_PATTERNS = [
    re.compile(rf"\b(?:{MONTH_RE})\s+\d{{1,2}}(?:st|nd|rd|th)?,\s+\d{{4}}\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b"),
    re.compile(r"\bPublished:\s*\d{1,2}\.\d{1,2}\.\d{4}\b", re.IGNORECASE),
]


SENATE_COMMITTEE_SOURCES: Dict[str, Dict[str, Any]] = {
    "senate_banking": {
        "label": "Senate Banking Committee",
        "organization": "Senate Committee on Banking, Housing, and Urban Affairs",
        "base_url": "https://www.banking.senate.gov",
        "listing_urls": [
            "https://www.banking.senate.gov/newsroom",
            "https://www.banking.senate.gov/newsroom/majority-press-releases",
            "https://www.banking.senate.gov/newsroom/minority-press-releases",
        ],
        "detail_path_patterns": [r"^/newsroom/(?:majority|minority)/[^/?#]+/?$"],
        "listing_path_patterns": [r"^/newsroom/?$", r"^/newsroom/(?:majority|minority)-press-releases/?$"],
        "tags_csv": "senate,congress,committee,banking,press-release",
    },
    "senate_finance": {
        "label": "Senate Finance Committee",
        "organization": "Senate Committee on Finance",
        "base_url": "https://www.finance.senate.gov",
        "listing_urls": [
            "https://www.finance.senate.gov/chairmans-news",
            "https://www.finance.senate.gov/ranking-members-news",
        ],
        "detail_path_patterns": [r"^/chairmans-news/[^/?#]+/?$", r"^/ranking-members-news/[^/?#]+/?$"],
        "listing_path_patterns": [r"^/chairmans-news/?$", r"^/ranking-members-news/?$"],
        "tags_csv": "senate,congress,committee,finance,press-release",
    },
    "senate_agriculture": {
        "label": "Senate Agriculture Committee",
        "organization": "Senate Committee on Agriculture, Nutrition, and Forestry",
        "base_url": "https://www.agriculture.senate.gov",
        "listing_urls": [
            "https://www.agriculture.senate.gov/newsroom",
            "https://www.agriculture.senate.gov/newsroom/majority-news",
            "https://www.agriculture.senate.gov/newsroom/minority-news",
        ],
        "detail_path_patterns": [r"^/newsroom/(?:rep|dem)/press/release/[^/?#]+/?$"],
        "listing_path_patterns": [
            r"^/newsroom/?$",
            r"^/newsroom/(?:majority|minority)-news/?$",
            r"^/newsroom/(?:majority|minority)-news/\d+/?$",
        ],
        "tags_csv": "senate,congress,committee,agriculture,press-release",
    },
    "senate_judiciary": {
        "label": "Senate Judiciary Committee",
        "organization": "Senate Committee on the Judiciary",
        "base_url": "https://www.judiciary.senate.gov",
        "listing_urls": [
            "https://www.judiciary.senate.gov/press",
            "https://www.judiciary.senate.gov/press/majority",
            "https://www.judiciary.senate.gov/press/minority",
        ],
        "detail_path_patterns": [r"^/press/(?:rep|dem)/releases/[^/?#]+/?$", r"^/press/releases/[^/?#]+/?$"],
        "listing_path_patterns": [r"^/press/?$", r"^/press/(?:majority|minority)/?$"],
        "tags_csv": "senate,congress,committee,judiciary,press-release",
    },
    "senate_hsgac": {
        "label": "Senate Homeland Security Committee",
        "organization": "Senate Committee on Homeland Security and Governmental Affairs",
        "base_url": "https://www.hsgac.senate.gov",
        "listing_urls": [
            "https://www.hsgac.senate.gov/media/",
            "https://www.hsgac.senate.gov/media/majority-news/",
            "https://www.hsgac.senate.gov/media/majority-news/current-congress",
            "https://www.hsgac.senate.gov/media/minority-news/",
            "https://www.hsgac.senate.gov/media/minority-news/current-congress",
        ],
        "detail_path_patterns": [r"^/media/(?:reps|dems)/[^/?#]+/?$"],
        "listing_path_patterns": [
            r"^/media/?$",
            r"^/media/(?:majority|minority)-news/?$",
            r"^/media/(?:majority|minority)-news/current-congress/?$",
        ],
        "tags_csv": "senate,congress,committee,homeland-security,press-release",
    },
    "senate_commerce": {
        "label": "Senate Commerce Committee",
        "organization": "Senate Committee on Commerce, Science, and Transportation",
        "base_url": "https://www.commerce.senate.gov",
        "listing_urls": [
            "https://www.commerce.senate.gov/press/republican-news/",
            "https://www.commerce.senate.gov/press/democratic-news/",
        ],
        "detail_path_patterns": [r"^/press/(?:rep|dem)/release/[^/?#]+/?$"],
        "listing_path_patterns": [r"^/press/(?:republican|democratic)-news(?:/\d+)?/?$"],
        "tags_csv": "senate,congress,committee,commerce,press-release",
    },
}


def _normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_multiline(value: Any) -> str:
    lines: List[str] = []
    for raw in str(value or "").splitlines():
        line = _normalize_space(raw)
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


SENATE_BOILERPLATE_LINES = {
    "about",
    "chairman biography",
    "chairman scott biography",
    "contact",
    "committee documents",
    "faq",
    "hearings",
    "home",
    "history of the chairman",
    "jurisdiction",
    "key issues",
    "legislation",
    "legislative calendar",
    "majority news",
    "majority press releases",
    "markups",
    "membership",
    "milestones",
    "minority news",
    "minority press releases",
    "nominations",
    "newsroom",
    "photo gallery",
    "press release archive",
    "press releases",
    "privacy policy",
    "ranking member",
    "resources",
    "submissions",
    "skip to content",
    "social media",
    "subcommittees",
    "ranking member biography",
    "ranking member warren biography",
    "witness list",
}

SENATE_BOILERPLATE_RE = re.compile(
    r"(?:"
    r"^about\s+about\s+jurisdiction\s+membership\b|"
    r"^hearings\s+hearings\s+witness list\b|"
    r"^legislative calendar\s+legislation\s+nominations\b|"
    r"^newsroom\s+majority press releases\b|"
    r"^majority press releases\s+minority press releases\b|"
    r"^photo gallery\s+press release archive\b|"
    r"^resources\s+resources\s+committee documents\b|"
    r"\b(?:privacy policy|accessibility|rss feed)\b"
    r")",
    re.IGNORECASE,
)

SENATE_NAV_TOKENS = {
    "about",
    "biography",
    "faq",
    "hearings",
    "history",
    "jurisdiction",
    "legislation",
    "membership",
    "milestones",
    "nominations",
    "ranking member",
    "subcommittees",
    "witness list",
}


def _title_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _clean_body_text(value: Any, title: str = "") -> str:
    lines = [_normalize_space(line) for line in str(value or "").splitlines()]
    lines = [line for line in lines if line]
    title_key = _title_token(title)
    if len(title_key) >= 18:
        for idx, line in enumerate(lines[:80]):
            line_key = _title_token(line)
            if len(line_key) >= 18 and (title_key in line_key or line_key in title_key):
                lines = lines[idx:]
                break

    cleaned: List[str] = []
    short_counts: Dict[str, int] = {}
    for line in lines:
        lowered = line.lower()
        if lowered in SENATE_BOILERPLATE_LINES:
            continue
        if SENATE_BOILERPLATE_RE.search(line):
            continue
        nav_hits = sum(1 for token in SENATE_NAV_TOKENS if token in lowered)
        if nav_hits >= 3:
            continue
        if len(line.split()) <= 4:
            count = short_counts.get(lowered, 0)
            short_counts[lowered] = count + 1
            if count >= 1:
                continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _url_without_fragment(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ""))


def _url_key(value: Any) -> str:
    raw = _url_without_fragment(value)
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}"


def _is_error_html(html: Any) -> bool:
    blob = str(html or "").lower()
    return (
        "enable javascript and cookies to continue" in blob
        or "just a moment" in blob and "cloudflare" in blob
        or ("<title>error</title>" in blob and len(blob) < 2000)
    )


def _parse_date_text(value: Any) -> Optional[datetime]:
    text = _normalize_space(value)
    if not text:
        return None
    text = re.sub(r"^Published:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d{1,2})(?:st|nd|rd|th)", r"\1", text)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        return parsed.replace(tzinfo=None)
    except ValueError:
        pass
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%m.%d.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        parsed = parsedate_to_datetime(text)
        if parsed:
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
    text = _normalize_space(value)
    if not text:
        return ""
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return _date_to_display(match.group(0))
    return ""


def _clean_title(value: Any) -> str:
    title = _normalize_space(value)
    title = re.sub(
        r"\s+(?:-|[|])\s+(?:United States Senate|United States Committee|U\.S\. Senate|Senate Committee|Committee on).*$",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(r"\s*->\s*$", "", title)
    title = re.sub(r"\s*[\u2192]\s*$", "", title)
    return title.strip()


def _is_generic_title(value: Any) -> bool:
    title = _normalize_space(value).lower()
    return not title or title in GENERIC_LINK_TEXT or len(title) < 8


def _path_matches(path: str, patterns: Iterable[str]) -> bool:
    normalized = path or "/"
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)


def _candidate_title(element: Tag, fallback_text: str = "") -> str:
    text = _clean_title(fallback_text)
    if not _is_generic_title(text):
        return text

    for selector in ("h1", "h2", "h3", "h4", "h5", ".title", ".post-title", ".entry-title"):
        found = element.select_one(selector)
        if found:
            candidate = _clean_title(found.get_text(" ", strip=True))
            if not _is_generic_title(candidate):
                return candidate

    parent = element.parent
    for _ in range(3):
        if not isinstance(parent, Tag):
            break
        for selector in ("h1", "h2", "h3", "h4", "h5", ".title", ".post-title", ".entry-title"):
            found = parent.select_one(selector)
            if found:
                candidate = _clean_title(found.get_text(" ", strip=True))
                if not _is_generic_title(candidate):
                    return candidate
        parent = parent.parent

    return text


def _candidate_context(element: Tag) -> str:
    parent = element.parent
    for _ in range(3):
        if not isinstance(parent, Tag):
            break
        text = _normalize_space(parent.get_text(" ", strip=True))
        if len(text) > 20:
            return text
        parent = parent.parent
    return _normalize_space(element.get_text(" ", strip=True))


class SenateCommitteeScraper:
    """Discover and extract official Senate committee news pages."""

    def __init__(self, timeout: int = 20, sleep_seconds: float = 0.35) -> None:
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self.session.headers.update(REQUEST_HEADERS)
        self.last_discovery_debug: Dict[str, Any] = {}

    def _fetch_text(self, url: str) -> Tuple[str, int, str]:
        time.sleep(max(0.0, self.sleep_seconds))
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text or "", int(response.status_code), str(response.url or url)

    def _select_sources(self, base_url: str = "") -> Dict[str, Dict[str, Any]]:
        raw = str(base_url or "").strip()
        if not raw or raw.rstrip("/") == SENATE_COMMITTEE_DEFAULT_URL.rstrip("/"):
            return SENATE_COMMITTEE_SOURCES

        base_netloc = urlparse(raw).netloc.lower()
        matched = {
            key: cfg
            for key, cfg in SENATE_COMMITTEE_SOURCES.items()
            if urlparse(str(cfg.get("base_url", ""))).netloc.lower() == base_netloc
        }
        if matched:
            return matched

        return {
            "senate_committee_custom": {
                "label": "Senate Committee Site",
                "organization": "Senate Committee",
                "base_url": raw,
                "listing_urls": [raw],
                "detail_path_patterns": [r"/(?:newsroom|press|media)/.+/[^/?#]+/?$"],
                "listing_path_patterns": [r"/(?:newsroom|press|media)/?.*$"],
                "tags_csv": "senate,congress,committee,press-release",
            }
        }

    def _iter_url_candidates(self, soup: BeautifulSoup, listing_url: str) -> Iterable[Tuple[str, str, str, Tag]]:
        for link in soup.find_all("a", href=True):
            if not isinstance(link, Tag):
                continue
            href = urljoin(listing_url, str(link.get("href", "") or ""))
            title = _candidate_title(link, link.get_text(" ", strip=True))
            context = _candidate_context(link)
            yield href, title, context, link

        for element in soup.find_all(attrs={"data-url": True}):
            if not isinstance(element, Tag):
                continue
            href = urljoin(listing_url, str(element.get("data-url", "") or ""))
            title = _candidate_title(element)
            context = _candidate_context(element)
            yield href, title, context, element

    def _is_same_source_domain(self, url: str, cfg: Dict[str, Any]) -> bool:
        target_netloc = urlparse(str(cfg.get("base_url", ""))).netloc.lower()
        candidate_netloc = urlparse(url).netloc.lower()
        return bool(target_netloc and candidate_netloc == target_netloc)

    def _is_detail_url(self, url: str, cfg: Dict[str, Any]) -> bool:
        if not self._is_same_source_domain(url, cfg):
            return False
        parsed = urlparse(url)
        path = parsed.path or "/"
        return _path_matches(path, cfg.get("detail_path_patterns", []))

    def _is_listing_url(self, url: str, cfg: Dict[str, Any]) -> bool:
        if not self._is_same_source_domain(url, cfg):
            return False
        parsed = urlparse(url)
        path = parsed.path or "/"
        return _path_matches(path, cfg.get("listing_path_patterns", []))

    def discover_documents(self, base_url: str = "", max_pages: int = 2) -> List[Dict[str, Any]]:
        max_listing_pages = max(1, int(max_pages or 1))
        docs: List[Dict[str, Any]] = []
        seen_docs = set()
        debug: Dict[str, Any] = {
            "source_counts": {},
            "listing_pages": [],
            "listing_errors": [],
            "skipped_error_pages": [],
        }

        for source_key, cfg in self._select_sources(base_url).items():
            queue = [str(url) for url in cfg.get("listing_urls", []) if str(url or "").strip()]
            if base_url and base_url.rstrip("/") != SENATE_COMMITTEE_DEFAULT_URL.rstrip("/"):
                override = str(base_url or "").strip()
                if override and self._is_same_source_domain(override, cfg):
                    queue = [url for url in queue if _url_key(url) != _url_key(override)]
                    queue.insert(0, override)
            seen_listing = set()
            source_count = 0

            while queue and len(seen_listing) < max_listing_pages:
                listing_url = queue.pop(0)
                listing_key = _url_key(listing_url)
                if not listing_key or listing_key in seen_listing:
                    continue
                seen_listing.add(listing_key)
                try:
                    html, status_code, final_url = self._fetch_text(listing_url)
                    debug["listing_pages"].append(
                        {
                            "source_key": source_key,
                            "url": listing_url,
                            "status_code": status_code,
                            "final_url": final_url,
                            "bytes": len(html),
                        }
                    )
                except Exception as exc:
                    debug["listing_errors"].append(
                        {"source_key": source_key, "url": listing_url, "error": str(exc)}
                    )
                    continue

                if _is_error_html(html):
                    debug["skipped_error_pages"].append({"source_key": source_key, "url": listing_url})
                    continue

                soup = BeautifulSoup(html, "html.parser")
                for href, title, context, element in self._iter_url_candidates(soup, listing_url):
                    canonical = _url_without_fragment(href)
                    if not canonical:
                        continue
                    if self._is_listing_url(canonical, cfg) and len(seen_listing) + len(queue) < max_listing_pages:
                        listing_candidate_key = _url_key(canonical)
                        if listing_candidate_key and listing_candidate_key not in seen_listing and canonical not in queue:
                            queue.append(canonical)
                        continue
                    if not self._is_detail_url(canonical, cfg):
                        continue
                    doc_key = _url_key(canonical)
                    if not doc_key or doc_key in seen_docs:
                        continue

                    seen_docs.add(doc_key)
                    source_count += 1
                    docs.append(
                        {
                            "url": canonical,
                            "title": title or cfg.get("label", "Senate Committee Item"),
                            "date": _extract_first_date(context),
                            "summary": context,
                            "source_key": source_key,
                            "source_label": str(cfg.get("label", "") or "Senate Committee Site"),
                            "organization": str(cfg.get("organization", "") or "Senate Committee"),
                            "doc_type": "Press Release",
                            "tags_csv": str(cfg.get("tags_csv", "") or "senate,congress,committee,press-release"),
                            "listing_page": listing_url,
                        }
                    )

            debug["source_counts"][source_key] = source_count

        self.last_discovery_debug = debug
        return docs

    def _extract_title(self, soup: BeautifulSoup, fallback_title: str = "") -> str:
        for selector in [
            'meta[property="og:title"]',
            'meta[name="twitter:title"]',
            'meta[name="DC.title"]',
        ]:
            tag = soup.select_one(selector)
            if tag:
                value = _clean_title(tag.get("content", ""))
                if not _is_generic_title(value):
                    return value
        for selector in ("h1", ".post-title", ".entry-title", ".article-title", "title"):
            tag = soup.select_one(selector)
            if tag:
                value = _clean_title(tag.get_text(" ", strip=True))
                if not _is_generic_title(value):
                    return value
        return _clean_title(fallback_title) or "Senate Committee Item"

    def _extract_date(self, soup: BeautifulSoup, fallback_date: str = "") -> str:
        for selector in [
            'meta[property="article:published_time"]',
            'meta[name="date"]',
            'meta[name="dc.date"]',
            'meta[name="DC.date"]',
            'meta[name="pubdate"]',
        ]:
            tag = soup.select_one(selector)
            if tag:
                value = _date_to_display(tag.get("content", ""))
                if value:
                    return value
        for tag in soup.find_all("time"):
            if not isinstance(tag, Tag):
                continue
            value = _date_to_display(tag.get("datetime", "") or tag.get_text(" ", strip=True))
            if value:
                return value
        for selector in (".date", ".published", ".post-date", ".entry-date", ".article-date"):
            tag = soup.select_one(selector)
            if tag:
                value = _extract_first_date(tag.get_text(" ", strip=True))
                if value:
                    return value
        value = _date_to_display(fallback_date)
        if value:
            return value
        return _extract_first_date(soup.get_text(" ", strip=True)[:2500])

    def _node_text(self, node: Tag) -> str:
        parts: List[str] = []
        for child in node.find_all(["p", "li", "blockquote", "h2", "h3"], recursive=True):
            line = _normalize_space(child.get_text(" ", strip=True))
            lowered = line.lower()
            if len(line) < 3:
                continue
            if lowered in GENERIC_LINK_TEXT:
                continue
            if lowered.startswith(("print", "share", "tweet", "email", "prev previous", "next next")):
                continue
            if "privacy policy" in lowered and len(line) < 80:
                continue
            parts.append(line)
        if parts:
            return "\n".join(parts).strip()
        return _clean_multiline(node.get_text("\n", strip=True))

    def _extract_body_text(self, soup: BeautifulSoup) -> str:
        for tag in soup(["script", "style", "noscript", "svg", "form", "iframe"]):
            tag.decompose()
        for selector in [
            "nav",
            "header",
            "footer",
            ".navigation",
            ".menu",
            ".share",
            ".social",
            ".sidebar",
            ".skip-link",
            ".breadcrumb",
            ".pagination",
        ]:
            for tag in soup.select(selector):
                tag.decompose()

        candidates: List[Tuple[int, str]] = []
        for selector in [
            "article",
            "main",
            ".entry-content",
            ".post-content",
            ".article-content",
            ".content",
            ".content-main",
            ".field--name-body",
            "#content",
            "body",
        ]:
            for node in soup.select(selector):
                if not isinstance(node, Tag):
                    continue
                text = self._node_text(node)
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
        fallback_summary: str = "",
    ) -> Dict[str, Any]:
        if isinstance(entry, dict):
            url = str(entry.get("url", "") or "").strip()
            fallback_title = str(entry.get("title", "") or fallback_title or "").strip()
            fallback_date = str(entry.get("date", "") or fallback_date or "").strip()
            fallback_summary = str(entry.get("summary", "") or fallback_summary or "").strip()
        else:
            url = str(entry or "").strip()

        if not url:
            return {"success": False, "error": "No Senate committee detail URL supplied."}

        try:
            html, status_code, final_url = self._fetch_text(url)
        except Exception as exc:
            return {"success": False, "error": str(exc)}

        if _is_error_html(html):
            return {"success": False, "error": "Senate committee detail page returned an error shell."}

        soup = BeautifulSoup(html, "html.parser")
        title = self._extract_title(soup, fallback_title=fallback_title)
        date_text = self._extract_date(soup, fallback_date=fallback_date)
        full_text = _clean_body_text(self._extract_body_text(soup), title=title)
        summary = fallback_summary or " ".join(full_text.split()[:60])
        if len(full_text.split()) < 20:
            return {"success": False, "error": "Senate committee detail page body extraction was too short."}

        return {
            "success": True,
            "data": {
                "url": final_url or url,
                "title": title,
                "date": date_text,
                "summary": summary,
                "full_text": full_text,
                "source_format": "html",
                "extraction_mode": "senate_committee_html",
                "status_code": status_code,
            },
        }
