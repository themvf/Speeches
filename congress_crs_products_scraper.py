#!/usr/bin/env python3
"""
Congress.gov CRS products scraper.

Discovery uses the public CRS quick-search listing. Extraction pulls the
product detail page HTML and preserves the linked PDF URL in metadata.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag


CONGRESS_HOME_URL = "https://www.congress.gov"
CRS_PRODUCTS_BROWSE_URL = f"{CONGRESS_HOME_URL}/crs-products"
CRS_PRODUCTS_SEARCH_URL = f"{CONGRESS_HOME_URL}/index.php/quick-search/crs-products"


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
    for fmt in ("%m/%d/%Y", "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
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
        r"(\b\d{2}/\d{2}/\d{4}\b)",
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


def _url_without_fragment(url: Any) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ""))


def _url_key(url: Any) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}"


def _listing_url_key(url: Any) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{scheme}://{netloc}{path}{query}"


def _is_crs_detail_url(url: Any) -> bool:
    raw = str(url or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw)
    if parsed.netloc and "congress.gov" not in parsed.netloc.lower():
        return False
    return (parsed.path or "").startswith("/crs-product/")


def _extract_product_number(url: Any, fallback_text: Any = "") -> str:
    match = re.search(r"/crs-product/([A-Za-z0-9.-]+)", str(url or ""))
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-Z]{1,4}\d{3,6})\b", str(fallback_text or ""))
    if match:
        return match.group(1).upper()
    return ""


def _infer_doc_type(product_number: str, fallback: str = "") -> str:
    if fallback:
        return fallback
    number = str(product_number or "").upper()
    if number.startswith("IF"):
        return "In Focus"
    if number.startswith("IN"):
        return "Insight"
    if number.startswith("R"):
        return "Report"
    if number.startswith("LSB"):
        return "Legal Sidebar"
    if number.startswith("RS"):
        return "Report"
    return "CRS Product"


def _split_semicolon_list(value: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in re.split(r"[;|]+", str(value or "")):
        item = _normalize_space(raw)
        if not item:
            continue
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(item)
    return out


def _topic_to_tag(topic: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(topic or "").strip().lower()).strip("-")
    return cleaned


def _normalize_listing_url(base_url: str) -> str:
    raw = str(base_url or "").strip() or CRS_PRODUCTS_BROWSE_URL
    parsed = urlparse(raw)
    if parsed.netloc and "congress.gov" not in parsed.netloc.lower():
        return raw
    if parsed.path.rstrip("/") == "/crs-products":
        query = parse_qs(parsed.query, keep_blank_values=True)
        query.setdefault("pageSize", ["100"])
        normalized_query = urlencode([(key, value) for key, values in query.items() for value in values], doseq=True)
        return urlunparse(("https", parsed.netloc or "www.congress.gov", "/index.php/quick-search/crs-products", "", normalized_query, ""))
    if parsed.path.rstrip("/") == "/index.php/quick-search/crs-products":
        query = parse_qs(parsed.query, keep_blank_values=True)
        query.setdefault("pageSize", ["100"])
        normalized_query = urlencode([(key, value) for key, values in query.items() for value in values], doseq=True)
        return urlunparse((parsed.scheme or "https", parsed.netloc or "www.congress.gov", parsed.path, "", normalized_query, ""))
    return raw


def _best_listing_container(anchor: Tag) -> Tag:
    candidate: Tag = anchor
    for parent in anchor.parents:
        if not isinstance(parent, Tag):
            continue
        if parent.name not in {"div", "li", "article", "tr", "section"}:
            continue
        text = _clean_multiline(parent.get_text("\n", strip=True))
        if "CRS Product Number:" in text or "Publication Date:" in text:
            candidate = parent
            break
    return candidate


def _extract_field(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return _normalize_space(match.group(1)) if match else ""


def _extract_listing_fields(row: Tag, anchor: Tag, detail_url: str) -> Dict[str, Any]:
    row_text = _clean_multiline(row.get_text("\n", strip=True))
    title = _normalize_space(anchor.get_text(" ", strip=True))
    product_number = _extract_product_number(detail_url, row_text)
    doc_type = _extract_field(r"CRS Product Type:\s*([^\n]+)", row_text)
    date_value = _date_to_display(_extract_field(r"Publication Date:\s*([^\n]+)", row_text))
    authors = _extract_field(r"Author(?:s)?:\s*([^\n]+)", row_text)
    topics = _split_semicolon_list(_extract_field(r"Topics?:\s*([^\n]+)", row_text))

    if not date_value:
        date_value = _extract_first_date(row_text)
    if not doc_type:
        doc_type = _infer_doc_type(product_number)

    return {
        "title": title or product_number or "CRS Product",
        "date": date_value,
        "doc_type": doc_type,
        "product_number": product_number,
        "authors": authors,
        "topics": topics,
    }


def _find_next_page_url(soup: BeautifulSoup, current_url: str) -> str:
    selectors = [
        "a[rel='next']",
        "a[title*='Go to next page']",
        "a[aria-label*='next page' i]",
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        if isinstance(node, Tag):
            href = _normalize_space(node.get("href", ""))
            if not href:
                continue
            next_url = _url_without_fragment(urljoin(current_url, href))
            if next_url and _listing_url_key(next_url) != _listing_url_key(current_url):
                return next_url

    for node in soup.select("a[href]"):
        if not isinstance(node, Tag):
            continue
        text = _normalize_space(node.get_text(" ", strip=True)).lower()
        title = _normalize_space(node.get("title", "")).lower()
        aria = _normalize_space(node.get("aria-label", "")).lower()
        if "next page" not in text and "next page" not in title and "next page" not in aria:
            continue
        href = _normalize_space(node.get("href", ""))
        if not href:
            continue
        next_url = _url_without_fragment(urljoin(current_url, href))
        if next_url and _listing_url_key(next_url) != _listing_url_key(current_url):
            return next_url
    return ""


def _content_score(node: Optional[Tag]) -> int:
    if not isinstance(node, Tag):
        return -1
    return len(_normalize_space(node.get_text(" ", strip=True)).split())


class CongressCRSProductsScraper:
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

    def _discover_from_listing_page(self, page_url: str) -> Tuple[List[Dict[str, Any]], str]:
        response = self._fetch_html(page_url, timeout=60)
        soup = BeautifulSoup(response.text, "html.parser")

        out: List[Dict[str, Any]] = []
        seen = set()
        for anchor in soup.select("a[href]"):
            if not isinstance(anchor, Tag):
                continue
            href = _normalize_space(anchor.get("href", ""))
            detail_url = _url_without_query(urljoin(CONGRESS_HOME_URL, href))
            if not _is_crs_detail_url(detail_url):
                continue
            key = _url_key(detail_url)
            if not key or key in seen:
                continue
            seen.add(key)

            container = _best_listing_container(anchor)
            fields = _extract_listing_fields(container, anchor, detail_url)
            out.append(
                {
                    "url": detail_url,
                    "title": fields.get("title", "") or fields.get("product_number", "") or "CRS Product",
                    "date": fields.get("date", ""),
                    "doc_type": fields.get("doc_type", ""),
                    "product_number": fields.get("product_number", ""),
                    "authors": fields.get("authors", ""),
                    "topics": fields.get("topics", []),
                    "source_format": "html",
                    "listing_page": str(page_url or ""),
                }
            )

        next_page = _find_next_page_url(soup, str(getattr(response, "url", page_url) or page_url))
        return out, next_page

    def discover_documents(self, base_url: str = CRS_PRODUCTS_BROWSE_URL, max_pages: int = 3) -> List[Dict[str, Any]]:
        start_url = _normalize_listing_url(base_url)
        max_pages = max(1, int(max_pages or 1))

        out: List[Dict[str, Any]] = []
        seen = set()
        current_url = start_url
        pages_scanned = 0
        debug: Dict[str, Any] = {
            "base_url": base_url,
            "normalized_start_url": start_url,
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
                "error_message": "",
            }
            try:
                discovered, next_page = self._discover_from_listing_page(current_url)
                page_debug["returned_items"] = len(discovered)
                page_debug["next_page_url"] = next_page
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
                page_debug["error_type"] = type(exc).__name__
                page_debug["error_message"] = str(exc)
                debug["pages"].append(page_debug)
                debug["stop_reason"] = "error"
                self.last_discovery_debug = debug
                raise

        if not debug["stop_reason"]:
            debug["stop_reason"] = "completed"

        def _sort_key(item: Dict[str, Any]):
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
        fallback_doc_type: str = "",
        fallback_authors: str = "",
        fallback_product_number: str = "",
    ) -> Dict[str, Any]:
        response = self._fetch_html(url, timeout=75)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup.find_all(["script", "style", "noscript", "svg"]):
            tag.decompose()

        canonical_url = ""
        canonical_link = soup.find("link", rel="canonical")
        if isinstance(canonical_link, Tag):
            canonical_url = _normalize_space(canonical_link.get("href", ""))
        final_url = canonical_url or _url_without_query(str(getattr(response, "url", url) or url))

        title = ""
        for selector in ("main h1", "article h1", "h1"):
            node = soup.select_one(selector)
            if not isinstance(node, Tag):
                continue
            title = _normalize_space(node.get_text(" ", strip=True))
            if title:
                break
        if not title:
            title = _normalize_space(fallback_title) or _extract_product_number(final_url) or "CRS Product"

        candidates: List[Tag] = []
        seen_candidates = set()
        for selector in ("main", "article", "body"):
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
                ".search-form",
                ".facets",
                ".footer",
                ".site-footer",
            ):
                for node in content_node.select(selector):
                    node.decompose()

        full_text = _clean_multiline(content_node.get_text("\n")) if isinstance(content_node, Tag) else ""
        if title and title in full_text:
            full_text = full_text[full_text.find(title):].strip()
        for marker in ("\nImage: Congress.gov", "\nSite Content", "\nWays to Connect", "\nResources"):
            if marker in full_text:
                full_text = full_text.split(marker, 1)[0].strip()
        if not full_text:
            full_text = _clean_multiline(soup.get_text("\n"))

        overview_text = "\n".join(full_text.splitlines()[:60])
        product_number = _extract_product_number(final_url, overview_text) or _normalize_space(fallback_product_number)
        doc_type = _extract_field(r"CRS Product Type:\s*([^\n]+)", overview_text)
        publication_date = _extract_field(r"Publication Date:\s*([^\n]+)", overview_text)
        authors = _extract_field(r"Author(?:s)?:\s*([^\n]+)", overview_text) or _normalize_space(fallback_authors)
        topics = _split_semicolon_list(_extract_field(r"Topics?:\s*([^\n]+)", overview_text))

        if not doc_type:
            doc_type = _normalize_space(fallback_doc_type) or _infer_doc_type(product_number)
        date_value = _date_to_display(publication_date or fallback_date or _extract_first_date(overview_text))

        pdf_url = ""
        for anchor in soup.select("a[href]"):
            if not isinstance(anchor, Tag):
                continue
            href = _normalize_space(anchor.get("href", ""))
            label = _normalize_space(anchor.get_text(" ", strip=True))
            full_url = urljoin(CONGRESS_HOME_URL, href)
            if "/pdf" in href or label.lower().startswith("download pdf"):
                pdf_url = full_url
                break

        return {
            "success": True,
            "data": {
                "url": final_url,
                "title": title,
                "date": date_value,
                "authors": authors,
                "doc_type": doc_type,
                "product_number": product_number,
                "topics": topics,
                "pdf_url": pdf_url,
                "full_text": full_text,
                "word_count": len(full_text.split()),
                "source_format": "html",
            },
        }
