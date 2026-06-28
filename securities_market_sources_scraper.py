#!/usr/bin/env python3
"""Focused securities-market official-source scraper.

This connector covers official feeds/listings that are close to securities
markets but do not warrant a bespoke scraper yet: selected SEC RSS feeds,
SEC PCAOB rulemaking, PCAOB updates, and MSRB press releases.
"""

from __future__ import annotations

import io
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from curl_cffi import requests as cffi_requests


SEC_BASE = "https://www.sec.gov"

SECURITIES_MARKET_SOURCES: Dict[str, Dict[str, Any]] = {
    "sec_press_release_rss": {
        "label": "SEC Press Releases",
        "organization": "SEC",
        "default_url": "https://www.sec.gov/news/pressreleases.rss",
        "discovery": "rss",
        "doc_type": "Press Release",
        "tags_csv": "sec,press-release,securities-regulation",
    },
    "sec_administrative_proceeding": {
        "label": "SEC Administrative Proceedings",
        "organization": "SEC",
        "default_url": "https://www.sec.gov/enforcement-litigation/administrative-proceedings/rss",
        "discovery": "rss",
        "doc_type": "Administrative Proceeding",
        "tags_csv": "sec,administrative-proceeding,enforcement,securities-regulation",
    },
    "sec_trading_suspension": {
        "label": "SEC Trading Suspensions",
        "organization": "SEC",
        "default_url": "https://www.sec.gov/enforcement-litigation/trading-suspensions/rss",
        "discovery": "rss",
        "doc_type": "Trading Suspension",
        "tags_csv": "sec,trading-suspension,enforcement,market-integrity",
    },
    "sec_federal_register": {
        "label": "SEC Federal Register Materials",
        "organization": "SEC",
        "default_url": "https://www.federalregister.gov/articles/search.rss?conditions%5Bagency_ids%5D%5B%5D=466&order=newest",
        "discovery": "rss",
        "doc_type": "Federal Register Notice",
        "tags_csv": "sec,federal-register,rulemaking,securities-regulation",
    },
    "sec_pcaob_rulemaking": {
        "label": "SEC PCAOB Rulemaking",
        "organization": "SEC",
        "default_url": "https://www.sec.gov/rules-regulations/public-company-accounting-oversight-board-rulemaking",
        "discovery": "sec_pcaob_table",
        "doc_type": "PCAOB Rulemaking",
        "tags_csv": "sec,pcaob,rulemaking,audit",
    },
    "pcaob_update": {
        "label": "PCAOB Updates",
        "organization": "PCAOB",
        "default_url": "https://pcaobus.org/all-updates-and-news-releases",
        "discovery": "html_links",
        "path_contains": "/news-events/news-releases/",
        "doc_type": "PCAOB Update",
        "tags_csv": "pcaob,audit,public-company-accounting",
    },
    "msrb_press_release": {
        "label": "MSRB Press Releases",
        "organization": "MSRB",
        "default_url": "https://www.msrb.org/Press-Releases",
        "discovery": "html_links",
        "path_contains": "/Press-Releases/",
        "doc_type": "Press Release",
        "tags_csv": "msrb,municipal-securities,market-transparency",
    },
}


def _normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clean_multiline(text: Any) -> str:
    lines = [_normalize_space(line) for line in str(text or "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _xml_local_name(tag: Any) -> str:
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
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%m/%d/%y"):
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
    return parsed.strftime("%B %d, %Y") if parsed is not None else str(value or "").strip()


def _extract_first_date(text: Any) -> str:
    match = re.search(
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)"
        r"\s+\d{1,2},\s+\d{4})",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    return _date_to_display(match.group(1)) if match else ""


def _url_key(url: Any) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}"


def _title_from_url(url: Any, fallback: str = "Document") -> str:
    slug = urlparse(str(url or "")).path.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"\.(html?|pdf)$", "", slug, flags=re.IGNORECASE)
    title = " ".join(part for part in re.split(r"[-_]+", slug) if part).strip()
    return title.title() if title else fallback


def _strip_html(value: Any) -> str:
    return _normalize_space(BeautifulSoup(str(value or ""), "html.parser").get_text(" ", strip=True))


def _is_generic_document_heading(value: Any) -> bool:
    return _normalize_space(value).lower() in {
        "notice",
        "rule",
        "proposed rule",
        "final rule",
        "interim final rule",
        "presidential document",
        "correction",
    }


def _looks_like_pdf_url(url: Any) -> bool:
    return str(url or "").lower().split("?", 1)[0].endswith(".pdf")


class SecuritiesMarketSourcesScraper:
    def __init__(self, min_delay_seconds: float = 0.5):
        self.sec_session = cffi_requests.Session(impersonate="chrome")
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

    def _fetch(self, url: str, timeout: int = 45) -> Any:
        target = str(url or "").strip()
        if not target:
            raise ValueError("URL is required")
        self._rate_limit()
        parsed = urlparse(target)
        if parsed.netloc.lower().endswith("sec.gov"):
            response = self.sec_session.get(target, timeout=timeout, allow_redirects=True)
        else:
            response = self.session.get(target, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        return response

    def _source_config(self, source_key: str) -> Dict[str, Any]:
        cfg = SECURITIES_MARKET_SOURCES.get(str(source_key or "").strip())
        if not cfg:
            raise ValueError(f"Unsupported securities market source: {source_key}")
        return cfg

    def discover_documents(self, source_key: str, base_url: str = "", max_pages: int = 1) -> List[Dict[str, Any]]:
        cfg = self._source_config(source_key)
        url = str(base_url or cfg["default_url"]).strip()
        mode = str(cfg.get("discovery", "") or "").strip()
        debug: Dict[str, Any] = {
            "source_key": source_key,
            "source_label": cfg.get("label", ""),
            "base_url": url,
            "mode": mode,
            "errors": [],
        }
        try:
            if mode == "rss":
                docs = self._discover_rss(source_key, cfg, url, max_items=max(10, int(max_pages or 1) * 25))
            elif mode == "sec_pcaob_table":
                docs = self._discover_sec_pcaob_rulemaking(source_key, cfg, url)
            elif mode == "html_links":
                docs = self._discover_html_links(source_key, cfg, url, max_items=max(10, int(max_pages or 1) * 25))
            else:
                raise ValueError(f"Unsupported discovery mode: {mode}")
        except Exception as exc:
            debug["errors"].append(str(exc))
            docs = []
        debug["items_found"] = len(docs)
        self.last_discovery_debug = debug
        return docs

    def _discover_rss(
        self, source_key: str, cfg: Dict[str, Any], feed_url: str, max_items: int
    ) -> List[Dict[str, Any]]:
        response = self._fetch(feed_url)
        root = ET.fromstring(str(response.text or "").lstrip("\ufeff").strip())
        out: List[Dict[str, Any]] = []
        seen = set()
        for item in root.iter():
            if _xml_local_name(item.tag).lower() != "item":
                continue
            title = ""
            link = ""
            date_text = ""
            description = ""
            for child in list(item):
                name = _xml_local_name(child.tag).lower()
                value = _normalize_space(child.text or "")
                if name == "title":
                    title = value
                elif name == "link":
                    link = value
                elif name == "pubdate":
                    date_text = _date_to_display(value)
                elif name in {"description", "summary"}:
                    description = _strip_html(child.text or "")
            if not link:
                continue
            doc_url = urljoin(feed_url, link)
            key = _url_key(doc_url)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                self._entry(
                    source_key,
                    cfg,
                    url=doc_url,
                    title=title or _title_from_url(doc_url, cfg.get("label", "Document")),
                    date=date_text,
                    summary=description,
                    listing_page=feed_url,
                    position=len(out) + 1,
                )
            )
            if len(out) >= max_items:
                break
        return out

    def _discover_sec_pcaob_rulemaking(
        self, source_key: str, cfg: Dict[str, Any], base_url: str
    ) -> List[Dict[str, Any]]:
        response = self._fetch(base_url)
        soup = BeautifulSoup(response.text, "html.parser")
        out: List[Dict[str, Any]] = []
        seen = set()
        for anchor in soup.select("a[href$='.pdf']"):
            href = str(anchor.get("href", "") or "")
            if "/rules/pcaob/" not in href.lower():
                continue
            doc_url = urljoin(SEC_BASE, href)
            key = _url_key(doc_url)
            if key in seen:
                continue
            seen.add(key)
            row = anchor.find_parent("tr") or anchor.parent
            row_text = _normalize_space(row.get_text(" ", strip=True) if row else anchor.get_text(" ", strip=True))
            title = row_text or _normalize_space(anchor.get_text(" ", strip=True)) or _title_from_url(doc_url)
            out.append(
                self._entry(
                    source_key,
                    cfg,
                    url=doc_url,
                    title=title[:300],
                    date=_extract_first_date(row_text),
                    summary=row_text,
                    listing_page=base_url,
                    position=len(out) + 1,
                )
            )
        return out

    def _discover_html_links(
        self, source_key: str, cfg: Dict[str, Any], base_url: str, max_items: int
    ) -> List[Dict[str, Any]]:
        response = self._fetch(base_url)
        soup = BeautifulSoup(response.text, "html.parser")
        path_contains = str(cfg.get("path_contains", "") or "").lower()
        out: List[Dict[str, Any]] = []
        seen = set()
        for anchor in soup.select("a[href]"):
            href = str(anchor.get("href", "") or "")
            if path_contains and path_contains not in href.lower():
                continue
            doc_url = urljoin(base_url, href)
            key = _url_key(doc_url)
            if key in seen or key == _url_key(base_url):
                continue
            seen.add(key)
            container = self._best_container(anchor)
            container_text = _normalize_space(container.get_text(" ", strip=True) if container else "")
            title = self._link_title(anchor, container_text, doc_url)
            out.append(
                self._entry(
                    source_key,
                    cfg,
                    url=doc_url,
                    title=title,
                    date=_extract_first_date(container_text),
                    summary=container_text,
                    listing_page=base_url,
                    position=len(out) + 1,
                )
            )
            if len(out) >= max_items:
                break
        return out

    def _best_container(self, anchor: Tag) -> Optional[Tag]:
        for parent in anchor.parents:
            if not isinstance(parent, Tag):
                continue
            if parent.name in {"article", "li", "tr"}:
                return parent
            if parent.name == "div":
                text = _normalize_space(parent.get_text(" ", strip=True))
                if len(text) >= 40:
                    return parent
        return anchor.parent if isinstance(anchor.parent, Tag) else None

    def _link_title(self, anchor: Tag, container_text: str, url: str) -> str:
        anchor_text = _normalize_space(anchor.get_text(" ", strip=True))
        if anchor_text and anchor_text.lower() not in {"read more", "read the report"}:
            return anchor_text
        if container_text:
            for separator in [" Read more", " Read the report"]:
                if separator in container_text:
                    return container_text.split(separator, 1)[0].strip()
            return container_text[:180].strip()
        return _title_from_url(url)

    def _entry(
        self,
        source_key: str,
        cfg: Dict[str, Any],
        *,
        url: str,
        title: str,
        date: str,
        summary: str,
        listing_page: str,
        position: int,
    ) -> Dict[str, Any]:
        return {
            "url": str(url or "").strip(),
            "title": _normalize_space(title),
            "date": _date_to_display(date),
            "summary": _normalize_space(summary),
            "source_key": source_key,
            "source_label": str(cfg.get("label", "") or "").strip(),
            "organization": str(cfg.get("organization", "") or "").strip(),
            "doc_type": str(cfg.get("doc_type", "Document") or "Document").strip(),
            "tags_csv": str(cfg.get("tags_csv", "") or "").strip(),
            "source_format": "pdf" if _looks_like_pdf_url(url) else "html",
            "listing_page": listing_page,
            "search_position": position,
        }

    def extract_document(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        url = _normalize_space(entry.get("url", ""))
        if not url:
            return {"success": False, "error": "No URL provided", "data": {}}
        try:
            response = self._fetch(url, timeout=90)
        except Exception as exc:
            title = _normalize_space(entry.get("title", ""))
            date_text = str(entry.get("date", "") or "").strip()
            summary = str(entry.get("summary", "") or "").strip()
            if title and (date_text or summary):
                text = "\n".join(
                    part
                    for part in [
                        title,
                        f"Published Date: {date_text}" if date_text else "",
                        summary,
                        f"Source URL: {url}",
                        f"Detail page fetch failed during ingest: {exc}",
                    ]
                    if part
                ).strip()
                return {
                    "success": True,
                    "data": {
                        "url": url,
                        "title": title,
                        "date": _date_to_display(date_text),
                        "summary": summary,
                        "full_text": text,
                        "source_format": str(entry.get("source_format", "html") or "html").strip(),
                        "extraction_mode": "metadata_fallback",
                        "word_count": len(text.split()),
                    },
                }
            return {"success": False, "error": str(exc), "data": {}}
        final_url = str(getattr(response, "url", url) or url)
        source_format = "pdf" if _looks_like_pdf_url(final_url) else "html"
        if source_format == "pdf":
            text = self._pdf_text(response.content)
            title = _normalize_space(entry.get("title", "")) or _title_from_url(final_url)
            date_text = str(entry.get("date", "") or "").strip()
            extraction_mode = "pdf"
        else:
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup.select("script, style, noscript, nav, footer, header"):
                tag.decompose()
            title_node = soup.find("h1") or soup.find("title")
            page_title = _normalize_space(title_node.get_text(" ", strip=True) if title_node else "")
            fallback_title = str(entry.get("title", "") or "").strip()
            if page_title.lower() in {"press releases", "news releases", "all updates and news releases"}:
                title = fallback_title or page_title
            elif _is_generic_document_heading(page_title):
                title = fallback_title or page_title
            else:
                title = page_title or fallback_title
            body = (
                soup.select_one("article")
                or soup.select_one("main")
                or soup.select_one("div.field--name-body")
                or soup.body
                or soup
            )
            text = _clean_multiline(body.get_text("\n"))
            date_text = str(entry.get("date", "") or "").strip() or _extract_first_date(text)
            extraction_mode = "html"
        if not text.strip():
            text = str(entry.get("summary", "") or "").strip()
        if not text.strip():
            return {"success": False, "error": "No usable text extracted", "data": {}}
        return {
            "success": True,
            "data": {
                "url": final_url,
                "title": title or str(entry.get("title", "") or "").strip(),
                "date": _date_to_display(date_text),
                "summary": str(entry.get("summary", "") or "").strip(),
                "full_text": text,
                "source_format": source_format,
                "extraction_mode": extraction_mode,
                "word_count": len(text.split()),
            },
        }

    def _pdf_text(self, content: bytes) -> str:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages[:30]:
            pages.append(page.extract_text() or "")
        return _clean_multiline("\n".join(pages))
