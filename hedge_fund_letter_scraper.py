#!/usr/bin/env python3
"""Public hedge fund and investor-letter source scraper."""

from __future__ import annotations

import io
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag


FISCAL_FUND_LETTERS_URL = "https://fiscal.ai/fund-letters/"

HEDGE_FUND_LETTER_SOURCES: Dict[str, Dict[str, Any]] = {
    "fiscal_ai_fund_letters": {
        "label": "Fiscal.ai Fund Letters",
        "organization": "Fiscal.ai",
        "default_url": FISCAL_FUND_LETTERS_URL,
        "mode": "fiscal_letters",
    },
    "investment_masters_letters": {
        "label": "Investment Masters Investor Letters",
        "organization": "Investment Masters",
        "default_url": "https://mastersinvest.com/new-page-16",
        "mode": "investor_letter_directory",
    },
    "pershing_square_materials": {
        "label": "Pershing Square Holdings Materials",
        "organization": "Pershing Square Holdings",
        "default_url": "https://pershingsquareholdings.com/materials/",
        "mode": "fund_materials",
    },
    "greenlight_capital_documents": {
        "label": "Greenlight Capital Documents",
        "organization": "Greenlight Capital",
        "default_url": "https://www.greenlightcapital.com/",
        "mode": "greenlight_downloads",
    },
}


def _normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_multiline(value: Any) -> str:
    lines = [_normalize_space(line) for line in str(value or "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _url_key(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    raw, _fragment = urldefrag(raw)
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{scheme}://{netloc}{path}{query}"


def _looks_like_pdf_url(value: Any) -> bool:
    path = urlparse(str(value or "")).path.lower()
    return path.endswith(".pdf")


def _title_from_url(value: Any, fallback: str = "Investor Letter") -> str:
    slug = urlparse(str(value or "")).path.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"\.(html?|pdf|aspx)$", "", slug, flags=re.IGNORECASE)
    slug = re.sub(r"[_+%-]+", " ", slug)
    title = _normalize_space(re.sub(r"\s+", " ", slug))
    return title.title() if title else fallback


def _parse_date_text(value: Any) -> Optional[datetime]:
    text = _normalize_space(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%B %Y", "%b %Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _date_to_display(value: Any) -> str:
    parsed = _parse_date_text(value)
    return parsed.strftime("%B %d, %Y") if parsed else _normalize_space(value)


def _extract_year(*parts: Any) -> str:
    for part in parts:
        match = re.search(r"\b(20\d{2}|19\d{2})\b", str(part or ""))
        if match:
            return match.group(1)
    return ""


def _extract_date(*parts: Any) -> str:
    joined = " ".join(str(part or "") for part in parts)
    full = re.search(
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)"
        r"\s+\d{1,2},\s+(?:20\d{2}|19\d{2}))\b",
        joined,
        flags=re.IGNORECASE,
    )
    if full:
        return _date_to_display(full.group(1).replace(".", ""))

    month_day = re.search(
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)"
        r"\s+\d{1,2})\b",
        joined,
        flags=re.IGNORECASE,
    )
    year = _extract_year(*parts)
    if month_day and year:
        return _date_to_display(f"{month_day.group(1).replace('.', '')}, {year}")

    year_month = re.search(
        r"\b((?:20\d{2}|19\d{2})[-_/](?:0?[1-9]|1[0-2]))\b",
        joined,
        flags=re.IGNORECASE,
    )
    if year_month:
        year, month = re.split(r"[-_/]", year_month.group(1))[:2]
        return datetime(int(year), int(month), 1).strftime("%B %d, %Y")
    return ""


def _best_container(anchor: Tag) -> Tag:
    for parent in anchor.parents:
        if not isinstance(parent, Tag):
            continue
        if parent.name in {"article", "li", "tr"}:
            return parent
        if parent.name == "div":
            text = _normalize_space(parent.get_text(" ", strip=True))
            if len(text) >= 35:
                return parent
    return anchor


def _is_probable_letter_link(text: str, href: str) -> bool:
    haystack = f"{text} {href}".lower()
    if not href or href.startswith("#") or href.startswith("mailto:"):
        return False
    if any(term in haystack for term in ["privacy", "terms", "login", "sign in", "contact"]):
        return False
    if _looks_like_pdf_url(href):
        return True
    return any(
        term in haystack
        for term in [
            "letter",
            "letters",
            "investor",
            "shareholder",
            "annual report",
            "quarterly",
            "commentary",
            "analysis",
        ]
    )


class HedgeFundLetterScraper:
    def __init__(self, min_delay_seconds: float = 0.75):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
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
        if response.status_code in {403, 429} and response.text and "fiscal.ai" in urlparse(target).netloc.lower():
            return response
        response.raise_for_status()
        return response

    def discover_documents(self, base_url: str = "", max_pages: int = 1) -> List[Dict[str, Any]]:
        max_items = max(10, int(max_pages or 1) * 25)
        source_count = max(1, len(HEDGE_FUND_LETTER_SOURCES))
        per_source_limit = max(5, (max_items + source_count - 1) // source_count)
        debug: Dict[str, Any] = {"errors": [], "sources": []}
        docs: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for source_key, cfg in HEDGE_FUND_LETTER_SOURCES.items():
            source_url = str(cfg.get("default_url", "") or "").strip()
            if source_key == "fiscal_ai_fund_letters" and base_url:
                source_url = str(base_url).strip()
            try:
                source_docs = self._discover_source(source_key, cfg, source_url, max_items=per_source_limit)
            except Exception as exc:
                debug["errors"].append({"source": source_key, "error": str(exc)})
                source_docs = []
            debug["sources"].append({"source": source_key, "items_found": len(source_docs), "url": source_url})
            for item in source_docs:
                key = _url_key(item.get("url", ""))
                if not key or key in seen:
                    continue
                seen.add(key)
                docs.append(item)

        docs = docs[:max_items]
        debug["items_found"] = len(docs)
        self.last_discovery_debug = debug
        return docs

    def _discover_source(self, source_key: str, cfg: Dict[str, Any], url: str, max_items: int) -> List[Dict[str, Any]]:
        response = self._fetch(url)
        soup = BeautifulSoup(response.text, "html.parser")
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        mode = str(cfg.get("mode", "") or "")

        for anchor in soup.select("a[href]"):
            href = str(anchor.get("href", "") or "").strip()
            text = _normalize_space(anchor.get_text(" ", strip=True))
            doc_url = urljoin(response.url, href)
            combined = f"{text} {href}"

            if mode == "fiscal_letters":
                if urlparse(doc_url).netloc.lower().endswith("fiscal.ai"):
                    continue
                if not _is_probable_letter_link(text, href):
                    continue
            elif mode == "investor_letter_directory":
                if not text:
                    continue
                if _url_key(doc_url) == _url_key(response.url) or not _is_probable_letter_link(text, href):
                    continue
            elif mode == "fund_materials":
                material_text = combined.lower()
                if not _looks_like_pdf_url(doc_url):
                    continue
                if "fact sheet" in material_text or "notice of" in material_text or "articles of incorporation" in material_text:
                    continue
                if not any(term in material_text for term in ["annual report", "interim", "investor presentation", "letter"]):
                    continue
            elif mode == "greenlight_downloads":
                if "download.aspx" not in doc_url.lower() and not _is_probable_letter_link(text, href):
                    continue
            else:
                if not _is_probable_letter_link(text, href):
                    continue

            key = _url_key(doc_url)
            if not key or key in seen:
                continue
            seen.add(key)

            container = _best_container(anchor)
            container_text = _normalize_space(container.get_text(" ", strip=True))
            title = self._title_for_anchor(text=text, container_text=container_text, url=doc_url)
            out.append(
                self._entry(
                    source_key=source_key,
                    cfg=cfg,
                    url=doc_url,
                    title=title,
                    date=_extract_date(text, container_text, doc_url),
                    summary=container_text,
                    listing_page=response.url,
                    position=len(out) + 1,
                )
            )
            if len(out) >= max_items:
                break
        return out

    def _title_for_anchor(self, *, text: str, container_text: str, url: str) -> str:
        clean = _normalize_space(text)
        if clean and clean.lower() not in {"pdf", "download", "read more", "view"}:
            return clean[:300]
        if container_text:
            return container_text[:300]
        return _title_from_url(url)

    def _entry(
        self,
        *,
        source_key: str,
        cfg: Dict[str, Any],
        url: str,
        title: str,
        date: str,
        summary: str,
        listing_page: str,
        position: int,
    ) -> Dict[str, Any]:
        source_format = "pdf" if _looks_like_pdf_url(url) else "html"
        fund_name = _normalize_space(re.sub(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}\b", "", title))
        return {
            "url": str(url or "").strip(),
            "title": _normalize_space(title) or _title_from_url(url),
            "date": _date_to_display(date),
            "summary": _normalize_space(summary),
            "source_key": source_key,
            "source_label": str(cfg.get("label", "") or "").strip(),
            "organization": str(cfg.get("organization", "") or "").strip(),
            "doc_type": "Investor Letter",
            "tags_csv": "hedge-fund,investor-letter,market-commentary",
            "source_format": source_format,
            "listing_page": listing_page,
            "search_position": position,
            "fund_name": fund_name,
        }

    def extract_document(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        url = _normalize_space(entry.get("url", ""))
        if not url:
            return {"success": False, "error": "No URL provided", "data": {}}

        try:
            response = self._fetch(url, timeout=90)
        except Exception as exc:
            return self._metadata_fallback(entry, error=str(exc))

        final_url = str(getattr(response, "url", url) or url)
        content_type = str(response.headers.get("content-type", "") or "").lower()
        is_pdf = _looks_like_pdf_url(final_url) or "application/pdf" in content_type or response.content[:4] == b"%PDF"

        if is_pdf:
            try:
                text = self._pdf_text(response.content)
            except Exception as exc:
                return self._metadata_fallback(entry, error=f"PDF extraction failed: {exc}")
            title = _normalize_space(entry.get("title", "")) or _title_from_url(final_url)
            date_text = _normalize_space(entry.get("date", "")) or _extract_date(title, final_url)
            source_format = "pdf"
            extraction_mode = "pdf"
        else:
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup.select("script, style, noscript, nav, footer, header, form"):
                tag.decompose()
            title_node = soup.find("h1") or soup.find("title")
            page_title = _normalize_space(title_node.get_text(" ", strip=True) if title_node else "")
            title = page_title or _normalize_space(entry.get("title", "")) or _title_from_url(final_url)
            body = soup.select_one("article") or soup.select_one("main") or soup.body or soup
            text = _clean_multiline(body.get_text("\n"))
            date_text = _normalize_space(entry.get("date", "")) or _extract_date(text, title, final_url)
            source_format = "html"
            extraction_mode = "html"

        if not text.strip():
            return self._metadata_fallback(entry, error="No usable text extracted")

        return {
            "success": True,
            "data": {
                "url": final_url,
                "title": title,
                "date": _date_to_display(date_text),
                "summary": _normalize_space(entry.get("summary", "")),
                "full_text": text,
                "source_format": source_format,
                "extraction_mode": extraction_mode,
                "word_count": len(text.split()),
                "fund_name": _normalize_space(entry.get("fund_name", "")),
                "source_label": _normalize_space(entry.get("source_label", "")),
                "source_key": _normalize_space(entry.get("source_key", "")),
            },
        }

    def _metadata_fallback(self, entry: Dict[str, Any], *, error: str) -> Dict[str, Any]:
        title = _normalize_space(entry.get("title", "")) or _title_from_url(entry.get("url", ""))
        date_text = _normalize_space(entry.get("date", ""))
        summary = _normalize_space(entry.get("summary", ""))
        url = _normalize_space(entry.get("url", ""))
        if not title and not summary:
            return {"success": False, "error": error, "data": {}}
        text = "\n".join(
            part
            for part in [
                title,
                f"Published Date: {date_text}" if date_text else "",
                summary,
                f"Source URL: {url}" if url else "",
                f"Detail fetch note: {error}" if error else "",
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
                "fund_name": _normalize_space(entry.get("fund_name", "")),
                "source_label": _normalize_space(entry.get("source_label", "")),
                "source_key": _normalize_space(entry.get("source_key", "")),
            },
        }

    def _pdf_text(self, content: bytes) -> str:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages[:40]:
            pages.append(page.extract_text() or "")
        return _clean_multiline("\n".join(pages))
