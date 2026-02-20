#!/usr/bin/env python3
"""
SEC Trading & Markets FAQ scraper.

Discovers FAQ links from the Trading and Markets FAQ index and extracts
document text from linked HTML/PDF pages.
"""

import io
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests


SEC_TM_FAQ_INDEX_URL = "https://www.sec.gov/rules-regulations/staff-guidance/trading-markets-frequently-asked-questions"


def _clean_whitespace(text: str) -> str:
    lines = []
    for raw in str(text or "").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _parse_date_text(value: str) -> Optional[datetime]:
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
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _date_to_display(value: str) -> str:
    parsed = _parse_date_text(value)
    if parsed is None:
        return str(value or "").strip()
    return parsed.strftime("%B %d, %Y")


def _extract_first_date(text: str) -> str:
    pattern = (
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|May|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)"
        r"\s+\d{1,2},\s+\d{4})"
    )
    m = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
    if not m:
        return ""
    return _date_to_display(m.group(1))


class TradingMarketsFAQScraper:
    def __init__(self, min_delay_seconds: float = 0.8):
        self.session = cffi_requests.Session(impersonate="chrome")
        self.min_delay_seconds = max(0.0, float(min_delay_seconds))
        self._last_request = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.min_delay_seconds:
            time.sleep(self.min_delay_seconds - elapsed)
        self._last_request = time.time()

    @staticmethod
    def _url_key(url: str) -> str:
        raw = str(url or "").strip()
        if not raw:
            return ""
        parsed = urlparse(raw)
        scheme = (parsed.scheme or "https").lower()
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/") or "/"
        return f"{scheme}://{netloc}{path}"

    @staticmethod
    def _extract_updated_from_text(text: str) -> str:
        blob = str(text or "")
        patterns = [
            r"Last Reviewed or Updated:\s*([A-Za-z]{3,9}\.?\s+\d{1,2},\s+\d{4})",
            r"UPDATED\s+([A-Za-z]{3,9}\.?\s+\d{1,2},\s+\d{4})",
            r"Updated:\s*([A-Za-z]{3,9}\.?\s+\d{1,2},\s+\d{4})",
        ]
        for p in patterns:
            m = re.search(p, blob, flags=re.IGNORECASE)
            if m:
                return _date_to_display(m.group(1))
        return ""

    def discover_documents(
        self,
        index_url: str = SEC_TM_FAQ_INDEX_URL,
        include_pdfs: bool = True,
    ) -> List[Dict[str, str]]:
        self._rate_limit()
        response = self.session.get(index_url, timeout=45)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        body = (
            soup.select_one("div.field--name-body")
            or soup.select_one("div.field--type-text-with-summary")
            or soup.find("main")
            or soup
        )

        found = []
        seen = set()
        for li in body.find_all("li"):
            anchor = li.find("a", href=True)
            if not anchor:
                continue
            href = anchor.get("href", "")
            full_url = urljoin("https://www.sec.gov", href)
            parsed = urlparse(full_url)
            if "sec.gov" not in parsed.netloc.lower():
                continue
            if not include_pdfs and parsed.path.lower().endswith(".pdf"):
                continue
            if self._url_key(full_url) == self._url_key(index_url):
                continue

            key = self._url_key(full_url)
            if key in seen:
                continue
            seen.add(key)

            title = anchor.get_text(" ", strip=True)
            li_text = re.sub(r"\s+", " ", li.get_text(" ", strip=True)).strip()

            published_date = ""
            m_pub = re.search(r"\(([^)]*\d{4}[^)]*)\)", li_text)
            if m_pub:
                published_date = _date_to_display(m_pub.group(1))

            updated_date = self._extract_updated_from_text(li_text)
            source_format = "pdf" if parsed.path.lower().endswith(".pdf") else "html"

            found.append(
                {
                    "url": full_url,
                    "title": title,
                    "published_date": published_date,
                    "updated_date": updated_date,
                    "source_format": source_format,
                }
            )

        def _sort_key(item: Dict[str, str]):
            dt = _parse_date_text(item.get("updated_date", "")) or _parse_date_text(item.get("published_date", ""))
            return dt or datetime.min

        found.sort(key=_sort_key, reverse=True)
        return found

    def extract_document(
        self,
        url: str,
        fallback_title: str = "",
        fallback_date: str = "",
    ) -> Dict[str, Any]:
        self._rate_limit()
        response = self.session.get(url, timeout=60)
        response.raise_for_status()
        final_url = str(getattr(response, "url", url) or url)

        lower_path = urlparse(final_url).path.lower()
        if lower_path.endswith(".pdf"):
            return self._extract_pdf(response.content, final_url, fallback_title, fallback_date)
        return self._extract_html(response.text, final_url, fallback_title, fallback_date)

    def _extract_pdf(self, content: bytes, final_url: str, fallback_title: str, fallback_date: str) -> Dict[str, Any]:
        try:
            from pypdf import PdfReader
        except Exception as e:
            raise RuntimeError(f"PDF extraction requires pypdf: {e}")

        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            txt = (page.extract_text() or "").strip()
            if txt:
                pages.append(txt)
        full_text = "\n\n".join(pages).strip()

        title = str(fallback_title or "").strip()
        if not title:
            title = urlparse(final_url).path.rsplit("/", 1)[-1] or "SEC FAQ PDF"

        return {
            "success": True,
            "data": {
                "url": final_url,
                "title": title,
                "date": _date_to_display(fallback_date),
                "last_reviewed_or_updated": "",
                "full_text": full_text,
                "word_count": len(full_text.split()),
                "source_format": "pdf",
            },
        }

    def _extract_html(self, html: str, final_url: str, fallback_title: str, fallback_date: str) -> Dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(["script", "style", "noscript"]):
            tag.decompose()

        heading = soup.select_one("h1.page-title__heading") or soup.find("h1")
        title = heading.get_text(" ", strip=True) if heading else ""
        if not title:
            t = soup.title.get_text(" ", strip=True) if soup.title else ""
            title = re.sub(r"^SEC\.gov\s*\|\s*", "", t, flags=re.IGNORECASE).strip()
        if not title:
            title = str(fallback_title or "").strip() or "SEC Trading & Markets FAQ"

        body = (
            soup.select_one("div.field--name-body")
            or soup.select_one("div.field--type-text-with-summary")
            or soup.find("article")
            or soup.find("main")
            or soup
        )
        full_text = _clean_whitespace(body.get_text("\n"))
        if not full_text:
            full_text = _clean_whitespace(soup.get_text("\n"))

        page_text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        last_reviewed = self._extract_updated_from_text(page_text)
        if not last_reviewed:
            last_reviewed = _extract_first_date(page_text)

        doc_date = _date_to_display(last_reviewed or fallback_date)

        return {
            "success": True,
            "data": {
                "url": final_url,
                "title": title,
                "date": doc_date,
                "last_reviewed_or_updated": last_reviewed,
                "full_text": full_text,
                "word_count": len(full_text.split()),
                "source_format": "html",
            },
        }

