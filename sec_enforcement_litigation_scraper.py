#!/usr/bin/env python3
"""
SEC Enforcement Litigation Releases scraper.

Discovers release detail-page links from the Litigation Releases listing and
extracts full release text from each detail page.
"""

import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests


SEC_LITIGATION_RELEASES_URL = "https://www.sec.gov/enforcement-litigation/litigation-releases"


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clean_multiline(text: str) -> str:
    lines = []
    for raw in str(text or "").splitlines():
        line = _normalize_space(raw)
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


def _url_key(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}"


def _release_no_from_text(text: str) -> str:
    blob = str(text or "")
    m = re.search(r"Release\s+No\.\s*LR[-\s]?(\d+)", blob, flags=re.IGNORECASE)
    if m:
        return f"LR-{m.group(1)}"
    return ""


def _release_no_from_url(url: str) -> str:
    m = re.search(r"/lr[-_]?(\d+)", str(url or "").lower())
    if m:
        return f"LR-{m.group(1)}"
    return ""


class SECEnforcementLitigationScraper:
    def __init__(self, min_delay_seconds: float = 0.8):
        self.session = cffi_requests.Session(impersonate="chrome")
        self.min_delay_seconds = max(0.0, float(min_delay_seconds))
        self._last_request_ts = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_delay_seconds:
            time.sleep(self.min_delay_seconds - elapsed)
        self._last_request_ts = time.time()

    def discover_documents(
        self,
        base_url: str = SEC_LITIGATION_RELEASES_URL,
        max_pages: int = 3,
    ) -> List[Dict[str, str]]:
        max_pages = max(1, int(max_pages or 1))
        out = []
        seen = set()

        for page in range(max_pages):
            page_url = f"{base_url}?page={page}" if page > 0 else base_url
            self._rate_limit()
            response = self.session.get(page_url, timeout=45)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            table = soup.select_one("table.views-table")
            if table is None:
                table = soup.find("table")
            if table is None:
                continue

            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue

                date_text = _normalize_space(cells[0].get_text(" ", strip=True))
                main_cell = cells[1]
                rel_link = main_cell.find(
                    "a",
                    href=re.compile(r"/enforcement-litigation/litigation-releases/lr-\d+", re.IGNORECASE),
                )
                if rel_link is None:
                    continue

                doc_url = urljoin("https://www.sec.gov", rel_link.get("href", ""))
                key = _url_key(doc_url)
                if key in seen:
                    continue
                seen.add(key)

                title = _normalize_space(rel_link.get_text(" ", strip=True))
                cell_text = _normalize_space(main_cell.get_text(" ", strip=True))
                release_no = _release_no_from_text(cell_text) or _release_no_from_url(doc_url)

                out.append(
                    {
                        "url": doc_url,
                        "title": title,
                        "date": _date_to_display(date_text),
                        "release_no": release_no,
                        "source_format": "html",
                        "listing_page": page_url,
                    }
                )

        def _sort_key(item: Dict[str, str]):
            return _parse_date_text(item.get("date", "")) or datetime.min

        out.sort(key=_sort_key, reverse=True)
        return out

    def extract_document(
        self,
        url: str,
        fallback_title: str = "",
        fallback_date: str = "",
        fallback_release_no: str = "",
    ) -> Dict[str, Any]:
        self._rate_limit()
        response = self.session.get(url, timeout=60)
        response.raise_for_status()
        final_url = str(getattr(response, "url", url) or url)

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup.find_all(["script", "style", "noscript"]):
            tag.decompose()

        h1 = soup.find("h1")
        title = _normalize_space(h1.get_text(" ", strip=True) if h1 else "")
        if not title:
            title_tag = soup.title.get_text(" ", strip=True) if soup.title else ""
            title = re.sub(r"^SEC\.gov\s*\|\s*", "", title_tag, flags=re.IGNORECASE).strip()
        if not title:
            title = str(fallback_title or "").strip() or "SEC Litigation Release"

        body = (
            soup.select_one("div.field--name-body")
            or soup.select_one("div.field--type-text-with-summary")
            or soup.find("article")
            or soup.find("main")
            or soup
        )
        full_text = _clean_multiline(body.get_text("\n"))

        page_text = _normalize_space(body.get_text(" ", strip=True))
        release_no = _release_no_from_text(page_text) or _release_no_from_url(final_url) or str(fallback_release_no or "")

        date_value = ""
        m = re.search(
            r"Litigation\s+Release\s+No\.\s*LR[-\s]?\d+\s*/\s*([A-Za-z]{3,9}\.?\s+\d{1,2},\s+\d{4})",
            page_text,
            flags=re.IGNORECASE,
        )
        if m:
            date_value = _date_to_display(m.group(1))
        if not date_value:
            # fallback: first date mention in body
            m2 = re.search(
                r"((?:January|February|March|April|May|June|July|August|September|October|November|December|"
                r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|May|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)"
                r"\s+\d{1,2},\s+\d{4})",
                page_text,
                flags=re.IGNORECASE,
            )
            if m2:
                date_value = _date_to_display(m2.group(1))
        if not date_value:
            date_value = _date_to_display(fallback_date)

        return {
            "success": True,
            "data": {
                "url": final_url,
                "title": title,
                "date": date_value,
                "release_no": release_no,
                "full_text": full_text,
                "word_count": len(full_text.split()),
                "source_format": "html",
            },
        }

