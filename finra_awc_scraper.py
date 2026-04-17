#!/usr/bin/env python3
"""
FINRA AWC (Acceptance, Waiver & Consent) scraper.

Discovery:  Scrapes FINRA's public disciplinary actions listing pages (HTML),
            collecting PDF links for AWC documents.
Extraction: Downloads each AWC PDF from FINRA's CDN and parses text via pypdf.
"""

from __future__ import annotations

import io
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from curl_cffi import requests as cffi_requests
from bs4 import BeautifulSoup


FINRA_AWC_INDEX_URL = (
    "https://www.finra.org/rules-guidance/oversight-enforcement/finra-disciplinary-actions"
)

# PDF CDN base — individual AWC docs live here
FINRA_PDF_CDN = "https://www.finra.org"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clean_multiline(text: Any) -> str:
    lines: List[str] = []
    for raw in str(text or "").splitlines():
        line = _normalize_space(raw)
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _parse_date(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    text = (
        text.replace("Jan.", "Jan").replace("Feb.", "Feb").replace("Mar.", "Mar")
        .replace("Apr.", "Apr").replace("Jun.", "Jun").replace("Jul.", "Jul")
        .replace("Aug.", "Aug").replace("Sep.", "Sep").replace("Sept.", "Sep")
        .replace("Oct.", "Oct").replace("Nov.", "Nov").replace("Dec.", "Dec")
    )
    for fmt in (
        "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%Y-%m-%d",
        "%B %Y", "%b %Y",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _date_display(value: Any) -> str:
    dt = _parse_date(value)
    return dt.strftime("%B %d, %Y") if dt else str(value or "").strip()


def _url_key(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    p = urlparse(raw)
    return f"{(p.scheme or 'https').lower()}://{p.netloc.lower()}{p.path.rstrip('/') or '/'}"


def _is_awc_url(href: str, text: str) -> bool:
    """Heuristic: does this link look like an AWC document?"""
    h = href.lower()
    t = text.lower()
    # Explicit AWC signals in URL or link text
    if "awc" in h or "awc" in t:
        return True
    # FINRA PDF document paths commonly contain fda_documents or disciplinary
    if "fda_documents" in h or "disciplinary" in h:
        return True
    return False


def _pdf_text(content: bytes) -> str:
    """Extract all text pages from a PDF byte blob via pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(f"PDF extraction requires pypdf: {exc}") from exc

    reader = PdfReader(io.BytesIO(content))
    pages: List[str] = []
    for page in reader.pages:
        txt = _clean_multiline(page.extract_text() or "")
        if txt:
            pages.append(txt)
    return "\n\n".join(pages)


def _title_from_pdf_url(url: str, fallback: str = "") -> str:
    """Derive a human-readable title from a PDF filename."""
    raw = str(url or "").strip()
    if not raw:
        return fallback or "FINRA AWC"
    filename = raw.rsplit("/", 1)[-1]
    stem = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    # e.g. "2026001234301" → keep as-is; "2026smith_awc" → pretty-print
    pretty = re.sub(r"[_\-]+", " ", stem).strip().title()
    return f"FINRA AWC — {pretty}" if pretty else (fallback or "FINRA AWC")


# ──────────────────────────────────────────────────────────────────────────────
# Scraper
# ──────────────────────────────────────────────────────────────────────────────

class FINRAAWCScraper:
    def __init__(self, min_delay_seconds: float = 1.0):
        # curl_cffi impersonates a real Chrome TLS fingerprint, bypassing Cloudflare
        self.session = cffi_requests.Session(impersonate="chrome")
        self.min_delay_seconds = max(0.0, float(min_delay_seconds))
        self._last_request_ts = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_delay_seconds:
            time.sleep(self.min_delay_seconds - elapsed)
        self._last_request_ts = time.time()

    def _fetch(self, url: str, timeout: int = 60) -> requests.Response:
        self._rate_limit()
        resp = self.session.get(url, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        return resp

    def _fetch_bytes(self, url: str, timeout: int = 120) -> bytes:
        self._rate_limit()
        resp = self.session.get(url, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        return resp.content

    @staticmethod
    def _page_url(base: str, page: int) -> str:
        if page == 0:
            return base
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}page={page}"

    # ── Discovery ─────────────────────────────────────────────────────────

    def discover_documents(
        self,
        base_url: str = FINRA_AWC_INDEX_URL,
        max_pages: int = 5,
        **_kwargs: Any,  # absorbs lookback_days / start_date / end_date if passed
    ) -> List[Dict[str, Any]]:
        """
        Scrape the FINRA disciplinary actions listing pages for AWC PDF links.

        The listing at finra.org/rules-guidance/oversight-enforcement/finra-disciplinary-actions
        is paginated HTML with Drupal views table rows. Each row contains a date and
        a link to an individual AWC PDF stored on FINRA's CDN.

        Returns:
            List of discovery dicts, newest first.
        """
        index_url = str(base_url or FINRA_AWC_INDEX_URL).strip() or FINRA_AWC_INDEX_URL
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for page_num in range(max_pages):
            page_url = self._page_url(index_url, page_num)
            try:
                resp = self._fetch(page_url, timeout=45)
            except Exception as exc:
                if page_num == 0:
                    raise RuntimeError(f"Failed to fetch FINRA AWC listing page: {exc}") from exc
                break  # Subsequent pages are optional; stop gracefully

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup.find_all(["script", "style", "noscript"]):
                tag.decompose()

            found_on_page = 0

            # ── Strategy 1: Drupal views-table rows (same structure as regulatory notices) ──
            rows = (
                soup.select("table.views-table tbody tr")
                or soup.select("div.view-content table tbody tr")
            )
            for row in rows:
                date_cell = (
                    row.select_one("td.views-field-field-core-official-dt")
                    or row.select_one("td.views-field-field-fda-action-date")
                    or row.select_one("td[class*='date']")
                )
                title_cell = row.select_one("td a[href]") or row.select_one("a[href]")
                if title_cell is None:
                    continue

                href = str(title_cell.get("href", "") or "").strip()
                link_text = _normalize_space(title_cell.get_text(" ", strip=True))

                if not href:
                    continue

                # Only AWC entries (skip Complaints, Decisions, etc.)
                if not _is_awc_url(href, link_text):
                    row_text = _normalize_space(row.get_text(" ", strip=True)).lower()
                    if "awc" not in row_text and "acceptance" not in row_text:
                        continue

                abs_url = urljoin(page_url, href)
                key = _url_key(abs_url)
                if key in seen:
                    continue
                seen.add(key)
                found_on_page += 1

                date_text = ""
                if date_cell:
                    time_el = date_cell.find("time")
                    if time_el and time_el.get("datetime"):
                        date_text = _date_display(time_el.get("datetime", ""))
                    if not date_text:
                        date_text = _date_display(date_cell.get_text(" ", strip=True))

                is_pdf = abs_url.lower().endswith(".pdf")
                title = link_text or _title_from_pdf_url(abs_url if is_pdf else "")

                out.append(
                    {
                        "url": abs_url,
                        "pdf_url": abs_url if is_pdf else "",
                        "title": title or "FINRA AWC",
                        "date": date_text,
                        "case_id": "",
                        "subject_text": "",
                        "case_summary": "",
                        "sanctions_text": "",
                        "source_format": "pdf" if is_pdf else "html",
                        "discovery_source": "html_table",
                        "listing_page": page_url,
                    }
                )

            # ── Strategy 2: Scan ALL PDF anchors on the page for AWC documents ──
            # (catches layouts that don't use a views-table)
            if found_on_page == 0:
                for anchor in soup.select("a[href]"):
                    href = str(anchor.get("href", "") or "").strip()
                    if not href.lower().endswith(".pdf"):
                        continue
                    link_text = _normalize_space(anchor.get_text(" ", strip=True))
                    if not _is_awc_url(href, link_text):
                        continue

                    abs_url = urljoin(page_url, href)
                    key = _url_key(abs_url)
                    if key in seen:
                        continue
                    seen.add(key)
                    found_on_page += 1

                    # Try to find a nearby date
                    date_text = ""
                    parent = anchor.parent
                    for _ in range(4):
                        if parent is None:
                            break
                        time_el = parent.find("time")
                        if time_el:
                            date_text = _date_display(
                                time_el.get("datetime", "") or time_el.get_text()
                            )
                            break
                        parent = parent.parent

                    out.append(
                        {
                            "url": abs_url,
                            "pdf_url": abs_url,
                            "title": link_text or _title_from_pdf_url(abs_url),
                            "date": date_text,
                            "case_id": "",
                            "subject_text": "",
                            "case_summary": "",
                            "sanctions_text": "",
                            "source_format": "pdf",
                            "discovery_source": "html_anchor_scan",
                            "listing_page": page_url,
                        }
                    )

            # If neither strategy found anything on page 0, the URL/structure is wrong
            if page_num == 0 and found_on_page == 0:
                raise RuntimeError(
                    f"No AWC documents found on FINRA listing page: {page_url}\n"
                    "The page structure may have changed. Check the URL and HTML selectors."
                )

            # If this page had no results, stop paginating
            if found_on_page == 0:
                break

        # Sort newest-first by parsed date, fall back to list order
        out.sort(
            key=lambda x: _parse_date(x.get("date", "")) or datetime.min,
            reverse=True,
        )
        return out

    # ── Extraction ────────────────────────────────────────────────────────

    def extract_document(
        self,
        url: str,
        fallback_title: str = "",
        fallback_date: str = "",
        fallback_case_id: str = "",
        fallback_subject: str = "",
        fallback_case_summary: str = "",
        fallback_sanctions: str = "",
    ) -> Dict[str, Any]:
        """
        Download a FINRA AWC PDF (or HTML page) and return its full text.
        """
        target = str(url or "").strip()
        if not target:
            raise ValueError("URL is required")

        is_pdf = target.lower().endswith(".pdf") or "/fda_documents/" in target.lower()
        full_text = ""

        if is_pdf:
            content = self._fetch_bytes(target, timeout=120)
            full_text = _pdf_text(content)
        else:
            # Attempt HTML extraction as fallback (e.g. a detail page)
            resp = self._fetch(target, timeout=60)
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup.find_all(["script", "style", "noscript"]):
                tag.decompose()
            article = soup.find("article") or soup.find("main") or soup
            full_text = _clean_multiline(article.get_text("\n"))

            # Look for a PDF link on the detail page and prefer that
            for anchor in soup.select("a[href]"):
                href = str(anchor.get("href", "") or "").strip()
                if href.lower().endswith(".pdf"):
                    pdf_abs = urljoin(target, href)
                    try:
                        content = self._fetch_bytes(pdf_abs, timeout=120)
                        pdf_extracted = _pdf_text(content)
                        if pdf_extracted:
                            full_text = pdf_extracted
                            target = pdf_abs
                            is_pdf = True
                    except Exception:
                        pass
                    break

        title = str(fallback_title or "").strip() or _title_from_pdf_url(target, "FINRA AWC")
        display_date = _date_display(fallback_date)

        header_parts = [
            "Document Type: AWC (Acceptance, Waiver & Consent)",
            f"Title: {title}",
            f"Case ID: {fallback_case_id}" if fallback_case_id else "",
            f"Subject: {fallback_subject}" if fallback_subject else "",
            f"Issue Date: {display_date}" if display_date else "",
            f"Sanctions: {fallback_sanctions}" if fallback_sanctions else "",
            f"Summary: {fallback_case_summary}" if fallback_case_summary else "",
            f"Source URL: {target}",
        ]
        header = "\n".join(p for p in header_parts if p)
        combined = f"{header}\n\n--- Document Text ---\n\n{full_text}".strip() if full_text else header

        return {
            "success": True,
            "data": {
                "url": target,
                "title": title,
                "date": display_date,
                "case_id": fallback_case_id,
                "subject_text": fallback_subject,
                "case_summary": fallback_case_summary,
                "sanctions_text": fallback_sanctions,
                "full_text": combined,
                "word_count": len(combined.split()),
                "source_format": "pdf" if is_pdf else "html",
                "pdf_url": target if is_pdf else "",
            },
        }
