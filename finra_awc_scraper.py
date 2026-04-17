#!/usr/bin/env python3
"""
FINRA AWC (Acceptance, Waiver & Consent) scraper.

Discovery:  Scrapes FINRA's public disciplinary actions listing page (HTML),
            parsing the Drupal views table for rich metadata (case ID, summary,
            firms/individuals, date) plus the linked PDF URL per row.
Extraction: Downloads each AWC PDF and extracts text via pypdf.
"""

from __future__ import annotations

import io
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse, urlencode, urlunparse, parse_qs, urlparse

from curl_cffi import requests as cffi_requests
from bs4 import BeautifulSoup, Tag


# Base listing URL — type filter restricts to AWC documents only
FINRA_AWC_INDEX_URL = (
    "https://www.finra.org/rules-guidance/oversight-enforcement/"
    "finra-disciplinary-actions?field_fda_document_type_tax=AWC"
)
FINRA_AWC_BASE = (
    "https://www.finra.org/rules-guidance/oversight-enforcement/finra-disciplinary-actions"
)


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
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%Y-%m-%d", "%B %Y", "%b %Y"):
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


def _cell_text(row: Tag, *partial_classes: str) -> str:
    """Return stripped text from the first <td> whose class contains any of the partial strings."""
    for partial in partial_classes:
        td = row.find("td", class_=lambda c: c and partial in c)
        if td:
            return _normalize_space(td.get_text(" ", strip=True))
    return ""


def _cell_link(row: Tag, *partial_classes: str) -> str:
    """Return the first href from a <td> matching any of the partial class strings."""
    for partial in partial_classes:
        td = row.find("td", class_=lambda c: c and partial in c)
        if td:
            a = td.find("a", href=True)
            if a:
                return str(a["href"]).strip()
    return ""


def _row_pdf_url(row: Tag, page_url: str) -> str:
    """Scan a table row for any PDF anchor and return its absolute URL."""
    for a in row.find_all("a", href=True):
        href = str(a["href"]).strip()
        if href.lower().endswith(".pdf"):
            return urljoin(page_url, href)
    return ""


def _row_detail_url(row: Tag, page_url: str) -> str:
    """Return the first internal detail-page link from the row (case ID link)."""
    for partial in ("fda-case-id", "case-id", "field-fda-case"):
        href = _cell_link(row, partial)
        if href:
            return urljoin(page_url, href)
    return ""


def _ensure_awc_filter(url: str) -> str:
    """Add ?field_fda_document_type_tax=AWC if not already present."""
    raw = str(url or "").strip()
    if not raw:
        return FINRA_AWC_INDEX_URL
    if "field_fda_document_type_tax" not in raw:
        sep = "&" if "?" in raw else "?"
        raw = f"{raw}{sep}field_fda_document_type_tax=AWC"
    return raw


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

    def _fetch(self, url: str, timeout: int = 60) -> Any:
        self._rate_limit()
        resp = self.session.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp

    def _fetch_bytes(self, url: str, timeout: int = 120) -> bytes:
        self._rate_limit()
        resp = self.session.get(url, timeout=timeout)
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
        **_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Scrape the FINRA disciplinary actions listing page for AWC documents.

        Each table row provides:
          - Case ID (field_fda_case_id_txt column)
          - Case Summary (text description of the violation)
          - Firms / Individuals subject to the action
          - Action Date (field_core_official_dt column)
          - PDF link (direct download of the AWC document)

        The AWC type filter is appended to the URL automatically.
        Returns a list of discovery dicts, newest first.
        """
        index_url = _ensure_awc_filter(str(base_url or FINRA_AWC_INDEX_URL).strip())
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for page_num in range(max_pages):
            page_url = self._page_url(index_url, page_num)
            try:
                resp = self._fetch(page_url, timeout=45)
            except Exception as exc:
                if page_num == 0:
                    raise RuntimeError(
                        f"Failed to fetch FINRA AWC listing page: {exc}"
                    ) from exc
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup.find_all(["script", "style", "noscript"]):
                tag.decompose()

            # ── Find the views table ──────────────────────────────────────
            table = (
                soup.select_one("table.views-table")
                or soup.select_one("div.view-content table")
                or soup.find("table")
            )
            rows = table.select("tbody tr") if table else []

            # ── Fallback: div-based views rows ────────────────────────────
            if not rows:
                rows = soup.select("div.views-row")  # type: ignore[assignment]

            found_on_page = 0
            for row in rows:
                # ── Extract metadata columns ──────────────────────────────
                # Case ID — Drupal class: views-field-field-fda-case-id-txt
                case_id = _cell_text(row, "fda-case-id-txt", "fda-case-id", "case-id")

                # Case Summary — Drupal class: views-field-field-fda-case-summary
                case_summary = _cell_text(row, "fda-case-summary", "case-summary", "fda-summary")

                # Firms / Individuals — combined entity column
                subject_text = _cell_text(
                    row,
                    "fda-firms-individuals",
                    "fda-entities",
                    "fda-firms",
                    "fda-individuals",
                    "firms-individuals",
                )

                # Action date
                date_raw = _cell_text(row, "field-core-official-dt", "official-dt", "action-date")

                # Document type (should be AWC, but capture for completeness)
                doc_type = _cell_text(
                    row, "field-fda-document-type", "fda-document-type", "document-type"
                ) or "AWC"

                # ── Find the PDF URL ──────────────────────────────────────
                pdf_url = _row_pdf_url(row, page_url)

                # ── Find detail page link as fallback ─────────────────────
                detail_url = _row_detail_url(row, page_url)

                # Primary URL for this document
                canonical_url = pdf_url or detail_url
                if not canonical_url:
                    # Last-resort: any link in the row
                    a = row.find("a", href=True)
                    if a:
                        canonical_url = urljoin(page_url, str(a["href"]).strip())
                if not canonical_url:
                    continue

                key = _url_key(canonical_url)
                if key in seen:
                    continue
                seen.add(key)
                found_on_page += 1

                # Build a descriptive title
                if subject_text:
                    title = f"FINRA AWC — {subject_text}"
                elif case_id:
                    title = f"FINRA AWC {case_id}"
                else:
                    title = "FINRA AWC"

                out.append(
                    {
                        "url": canonical_url,
                        "pdf_url": pdf_url,
                        "detail_url": detail_url,
                        "title": title,
                        "date": _date_display(date_raw),
                        "case_id": case_id,
                        "subject_text": subject_text,
                        "case_summary": case_summary[:500] if case_summary else "",
                        "doc_type": _normalize_space(doc_type),
                        "source_format": "pdf" if pdf_url else "html",
                        "discovery_source": "html_table",
                        "listing_page": page_url,
                    }
                )

            # If page 0 found nothing, raise a clear error for debugging
            if page_num == 0 and found_on_page == 0:
                raise RuntimeError(
                    f"No AWC rows found on FINRA listing page: {page_url}\n"
                    "The page HTML structure may have changed. "
                    "Inspect the page and update CSS selectors in finra_awc_scraper.py."
                )

            if found_on_page == 0:
                break  # No more pages

        out.sort(
            key=lambda x: _parse_date(x.get("date", "")) or datetime.min,
            reverse=True,
        )
        return out

    # ── Detail-page fallback ──────────────────────────────────────────────────

    def _resolve_pdf_from_detail(self, detail_url: str) -> str:
        """Fetch a case detail page and return the first PDF link found."""
        try:
            resp = self._fetch(detail_url, timeout=45)
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = str(a["href"]).strip()
                if href.lower().endswith(".pdf"):
                    return urljoin(detail_url, href)
        except Exception:
            pass
        return ""

    # ── Extraction ────────────────────────────────────────────────────────────

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
        Download a FINRA AWC PDF (or HTML detail page) and return its full text.

        If the URL points to a detail page rather than a PDF, the method tries
        to find the PDF link on that page first.
        """
        target = str(url or "").strip()
        if not target:
            raise ValueError("URL is required")

        is_pdf = target.lower().endswith(".pdf")
        full_text = ""

        if is_pdf:
            content = self._fetch_bytes(target, timeout=120)
            full_text = _pdf_text(content)
        else:
            # It's a detail page — try to find and download the linked PDF
            pdf_found = self._resolve_pdf_from_detail(target)
            if pdf_found:
                target = pdf_found
                is_pdf = True
                content = self._fetch_bytes(target, timeout=120)
                full_text = _pdf_text(content)
            else:
                # Fall back to extracting the detail page HTML text
                resp = self._fetch(target, timeout=60)
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup.find_all(["script", "style", "noscript"]):
                    tag.decompose()
                article = soup.find("article") or soup.find("main") or soup
                full_text = _clean_multiline(article.get_text("\n"))

        title = str(fallback_title or "").strip() or "FINRA AWC"
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
