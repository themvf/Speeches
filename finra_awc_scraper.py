#!/usr/bin/env python3
"""
FINRA AWC (Acceptance, Waiver & Consent) scraper.

Discovers AWC enforcement documents from FINRA's public disciplinary actions
search API (EFTS) and extracts full text from the associated PDF documents.

Discovery:  FINRA EFTS JSON API — returns structured metadata per AWC
Extraction: Download PDF from FINRA CDN, parse text with pypdf
"""

from __future__ import annotations

import io
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests


# FINRA EFTS (Enterprise Full-Text Search) disciplinary actions API
FINRA_AWC_API_URL = "https://efts.finra.org/EFTS/v2/broker-dealer/disciplinary-actions"

# Human-readable landing page (used as base_url sentinel / default)
FINRA_AWC_INDEX_URL = "https://www.finra.org/rules-guidance/oversight-enforcement/finra-disciplinary-actions"

# Base URL for FINRA-hosted PDFs
FINRA_PDF_CDN = "https://www.finra.org/sites/default/files"

DEFAULT_LOOKBACK_DAYS = 90
API_PAGE_SIZE = 20  # FINRA API practical maximum per page


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
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
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


def _build_pdf_url(source: Dict[str, Any]) -> str:
    """
    Attempt to reconstruct the AWC PDF URL from the EFTS API _source dict.
    FINRA has changed its CDN structure over time, so we try several patterns.
    """
    # Direct URL fields
    for key in ("Document URL", "document_url", "URL", "url", "PDF URL", "pdf_url"):
        val = str(source.get(key, "") or "").strip()
        if val:
            return val if val.startswith("http") else f"https://www.finra.org{val}"

    # AWS S3 filename + year
    s3_name = str(source.get("AWS S3 Filename", "") or source.get("S3Filename", "") or "").strip()
    if s3_name:
        issue = str(source.get("Issue Date", "") or "").strip()
        dt = _parse_date(issue)
        year = str(dt.year) if dt else ""
        if year:
            return f"{FINRA_PDF_CDN}/{year}/{s3_name}"
        return f"{FINRA_PDF_CDN}/{s3_name}"

    # Nested Document object
    doc = source.get("Document", {})
    if isinstance(doc, dict):
        fp = str(doc.get("FilePath", "") or doc.get("FileName", "") or "").strip()
        if fp:
            return f"https://www.finra.org/{fp.lstrip('/')}"

    return ""


def _sanctions_summary(sanctions: Any) -> str:
    if not isinstance(sanctions, list):
        return ""
    parts: List[str] = []
    for s in sanctions:
        if isinstance(s, dict):
            kind = str(s.get("Sanction Type", "") or s.get("type", "") or "").strip()
            amount = str(s.get("Fine Amount", "") or s.get("amount", "") or "").strip()
            desc = str(s.get("Description", "") or s.get("description", "") or "").strip()
            piece = kind
            if amount:
                piece += f": ${amount}" if not amount.startswith("$") else f": {amount}"
            if desc:
                piece += f" — {desc}"
            if piece:
                parts.append(piece)
        elif isinstance(s, str) and s.strip():
            parts.append(s.strip())
    return "; ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Scraper class
# ──────────────────────────────────────────────────────────────────────────────

class FINRAAWCScraper:
    def __init__(self, min_delay_seconds: float = 1.0):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Origin": "https://www.finra.org",
                "Referer": "https://www.finra.org/",
            }
        )
        self.min_delay_seconds = max(0.0, float(min_delay_seconds))
        self._last_request_ts = 0.0

    # ── Internal HTTP ──────────────────────────────────────────────────────

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_delay_seconds:
            time.sleep(self.min_delay_seconds - elapsed)
        self._last_request_ts = time.time()

    def _get_json(self, url: str, params: Optional[Dict] = None, timeout: int = 45) -> Any:
        self._rate_limit()
        resp = self.session.get(url, params=params, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        return resp.json()

    def _get_bytes(self, url: str, timeout: int = 120) -> bytes:
        self._rate_limit()
        resp = self.session.get(url, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        return resp.content

    # ── Discovery ─────────────────────────────────────────────────────────

    def discover_documents(
        self,
        base_url: str = FINRA_AWC_INDEX_URL,
        max_pages: int = 3,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        start_date: str = "",
        end_date: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Query FINRA's EFTS API for recent AWC documents.

        Args:
            base_url:     Ignored if it's the human-facing page; API URL is used.
            max_pages:    How many paginated API pages to fetch (20 docs each).
            lookback_days: How far back to search when no explicit dates given.
            start_date:   ISO date override (YYYY-MM-DD).
            end_date:     ISO date override (YYYY-MM-DD).

        Returns:
            List of discovery dicts, sorted newest-first.
        """
        now = datetime.now(timezone.utc)
        if not end_date:
            end_date = now.strftime("%Y-%m-%d")
        if not start_date:
            start_date = (now - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        # Normalise: if caller passed the human-readable page, use the API URL
        api_url = str(base_url or FINRA_AWC_INDEX_URL).strip()
        if "rules-guidance" in api_url or "disciplinary-actions" in api_url and "efts" not in api_url:
            api_url = FINRA_AWC_API_URL

        out: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for page_num in range(max_pages):
            skip = page_num * API_PAGE_SIZE
            params = {
                "q": "",
                "dateRange": "custom",
                "startDate": start_date,
                "endDate": end_date,
                "firms": "",
                "individuals": "",
                "skip": skip,
                "limit": API_PAGE_SIZE,
                "action": "AWC",
            }

            try:
                data = self._get_json(api_url, params=params)
            except Exception as exc:
                raise RuntimeError(f"FINRA AWC API request failed (page {page_num}): {exc}") from exc

            hits_block = data.get("hits", {}) if isinstance(data, dict) else {}
            hits = hits_block.get("hits", []) if isinstance(hits_block, dict) else []

            if not hits:
                break

            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                source = hit.get("_source", {})
                if not isinstance(source, dict):
                    continue

                doc_id = _normalize_space(source.get("Document ID", "") or hit.get("_id", ""))
                doc_type = _normalize_space(source.get("Document Type", "AWC"))
                issue_date = _normalize_space(
                    source.get("Issue Date", "") or source.get("Resolution Date", "")
                )

                pdf_url = _build_pdf_url(source)

                # Canonical URL: PDF if available, otherwise a fragment anchor
                canonical_url = pdf_url
                if not canonical_url and doc_id:
                    canonical_url = f"{FINRA_AWC_INDEX_URL}#{doc_id}"
                if not canonical_url:
                    continue

                key = _url_key(canonical_url)
                if key in seen:
                    continue
                seen.add(key)

                # Build subject string from firms + individuals
                firms = source.get("Firms", [])
                individuals = source.get("Individuals", [])
                subject_parts: List[str] = []
                if isinstance(firms, list):
                    subject_parts += [
                        _normalize_space(f.get("Name", "") if isinstance(f, dict) else str(f))
                        for f in firms
                        if f
                    ]
                if isinstance(individuals, list):
                    subject_parts += [
                        _normalize_space(i.get("Name", "") if isinstance(i, dict) else str(i))
                        for i in individuals
                        if i
                    ]
                subject_text = "; ".join(s for s in subject_parts if s)

                case_summary = _normalize_space(source.get("Case Summary", ""))
                sanctions_text = _sanctions_summary(source.get("Sanctions", []))
                case_id = _normalize_space(
                    source.get("Case ID", "") or source.get("Case Number", "") or doc_id
                )

                title = (
                    f"FINRA AWC — {subject_text}" if subject_text else f"FINRA AWC {doc_id or case_id}"
                )

                out.append(
                    {
                        "url": canonical_url,
                        "pdf_url": pdf_url,
                        "title": title,
                        "date": _date_display(issue_date),
                        "doc_id": doc_id,
                        "case_id": case_id,
                        "doc_type": doc_type,
                        "subject_text": subject_text,
                        "case_summary": case_summary,
                        "sanctions_text": sanctions_text,
                        "source_format": "pdf" if pdf_url else "html",
                        "discovery_source": "api",
                    }
                )

            # Pagination check
            total_block = hits_block.get("total", {})
            if isinstance(total_block, dict):
                total_count = int(total_block.get("value", 0) or 0)
            else:
                total_count = int(total_block or 0)
            if skip + API_PAGE_SIZE >= total_count:
                break

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
        Download a FINRA AWC PDF and extract its full text.

        Returns:
            {'success': True, 'data': {...}} or raises RuntimeError on failure.
        """
        target = str(url or "").strip()
        if not target:
            raise ValueError("URL is required")

        is_pdf = target.lower().endswith(".pdf") or "sites/default/files" in target.lower()

        full_text = ""
        if is_pdf:
            content = self._get_bytes(target, timeout=120)
            full_text = _pdf_text(content)

        title = str(fallback_title or "").strip() or "FINRA AWC"
        display_date = _date_display(fallback_date)

        # Structured header block that enrichment can use even if PDF parsing fails
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
