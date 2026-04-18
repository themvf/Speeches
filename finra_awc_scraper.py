#!/usr/bin/env python3
"""
FINRA AWC (Acceptance, Waiver & Consent) scraper.

Discovery:  Scrapes FINRA's public disciplinary actions listing page (HTML),
            finding PDF links per row.  Metadata (case ID, subject, doc type)
            is parsed directly from the PDF filename, which FINRA encodes as:
              {case_id} {subject_name} CRD {crd_number} {doc_type} {initials}.pdf
Extraction: Downloads each AWC PDF and extracts text via pypdf.
"""

from __future__ import annotations

import io
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse, unquote

import requests as std_requests
from curl_cffi import requests as cffi_requests
from bs4 import BeautifulSoup, Tag


FINRA_AWC_INDEX_URL = (
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


def _parse_pdf_filename(pdf_url: str) -> Dict[str, str]:
    """
    Extract metadata from a FINRA disciplinary-action PDF filename.

    FINRA encodes filenames as:
      {case_id} {subject_name} CRD {crd_number} {doc_type} {initials}.pdf

    Examples:
      2022076873501 Peter M. Rosten CRD 2973115 AWC vrp.pdf
      2023077012601 Brentwood Capital Advisors LLC CRD 118712 AWC lp.pdf
      2018056490315 Shadi T Barakat CRD 5031281 OHO Decision df.pdf

    Returns a dict with keys: case_id, subject_text, crd_number, doc_type.
    Returns {} if the filename does not match the expected pattern.
    """
    if not pdf_url:
        return {}
    filename = unquote(pdf_url.rstrip("/").split("/")[-1])
    if not filename.lower().endswith(".pdf"):
        return {}
    stem = filename[:-4]

    # Strip trailing parenthetical timestamp, e.g. " (2025-1738282793208)"
    stem = re.sub(r"\s*\(\d{4}-\d+\)\s*$", "", stem).strip()

    # Space-separated format (most common):
    #   {case_id} {subject} CRD [No.] {crd_number} {doc_type} {initials}
    # Subject may end with a comma: "LaBarbara, CRD …" → strip trailing comma from subject.
    m = re.match(
        r"^(\d+)\s+"                    # case_id
        r"(.+?),?\s+"                   # subject (optional trailing comma)
        r"CRD(?:\s+No\.?)?\s+"          # "CRD", "CRD No.", or "CRD No"
        r"(\d+)\s+"                     # crd_number
        r"(.+?)\s+"                     # doc_type (may be multi-word e.g. "OHO Decision")
        r"([a-z]{2,6})$",               # initials
        stem,
    )
    if m:
        return {
            "case_id": m.group(1),
            "subject_text": m.group(2).rstrip(",").strip(),
            "crd_number": m.group(3),
            "doc_type": m.group(4).strip(),
        }

    # Underscore-separated format (used by some NAC decisions):
    #   {case_id}_{subject}_{crd_number}_{doc_type}_{initials}
    m = re.match(
        r"^(\d+)_"
        r"(.+?)_"
        r"(\d+)_"
        r"(.+?)_"
        r"([a-z]{2,6})$",
        stem,
    )
    if m:
        return {
            "case_id": m.group(1),
            "subject_text": m.group(2).replace("_", " ").strip(),
            "crd_number": m.group(3),
            "doc_type": m.group(4).replace("_", " ").strip(),
        }

    # Space-separated without CRD number (older FINRA naming):
    #   {case_id} {subject} {doc_type} {initials}
    m = re.match(
        r"^(\d+)\s+"
        r"(.+?)\s+"
        r"(AWC|OHO\s+Decision|OHO|NAC|SC|Settlement)\s+"
        r"([a-z]{2,6})$",
        stem,
    )
    if m:
        return {
            "case_id": m.group(1),
            "subject_text": m.group(2).strip(),
            "doc_type": m.group(3).strip(),
        }

    # Fallback: at minimum extract case_id from the leading digits
    m2 = re.match(r"^(\d+)", stem)
    if m2:
        return {"case_id": m2.group(1)}

    return {}


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


def fetch_page_with_browser(url: str, headless: bool = True, wait_seconds: int = 4) -> str:
    """Fetch *url* using a real Chrome instance so FINRA WAF/JS challenges pass.

    Uses Selenium with Chrome's built-in chromedriver manager (Selenium 4.6+).
    Returns the rendered page HTML (``driver.page_source``).
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=opts)
    try:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"},
        )
        driver.get(url)
        time.sleep(wait_seconds)
        return driver.page_source
    finally:
        driver.quit()


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

    def _fetch(self, url: str, timeout: int = 60, retries: int = 3) -> Any:
        """
        Fetch a URL with retry/backoff.
        Tries curl_cffi first (Chrome TLS fingerprint); on 5xx falls back to
        plain requests with a standard browser User-Agent.
        """
        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(1, retries + 1):
            self._rate_limit()
            try:
                resp = self.session.get(url, timeout=timeout)
                if resp.status_code < 500:
                    resp.raise_for_status()
                    return resp
                # 5xx — try plain requests as fallback on last attempt
                if attempt == retries:
                    return self._fetch_plain(url, timeout)
                # back off before retry
                time.sleep(2 ** attempt)
            except Exception as exc:
                last_exc = exc
                if attempt == retries:
                    try:
                        return self._fetch_plain(url, timeout)
                    except Exception as plain_exc:
                        raise RuntimeError(
                            f"HTTP Error {plain_exc}"
                        ) from plain_exc
                time.sleep(2 ** attempt)
        raise last_exc

    def _fetch_plain(self, url: str, timeout: int = 60) -> Any:
        """Fallback fetch using plain requests with a browser User-Agent."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = std_requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
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

    # Drupal taxonomy term IDs for the document-type filter
    DOC_TYPE_TERM_IDS: Dict[str, str] = {
        "AWC": "4610",
        "Complaints": "4611",
        "Court of Appeals": "4612",
        "NAC": "4613",
        "OHO": "4614",
        "Settlement": "4615",
        "SEC": "4616",
        "SC": "4617",
    }

    def _build_filter_url(
        self,
        page: int = 0,
        doc_type: str = "AWC",
        date_min: str = "",
        date_max: str = "",
    ) -> str:
        """Build a FINRA disciplinary-actions listing URL with Drupal Views
        filter parameters.

        FINRA's filter params require the Drupal taxonomy *term ID* for the
        document type (e.g. 4610 for AWC), not the human-readable label.
        Date values should be in YYYY-MM-DD format.
        """
        from urllib.parse import urlencode, quote

        params: dict[str, str] = {}
        term_id = self.DOC_TYPE_TERM_IDS.get(doc_type, "") if doc_type else ""
        if term_id:
            params["field_fda_document_type_tax"] = term_id
        if date_min:
            params["field_core_official_dt[min]"] = date_min
        if date_max:
            params["field_core_official_dt[max]"] = date_max
        if page > 0:
            params["page"] = str(page)

        if params:
            qs = urlencode(params)
            return f"{FINRA_AWC_INDEX_URL}?{qs}"
        return FINRA_AWC_INDEX_URL

    def discover_documents(
        self,
        base_url: str = FINRA_AWC_INDEX_URL,
        max_pages: int = 5,
        doc_type: str = "AWC",
        date_min: str = "",
        date_max: str = "",
        **_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Scrape the FINRA disciplinary actions listing page for documents.

        The filter uses Drupal taxonomy term IDs via GET query parameters
        (e.g. ``?field_fda_document_type_tax=4610`` for AWC).  Pagination
        appends ``&page=N``.  Both work reliably with curl_cffi Chrome
        impersonation — only the human-readable string form of the filter
        (e.g. ``?field_fda_document_type_tax=AWC``) returns 503.

        Metadata (case ID, firm/individual name, doc type) is parsed from the
        PDF filename, which FINRA encodes with all key fields.  HTML table
        columns are used only for the action date and case summary.

        Returns a list of discovery dicts, newest first.
        """
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def _process_page(page_url: str, resp_obj: Any) -> int:
            nonlocal out
            soup = BeautifulSoup(resp_obj.text, "html.parser")
            for tag in soup.find_all(["script", "style", "noscript"]):
                tag.decompose()

            table = (
                soup.select_one("table.views-table")
                or soup.select_one("div.view-content table")
                or soup.find("table")
            )
            rows = table.select("tbody tr") if table else []
            if not rows:
                rows = soup.select("div.views-row")  # type: ignore[assignment]

            found = 0
            for row in rows:
                pdf_url = _row_pdf_url(row, page_url)
                file_meta = _parse_pdf_filename(pdf_url)
                case_id = file_meta.get("case_id", "")
                subject_text = file_meta.get("subject_text", "")
                date_raw = _cell_text(row, "field-core-official-dt", "official-dt", "action-date")
                case_summary = _cell_text(row, "fda-case-summary", "case-summary", "fda-summary")
                detail_url = _row_detail_url(row, page_url)

                canonical_url = pdf_url or detail_url
                if not canonical_url:
                    a = row.find("a", href=True)
                    if a:
                        canonical_url = urljoin(page_url, str(a["href"]).strip())
                if not canonical_url:
                    continue

                key = _url_key(canonical_url)
                if key in seen:
                    continue
                seen.add(key)
                found += 1

                if subject_text and case_id:
                    title = f"{subject_text} ({case_id})"
                elif subject_text:
                    title = subject_text
                elif case_id:
                    title = case_id
                else:
                    title = doc_type or "AWC"

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
                        "doc_type": file_meta.get("doc_type", doc_type or "AWC"),
                        "source_format": "pdf" if pdf_url else "html",
                        "discovery_source": "listing_get",
                        "listing_page": page_url,
                    }
                )
            return found

        for page_num in range(max_pages):
            page_url = self._build_filter_url(
                page=page_num, doc_type=doc_type,
                date_min=date_min, date_max=date_max,
            )
            try:
                resp = self._fetch(page_url, timeout=45)
            except Exception as exc:
                if page_num == 0:
                    raise RuntimeError(f"Failed to fetch FINRA listing page: {exc}") from exc
                break

            found = _process_page(page_url, resp)

            if page_num == 0 and found == 0:
                raise RuntimeError(
                    f"No rows found on FINRA listing page: {page_url}\n"
                    "The page structure may have changed — inspect and update selectors."
                )
            if found == 0:
                break

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

        # If no fallback metadata, try parsing from the URL itself
        if not fallback_case_id or not fallback_subject:
            file_meta = _parse_pdf_filename(target)
            fallback_case_id = fallback_case_id or file_meta.get("case_id", "")
            fallback_subject = fallback_subject or file_meta.get("subject_text", "")

        title = str(fallback_title or "").strip() or (
            f"{fallback_subject} ({fallback_case_id})" if fallback_subject and fallback_case_id
            else fallback_subject or fallback_case_id or "AWC"
        )
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
