#!/usr/bin/env python3
"""
FINRA comment letter scraper.

Discovers comment letters from a specific FINRA Regulatory Notice comments table and
extracts full text from either comment pages or linked PDF files.
"""

from __future__ import annotations

import io
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

try:
    from curl_cffi import requests as _cffi_requests

    _CURL_CFFI_AVAILABLE = True
except ImportError:
    _CURL_CFFI_AVAILABLE = False

logger = logging.getLogger(__name__)


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
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _date_to_display(value: Any) -> str:
    parsed = _parse_date_text(value)
    if parsed is None:
        return str(value or "").strip()
    return parsed.strftime("%B %d, %Y")


def _notice_number_from_text(text: Any) -> str:
    match = re.search(r"\b(\d{2}-\d{2})\b", str(text or ""))
    return str(match.group(1) if match else "").strip()


def _is_pdf_url(url: str) -> bool:
    raw = str(url or "").strip().lower()
    if not raw:
        return False
    return raw.endswith(".pdf") or "/noticecomment/" in raw


def _commenter_from_label(text: str) -> str:
    label = _normalize_space(text)
    lowered = label.lower()
    if lowered.startswith("letter from "):
        return _normalize_space(label[12:])
    # Common FINRA link labels: "<Name> Comment On Regulatory Notice 26-06"
    notice_suffix = re.compile(
        r"\s+comment\s+on\s+(?:regulatory\s+)?notice(?:\s+\d{2}-\d{2})?\s*$",
        flags=re.IGNORECASE,
    )
    normalized = notice_suffix.sub("", label).strip()
    if normalized:
        return _normalize_space(normalized)
    return label


class FINRACommentLetterScraper:
    # Impersonation profiles to try in order when curl_cffi is available.
    _IMPERSONATE_PROFILES = ("safari17_0", "chrome124", "chrome120")

    def __init__(self, min_delay_seconds: float = 2.0):
        self._use_cffi = _CURL_CFFI_AVAILABLE
        self._cffi_session: Any = None
        self._cffi_impersonate: str = self._IMPERSONATE_PROFILES[0]
        self._cffi_profile_locked = False  # Once a profile works, stop rotating

        # Fallback plain-requests session (used when curl_cffi unavailable or explicitly disabled)
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

    def _get_cffi_session(self) -> Any:
        """Lazily create a curl_cffi session with browser impersonation."""
        if self._cffi_session is None:
            self._cffi_session = _cffi_requests.Session(impersonate=self._cffi_impersonate)
        return self._cffi_session

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_delay_seconds:
            time.sleep(self.min_delay_seconds - elapsed)
        self._last_request_ts = time.time()

    def _fetch(self, url: str, timeout: int = 60) -> requests.Response:
        target = str(url or "").strip()
        if not target:
            raise ValueError("URL is required")
        self._rate_limit()

        # Try curl_cffi with browser impersonation first (bypasses Cloudflare)
        if self._use_cffi:
            profiles = (self._cffi_impersonate,) if self._cffi_profile_locked else self._IMPERSONATE_PROFILES
            for profile in profiles:
                try:
                    if profile != self._cffi_impersonate or self._cffi_session is None:
                        self._cffi_impersonate = profile
                        self._cffi_session = None
                    cffi_session = self._get_cffi_session()
                    response = cffi_session.get(target, timeout=timeout, allow_redirects=True)
                    if response.status_code == 403 and "Just a moment" in response.text[:500]:
                        logger.debug("curl_cffi profile %s got Cloudflare challenge, trying next", profile)
                        continue
                    response.raise_for_status()
                    # Lock to this profile for subsequent requests
                    self._cffi_profile_locked = True
                    self._cffi_impersonate = profile
                    return response
                except Exception as e:
                    logger.debug("curl_cffi profile %s failed: %s", profile, e)
                    continue
            # If all curl_cffi profiles fail, fall through to plain requests
            logger.warning("All curl_cffi profiles failed for %s, falling back to requests", target)

        response = self.session.get(target, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        return response

    @staticmethod
    def _extract_pdf_text(content: bytes) -> str:
        try:
            from pypdf import PdfReader
        except Exception as e:
            raise RuntimeError(f"PDF extraction requires pypdf: {e}") from e

        reader = PdfReader(io.BytesIO(content))
        pages: List[str] = []
        for page in reader.pages:
            txt = _clean_multiline(page.extract_text() or "")
            if txt:
                pages.append(txt)
        return "\n\n".join(pages).strip()

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

    def discover_documents(self, notice_url: str, include_pdfs: bool = True) -> List[Dict[str, str]]:
        initial_url = str(notice_url or "").strip()
        initial_path = urlparse(initial_url).path.lower()
        if "/rules-guidance/notices/comment/" in initial_path or "/noticecomment/" in initial_path:
            is_pdf = _is_pdf_url(initial_url)
            if is_pdf and not include_pdfs:
                return []
            notice_number = _notice_number_from_text(initial_url)
            title = _commenter_from_label(initial_url.rsplit("/", 1)[-1].replace("-", " "))
            return [
                {
                    "url": initial_url,
                    "title": title or "Comment Letter",
                    "date": "",
                    "commenter_name": title,
                    "notice_number": notice_number,
                    "notice_title": "",
                    "notice_url": "",
                    "comments_url": "",
                    "source_format": "pdf" if is_pdf else "html",
                    "discovery_source": "direct_comment_url",
                }
            ]

        response = self._fetch(notice_url, timeout=60)
        final_url = str(getattr(response, "url", notice_url) or notice_url)
        soup = BeautifulSoup(response.text, "html.parser")

        h1 = soup.find("h1")
        notice_title = _normalize_space(h1.get_text(" ", strip=True) if h1 else "")
        notice_number = _notice_number_from_text(notice_title) or _notice_number_from_text(final_url)
        comments_url = f"{final_url}#comments"

        out: List[Dict[str, str]] = []
        seen = set()

        comments_table = soup.select_one(".view-notice-comments table.views-table")
        if comments_table is not None:
            for row in comments_table.select("tbody tr"):
                cells = row.select("td")
                if len(cells) < 2:
                    continue
                date_text = _date_to_display(cells[0].get_text(" ", strip=True))
                link = cells[1].select_one("a[href]")
                if link is None:
                    continue

                href = str(link.get("href", "") or "").strip()
                if not href:
                    continue
                comment_url = urljoin(final_url, href)
                is_pdf = _is_pdf_url(comment_url)
                if is_pdf and not include_pdfs:
                    continue

                commenter_label = _normalize_space(link.get_text(" ", strip=True))
                commenter_name = _commenter_from_label(commenter_label)
                key = self._url_key(comment_url)
                if key in seen:
                    continue
                seen.add(key)

                out.append(
                    {
                        "url": comment_url,
                        "title": commenter_name or commenter_label or "Comment Letter",
                        "date": date_text,
                        "commenter_name": commenter_name,
                        "notice_number": notice_number,
                        "notice_title": notice_title,
                        "notice_url": final_url,
                        "comments_url": comments_url,
                        "source_format": "pdf" if is_pdf else "html",
                        "discovery_source": "comments_table",
                    }
                )

        if not out:
            for a in soup.select("a[href]"):
                href = str(a.get("href", "") or "").strip()
                if not href:
                    continue
                link_url = urljoin(final_url, href)
                link_text = _normalize_space(a.get_text(" ", strip=True))
                is_comment_page = "/rules-guidance/notices/comment/" in link_url.lower()
                is_notice_comment_pdf = "/noticecomment/" in link_url.lower()
                if not (is_comment_page or is_notice_comment_pdf):
                    continue
                if notice_number and notice_number not in f"{link_text} {link_url}":
                    continue
                is_pdf = _is_pdf_url(link_url)
                if is_pdf and not include_pdfs:
                    continue
                key = self._url_key(link_url)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "url": link_url,
                        "title": link_text or "Comment Letter",
                        "date": "",
                        "commenter_name": _commenter_from_label(link_text),
                        "notice_number": notice_number,
                        "notice_title": notice_title,
                        "notice_url": final_url,
                        "comments_url": comments_url,
                        "source_format": "pdf" if is_pdf else "html",
                        "discovery_source": "anchor_fallback",
                    }
                )

        out.sort(key=lambda item: _parse_date_text(item.get("date", "")) or datetime.min, reverse=True)
        return out

    def extract_document(
        self,
        url: str,
        fallback_title: str = "",
        fallback_date: str = "",
        fallback_commenter_name: str = "",
        fallback_notice_number: str = "",
        fallback_notice_title: str = "",
        fallback_notice_url: str = "",
    ) -> Dict[str, Any]:
        response = self._fetch(url, timeout=60)
        final_url = str(getattr(response, "url", url) or url)

        notice_number = _normalize_space(fallback_notice_number)
        notice_title = _normalize_space(fallback_notice_title)
        notice_url = _normalize_space(fallback_notice_url)
        date_text = _date_to_display(fallback_date)
        commenter_name = _normalize_space(fallback_commenter_name)
        commenter_org = ""

        if _is_pdf_url(final_url):
            full_text_body = self._extract_pdf_text(response.content)
            if not full_text_body:
                raise RuntimeError("No text extracted from comment letter PDF.")

            if not notice_number:
                notice_number = _notice_number_from_text(final_url) or _notice_number_from_text(fallback_title)

            title = _normalize_space(fallback_title)
            if not title:
                filename = urlparse(final_url).path.rsplit("/", 1)[-1]
                title = _normalize_space(filename.replace(".pdf", "").replace("_", " ").replace("-", " "))
            if not title:
                title = f"Comment Letter {notice_number}".strip()

            header_lines = [
                f"Notice Number: {notice_number}" if notice_number else "",
                f"Notice Title: {notice_title}" if notice_title else "",
                f"Notice URL: {notice_url}" if notice_url else "",
                f"Commenter: {commenter_name}" if commenter_name else "",
                f"Date: {date_text}" if date_text else "",
                f"Source URL: {final_url}",
            ]
            header_lines = [line for line in header_lines if line]
            full_text = "\n".join(header_lines).strip()
            if full_text_body:
                full_text = f"{full_text}\n\n{full_text_body}".strip()

            return {
                "success": True,
                "data": {
                    "url": final_url,
                    "title": title or "Comment Letter",
                    "date": date_text,
                    "commenter_name": commenter_name,
                    "commenter_org": commenter_org,
                    "notice_number": notice_number,
                    "notice_title": notice_title,
                    "notice_url": notice_url,
                    "pdf_url": final_url,
                    "comment_url": "",
                    "full_text": full_text,
                    "word_count": len(full_text.split()),
                    "source_format": "pdf",
                },
            }

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup.find_all(["script", "style", "noscript"]):
            tag.decompose()

        layout = soup.select_one(".layout--twocol-bricks") or soup
        heading = layout.find("h1")
        title = _normalize_space(heading.get_text(" ", strip=True) if heading else "")
        if not title:
            title = _normalize_space(fallback_title) or "Comment Letter"

        if not notice_number:
            notice_number = _notice_number_from_text(title) or _notice_number_from_text(final_url)

        commenter_el = layout.select_one(".field--name-field-commenter .field__item")
        if commenter_el is not None:
            commenter_name = _normalize_space(commenter_el.get_text(" ", strip=True))
        commenter_name = commenter_name or _normalize_space(fallback_commenter_name)

        affiliation_el = layout.select_one(".field--name-field-professional-affiliation .field__item")
        if affiliation_el is not None:
            commenter_org = _normalize_space(affiliation_el.get_text(" ", strip=True))

        body = layout.select_one(".block-entity-fieldnodebody .field--name-body")
        if body is None:
            body = layout.select_one(".field--name-body")
        body_text = _clean_multiline(body.get_text("\n") if body is not None else "")
        if not body_text:
            raise RuntimeError("No text extracted from FINRA comment page.")

        header_lines = [
            f"Notice Number: {notice_number}" if notice_number else "",
            f"Notice Title: {notice_title}" if notice_title else "",
            f"Notice URL: {notice_url}" if notice_url else "",
            f"Commenter: {commenter_name}" if commenter_name else "",
            f"Professional Affiliation: {commenter_org}" if commenter_org else "",
            f"Date: {date_text}" if date_text else "",
            f"Source URL: {final_url}",
        ]
        header_lines = [line for line in header_lines if line]
        full_text = "\n".join(header_lines).strip()
        if body_text:
            full_text = f"{full_text}\n\n{body_text}".strip()

        return {
            "success": True,
            "data": {
                "url": final_url,
                "title": title,
                "date": date_text,
                "commenter_name": commenter_name,
                "commenter_org": commenter_org,
                "notice_number": notice_number,
                "notice_title": notice_title,
                "notice_url": notice_url,
                "pdf_url": "",
                "comment_url": final_url,
                "full_text": full_text,
                "word_count": len(full_text.split()),
                "source_format": "html",
            },
        }
