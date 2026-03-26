#!/usr/bin/env python3
"""
SEC rules + public comments scraper.

Starts from a specific SEC rule/release page, extracts the parent rule metadata,
discovers the linked public comments page, and extracts individual comments from
direct SEC comment links (PDF/HTML/TXT).
"""

from __future__ import annotations

import io
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests


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


def _extract_first_date(text: Any) -> str:
    blob = str(text or "")
    pattern = (
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|May|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)"
        r"\s+\d{1,2},\s+\d{4})"
    )
    match = re.search(pattern, blob, flags=re.IGNORECASE)
    if not match:
        return ""
    return _date_to_display(match.group(1))


def _file_number_from_text(text: Any) -> str:
    match = re.search(r"\b([A-Z]\d+-\d{4}-\d{2})\b", str(text or ""), flags=re.IGNORECASE)
    return _normalize_space(match.group(1)).upper() if match else ""


def _release_numbers_from_text(text: Any) -> List[str]:
    found = re.findall(r"\b\d{2}-\d{5,}\b", str(text or ""))
    out: List[str] = []
    seen = set()
    for item in found:
        normalized = _normalize_space(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _lines_from_soup(soup: BeautifulSoup) -> List[str]:
    return [line for line in (_normalize_space(raw) for raw in soup.get_text("\n").splitlines()) if line]


def _extract_value_after_label(lines: List[str], label: str) -> str:
    wanted = _normalize_space(label).lower()
    for idx, line in enumerate(lines):
        if line.lower() != wanted:
            continue
        for candidate in lines[idx + 1 : idx + 6]:
            if candidate and candidate.lower() != wanted:
                return candidate
    return ""


def _extract_release_numbers_from_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for idx, line in enumerate(lines):
        if line.lower() != "release number":
            continue
        for candidate in lines[idx + 1 : idx + 6]:
            if re.fullmatch(r"\d{2}-\d{5,}", candidate):
                if candidate not in seen:
                    seen.add(candidate)
                    out.append(candidate)
            elif out:
                break
    return out


def _html_title(soup: BeautifulSoup) -> str:
    heading = soup.find("h1")
    if heading is not None:
        title = _normalize_space(heading.get_text(" ", strip=True))
        if title:
            return title
    title_tag = soup.title.get_text(" ", strip=True) if soup.title else ""
    return re.sub(r"^SEC\.gov\s*\|\s*", "", _normalize_space(title_tag), flags=re.IGNORECASE)


def _url_key(url: Any) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}"


def _normalize_request_url(url: Any) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    normalized, _fragment = urldefrag(raw)
    return normalized.strip()


def _is_pdf_url(url: Any) -> bool:
    return str(url or "").strip().lower().endswith(".pdf")


def _comment_source_format(url: Any) -> str:
    lowered = str(url or "").strip().lower()
    if lowered.endswith(".pdf"):
        return "pdf"
    if lowered.endswith(".txt"):
        return "txt"
    if lowered.endswith(".htm") or lowered.endswith(".html"):
        return "html"
    return "html"


def _looks_like_sec_comment_href(url: Any) -> bool:
    target = str(url or "").strip()
    if not target:
        return False
    parsed = urlparse(target)
    host = (parsed.netloc or "").lower()
    if "sec.gov" not in host:
        return False
    path = (parsed.path or "").lower()
    if path.startswith("/comments/"):
        return True
    return path.endswith((".pdf", ".txt", ".htm", ".html")) and "/comments/" in path


def _extract_text_from_html(html_text: str) -> str:
    soup = BeautifulSoup(str(html_text or ""), "html.parser")
    for tag in soup.find_all(["script", "style", "noscript", "svg"]):
        tag.decompose()
    root = (
        soup.select_one("main")
        or soup.select_one("article")
        or soup.select_one("#main-content")
        or soup.body
        or soup
    )
    return _clean_multiline(root.get_text("\n"))


class SECRuleCommentsScraper:
    def __init__(self, min_delay_seconds: float = 0.8):
        self.session = cffi_requests.Session(impersonate="chrome")
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.sec.gov/",
            }
        )
        self.min_delay_seconds = max(0.0, float(min_delay_seconds))
        self._last_request_ts = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_delay_seconds:
            time.sleep(self.min_delay_seconds - elapsed)
        self._last_request_ts = time.time()

    def _fetch(self, url: str, timeout: int = 60):
        target = _normalize_request_url(url)
        if not target:
            raise ValueError("URL is required")
        self._rate_limit()
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
            text = _clean_multiline(page.extract_text() or "")
            if text:
                pages.append(text)
        return "\n\n".join(pages).strip()

    def _parse_rule_page_metadata(self, soup: BeautifulSoup, final_url: str) -> Dict[str, Any]:
        lines = _lines_from_soup(soup)
        title = _html_title(soup) or "SEC Rule Release"
        file_number = _extract_value_after_label(lines, "File Number") or _file_number_from_text(f"{title} {final_url}")
        release_numbers = _extract_release_numbers_from_lines(lines) or _release_numbers_from_text(" ".join(lines))
        rule_type = _extract_value_after_label(lines, "Rule Type")
        sec_issue_date = _extract_value_after_label(lines, "SEC Issue Date")
        effective_date = _extract_value_after_label(lines, "Effective Date")
        federal_register_publish_date = _extract_value_after_label(lines, "Federal Register Publish Date")
        last_reviewed = _extract_value_after_label(lines, "Last Reviewed or Updated")

        comments_url = ""
        pdf_url = ""
        for anchor in soup.select("a[href]"):
            href = urljoin(final_url, str(anchor.get("href", "") or "").strip())
            label = _normalize_space(anchor.get_text(" ", strip=True))
            lower_label = label.lower()
            if not comments_url and (lower_label == "view received comments" or "/public-comments/" in href.lower()):
                comments_url = href
            if not pdf_url and href.lower().endswith(".pdf"):
                if "issued version" in lower_label or "release" in lower_label:
                    pdf_url = href

        if not comments_url and file_number:
            comments_url = f"https://www.sec.gov/rules-regulations/public-comments/{file_number.lower()}"

        published_date = _date_to_display(sec_issue_date or federal_register_publish_date or last_reviewed or "")
        return {
            "title": title,
            "file_number": file_number,
            "release_numbers": release_numbers,
            "rule_type": rule_type,
            "sec_issue_date": _date_to_display(sec_issue_date),
            "effective_date": _date_to_display(effective_date),
            "federal_register_publish_date": _date_to_display(federal_register_publish_date),
            "last_reviewed_or_updated": _date_to_display(last_reviewed),
            "comments_url": comments_url,
            "pdf_url": pdf_url,
            "published_date": published_date,
        }

    def _parse_comment_listing(
        self,
        soup: BeautifulSoup,
        comments_url: str,
        rule_metadata: Dict[str, Any],
        include_pdfs: bool,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        file_number = _normalize_space(rule_metadata.get("file_number", ""))
        release_numbers = list(rule_metadata.get("release_numbers", []) or [])
        rule_title = _normalize_space(rule_metadata.get("title", ""))
        letter_type_values = {
            "public comment",
            "staff study",
            "meeting with sec officials",
            "investor roundtable",
        }

        for row in soup.select("tr"):
            link = row.select_one("a[href]")
            if link is None:
                continue
            href = urljoin(comments_url, str(link.get("href", "") or "").strip())
            if not _looks_like_sec_comment_href(href):
                continue
            if _is_pdf_url(href) and not include_pdfs:
                continue

            key = _url_key(href)
            if key in seen:
                continue
            seen.add(key)

            row_text = _normalize_space(row.get_text(" ", strip=True))
            commenter_name = _normalize_space(link.get_text(" ", strip=True)) or "Commenter"
            date_text = _extract_first_date(row_text)
            letter_type = ""
            for candidate in letter_type_values:
                if candidate in row_text.lower():
                    letter_type = candidate.title()
                    break

            title = commenter_name
            if title and not title.lower().startswith("comment from "):
                title = f"Comment from {title}"

            out.append(
                {
                    "entry_kind": "comment",
                    "url": href,
                    "title": title or "Public Comment",
                    "date": _date_to_display(date_text),
                    "commenter_name": commenter_name,
                    "letter_type": letter_type,
                    "file_number": file_number,
                    "notice_number": file_number,
                    "release_numbers": release_numbers,
                    "rule_title": rule_title,
                    "notice_title": rule_title,
                    "rule_url": _normalize_space(rule_metadata.get("rule_url", "")),
                    "notice_url": _normalize_space(rule_metadata.get("rule_url", "")),
                    "comments_url": comments_url,
                    "pdf_url": href if _comment_source_format(href) == "pdf" else "",
                    "comment_url": href if _comment_source_format(href) != "pdf" else "",
                    "source_format": _comment_source_format(href),
                    "discovery_source": "comments_table",
                }
            )

        out.sort(key=lambda item: _parse_date_text(item.get("date", "")) or datetime.min, reverse=True)
        return out

    def discover_documents(self, rule_url: str, include_pdfs: bool = True) -> List[Dict[str, Any]]:
        response = self._fetch(rule_url, timeout=60)
        final_rule_url = _normalize_request_url(getattr(response, "url", rule_url) or rule_url)
        soup = BeautifulSoup(response.text, "html.parser")
        rule_metadata = self._parse_rule_page_metadata(soup, final_rule_url)
        rule_metadata["rule_url"] = final_rule_url

        out: List[Dict[str, Any]] = [
            {
                "entry_kind": "rule",
                "url": final_rule_url,
                "title": _normalize_space(rule_metadata.get("title", "")) or "SEC Rule Release",
                "date": _normalize_space(rule_metadata.get("published_date", "")),
                "file_number": _normalize_space(rule_metadata.get("file_number", "")),
                "notice_number": _normalize_space(rule_metadata.get("file_number", "")),
                "release_numbers": list(rule_metadata.get("release_numbers", []) or []),
                "rule_type": _normalize_space(rule_metadata.get("rule_type", "")),
                "sec_issue_date": _normalize_space(rule_metadata.get("sec_issue_date", "")),
                "effective_date": _normalize_space(rule_metadata.get("effective_date", "")),
                "federal_register_publish_date": _normalize_space(rule_metadata.get("federal_register_publish_date", "")),
                "comments_url": _normalize_space(rule_metadata.get("comments_url", "")),
                "pdf_url": _normalize_space(rule_metadata.get("pdf_url", "")),
                "source_format": "pdf" if _normalize_space(rule_metadata.get("pdf_url", "")) else "html",
                "discovery_source": "rule_page",
            }
        ]

        comments_url = _normalize_space(rule_metadata.get("comments_url", ""))
        if comments_url:
            try:
                comments_response = self._fetch(comments_url, timeout=60)
                comments_final_url = _normalize_request_url(
                    getattr(comments_response, "url", comments_url) or comments_url
                )
                comments_soup = BeautifulSoup(comments_response.text, "html.parser")
                out.extend(
                    self._parse_comment_listing(
                        comments_soup,
                        comments_final_url,
                        {**rule_metadata, "rule_url": final_rule_url},
                        include_pdfs=include_pdfs,
                    )
                )
            except Exception:
                pass

        return out

    def extract_rule(
        self,
        url: str,
        fallback_title: str = "",
        fallback_date: str = "",
        fallback_file_number: str = "",
        fallback_release_numbers: Optional[List[str]] = None,
        fallback_rule_type: str = "",
        fallback_comments_url: str = "",
        fallback_pdf_url: str = "",
        fallback_effective_date: str = "",
        fallback_sec_issue_date: str = "",
        fallback_federal_register_publish_date: str = "",
    ) -> Dict[str, Any]:
        response = self._fetch(url, timeout=60)
        final_url = _normalize_request_url(getattr(response, "url", url) or url)
        soup = BeautifulSoup(response.text, "html.parser")
        metadata = self._parse_rule_page_metadata(soup, final_url)

        title = _normalize_space(metadata.get("title", "")) or _normalize_space(fallback_title) or "SEC Rule Release"
        file_number = _normalize_space(metadata.get("file_number", "")) or _normalize_space(fallback_file_number).upper()
        release_numbers = list(metadata.get("release_numbers", []) or fallback_release_numbers or [])
        rule_type = _normalize_space(metadata.get("rule_type", "")) or _normalize_space(fallback_rule_type)
        comments_url = _normalize_space(metadata.get("comments_url", "")) or _normalize_space(fallback_comments_url)
        pdf_url = _normalize_space(metadata.get("pdf_url", "")) or _normalize_space(fallback_pdf_url)
        sec_issue_date = _normalize_space(metadata.get("sec_issue_date", "")) or _date_to_display(fallback_sec_issue_date)
        effective_date = _normalize_space(metadata.get("effective_date", "")) or _date_to_display(fallback_effective_date)
        federal_register_publish_date = _normalize_space(metadata.get("federal_register_publish_date", "")) or _date_to_display(
            fallback_federal_register_publish_date
        )
        doc_date = (
            _normalize_space(metadata.get("published_date", ""))
            or _date_to_display(fallback_date)
            or sec_issue_date
            or federal_register_publish_date
        )

        full_text_body = ""
        source_format = "html"
        if pdf_url:
            try:
                pdf_response = self._fetch(pdf_url, timeout=90)
                full_text_body = self._extract_pdf_text(pdf_response.content)
                source_format = "pdf"
            except Exception:
                full_text_body = ""

        if not full_text_body:
            main = soup.select_one("main") or soup.find("article") or soup.body or soup
            for tag in main.find_all(["script", "style", "noscript"]):
                tag.decompose()
            full_text_body = _clean_multiline(main.get_text("\n"))
            source_format = "html"

        header_lines = [
            f"Title: {title}" if title else "",
            f"File Number: {file_number}" if file_number else "",
            f"Release Numbers: {', '.join(release_numbers)}" if release_numbers else "",
            f"Rule Type: {rule_type}" if rule_type else "",
            f"SEC Issue Date: {sec_issue_date}" if sec_issue_date else "",
            f"Effective Date: {effective_date}" if effective_date else "",
            f"Federal Register Publish Date: {federal_register_publish_date}" if federal_register_publish_date else "",
            f"Rule URL: {final_url}",
            f"Comments URL: {comments_url}" if comments_url else "",
            f"PDF URL: {pdf_url}" if pdf_url else "",
            f"Source URL: {pdf_url or final_url}",
        ]
        full_text = "\n".join(line for line in header_lines if line).strip()
        if full_text_body:
            full_text = f"{full_text}\n\n{full_text_body}".strip()

        return {
            "success": True,
            "data": {
                "url": final_url,
                "title": title,
                "date": doc_date,
                "file_number": file_number,
                "notice_number": file_number,
                "release_numbers": release_numbers,
                "rule_type": rule_type,
                "sec_issue_date": sec_issue_date,
                "effective_date": effective_date,
                "federal_register_publish_date": federal_register_publish_date,
                "comments_url": comments_url,
                "rule_url": final_url,
                "notice_url": final_url,
                "pdf_url": pdf_url,
                "full_text": full_text,
                "word_count": len(full_text.split()),
                "source_format": source_format,
            },
        }

    def extract_comment(
        self,
        url: str,
        fallback_title: str = "",
        fallback_date: str = "",
        fallback_commenter_name: str = "",
        fallback_file_number: str = "",
        fallback_release_numbers: Optional[List[str]] = None,
        fallback_rule_title: str = "",
        fallback_rule_url: str = "",
        fallback_comments_url: str = "",
        fallback_letter_type: str = "",
    ) -> Dict[str, Any]:
        response = self._fetch(url, timeout=90)
        final_url = _normalize_request_url(getattr(response, "url", url) or url)
        content_type = str(response.headers.get("Content-Type", "") or "").lower()
        source_format = _comment_source_format(final_url)
        if "application/pdf" in content_type:
            source_format = "pdf"
        elif "text/plain" in content_type and source_format != "pdf":
            source_format = "txt"

        if source_format == "pdf":
            full_text_body = self._extract_pdf_text(response.content)
        elif source_format == "txt":
            full_text_body = _clean_multiline(response.text)
        else:
            full_text_body = _extract_text_from_html(response.text)

        title = _normalize_space(fallback_title)
        if source_format == "html":
            soup = BeautifulSoup(response.text, "html.parser")
            extracted_title = _html_title(soup)
            if extracted_title:
                title = extracted_title

        if not title:
            commenter_name = _normalize_space(fallback_commenter_name)
            title = f"Comment from {commenter_name}" if commenter_name else "Public Comment"

        commenter_name = _normalize_space(fallback_commenter_name)
        file_number = _normalize_space(fallback_file_number).upper() or _file_number_from_text(f"{title} {final_url}")
        release_numbers = list(fallback_release_numbers or _release_numbers_from_text(full_text_body))
        rule_title = _normalize_space(fallback_rule_title)
        rule_url = _normalize_space(fallback_rule_url)
        comments_url = _normalize_space(fallback_comments_url)
        letter_type = _normalize_space(fallback_letter_type)
        date_text = _date_to_display(fallback_date or _extract_first_date(full_text_body))

        header_lines = [
            f"Title: {title}" if title else "",
            f"Rule Title: {rule_title}" if rule_title else "",
            f"File Number: {file_number}" if file_number else "",
            f"Release Numbers: {', '.join(release_numbers)}" if release_numbers else "",
            f"Letter Type: {letter_type}" if letter_type else "",
            f"Commenter: {commenter_name}" if commenter_name else "",
            f"Date: {date_text}" if date_text else "",
            f"Rule URL: {rule_url}" if rule_url else "",
            f"Comments URL: {comments_url}" if comments_url else "",
            f"Comment URL: {final_url}" if source_format != "pdf" else "",
            f"PDF URL: {final_url}" if source_format == "pdf" else "",
            f"Source URL: {final_url}",
        ]
        full_text = "\n".join(line for line in header_lines if line).strip()
        if full_text_body:
            full_text = f"{full_text}\n\n{full_text_body}".strip()

        return {
            "success": True,
            "data": {
                "url": final_url,
                "title": title,
                "date": date_text,
                "file_number": file_number,
                "notice_number": file_number,
                "release_numbers": release_numbers,
                "rule_title": rule_title,
                "notice_title": rule_title,
                "rule_url": rule_url,
                "notice_url": rule_url,
                "comments_url": comments_url,
                "comment_url": final_url if source_format != "pdf" else "",
                "pdf_url": final_url if source_format == "pdf" else "",
                "commenter_name": commenter_name,
                "commenter_org": "",
                "letter_type": letter_type,
                "full_text": full_text,
                "word_count": len(full_text.split()),
                "source_format": source_format,
            },
        }
