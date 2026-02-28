#!/usr/bin/env python3
"""
FINRA Regulatory Notice scraper.

Discovers Regulatory Notices from FINRA's notices index and notices RSS feed,
then extracts full notice text and key metadata from individual notice pages.
"""

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


FINRA_NOTICES_URL = "https://www.finra.org/rules-guidance/notices"
FINRA_NOTICES_RSS_URL = "http://feeds.finra.org/FINRANotices"


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
    for fmt in ("%A, %B %d, %Y", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ"):
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


def _notice_number_from_text(text: str) -> str:
    match = re.search(r"\b(\d{2}-\d{2})\b", str(text or ""))
    return str(match.group(1) if match else "").strip()


def _is_regulatory_notice_label(text: str) -> bool:
    return _normalize_space(text).lower().startswith("regulatory notice")


def _xml_local_name(tag: str) -> str:
    raw = str(tag or "")
    if "}" in raw:
        return raw.rsplit("}", 1)[-1]
    return raw


def _parse_xml_root(text: str) -> ET.Element:
    blob = str(text or "").lstrip("\ufeff").strip()
    if not blob:
        raise RuntimeError("XML payload is empty")
    try:
        return ET.fromstring(blob)
    except ET.ParseError as e:
        raise RuntimeError(f"Invalid XML payload: {e}") from e


def _extract_labeled_date(text: str, label: str) -> str:
    pattern = rf"{re.escape(label)}\s*:\s*([A-Za-z]{{3,9}}\.?\s+\d{{1,2}},\s+\d{{4}})"
    match = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
    if match:
        return _date_to_display(match.group(1))
    return ""


class FINRARegulatoryNoticeScraper:
    def __init__(self, min_delay_seconds: float = 0.8):
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
        response = self.session.get(target, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        return response

    @staticmethod
    def _build_page_url(base_url: str, page: int) -> str:
        root = str(base_url or FINRA_NOTICES_URL).strip() or FINRA_NOTICES_URL
        return f"{root}&page={page}" if "?" in root else (f"{root}?page={page}" if page > 0 else root)

    def _discover_from_rss(self, rss_url: str) -> List[Dict[str, str]]:
        response = self._fetch(rss_url, timeout=45)
        root = _parse_xml_root(response.text)

        out = []
        seen = set()
        for item in root.iter():
            if _xml_local_name(item.tag).lower() != "item":
                continue

            item_text = ""
            title = ""
            link = ""
            description = ""
            pub_date = ""
            for child in list(item):
                name = _xml_local_name(child.tag).lower()
                value = _normalize_space(child.text or "")
                item_text += f"\n{name}: {value}"
                if name == "title":
                    title = value
                elif name == "link":
                    link = value
                elif name == "description":
                    description = _clean_multiline(BeautifulSoup(value, "html.parser").get_text("\n"))
                elif name == "pubdate":
                    pub_date = _date_to_display(value)

            if not _is_regulatory_notice_label(title):
                continue
            if not link:
                continue
            key = _url_key(link)
            if key in seen:
                continue
            seen.add(key)

            out.append(
                {
                    "url": link,
                    "title": description or title,
                    "date": pub_date,
                    "notice_number": _notice_number_from_text(title) or _notice_number_from_text(link),
                    "effective_date": _extract_labeled_date(description, "Effective Date"),
                    "comment_deadline": _extract_labeled_date(description, "Comment Period Expires"),
                    "notice_type": "Regulatory Notice",
                    "source_format": "html",
                    "discovery_source": "rss",
                }
            )

        return out

    def _discover_from_index(self, base_url: str, max_pages: int) -> List[Dict[str, str]]:
        max_pages = max(1, int(max_pages or 1))
        out = []
        seen = set()

        for page in range(max_pages):
            page_url = self._build_page_url(base_url, page)
            response = self._fetch(page_url, timeout=45)
            soup = BeautifulSoup(response.text, "html.parser")
            rows = soup.select("table.views-table tbody tr")
            if not rows:
                continue

            for row in rows:
                date_cell = row.select_one("td.views-field-field-core-official-dt")
                title_cell = row.select_one("td.views-field-title a[href]")
                desc_cell = row.select_one("td.views-field-field-notice-title-tx")
                if title_cell is None:
                    continue

                notice_label = _normalize_space(title_cell.get_text(" ", strip=True))
                if not _is_regulatory_notice_label(notice_label):
                    continue

                detail_url = urljoin("https://www.finra.org", title_cell.get("href", ""))
                key = _url_key(detail_url)
                if key in seen:
                    continue
                seen.add(key)

                date_text = ""
                time_el = date_cell.find("time") if date_cell else None
                if time_el is not None and time_el.get("datetime"):
                    date_text = _date_to_display(time_el.get("datetime", ""))
                if not date_text and date_cell is not None:
                    date_text = _date_to_display(date_cell.get_text(" ", strip=True))

                desc_text = _normalize_space(desc_cell.get_text(" ", strip=True) if desc_cell else "")
                desc_lines = [
                    _normalize_space(div.get_text(" ", strip=True))
                    for div in (desc_cell.find_all("div", recursive=False) if desc_cell else [])
                    if _normalize_space(div.get_text(" ", strip=True))
                ]
                descriptive_title = desc_lines[0] if desc_lines else desc_text or notice_label

                out.append(
                    {
                        "url": detail_url,
                        "title": descriptive_title,
                        "date": date_text,
                        "notice_number": _notice_number_from_text(notice_label) or _notice_number_from_text(detail_url),
                        "effective_date": _extract_labeled_date(desc_text, "Effective Date"),
                        "comment_deadline": _extract_labeled_date(desc_text, "Comment Period Expires"),
                        "notice_type": "Regulatory Notice",
                        "source_format": "html",
                        "listing_page": page_url,
                        "discovery_source": "index",
                    }
                )

        return out

    def discover_documents(
        self,
        base_url: str = FINRA_NOTICES_URL,
        max_pages: int = 3,
        include_rss: bool = True,
        rss_url: str = FINRA_NOTICES_RSS_URL,
    ) -> List[Dict[str, str]]:
        merged = {}
        sources = []
        if include_rss:
            try:
                sources.extend(self._discover_from_rss(rss_url))
            except Exception:
                pass
        sources.extend(self._discover_from_index(base_url, max_pages=max_pages))

        for item in sources:
            key = _url_key(item.get("url", ""))
            if not key:
                continue
            existing = merged.get(key, {})
            combined = dict(existing)
            combined.update({k: v for k, v in item.items() if str(v or "").strip()})
            # Prefer descriptive index titles over RSS labels and HTML snippets.
            if str(item.get("discovery_source", "") or "").strip() == "index":
                combined["discovery_source"] = "index"
            merged[key] = combined

        def _sort_key(item: Dict[str, str]):
            return _parse_date_text(item.get("date", "")) or datetime.min

        out = list(merged.values())
        out.sort(key=_sort_key, reverse=True)
        return out

    def extract_document(
        self,
        url: str,
        fallback_title: str = "",
        fallback_date: str = "",
        fallback_notice_number: str = "",
        fallback_effective_date: str = "",
        fallback_comment_deadline: str = "",
    ) -> Dict[str, Any]:
        response = self._fetch(url, timeout=60)
        final_url = str(getattr(response, "url", url) or url)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup.find_all(["script", "style", "noscript"]):
            tag.decompose()

        article = soup.find("article")
        if article is None:
            article = soup.find("main") or soup

        heading = soup.find("h1")
        title = _normalize_space(heading.get_text(" ", strip=True) if heading else "")
        if not title:
            title = str(fallback_title or "").strip() or "FINRA Regulatory Notice"

        notice_number = ""
        notice_number_el = article.select_one(".field--name-title")
        if notice_number_el is not None:
            notice_number = _notice_number_from_text(notice_number_el.get_text(" ", strip=True))
        if not notice_number:
            title_tag = soup.title.get_text(" ", strip=True) if soup.title else ""
            notice_number = _notice_number_from_text(title_tag) or str(fallback_notice_number or "").strip()

        published_date = ""
        pub_el = article.select_one(".field--name-field-core-official-dt")
        if pub_el is not None:
            pub_text = _normalize_space(pub_el.get_text(" ", strip=True))
            published_date = _extract_labeled_date(pub_text, "Published Date") or _date_to_display(pub_text)
        if not published_date:
            published_date = _date_to_display(fallback_date)

        subtitle_texts = [
            _normalize_space(el.get_text(" ", strip=True))
            for el in article.select(".field--name-field-notice-subtitle-tx")
            if _normalize_space(el.get_text(" ", strip=True))
        ]
        subtitle_blob = "\n".join(subtitle_texts)
        effective_date = _extract_labeled_date(subtitle_blob, "Effective Date") or str(fallback_effective_date or "").strip()
        comment_deadline = _extract_labeled_date(subtitle_blob, "Comment Period Expires") or str(fallback_comment_deadline or "").strip()

        body = article.select_one(".field--name-field-tab-content")
        if body is None:
            body = article
        body_text = _clean_multiline(body.get_text("\n"))

        pdf_url = ""
        for anchor in article.select("a[href]"):
            href = str(anchor.get("href", "") or "").strip()
            link_text = _normalize_space(anchor.get_text(" ", strip=True))
            if href.lower().endswith(".pdf") and "download as" in link_text.lower():
                pdf_url = urljoin(final_url, href)
                break
        if not pdf_url:
            for anchor in article.select("a[href]"):
                href = str(anchor.get("href", "") or "").strip()
                if href.lower().endswith(".pdf"):
                    pdf_url = urljoin(final_url, href)
                    break

        header_lines = [
            f"Notice Number: {notice_number}" if notice_number else "",
            f"Title: {title}" if title else "",
            f"Published Date: {published_date}" if published_date else "",
            f"Effective Date: {effective_date}" if effective_date else "",
            f"Comment Period Expires: {comment_deadline}" if comment_deadline else "",
            f"Source URL: {final_url}",
        ]
        if pdf_url:
            header_lines.append(f"PDF URL: {pdf_url}")
        header_lines = [line for line in header_lines if line]
        full_text = "\n".join(header_lines).strip()
        if body_text:
            full_text = f"{full_text}\n\n{body_text}".strip()

        return {
            "success": True,
            "data": {
                "url": final_url,
                "title": title,
                "date": published_date,
                "notice_number": notice_number,
                "effective_date": effective_date,
                "comment_deadline": comment_deadline,
                "pdf_url": pdf_url,
                "full_text": full_text,
                "word_count": len(full_text.split()),
                "source_format": "html",
            },
        }
