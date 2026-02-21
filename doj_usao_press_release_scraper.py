#!/usr/bin/env python3
"""
DOJ U.S. Attorneys' Office press-release scraper.

Discovers press-release links from the USAO listing page and extracts full
press-release text from detail pages. Handles DOJ's Akamai bm-verify
interstitial challenge automatically.
"""

import re
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup


DOJ_USAO_PRESS_RELEASES_URL = "https://www.justice.gov/usao/pressreleases"
DOJ_USAO_PRESS_RELEASES_RSS_URL = (
    "https://www.justice.gov/news/rss?"
    "field_component=1681&require_all=0&search_api_language=en&type=press_release"
)


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
    for fmt in ("%A, %B %d, %Y", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%Y-%m-%d"):
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


def _url_without_query(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _url_key(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}"


def _looks_like_akamai_challenge(html: str) -> bool:
    blob = str(html or "").lower()
    return ("akamai-logo" in blob or "powered and protected by" in blob) and "bm-verify=" in blob


def _extract_bm_verify_url(html: str, current_url: str) -> str:
    blob = str(html or "")
    patterns = [
        r"""URL\s*=\s*['"]([^'"]*bm-verify=[^'"]+)['"]""",
        r"""http-equiv=["']refresh["'][^>]*content=(["'])(.*?)\1""",
        r"""window\.location\.replace\(["']([^"']*bm-verify=[^"']+)["']\)""",
    ]
    for pattern in patterns:
        m = re.search(pattern, blob, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        if "http-equiv" in pattern:
            content = str(m.group(2) or "")
            url_match = re.search(r"""url\s*=\s*['"]?([^'"]*bm-verify=[^'"]+)""", content, flags=re.IGNORECASE)
            href = str(url_match.group(1) if url_match else "").strip().strip("'\"")
        else:
            href = str(m.group(1) or "").strip().strip("'\"")
        if not href:
            continue
        return urljoin(current_url, href)
    return ""


def _is_usao_press_release_url(url: str) -> bool:
    lower = str(url or "").lower()
    return ("/usao-" in lower or "/usao/" in lower) and "/pr/" in lower


class DOJUSAOPressReleaseScraper:
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

    def _build_page_url(self, base_url: str, page: int) -> str:
        parsed = urlparse(str(base_url or "").strip() or DOJ_USAO_PRESS_RELEASES_URL)
        pairs = [(k, v) for (k, v) in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() != "page"]
        if page > 0:
            pairs.append(("page", str(page)))
        query = urlencode(pairs, doseq=True)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, query, parsed.fragment))

    def _fetch_html(self, url: str, timeout: int = 60, max_verify_hops: int = 3) -> requests.Response:
        current_url = str(url or "").strip()
        if not current_url:
            raise ValueError("URL is required")

        response: Optional[requests.Response] = None
        for _ in range(max_verify_hops + 1):
            self._rate_limit()
            response = self.session.get(current_url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            html = str(response.text or "")
            if not _looks_like_akamai_challenge(html):
                return response

            bm_url = _extract_bm_verify_url(html, str(response.url or current_url))
            if not bm_url:
                break
            current_url = bm_url

        if response is not None and _looks_like_akamai_challenge(response.text):
            raise RuntimeError("DOJ returned an Akamai challenge page that could not be bypassed automatically.")
        if response is None:
            raise RuntimeError("No response received from DOJ.")
        return response

    def _discover_from_listing_page(self, page_url: str) -> List[Dict[str, str]]:
        response = self._fetch_html(page_url, timeout=60)
        soup = BeautifulSoup(response.text, "html.parser")
        rows = soup.select("div.views-row")

        found = []
        for row in rows:
            link = row.select_one("h2.news-title a[href], h2 a[href], a[rel='bookmark'][href]")
            if not link:
                continue
            doc_url = urljoin("https://www.justice.gov", link.get("href", ""))
            if not _is_usao_press_release_url(doc_url):
                continue

            title = _normalize_space(link.get_text(" ", strip=True))
            date_el = row.select_one("time")
            date_text = _normalize_space(date_el.get_text(" ", strip=True) if date_el else "")

            teaser_el = row.select_one("div.field_teaser, div.field-formatter--smart-trim")
            teaser = _normalize_space(teaser_el.get_text(" ", strip=True) if teaser_el else "")

            found.append(
                {
                    "url": _url_without_query(doc_url),
                    "title": title,
                    "date": _date_to_display(date_text),
                    "office": "",
                    "teaser": teaser,
                    "source_format": "html",
                    "listing_page": str(page_url or ""),
                }
            )
        return found

    def _discover_from_rss(self, rss_url: str = DOJ_USAO_PRESS_RELEASES_RSS_URL) -> List[Dict[str, str]]:
        self._rate_limit()
        response = self.session.get(rss_url, timeout=45)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "xml")
        found = []
        for item in soup.find_all("item"):
            link_tag = item.find("link")
            link = _normalize_space(link_tag.get_text(" ", strip=True) if link_tag else "")
            if not _is_usao_press_release_url(link):
                continue
            title_tag = item.find("title")
            pub_date_tag = item.find("pubDate")
            description_tag = item.find("description")
            title = _normalize_space(title_tag.get_text(" ", strip=True) if title_tag else "")
            pub_date = _normalize_space(pub_date_tag.get_text(" ", strip=True) if pub_date_tag else "")
            description = _normalize_space(description_tag.get_text(" ", strip=True) if description_tag else "")
            found.append(
                {
                    "url": _url_without_query(link),
                    "title": title,
                    "date": _date_to_display(pub_date),
                    "office": "",
                    "teaser": description,
                    "source_format": "html",
                    "listing_page": rss_url,
                }
            )
        return found

    def discover_documents(
        self,
        base_url: str = DOJ_USAO_PRESS_RELEASES_URL,
        max_pages: int = 3,
        fallback_to_rss: bool = True,
    ) -> List[Dict[str, str]]:
        max_pages = max(1, int(max_pages or 1))
        out = []
        seen = set()

        for page in range(max_pages):
            page_url = self._build_page_url(base_url, page)
            discovered = self._discover_from_listing_page(page_url)
            if not discovered and page == 0 and fallback_to_rss:
                discovered = self._discover_from_rss()
            if not discovered and page > 0:
                break

            for item in discovered:
                key = _url_key(item.get("url", ""))
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(item)

        def _sort_key(item: Dict[str, str]):
            return _parse_date_text(item.get("date", "")) or datetime.min

        out.sort(key=_sort_key, reverse=True)
        return out

    def extract_document(
        self,
        url: str,
        fallback_title: str = "",
        fallback_date: str = "",
        fallback_office: str = "",
    ) -> Dict[str, Any]:
        response = self._fetch_html(url, timeout=75)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup.find_all(["script", "style", "noscript"]):
            tag.decompose()

        canonical_link = soup.find("link", rel="canonical")
        canonical_url = ""
        if canonical_link is not None:
            canonical_url = _normalize_space(canonical_link.get("href", ""))
        final_url = canonical_url or _url_without_query(str(getattr(response, "url", url) or url))

        h1 = soup.select_one("h1.page-title") or soup.find("h1")
        title = _normalize_space(h1.get_text(" ", strip=True) if h1 else "")
        if not title:
            title_tag = _normalize_space(soup.title.get_text(" ", strip=True) if soup.title else "")
            title_tag = re.sub(
                r"\s*\|\s*United States Department of Justice\s*$",
                "",
                title_tag,
                flags=re.IGNORECASE,
            ).strip()
            if "|" in title_tag:
                title = _normalize_space(title_tag.split("|")[-1])
            else:
                title = title_tag
        if not title:
            title = _normalize_space(fallback_title) or "DOJ Press Release"

        node_date = soup.select_one(".node-date time")
        if node_date is not None:
            date_text = _normalize_space(node_date.get_text(" ", strip=True))
        else:
            any_time = soup.find("time")
            date_text = _normalize_space(any_time.get_text(" ", strip=True) if any_time else "")
        date_value = _date_to_display(date_text or fallback_date)

        office_node = soup.select_one(".node-office")
        office = _normalize_space(office_node.get_text(" ", strip=True) if office_node else "")
        if not office:
            office = _normalize_space(fallback_office) or "U.S. Attorney's Office"

        body = (
            soup.select_one("div.node-body")
            or soup.select_one("div.field_body")
            or soup.select_one("div.field--name-body")
        )
        if body is None:
            body = soup.find("article") or soup.find("main") or soup
        full_text = _clean_multiline(body.get_text("\n"))

        updated_date = ""
        updated_node = soup.select_one(".node-updated-date")
        if updated_node is not None:
            updated_raw = _normalize_space(updated_node.get_text(" ", strip=True))
            m = re.search(r"Updated\s+(.+)$", updated_raw, flags=re.IGNORECASE)
            if m:
                updated_date = _date_to_display(m.group(1))
            else:
                updated_date = _date_to_display(updated_raw)

        return {
            "success": True,
            "data": {
                "url": final_url,
                "title": title,
                "date": date_value,
                "office": office,
                "updated_date": updated_date,
                "full_text": full_text,
                "word_count": len(full_text.split()),
                "source_format": "html",
            },
        }
