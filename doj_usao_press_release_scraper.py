#!/usr/bin/env python3
"""
DOJ U.S. Attorneys' Office press-release scraper.

Discovers press-release links from the USAO listing page and extracts full
press-release text from detail pages. Handles DOJ's Akamai bm-verify
interstitial challenge automatically.
"""

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from itertools import zip_longest

import requests
from bs4 import BeautifulSoup


DOJ_USAO_PRESS_RELEASES_URL = "https://www.justice.gov/usao/pressreleases"
DOJ_USAO_PRESS_RELEASES_RSS_URL = (
    "https://www.justice.gov/news/rss?"
    "field_component=1681&require_all=0&search_api_language=en&type=press_release"
)
DOJ_NEWS_PRESS_RELEASES_URL = "https://www.justice.gov/news/press-releases"
DOJ_SITEMAP_INDEX_URL = "https://www.justice.gov/sitemap.xml"


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


def _canonical_press_url(url: str) -> str:
    clean = _url_without_query(url)
    if clean.endswith("/alias"):
        clean = clean[: -len("/alias")]
    return clean


def _title_from_url(url: str) -> str:
    path = urlparse(str(url or "")).path
    slug = path.rstrip("/").rsplit("/", 1)[-1].strip()
    if not slug or slug in {"pr", "alias"}:
        return "DOJ USAO Press Release"
    words = [w for w in slug.replace("-", " ").split(" ") if w]
    if not words:
        return "DOJ USAO Press Release"
    return " ".join(words[:24]).strip().title()


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


def _xml_find_all_text(root: ET.Element, local_name: str) -> List[str]:
    out = []
    target = str(local_name or "").strip().lower()
    for el in root.iter():
        if _xml_local_name(el.tag).lower() != target:
            continue
        txt = _normalize_space(el.text or "")
        if txt:
            out.append(txt)
    return out


def _xml_find_child_text(parent: ET.Element, local_name: str) -> str:
    target = str(local_name or "").strip().lower()
    for child in parent.iter():
        if _xml_local_name(child.tag).lower() != target:
            continue
        txt = _normalize_space(child.text or "")
        if txt:
            return txt
    return ""


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
        self.last_discovery_debug: Dict[str, Any] = {}
        self._last_bm_verify_token: str = ""

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

    def _update_bm_token_from_url(self, url: str):
        try:
            q = dict(parse_qsl(urlparse(str(url or "")).query, keep_blank_values=True))
            token = str(q.get("bm-verify", "") or "").strip()
            if token:
                self._last_bm_verify_token = token
        except Exception:
            pass

    def _with_bm_token(self, url: str) -> str:
        token = str(self._last_bm_verify_token or "").strip()
        if not token:
            return str(url or "")
        parsed = urlparse(str(url or ""))
        pairs = [(k, v) for (k, v) in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() != "bm-verify"]
        pairs.append(("bm-verify", token))
        query = urlencode(pairs, doseq=True)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, query, parsed.fragment))

    def _prime_bm_token(self):
        # Best-effort attempt to refresh Akamai token state for query URLs.
        try:
            self._rate_limit()
            response = self.session.get(DOJ_USAO_PRESS_RELEASES_URL, timeout=45, allow_redirects=True)
            self._update_bm_token_from_url(str(getattr(response, "url", "") or ""))
            html = str(response.text or "")
            if _looks_like_akamai_challenge(html):
                bm_url = _extract_bm_verify_url(html, str(getattr(response, "url", DOJ_USAO_PRESS_RELEASES_URL)))
                if bm_url:
                    self._update_bm_token_from_url(bm_url)
                    self._rate_limit()
                    r2 = self.session.get(bm_url, timeout=45, allow_redirects=True)
                    self._update_bm_token_from_url(str(getattr(r2, "url", "") or ""))
        except Exception:
            pass

    def _fetch_html(self, url: str, timeout: int = 60, max_verify_hops: int = 3) -> requests.Response:
        current_url = str(url or "").strip()
        if not current_url:
            raise ValueError("URL is required")

        response: Optional[requests.Response] = None
        tried_bm_retry = False
        tried_prime = False
        for _ in range(max_verify_hops + 1):
            self._rate_limit()
            response = self.session.get(current_url, timeout=timeout, allow_redirects=True)
            self._update_bm_token_from_url(str(getattr(response, "url", "") or ""))
            html = str(response.text or "")
            if _looks_like_akamai_challenge(html):
                bm_url = _extract_bm_verify_url(html, str(response.url or current_url))
                if bm_url:
                    self._update_bm_token_from_url(bm_url)
                    current_url = bm_url
                    continue
                break
            status_code = int(response.status_code or 0)
            if status_code >= 400:
                if status_code in {401, 403} and not tried_bm_retry:
                    with_token = self._with_bm_token(current_url)
                    if with_token and with_token != current_url:
                        tried_bm_retry = True
                        current_url = with_token
                        continue
                if status_code in {401, 403} and not tried_prime:
                    tried_prime = True
                    self._prime_bm_token()
                    with_token = self._with_bm_token(current_url)
                    if with_token and with_token != current_url:
                        tried_bm_retry = True
                        current_url = with_token
                        continue
                response.raise_for_status()
            return response

        if response is not None and _looks_like_akamai_challenge(response.text):
            raise RuntimeError("DOJ returned an Akamai challenge page that could not be bypassed automatically.")
        if response is None:
            raise RuntimeError("No response received from DOJ.")
        if int(response.status_code or 0) >= 400:
            response.raise_for_status()
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
        # Use the same Akamai-aware fetch path as listing pages.
        response = self._fetch_html(rss_url, timeout=45, max_verify_hops=3)
        xml_head = str(response.text or "")[:2000].lower()
        if "<rss" not in xml_head and "<feed" not in xml_head:
            raise RuntimeError(f"RSS endpoint did not return XML (status={getattr(response, 'status_code', 'unknown')})")

        root = _parse_xml_root(response.text)
        found = []
        items = [el for el in root.iter() if _xml_local_name(el.tag).lower() == "item"]
        for item in items:
            link = _xml_find_child_text(item, "link")
            if not _is_usao_press_release_url(link):
                continue
            title = _xml_find_child_text(item, "title")
            pub_date = _xml_find_child_text(item, "pubDate")
            description = _xml_find_child_text(item, "description")
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
        if not found:
            raise RuntimeError("RSS returned zero USAO press-release items")
        return found

    def _discover_from_news_press_listing(
        self,
        base_url: str = DOJ_NEWS_PRESS_RELEASES_URL,
        max_pages: int = 3,
    ) -> List[Dict[str, str]]:
        max_pages = max(1, int(max_pages or 1))
        found: List[Dict[str, str]] = []
        seen = set()

        for page in range(max_pages):
            page_url = self._build_page_url(base_url, page)
            try:
                response = self._fetch_html(page_url, timeout=60)
            except requests.HTTPError as e:
                status_code = int(getattr(getattr(e, "response", None), "status_code", 0) or 0)
                if page > 0 and status_code in {401, 403, 404, 429, 503}:
                    break
                if page > 0:
                    break
                raise

            soup = BeautifulSoup(response.text, "html.parser")
            rows = soup.select("div.views-row")
            if page > 0 and not rows:
                break

            page_found = 0
            for row in rows:
                link = row.select_one("h2.news-title a[href], h2 a[href], a[rel='bookmark'][href]")
                if not link:
                    continue
                doc_url = urljoin("https://www.justice.gov", link.get("href", ""))
                if not _is_usao_press_release_url(doc_url):
                    continue

                key = _url_key(doc_url)
                if not key or key in seen:
                    continue
                seen.add(key)

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
                page_found += 1

            if page > 0 and page_found <= 0:
                break

        return found

    def _discover_from_sitemap(
        self,
        sitemap_index_url: str = DOJ_SITEMAP_INDEX_URL,
        max_sitemap_pages: int = 6,
        max_records: int = 250,
    ) -> Dict[str, Any]:
        index_response = self._fetch_html(sitemap_index_url, timeout=60)
        index_root = _parse_xml_root(index_response.text)
        sitemap_urls = _xml_find_all_text(index_root, "loc")
        sitemap_urls = [u for u in sitemap_urls if u]

        page_urls = [u for u in sitemap_urls if "/sitemap.xml?page=" in u.lower()]
        if not page_urls:
            page_urls = [sitemap_index_url]

        def _page_num(u: str) -> int:
            try:
                q = dict(parse_qsl(urlparse(u).query, keep_blank_values=True))
                return int(q.get("page", "0") or 0)
            except Exception:
                return 0

        max_sitemap_pages = max(1, int(max_sitemap_pages or 1))
        page_urls_sorted = sorted(page_urls, key=_page_num)
        low_first = [u for u in page_urls_sorted if _page_num(u) >= 2] + [u for u in page_urls_sorted if _page_num(u) < 2]
        high_first = list(reversed(page_urls_sorted))
        ordered_candidates: List[str] = []
        seen_candidates = set()
        for lo, hi in zip_longest(low_first, high_first):
            for candidate in (lo, hi):
                if not candidate:
                    continue
                if candidate in seen_candidates:
                    continue
                seen_candidates.add(candidate)
                ordered_candidates.append(candidate)
                if len(ordered_candidates) >= max_sitemap_pages:
                    break
            if len(ordered_candidates) >= max_sitemap_pages:
                break

        out: List[Dict[str, str]] = []
        seen = set()
        pages_attempted = 0
        pages_ok = 0
        pages_with_hits = 0
        for sitemap_url in ordered_candidates:
            pages_attempted += 1
            try:
                response = self._fetch_html(sitemap_url, timeout=60)
            except requests.HTTPError:
                continue
            except Exception:
                continue

            pages_ok += 1
            page_root = _parse_xml_root(response.text)
            locs = _xml_find_all_text(page_root, "loc")
            before_count = len(out)
            for loc_url in locs:
                canonical = _canonical_press_url(loc_url)
                if not _is_usao_press_release_url(canonical):
                    continue
                key = _url_key(canonical)
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "url": canonical,
                        "title": _title_from_url(canonical),
                        "date": "",
                        "office": "",
                        "teaser": "",
                        "source_format": "xml_sitemap",
                        "listing_page": sitemap_url,
                    }
                )
                if len(out) >= max_records:
                    return {
                        "items": out,
                        "pages_attempted": pages_attempted,
                        "pages_ok": pages_ok,
                        "pages_with_hits": pages_with_hits + (1 if len(out) > before_count else 0),
                        "candidates_considered": len(ordered_candidates),
                    }
            if len(out) > before_count:
                pages_with_hits += 1

        return {
            "items": out,
            "pages_attempted": pages_attempted,
            "pages_ok": pages_ok,
            "pages_with_hits": pages_with_hits,
            "candidates_considered": len(ordered_candidates),
        }

    def discover_documents(
        self,
        base_url: str = DOJ_USAO_PRESS_RELEASES_URL,
        max_pages: int = 3,
        fallback_to_rss: bool = True,
    ) -> List[Dict[str, str]]:
        max_pages = max(1, int(max_pages or 1))
        out = []
        seen = set()
        pagination_blocked = False
        target_count = max(12, min(400, max_pages * 12))
        listing_added = 0
        rss_added = 0
        news_added = 0
        sitemap_added = 0

        debug: Dict[str, Any] = {
            "base_url": str(base_url or DOJ_USAO_PRESS_RELEASES_URL),
            "max_pages_requested": max_pages,
            "target_count": target_count,
            "fallback_to_rss": bool(fallback_to_rss),
            "pages": [],
            "pagination_blocked": False,
            "rss_page0_used": False,
            "rss_supplement_used": False,
            "rss_supplement_error": "",
            "news_supplement_used": False,
            "news_supplement_error": "",
            "sitemap_supplement_used": False,
            "sitemap_supplement_error": "",
            "sitemap_pages_attempted": 0,
            "sitemap_pages_ok": 0,
            "sitemap_pages_with_hits": 0,
            "sitemap_candidates_considered": 0,
            "stop_reason": "",
            "listing_added": 0,
            "rss_added": 0,
            "news_added": 0,
            "sitemap_added": 0,
            "total_unique": 0,
        }

        for page in range(max_pages):
            page_url = self._build_page_url(base_url, page)
            discovered = []
            page_attempts = 0
            page_debug: Dict[str, Any] = {
                "page": page,
                "page_url": page_url,
                "attempts": 0,
                "error_type": "",
                "error_status": 0,
                "error_message": "",
                "returned_items": 0,
                "unique_added": 0,
            }
            while page_attempts < 2:
                page_attempts += 1
                page_debug["attempts"] = page_attempts
                try:
                    discovered = self._discover_from_listing_page(page_url)
                    break
                except requests.HTTPError as e:
                    status_code = int(getattr(getattr(e, "response", None), "status_code", 0) or 0)
                    page_debug["error_type"] = "HTTPError"
                    page_debug["error_status"] = status_code
                    page_debug["error_message"] = str(e)
                    if page == 0 and fallback_to_rss:
                        try:
                            discovered = self._discover_from_rss()
                            debug["rss_page0_used"] = True
                            break
                        except Exception as rss_e:
                            page_debug["error_type"] = type(rss_e).__name__
                            page_debug["error_message"] = f"listing+rss_page0_failed: {rss_e}"
                            discovered = []
                            break
                    if status_code in {401, 403, 404, 429, 503}:
                        if page > 0 and page_attempts == 1:
                            # Re-prime session cookies on first blocked paged request, then retry once.
                            try:
                                self._fetch_html(self._build_page_url(base_url, 0), timeout=45)
                            except Exception:
                                pass
                            continue
                        pagination_blocked = pagination_blocked or page > 0
                        debug["pagination_blocked"] = bool(pagination_blocked)
                        discovered = []
                        if page > 0:
                            debug["stop_reason"] = f"blocked_page_{page}_status_{status_code}"
                        break
                    if page > 0:
                        discovered = []
                        debug["stop_reason"] = f"page_{page}_http_error_{status_code}"
                        break
                    self.last_discovery_debug = debug
                    raise
                except Exception as e:
                    page_debug["error_type"] = type(e).__name__
                    page_debug["error_message"] = str(e)
                    if page == 0 and fallback_to_rss:
                        try:
                            discovered = self._discover_from_rss()
                            debug["rss_page0_used"] = True
                            break
                        except Exception as rss_e:
                            page_debug["error_type"] = type(rss_e).__name__
                            page_debug["error_message"] = f"listing+rss_page0_failed: {rss_e}"
                            discovered = []
                            break
                    if page > 0:
                        discovered = []
                        debug["stop_reason"] = f"page_{page}_error"
                        break
                    self.last_discovery_debug = debug
                    raise
            page_debug["returned_items"] = len(discovered)
            if not discovered and page == 0 and fallback_to_rss:
                try:
                    discovered = self._discover_from_rss()
                    debug["rss_page0_used"] = True
                except Exception as rss_e:
                    page_debug["error_type"] = type(rss_e).__name__
                    page_debug["error_message"] = f"rss_page0_failed: {rss_e}"
                    discovered = []
            if not discovered and page > 0:
                if not debug.get("stop_reason", ""):
                    debug["stop_reason"] = f"no_results_page_{page}"
                debug["pages"].append(page_debug)
                break

            for item in discovered:
                key = _url_key(item.get("url", ""))
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(item)
                page_debug["unique_added"] += 1
                if page_debug.get("error_type") == "HTTPError" and page == 0 and debug["rss_page0_used"]:
                    rss_added += 1
                elif page == 0 and debug["rss_page0_used"] and str(item.get("listing_page", "")).startswith("https://www.justice.gov/news/rss"):
                    rss_added += 1
                else:
                    listing_added += 1
            debug["pages"].append(page_debug)

        # If paginated listing is blocked, supplement from RSS so discovery exceeds page 0.
        if fallback_to_rss and (pagination_blocked or (max_pages > 1 and len(out) <= 12)):
            try:
                rss_items = self._discover_from_rss()
                debug["rss_supplement_used"] = True
                for item in rss_items:
                    key = _url_key(item.get("url", ""))
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    out.append(item)
                    rss_added += 1
            except Exception as e:
                debug["rss_supplement_error"] = str(e)
                debug["rss_supplement_used"] = True

        # If pagination is blocked and still below requested depth, try DOJ News listing.
        if max_pages > 1 and len(out) < target_count and (pagination_blocked or len(out) <= 12):
            try:
                news_items = self._discover_from_news_press_listing(
                    base_url=DOJ_NEWS_PRESS_RELEASES_URL,
                    max_pages=max_pages,
                )
                debug["news_supplement_used"] = True
                for item in news_items:
                    key = _url_key(item.get("url", ""))
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    out.append(item)
                    news_added += 1
            except Exception as e:
                debug["news_supplement_used"] = True
                debug["news_supplement_error"] = str(e)

        # Final fallback: sitemap scan for USAO /pr/ URLs to approach requested depth.
        if max_pages > 1 and len(out) < target_count and (pagination_blocked or len(out) <= 12):
            try:
                records_needed = max(120, min(1200, (target_count - len(out)) + 120))
                pages_needed = max(20, min(120, ((target_count - len(out)) // 5) + 10))
                sitemap_result = self._discover_from_sitemap(
                    sitemap_index_url=DOJ_SITEMAP_INDEX_URL,
                    max_sitemap_pages=pages_needed,
                    max_records=records_needed,
                )
                debug["sitemap_supplement_used"] = True
                if isinstance(sitemap_result, dict):
                    debug["sitemap_pages_attempted"] = int(sitemap_result.get("pages_attempted", 0) or 0)
                    debug["sitemap_pages_ok"] = int(sitemap_result.get("pages_ok", 0) or 0)
                    debug["sitemap_pages_with_hits"] = int(sitemap_result.get("pages_with_hits", 0) or 0)
                    debug["sitemap_candidates_considered"] = int(sitemap_result.get("candidates_considered", 0) or 0)
                    sitemap_items = sitemap_result.get("items", [])
                else:
                    sitemap_items = []
                for item in sitemap_items:
                    key = _url_key(item.get("url", ""))
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    out.append(item)
                    sitemap_added += 1
            except Exception as e:
                debug["sitemap_supplement_used"] = True
                debug["sitemap_supplement_error"] = str(e)

        def _sort_key(item: Dict[str, str]):
            return _parse_date_text(item.get("date", "")) or datetime.min

        debug["pagination_blocked"] = bool(pagination_blocked)
        debug["listing_added"] = int(listing_added)
        debug["rss_added"] = int(rss_added)
        debug["news_added"] = int(news_added)
        debug["sitemap_added"] = int(sitemap_added)
        debug["total_unique"] = int(len(out))
        if not debug.get("stop_reason", "") and max_pages > 1 and len(out) <= 12:
            debug["stop_reason"] = "low_yield"
        if not debug.get("stop_reason", ""):
            debug["stop_reason"] = "completed"
        self.last_discovery_debug = debug

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
