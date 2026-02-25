#!/usr/bin/env python3
"""
NewsAPI financial/regulatory news scraper.

Discovers article URLs through NewsAPI's /v2/everything endpoint and extracts
article full text from publisher pages with a snippet fallback.
"""

import io
import re
import time
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


NEWSAPI_EVERYTHING_URL = "https://newsapi.org/v2/everything"


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clean_multiline(text: str) -> str:
    lines = []
    for raw in str(text or "").splitlines():
        line = _normalize_space(raw)
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


_INLINE_REMOVE_PHRASES = [
    "share this article",
    "copy link",
    "x (twitter)",
    "twitter",
    "linkedin",
    "facebook",
    "email",
    "whatsapp",
    "telegram",
    "share",
]


def _strip_inline_boilerplate(line: str) -> str:
    out = str(line or "")
    for phrase in _INLINE_REMOVE_PHRASES:
        out = re.sub(rf"(?i)\b{re.escape(phrase)}\b", " ", out)
    out = re.sub(r"\s{2,}", " ", out).strip(" |-:\t")
    return _normalize_space(out)


def _is_boilerplate_line(line: str) -> bool:
    text = _normalize_space(line)
    if not text:
        return True

    low = text.lower()
    if low in {
        "by",
        "edited by",
        "updated",
        "published",
        "what to know",
        "next",
        "read more",
        "advertisement",
    }:
        return True
    if re.fullmatch(r"[|:\-•]+", low):
        return True
    if re.fullmatch(r"by\s+[a-z][a-z .'-]{1,60}", low):
        return True
    if re.fullmatch(r"(updated|published)\s*(on)?\s*[:\-]?\s*.+", low) and len(low.split()) <= 8:
        return True
    if "share this article" in low:
        return True
    return False


def _clean_article_body_text(text: str) -> str:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip():
        return ""

    lines = []
    seen = set()
    for raw_line in raw.splitlines():
        line = _strip_inline_boilerplate(raw_line)
        if not line or _is_boilerplate_line(line):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)

    if not lines:
        return ""

    paragraphs = []
    current = ""
    for line in lines:
        if not current:
            current = line
            continue

        # Merge short, broken lines into one paragraph for readability.
        if len(line.split()) <= 6 or len(current.split()) < 28:
            current = f"{current} {line}".strip()
        else:
            paragraphs.append(current)
            current = line
    if current:
        paragraphs.append(current)

    out = "\n\n".join(_normalize_space(p) for p in paragraphs if _normalize_space(p))
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def _url_key(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}"


def _parse_date_text(value: str) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).replace(tzinfo=None)
    except Exception:
        pass

    for fmt in (
        "%Y-%m-%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%m/%d/%Y",
        "%m/%d/%y",
    ):
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


def _extract_first_date(text: str) -> str:
    pattern = (
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|May|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)"
        r"\s+\d{1,2},\s+\d{4})"
    )
    m = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
    if not m:
        return ""
    return _date_to_display(m.group(1))


def _extract_meta_content(soup: BeautifulSoup, attrs_options: List[Dict[str, str]]) -> str:
    for attrs in attrs_options:
        tag = soup.find("meta", attrs=attrs)
        if not tag:
            continue
        content = _normalize_space(tag.get("content", ""))
        if content:
            return content
    return ""


def _best_article_text(soup: BeautifulSoup) -> str:
    selectors = [
        "article",
        "main",
        '[itemprop="articleBody"]',
        "div.article-content",
        "div.post-content",
        "div.entry-content",
        "div.story-body",
    ]

    best_text = ""
    best_words = 0
    for selector in selectors:
        for node in soup.select(selector):
            paragraphs = []
            for block in node.select("p, li"):
                block_text = _normalize_space(block.get_text(" ", strip=True))
                if len(block_text.split()) < 5:
                    continue
                paragraphs.append(block_text)
            para_text = "\n\n".join(paragraphs).strip()

            full_text = _clean_multiline(node.get_text("\n"))
            txt = para_text if len(para_text.split()) >= 100 else full_text
            words = len(str(txt or "").split())
            if words > best_words:
                best_text = txt
                best_words = words

    if best_words >= 80:
        return _clean_article_body_text(best_text)

    body = soup.body or soup
    return _clean_article_body_text(_clean_multiline(body.get_text("\n")))


class NewsAPIFinancialScraper:
    def __init__(self, api_key: str, min_delay_seconds: float = 0.25):
        key = str(api_key or "").strip()
        if not key:
            raise ValueError("NewsAPI key is required")

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "Accept": "application/json,text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "X-Api-Key": key,
            }
        )
        self.min_delay_seconds = max(0.0, float(min_delay_seconds))
        self._last_request_ts = 0.0
        self.last_discovery_debug: Dict[str, Any] = {}

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_delay_seconds:
            time.sleep(self.min_delay_seconds - elapsed)
        self._last_request_ts = time.time()

    def _fetch_json(self, endpoint: str, params: Dict[str, Any], timeout: int = 45) -> Dict[str, Any]:
        self._rate_limit()
        response = self.session.get(endpoint, params=params, timeout=timeout)
        if response.status_code >= 400:
            detail = ""
            try:
                payload = response.json()
                detail = str(payload.get("message", "") or "").strip()
            except Exception:
                detail = _normalize_space(response.text[:300])
            raise RuntimeError(f"NewsAPI request failed ({response.status_code}): {detail or 'unknown error'}")

        payload = response.json()
        if str(payload.get("status", "") or "").lower() != "ok":
            msg = str(payload.get("message", "") or "Unexpected NewsAPI response status").strip()
            raise RuntimeError(msg)
        return payload

    def discover_documents(
        self,
        query: str,
        max_pages: int = 1,
        page_size: int = 100,
        from_date: str = "",
        to_date: str = "",
        language: str = "en",
        sort_by: str = "publishedAt",
        domains: str = "",
        exclude_domains: str = "",
        search_in: str = "title,description",
        target_count: int = 0,
        endpoint: str = NEWSAPI_EVERYTHING_URL,
    ) -> List[Dict[str, str]]:
        query = str(query or "").strip()
        if not query:
            raise ValueError("Search query is required")

        max_pages = max(1, int(max_pages or 1))
        page_size = max(10, min(100, int(page_size or 100)))
        target_count = max(0, int(target_count or 0))
        endpoint = str(endpoint or NEWSAPI_EVERYTHING_URL).strip() or NEWSAPI_EVERYTHING_URL

        base_params: Dict[str, Any] = {
            "q": query,
            "language": str(language or "en").strip() or "en",
            "sortBy": str(sort_by or "publishedAt").strip() or "publishedAt",
            "pageSize": page_size,
        }
        if str(search_in or "").strip():
            base_params["searchIn"] = str(search_in).strip()
        if str(from_date or "").strip():
            base_params["from"] = str(from_date).strip()
        if str(to_date or "").strip():
            base_params["to"] = str(to_date).strip()
        if str(domains or "").strip():
            base_params["domains"] = str(domains).strip()
        if str(exclude_domains or "").strip():
            base_params["excludeDomains"] = str(exclude_domains).strip()

        debug: Dict[str, Any] = {
            "endpoint": endpoint,
            "query": query,
            "max_pages_requested": max_pages,
            "page_size": page_size,
            "target_count": target_count,
            "from": str(from_date or "").strip(),
            "to": str(to_date or "").strip(),
            "language": base_params.get("language", ""),
            "sort_by": base_params.get("sortBy", ""),
            "domains": str(domains or "").strip(),
            "exclude_domains": str(exclude_domains or "").strip(),
            "search_in": base_params.get("searchIn", ""),
            "passes_run": [],
            "fallback_no_domains_used": False,
            "fallback_no_domains_reason": "",
            "fallback_no_search_in_used": False,
            "fallback_no_search_in_reason": "",
            "fallback_widened_window_used": False,
            "fallback_widened_window_reason": "",
            "fallback_widened_from": "",
            "pages": [],
            "stop_reason": "",
            "total_results_reported": 0,
            "total_unique": 0,
        }

        discovered: List[Dict[str, str]] = []
        seen = set()

        def _scan_pass(pass_name: str, pass_params: Dict[str, Any], pages_limit: int) -> str:
            pages_limit = max(1, int(pages_limit or 1))
            debug["passes_run"].append(pass_name)

            for page in range(1, pages_limit + 1):
                params = dict(pass_params)
                params["page"] = page
                prepared = requests.Request("GET", endpoint, params=params).prepare()
                page_log = {
                    "pass": pass_name,
                    "page": page,
                    "page_url": str(getattr(prepared, "url", "") or ""),
                    "error_type": "",
                    "error_message": "",
                    "returned_items": 0,
                    "unique_added": 0,
                }

                try:
                    payload = self._fetch_json(endpoint, params=params)
                    total_results = int(payload.get("totalResults", 0) or 0)
                    if total_results:
                        debug["total_results_reported"] = max(int(debug.get("total_results_reported", 0) or 0), total_results)

                    articles = payload.get("articles", [])
                    if not isinstance(articles, list):
                        articles = []
                    page_log["returned_items"] = len(articles)

                    page_unique = 0
                    for article in articles:
                        if not isinstance(article, dict):
                            continue
                        article_url = str(article.get("url", "") or "").strip()
                        if not article_url:
                            continue
                        key = _url_key(article_url)
                        if not key or key in seen:
                            continue
                        seen.add(key)
                        page_unique += 1

                        source_obj = article.get("source", {}) if isinstance(article.get("source", {}), dict) else {}
                        published_at = str(article.get("publishedAt", "") or "").strip()
                        discovered.append(
                            {
                                "url": article_url,
                                "title": _normalize_space(article.get("title", "")) or "Untitled News Article",
                                "date": _date_to_display(published_at),
                                "published_at": published_at,
                                "source_name": _normalize_space(source_obj.get("name", "")),
                                "source_id": _normalize_space(source_obj.get("id", "")),
                                "author": _normalize_space(article.get("author", "")),
                                "description": _normalize_space(article.get("description", "")),
                                "content_snippet": _normalize_space(article.get("content", "")),
                            }
                        )

                    page_log["unique_added"] = page_unique
                    debug["pages"].append(page_log)

                    if target_count > 0 and len(discovered) >= target_count:
                        return f"target_count_reached_{target_count}"
                    if len(articles) < page_size:
                        return f"short_page_{page}"
                    if page_unique == 0:
                        return f"no_unique_results_page_{page}"

                except Exception as e:
                    page_log["error_type"] = type(e).__name__
                    page_log["error_message"] = str(e)
                    debug["pages"].append(page_log)
                    return f"error_page_{page}"

            return "max_pages_reached"

        reason = _scan_pass("primary", base_params, max_pages)

        if not discovered and str(base_params.get("domains", "") or "").strip():
            debug["fallback_no_domains_used"] = True
            no_domains_params = dict(base_params)
            no_domains_params.pop("domains", None)
            reason = _scan_pass("fallback_no_domains", no_domains_params, min(max_pages, 2))
            debug["fallback_no_domains_reason"] = reason

        if not discovered and str(base_params.get("searchIn", "") or "").strip():
            debug["fallback_no_search_in_used"] = True
            no_search_params = dict(base_params)
            no_search_params.pop("searchIn", None)
            reason = _scan_pass("fallback_no_search_in", no_search_params, min(max_pages, 2))
            debug["fallback_no_search_in_reason"] = reason

        if not discovered and str(base_params.get("from", "") or "").strip():
            base_from_dt = _parse_date_text(str(base_params.get("from", "") or "").strip())
            if base_from_dt is not None:
                widened_from_dt = base_from_dt - timedelta(days=7)
                widened_from = widened_from_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                debug["fallback_widened_window_used"] = True
                debug["fallback_widened_from"] = widened_from
                widened_params = dict(base_params)
                widened_params["from"] = widened_from
                reason = _scan_pass("fallback_widened_window", widened_params, min(max_pages, 2))
                debug["fallback_widened_window_reason"] = reason

        if not debug.get("stop_reason"):
            debug["stop_reason"] = reason

        discovered.sort(
            key=lambda x: _parse_date_text(x.get("published_at", "")) or datetime.min,
            reverse=True,
        )
        if target_count > 0:
            discovered = discovered[:target_count]
        debug["total_unique"] = len(discovered)
        self.last_discovery_debug = debug
        return discovered

    def _extract_pdf_text(self, content: bytes) -> str:
        try:
            from pypdf import PdfReader
        except Exception as e:
            raise RuntimeError(f"PDF extraction requires pypdf: {e}")

        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            txt = _clean_multiline(page.extract_text() or "")
            if txt:
                pages.append(txt)
        return "\n\n".join(pages).strip()

    def _extract_html(self, html: str) -> Dict[str, str]:
        soup = BeautifulSoup(str(html or ""), "html.parser")
        for tag in soup.find_all(["script", "style", "noscript", "svg", "form", "iframe"]):
            tag.decompose()

        title = _extract_meta_content(
            soup,
            [
                {"property": "og:title"},
                {"name": "twitter:title"},
            ],
        )
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = _normalize_space(h1.get_text(" ", strip=True))
        if not title and soup.title:
            title = _normalize_space(soup.title.get_text(" ", strip=True))

        published = _extract_meta_content(
            soup,
            [
                {"property": "article:published_time"},
                {"name": "article:published_time"},
                {"name": "publish-date"},
                {"name": "pubdate"},
                {"name": "date"},
            ],
        )
        if not published:
            time_tag = soup.find("time")
            if time_tag:
                published = _normalize_space(time_tag.get("datetime", "") or time_tag.get_text(" ", strip=True))

        description = _extract_meta_content(
            soup,
            [
                {"property": "og:description"},
                {"name": "description"},
            ],
        )

        full_text = _best_article_text(soup)
        if not published:
            published = _extract_first_date(full_text)

        return {
            "title": title,
            "date": published,
            "description": description,
            "full_text": full_text,
        }

    @staticmethod
    def _fallback_text(title: str, description: str, content_snippet: str) -> str:
        parts = []
        clean_title = _normalize_space(title)
        clean_desc = _normalize_space(description)
        clean_snippet = _normalize_space(content_snippet)
        if clean_title:
            parts.append(f"Title: {clean_title}")
        if clean_desc:
            parts.append(f"Summary: {clean_desc}")
        if clean_snippet:
            parts.append(f"Snippet: {clean_snippet}")
        return "\n".join(parts).strip()

    def extract_document(
        self,
        url: str,
        fallback_title: str = "",
        fallback_date: str = "",
        fallback_description: str = "",
        fallback_content: str = "",
        fallback_source_name: str = "",
        fallback_author: str = "",
    ) -> Dict[str, Any]:
        target = str(url or "").strip()
        if not target:
            raise ValueError("URL is required")

        fallback_text = self._fallback_text(
            title=fallback_title,
            description=fallback_description,
            content_snippet=fallback_content,
        )

        try:
            self._rate_limit()
            response = self.session.get(target, timeout=60, allow_redirects=True)
            response.raise_for_status()
        except Exception:
            if not fallback_text:
                raise
            doc_date = _date_to_display(fallback_date)
            return {
                "success": True,
                "data": {
                    "url": target,
                    "title": str(fallback_title or "").strip() or "News Article",
                    "date": doc_date,
                    "source_name": str(fallback_source_name or "").strip(),
                    "author": str(fallback_author or "").strip(),
                    "description": str(fallback_description or "").strip(),
                    "full_text": fallback_text,
                    "word_count": len(fallback_text.split()),
                    "source_format": "snippet",
                    "extraction_mode": "api_snippet_fallback",
                },
            }

        final_url = str(getattr(response, "url", target) or target)
        content_type = str(response.headers.get("Content-Type", "") or "").lower()
        is_pdf = final_url.lower().endswith(".pdf") or "application/pdf" in content_type

        doc_title = ""
        doc_date_raw = ""
        full_text = ""
        source_format = "html"
        extraction_mode = "fetched_html"
        description = str(fallback_description or "").strip()

        if is_pdf:
            source_format = "pdf"
            extraction_mode = "fetched_pdf"
            full_text = self._extract_pdf_text(response.content)
        else:
            parsed = self._extract_html(response.text)
            doc_title = parsed.get("title", "")
            doc_date_raw = parsed.get("date", "")
            description = parsed.get("description", "") or description
            full_text = parsed.get("full_text", "")

        doc_title = _normalize_space(doc_title) or _normalize_space(fallback_title) or "News Article"
        doc_date = _date_to_display(doc_date_raw or fallback_date)

        if fallback_text and len(full_text.split()) < 120:
            if full_text:
                full_text = f"{full_text}\n\nNewsAPI Summary\n{fallback_text}".strip()
                extraction_mode = f"{extraction_mode}_plus_api_summary"
            else:
                full_text = fallback_text
                extraction_mode = "api_snippet_fallback"

        if not full_text:
            raise RuntimeError("No text extracted from article URL.")

        return {
            "success": True,
            "data": {
                "url": final_url,
                "title": doc_title,
                "date": doc_date,
                "source_name": str(fallback_source_name or "").strip(),
                "author": str(fallback_author or "").strip(),
                "description": description,
                "full_text": full_text,
                "word_count": len(full_text.split()),
                "source_format": source_format,
                "extraction_mode": extraction_mode,
            },
        }
