#!/usr/bin/env python3
"""
Trade-media scraper for JD Supra, InvestmentNews, and Citywire.

This connector discovers article URLs from RSS/Atom feeds when available, with
an HTML listing fallback, then extracts full text from article pages.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


TRADE_MEDIA_SOURCES: Dict[str, Dict[str, Any]] = {
    "jdsupra_article": {
        "label": "JD Supra",
        "organization": "JD Supra",
        "default_url": "https://www.jdsupra.com/legalnews/",
        "tags_csv": "jdsupra,legal-analysis,regulatory-commentary",
        "rss_candidates": [
            "https://www.jdsupra.com/legalnews/rss-law-feeds.aspx",
            "https://www.jdsupra.com/feed/",
            "https://www.jdsupra.com/rss/",
        ],
        "article_path_keywords": ["legalnews"],
    },
    "investmentnews_article": {
        "label": "InvestmentNews",
        "organization": "InvestmentNews",
        "default_url": "https://www.investmentnews.com/",
        "tags_csv": "investmentnews,wealth-management,industry-news",
        "rss_candidates": [
            "https://www.investmentnews.com/feed/",
            "https://www.investmentnews.com/rss",
            "https://www.investmentnews.com/rss.xml",
        ],
        "article_path_keywords": ["investmentnews"],
    },
    "citywire_article": {
        "label": "Citywire",
        "organization": "Citywire",
        "default_url": "https://citywire.com/us/news",
        "tags_csv": "citywire,asset-management,industry-news",
        "rss_candidates": [
            "https://citywire.com/rss",
            "https://citywire.com/us/rss",
            "https://citywire.com/us/news/rss",
        ],
        "article_path_keywords": ["news", "article"],
    },
}


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
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).replace(tzinfo=None)
    except Exception:
        pass
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
    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%B %d, %Y",
        "%b %d, %Y",
        "%m/%d/%Y",
        "%m/%d/%y",
    ):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=None)
        except ValueError:
            continue
    try:
        parsed = parsedate_to_datetime(text)
        if parsed is not None:
            return parsed.replace(tzinfo=None)
    except Exception:
        pass
    return None


def _date_to_display(value: Any) -> str:
    parsed = _parse_date_text(value)
    if parsed is None:
        return str(value or "").strip()
    return parsed.strftime("%B %d, %Y")


def _canonical_host(value: str) -> str:
    parsed = urlparse(str(value or "").strip())
    host = str(parsed.netloc or parsed.path or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _host_matches(host_or_url: str, domain_or_url: str) -> bool:
    host = _canonical_host(host_or_url)
    target = _canonical_host(domain_or_url)
    return bool(host and target and (host == target or host.endswith(f".{target}")))


def _url_key(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}"


def _xml_local_name(tag: str) -> str:
    raw = str(tag or "")
    if "}" in raw:
        return raw.rsplit("}", 1)[-1]
    return raw


def _title_from_url(url: str, fallback: str = "Article") -> str:
    path = urlparse(str(url or "")).path
    slug = path.rstrip("/").rsplit("/", 1)[-1].strip()
    if not slug:
        return fallback
    words = [part for part in slug.replace("-", " ").replace("_", " ").split(" ") if part]
    if not words:
        return fallback
    return " ".join(words[:18]).strip().title()


def _looks_like_article_url(url: str, source_url: str, path_keywords: List[str]) -> bool:
    parsed = urlparse(str(url or "").strip())
    if not parsed.scheme.startswith("http"):
        return False
    if not _host_matches(parsed.netloc, source_url):
        return False

    path = parsed.path.lower().strip()
    if not path or path in {"/", ""}:
        return False
    if any(path.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".svg", ".css", ".js", ".xml")):
        return False

    skip_tokens = (
        "/tag/",
        "/tags/",
        "/topic/",
        "/topics/",
        "/author/",
        "/authors/",
        "/category/",
        "/categories/",
        "/events",
        "/video",
        "/podcast",
        "/about",
        "/contact",
        "/subscribe",
        "/search",
        "/rss",
        "/feed",
        "/privacy",
        "/terms",
    )
    if any(token in path for token in skip_tokens):
        return False

    segments = [seg for seg in path.split("/") if seg]
    if len(segments) < 2:
        return False

    if re.search(r"/\d{4}/\d{2}/", path):
        return True
    if any(keyword in path for keyword in path_keywords):
        return True
    return len(segments[-1]) >= 12


class TradeMediaScraper:
    def __init__(self, min_delay_seconds: float = 0.7):
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

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_delay_seconds:
            time.sleep(self.min_delay_seconds - elapsed)
        self._last_request_ts = time.time()

    def _fetch(self, url: str, timeout: int = 45) -> requests.Response:
        target = str(url or "").strip()
        if not target:
            raise ValueError("URL is required")
        self._rate_limit()
        response = self.session.get(target, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        return response

    def _source_config(self, source_key: str) -> Dict[str, Any]:
        cfg = TRADE_MEDIA_SOURCES.get(str(source_key or "").strip())
        if not isinstance(cfg, dict):
            raise ValueError(f"Unsupported trade-media source: {source_key}")
        return cfg

    @staticmethod
    def _extract_feed_urls_from_html(soup: BeautifulSoup, base_url: str) -> List[str]:
        out: List[str] = []
        for link in soup.select("link[rel][type][href]"):
            rel = str(link.get("rel", "") or "").lower()
            typ = str(link.get("type", "") or "").lower()
            if "alternate" not in rel:
                continue
            if "rss" not in typ and "atom" not in typ and "xml" not in typ:
                continue
            href = str(link.get("href", "") or "").strip()
            if not href:
                continue
            out.append(urljoin(base_url, href))
        return out

    def _discover_from_feed(self, feed_url: str, source_key: str, source_label: str, source_url: str) -> List[Dict[str, str]]:
        response = self._fetch(feed_url, timeout=45)
        root = ET.fromstring(str(response.text or "").lstrip("\ufeff").strip())

        out: List[Dict[str, str]] = []
        seen: set[str] = set()
        for node in root.iter():
            node_name = _xml_local_name(node.tag).lower()
            if node_name not in {"item", "entry"}:
                continue

            title = ""
            link = ""
            date_text = ""
            description = ""
            for child in list(node):
                name = _xml_local_name(child.tag).lower()
                value = _normalize_space(child.text or "")
                if name == "title":
                    title = value
                elif name == "link":
                    href = str(child.get("href", "") or "").strip()
                    link = href or value
                elif name in {"pubdate", "published", "updated", "date"} and not date_text:
                    date_text = _date_to_display(value)
                elif name in {"description", "summary", "content"} and not description:
                    description = _clean_multiline(BeautifulSoup(value, "html.parser").get_text("\n"))

            item_url = urljoin(feed_url, link)
            if not _looks_like_article_url(
                item_url,
                source_url=source_url,
                path_keywords=list(self._source_config(source_key).get("article_path_keywords", [])),
            ):
                continue
            key = _url_key(item_url)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "url": item_url,
                    "title": title or _title_from_url(item_url, fallback=f"{source_label} Article"),
                    "date": date_text,
                    "source_name": source_label,
                    "source_key": source_key,
                    "description": description,
                    "source_format": "html",
                    "discovery_source": "rss",
                    "feed_url": feed_url,
                }
            )
        return out

    def _discover_from_listing(self, listing_url: str, source_key: str, source_label: str, max_pages: int) -> List[Dict[str, str]]:
        max_pages = max(1, int(max_pages or 1))
        queue: List[str] = [listing_url]
        visited_pages: set[str] = set()
        seen_urls: set[str] = set()
        out: List[Dict[str, str]] = []
        cfg = self._source_config(source_key)
        path_keywords = list(cfg.get("article_path_keywords", []))

        while queue and len(visited_pages) < max_pages:
            page_url = queue.pop(0)
            page_key = _url_key(page_url)
            if not page_key or page_key in visited_pages:
                continue
            visited_pages.add(page_key)

            try:
                response = self._fetch(page_url, timeout=45)
            except Exception:
                continue
            soup = BeautifulSoup(response.text, "html.parser")

            for article in soup.select("article"):
                link = article.select_one("a[href]")
                if link is None:
                    continue
                article_url = urljoin(page_url, str(link.get("href", "") or "").strip())
                if not _looks_like_article_url(article_url, listing_url, path_keywords):
                    continue
                item_key = _url_key(article_url)
                if not item_key or item_key in seen_urls:
                    continue
                seen_urls.add(item_key)

                heading = article.select_one("h1, h2, h3, h4")
                title = _normalize_space(heading.get_text(" ", strip=True) if heading else link.get_text(" ", strip=True))
                time_el = article.find("time")
                date_text = _date_to_display(time_el.get("datetime", "") if time_el and time_el.get("datetime") else (time_el.get_text(" ", strip=True) if time_el else ""))
                desc_el = article.select_one("p")
                description = _normalize_space(desc_el.get_text(" ", strip=True) if desc_el else "")
                out.append(
                    {
                        "url": article_url,
                        "title": title or _title_from_url(article_url, fallback=f"{source_label} Article"),
                        "date": date_text,
                        "source_name": source_label,
                        "source_key": source_key,
                        "description": description,
                        "source_format": "html",
                        "discovery_source": "listing",
                        "listing_page": page_url,
                    }
                )

            for link in soup.select("a[href]"):
                href = str(link.get("href", "") or "").strip()
                if not href:
                    continue
                target_url = urljoin(page_url, href)
                text = _normalize_space(link.get_text(" ", strip=True)).lower()

                if _looks_like_article_url(target_url, listing_url, path_keywords):
                    item_key = _url_key(target_url)
                    if not item_key or item_key in seen_urls:
                        continue
                    seen_urls.add(item_key)
                    out.append(
                        {
                            "url": target_url,
                            "title": _normalize_space(link.get_text(" ", strip=True))
                            or _title_from_url(target_url, fallback=f"{source_label} Article"),
                            "date": "",
                            "source_name": source_label,
                            "source_key": source_key,
                            "description": "",
                            "source_format": "html",
                            "discovery_source": "listing_link",
                            "listing_page": page_url,
                        }
                    )
                    continue

                if len(visited_pages) + len(queue) >= max_pages:
                    continue
                if not _host_matches(target_url, listing_url):
                    continue
                if text in {"next", "next page", "older", "older posts", "more"}:
                    next_key = _url_key(target_url)
                    if next_key and next_key not in visited_pages and target_url not in queue:
                        queue.append(target_url)
                elif re.search(r"[?&](page|p)=\d+", target_url):
                    next_key = _url_key(target_url)
                    if next_key and next_key not in visited_pages and target_url not in queue:
                        queue.append(target_url)

        return out

    def discover_documents(
        self,
        source_key: str,
        base_url: str = "",
        max_pages: int = 3,
        include_rss: bool = True,
    ) -> List[Dict[str, str]]:
        cfg = self._source_config(source_key)
        source_label = str(cfg.get("label", "Trade Media")).strip() or "Trade Media"
        source_url = str(base_url or cfg.get("default_url", "")).strip()
        if not source_url:
            raise ValueError("A source URL is required.")

        merged: Dict[str, Dict[str, str]] = {}
        debug: Dict[str, Any] = {
            "source_key": source_key,
            "source_label": source_label,
            "base_url": source_url,
            "max_pages": int(max_pages or 1),
            "include_rss": bool(include_rss),
            "feed_urls_attempted": [],
            "feed_urls_succeeded": [],
            "listing_used": False,
            "discovered_count": 0,
        }

        feed_urls: List[str] = []
        try:
            listing_response = self._fetch(source_url, timeout=45)
            listing_soup = BeautifulSoup(listing_response.text, "html.parser")
            feed_urls.extend(self._extract_feed_urls_from_html(listing_soup, source_url))
        except Exception:
            pass

        for candidate in cfg.get("rss_candidates", []):
            resolved = urljoin(source_url, str(candidate or "").strip())
            if resolved:
                feed_urls.append(resolved)

        feed_urls = [u for idx, u in enumerate(feed_urls) if u and u not in feed_urls[:idx]]
        debug["feed_urls_attempted"] = list(feed_urls)

        if include_rss:
            for feed_url in feed_urls:
                try:
                    feed_docs = self._discover_from_feed(
                        feed_url=feed_url,
                        source_key=source_key,
                        source_label=source_label,
                        source_url=source_url,
                    )
                    if feed_docs:
                        debug["feed_urls_succeeded"].append(feed_url)
                    for item in feed_docs:
                        key = _url_key(item.get("url", ""))
                        if not key:
                            continue
                        existing = merged.get(key, {})
                        combined = dict(existing)
                        combined.update({k: v for k, v in item.items() if str(v or "").strip()})
                        merged[key] = combined
                except Exception:
                    continue

        if not merged:
            debug["listing_used"] = True
            listing_docs = self._discover_from_listing(
                listing_url=source_url,
                source_key=source_key,
                source_label=source_label,
                max_pages=max_pages,
            )
            for item in listing_docs:
                key = _url_key(item.get("url", ""))
                if not key:
                    continue
                existing = merged.get(key, {})
                combined = dict(existing)
                combined.update({k: v for k, v in item.items() if str(v or "").strip()})
                merged[key] = combined

        out = list(merged.values())
        out.sort(key=lambda row: _parse_date_text(row.get("date", "")) or datetime.min, reverse=True)
        debug["discovered_count"] = len(out)
        self.last_discovery_debug = debug
        return out

    @staticmethod
    def _extract_meta_content(soup: BeautifulSoup, attrs_options: List[Dict[str, str]]) -> str:
        for attrs in attrs_options:
            tag = soup.find("meta", attrs=attrs)
            if not tag:
                continue
            content = _normalize_space(tag.get("content", ""))
            if content:
                return content
        return ""

    def _extract_html(self, html_text: str) -> Dict[str, str]:
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup.find_all(["script", "style", "noscript"]):
            tag.decompose()

        title = self._extract_meta_content(
            soup,
            [{"property": "og:title"}, {"name": "twitter:title"}],
        )
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = _normalize_space(h1.get_text(" ", strip=True))
        if not title and soup.title:
            title = _normalize_space(soup.title.get_text(" ", strip=True))

        published = self._extract_meta_content(
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
            if time_tag is not None:
                published = _normalize_space(time_tag.get("datetime", "") or time_tag.get_text(" ", strip=True))

        description = self._extract_meta_content(
            soup,
            [{"property": "og:description"}, {"name": "description"}],
        )

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
                paragraphs: List[str] = []
                for block in node.select("p, li"):
                    block_text = _normalize_space(block.get_text(" ", strip=True))
                    if len(block_text.split()) < 5:
                        continue
                    paragraphs.append(block_text)
                txt = "\n\n".join(paragraphs).strip()
                words = len(txt.split())
                if words > best_words:
                    best_words = words
                    best_text = txt

        if best_words < 40:
            body = soup.body or soup
            body_text = _clean_multiline(body.get_text("\n"))
            if len(body_text.split()) > best_words:
                best_text = body_text

        return {
            "title": title,
            "date": published,
            "description": description,
            "full_text": best_text,
        }

    @staticmethod
    def _fallback_text(title: str, description: str) -> str:
        parts: List[str] = []
        clean_title = _normalize_space(title)
        clean_desc = _normalize_space(description)
        if clean_title:
            parts.append(f"Title: {clean_title}")
        if clean_desc:
            parts.append(f"Summary: {clean_desc}")
        return "\n".join(parts).strip()

    def extract_document(
        self,
        url: str,
        fallback_title: str = "",
        fallback_date: str = "",
        fallback_description: str = "",
        fallback_source_name: str = "",
    ) -> Dict[str, Any]:
        target = str(url or "").strip()
        if not target:
            raise ValueError("URL is required")

        fallback_text = self._fallback_text(fallback_title, fallback_description)

        try:
            response = self._fetch(target, timeout=60)
        except Exception:
            if not fallback_text:
                raise
            doc_date = _date_to_display(fallback_date)
            return {
                "success": True,
                "data": {
                    "url": target,
                    "title": str(fallback_title or "").strip() or "Trade Media Article",
                    "date": doc_date,
                    "source_name": str(fallback_source_name or "").strip(),
                    "description": str(fallback_description or "").strip(),
                    "full_text": fallback_text,
                    "word_count": len(fallback_text.split()),
                    "source_format": "snippet",
                },
            }

        final_url = str(getattr(response, "url", target) or target)
        content_type = str(response.headers.get("Content-Type", "") or "").lower()
        is_pdf = final_url.lower().endswith(".pdf") or "application/pdf" in content_type

        source_format = "html"
        doc_title = ""
        doc_date_raw = ""
        description = str(fallback_description or "").strip()
        full_text = ""

        if is_pdf:
            source_format = "pdf"
            try:
                from pypdf import PdfReader
                import io
            except Exception as e:
                raise RuntimeError(f"PDF extraction requires pypdf: {e}") from e

            reader = PdfReader(io.BytesIO(response.content))
            pages: List[str] = []
            for page in reader.pages:
                txt = _clean_multiline(page.extract_text() or "")
                if txt:
                    pages.append(txt)
            full_text = "\n\n".join(pages).strip()
        else:
            parsed = self._extract_html(response.text)
            doc_title = parsed.get("title", "")
            doc_date_raw = parsed.get("date", "")
            description = parsed.get("description", "") or description
            full_text = parsed.get("full_text", "")

        doc_title = _normalize_space(doc_title) or _normalize_space(fallback_title) or "Trade Media Article"
        doc_date = _date_to_display(doc_date_raw or fallback_date)

        if fallback_text and len(full_text.split()) < 80:
            if full_text:
                full_text = f"{full_text}\n\nSource Summary\n{fallback_text}".strip()
            else:
                full_text = fallback_text
                source_format = "snippet"

        if not full_text:
            raise RuntimeError("No text extracted from article URL.")

        return {
            "success": True,
            "data": {
                "url": final_url,
                "title": doc_title,
                "date": doc_date,
                "source_name": str(fallback_source_name or "").strip(),
                "description": description,
                "full_text": full_text,
                "word_count": len(full_text.split()),
                "source_format": source_format,
            },
        }
