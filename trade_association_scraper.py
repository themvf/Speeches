#!/usr/bin/env python3
"""Trade-association source scraper.

Discovers and extracts public news, press-release, policy, and advocacy items
from financial-services trade associations. The scraper intentionally uses a
configuration map so new associations can be added without creating a bespoke
scraper for each site.
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
from bs4 import BeautifulSoup, Tag


TRADE_ASSOCIATION_DEFAULT_QUERY = "SEC OR FINRA OR CFTC OR Treasury OR CFPB OR OCC OR Federal Reserve"

TRADE_ASSOCIATION_SOURCES: Dict[str, Dict[str, Any]] = {
    "ici_news_item": {
        "label": "Investment Company Institute",
        "organization": "ICI",
        "default_url": "https://www.ici.org/news_%26_opinions/news-releases",
        "doc_type": "News Release",
        "tags_csv": "ici,trade-association,asset-management,mutual-funds,etfs",
        "article_path_keywords": ["/news-release", "/news-releases", "/viewpoints", "/comment-letter"],
    },
    "isda_news_item": {
        "label": "ISDA",
        "organization": "ISDA",
        "default_url": "https://www.isda.org/category/news/?subcategories=24",
        "doc_type": "News Item",
        "tags_csv": "isda,trade-association,derivatives,swaps,market-structure",
        "rss_candidates": ["https://www.isda.org/feed/"],
        "article_path_keywords": ["/category/news", "/a/"],
    },
    "mfa_news_item": {
        "label": "Managed Funds Association",
        "organization": "MFA",
        "default_url": "https://www.mfaalts.org/newsroom/",
        "doc_type": "News Item",
        "tags_csv": "mfa,trade-association,private-funds,hedge-funds,alternatives",
        "article_path_keywords": ["/newsroom/", "/press-release", "/policy"],
    },
    "fia_news_item": {
        "label": "FIA",
        "organization": "FIA",
        "default_url": "https://www.fia.org/news",
        "doc_type": "News Item",
        "tags_csv": "fia,trade-association,futures,clearing,derivatives",
        "rss_candidates": ["https://www.fia.org/rss.xml"],
        "article_path_keywords": ["/articles/", "/news/", "/resources/"],
    },
    "aba_news_item": {
        "label": "American Bankers Association",
        "organization": "ABA",
        "default_url": "https://www.aba.com/about-us/press-room/press-releases",
        "doc_type": "Press Release",
        "tags_csv": "aba,trade-association,banking,policy,press-release",
        "article_path_keywords": ["/about-us/press-room/press-releases/"],
    },
    "bpi_news_item": {
        "label": "Bank Policy Institute",
        "organization": "BPI",
        "default_url": "https://bpi.com/news/",
        "doc_type": "News Item",
        "tags_csv": "bpi,trade-association,banking,policy,capital,regulation",
        "rss_candidates": ["https://bpi.com/feed/"],
        "article_path_keywords": ["/news/", "/press-release", "/policy"],
    },
    "icba_news_item": {
        "label": "Independent Community Bankers of America",
        "organization": "ICBA",
        "default_url": "https://www.icba.org/newsroom/news-and-articles",
        "doc_type": "News Item",
        "tags_csv": "icba,trade-association,community-banking,policy",
        "article_path_keywords": ["/newsroom/news-and-articles/", "/newsroom/news-details/"],
    },
    "lsta_news_item": {
        "label": "LSTA",
        "organization": "LSTA",
        "default_url": "https://www.lsta.org/news-resources/",
        "doc_type": "News Item",
        "tags_csv": "lsta,trade-association,loans,private-credit,clo",
        "article_path_keywords": ["/news-resources/", "/news/"],
    },
    # Capital formation associations. Neither publishes an RSS feed (checked
    # 2026-08-16: ipa.com/feed and adisa.org/feed both 404), so both rely on
    # the listing-scrape path and carry no rss_candidates.
    "ipa_news_item": {
        "label": "Institute for Portfolio Alternatives",
        "organization": "IPA",
        "default_url": "https://www.ipa.com/about/news",
        "doc_type": "News Item",
        "tags_csv": (
            "ipa,trade-association,capital-formation,alternatives,"
            "non-traded-reit,bdc,direct-participation,interval-fund"
        ),
        "article_path_keywords": ["/articles/"],
    },
    "adisa_news_item": {
        "label": "ADISA",
        "organization": "ADISA",
        "default_url": "https://www.adisa.org/news-advocacy",
        "doc_type": "News Item",
        "tags_csv": (
            "adisa,trade-association,capital-formation,alternatives,"
            "direct-participation,1031-exchange,accredited-investor"
        ),
        # Only /article/ detail pages. A bare /news-advocacy/ prefix also let
        # section landing pages through as if they were articles.
        "article_path_keywords": ["/news-advocacy/article/"],
    },
}


def _normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clean_multiline(text: Any) -> str:
    lines = [_normalize_space(line) for line in str(text or "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _xml_local_name(tag: Any) -> str:
    return str(tag or "").rsplit("}", 1)[-1]


def _parse_date_text(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed
    except ValueError:
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
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        parsed = parsedate_to_datetime(text)
        return parsed.replace(tzinfo=None) if parsed is not None else None
    except Exception:
        return None


def _date_to_display(value: Any) -> str:
    parsed = _parse_date_text(value)
    return parsed.strftime("%B %d, %Y") if parsed is not None else str(value or "").strip()


def _extract_first_date(text: Any) -> str:
    match = re.search(
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)"
        r"\s+\d{1,2},\s+\d{4}|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b)",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    return _date_to_display(match.group(1)) if match else ""


def _strip_html(value: Any) -> str:
    return _normalize_space(BeautifulSoup(str(value or ""), "html.parser").get_text(" ", strip=True))


def _url_key(url: Any) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}"


def _same_host(left: str, right: str) -> bool:
    a = urlparse(left).netloc.lower().removeprefix("www.")
    b = urlparse(right).netloc.lower().removeprefix("www.")
    return bool(a and b and (a == b or a.endswith(f".{b}") or b.endswith(f".{a}")))


def _title_from_url(url: Any, fallback: str = "Trade Association Item") -> str:
    slug = urlparse(str(url or "")).path.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"\.(html?|pdf)$", "", slug, flags=re.IGNORECASE)
    title = " ".join(part for part in re.split(r"[-_]+", slug) if part).strip()
    return title.title() if title else fallback


def _looks_like_detail_path(url: str, cfg: Dict[str, Any]) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    if not path or path == "/":
        return False
    if any(part in path for part in ("/login", "/contact", "/events", "/tag/", "/author/", "/page/")):
        return False
    keywords = [str(item).lower() for item in cfg.get("article_path_keywords", []) if str(item).strip()]
    if keywords:
        return any(keyword in path for keyword in keywords)
    return bool(re.search(r"/20\d{2}/|/news|/press|/policy|/advocacy", path))


def _best_article_text(soup: BeautifulSoup) -> str:
    selectors = [
        "article",
        "main",
        '[itemprop="articleBody"]',
        ".entry-content",
        ".post-content",
        ".article-content",
        ".body-content",
        ".content",
    ]
    best = ""
    best_words = 0
    for selector in selectors:
        for node in soup.select(selector):
            paragraphs = [
                _normalize_space(p.get_text(" ", strip=True))
                for p in node.select("p, li")
                if len(_normalize_space(p.get_text(" ", strip=True)).split()) >= 5
            ]
            candidate = "\n\n".join(paragraphs) if paragraphs else _clean_multiline(node.get_text("\n"))
            words = len(candidate.split())
            if words > best_words:
                best = candidate
                best_words = words
    return best


class TradeAssociationScraper:
    def __init__(self, min_delay_seconds: float = 0.5):
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

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_delay_seconds:
            time.sleep(self.min_delay_seconds - elapsed)
        self._last_request_ts = time.time()

    def _fetch(self, url: str, timeout: int = 45) -> requests.Response:
        self._rate_limit()
        response = self.session.get(url, timeout=timeout)
        response.raise_for_status()
        return response

    def _discover_from_rss(self, source_key: str, cfg: Dict[str, Any], feed_url: str, max_items: int) -> List[Dict[str, Any]]:
        response = self._fetch(feed_url)
        root = ET.fromstring(response.text)
        items = [el for el in root.iter() if _xml_local_name(el.tag).lower() in {"item", "entry"}]
        docs: List[Dict[str, Any]] = []
        for item in items[:max_items]:
            fields: Dict[str, str] = {}
            for child in list(item):
                name = _xml_local_name(child.tag).lower()
                text = _normalize_space(child.text or "")
                if name == "link":
                    href = child.attrib.get("href") or text
                    fields.setdefault("link", href)
                elif name in {"title", "pubdate", "published", "updated", "description", "summary", "content"}:
                    fields[name] = text
            url = str(fields.get("link", "") or "").strip()
            if not url:
                continue
            docs.append(
                {
                    "url": url,
                    "title": _strip_html(fields.get("title", "")) or _title_from_url(url),
                    "date": _date_to_display(fields.get("pubdate") or fields.get("published") or fields.get("updated") or ""),
                    "description": _strip_html(fields.get("description") or fields.get("summary") or fields.get("content") or ""),
                    "source_key": source_key,
                    "source_label": str(cfg.get("label", "") or source_key),
                    "organization": str(cfg.get("organization", "") or ""),
                    "doc_type": str(cfg.get("doc_type", "") or "News Item"),
                    "tags_csv": str(cfg.get("tags_csv", "") or "trade-association"),
                    "source_format": "html",
                    "discovery_source": "rss",
                    "listing_page": feed_url,
                }
            )
        return docs

    def _discover_from_html(self, source_key: str, cfg: Dict[str, Any], base_url: str, max_items: int) -> List[Dict[str, Any]]:
        response = self._fetch(base_url)
        soup = BeautifulSoup(response.text, "html.parser")
        docs: List[Dict[str, Any]] = []
        seen = set()
        for link in soup.select("a[href]"):
            href = str(link.get("href", "") or "").strip()
            url = urljoin(base_url, href)
            if not url.startswith(("http://", "https://")) or not _same_host(url, base_url):
                continue
            if not _looks_like_detail_path(url, cfg):
                continue
            key = _url_key(url)
            if not key or key in seen:
                continue
            seen.add(key)
            container = link
            for parent in link.parents:
                if isinstance(parent, Tag) and parent.name in {"article", "li", "div", "section"}:
                    container = parent
                    break
            title = _normalize_space(link.get_text(" ", strip=True))
            heading = container.select_one("h1, h2, h3, h4")
            if heading:
                heading_text = _normalize_space(heading.get_text(" ", strip=True))
                if len(heading_text) > len(title):
                    title = heading_text
            blob = _normalize_space(container.get_text(" ", strip=True))
            docs.append(
                {
                    "url": url,
                    "title": title or _title_from_url(url),
                    "date": _extract_first_date(blob),
                    "description": blob[:800],
                    "source_key": source_key,
                    "source_label": str(cfg.get("label", "") or source_key),
                    "organization": str(cfg.get("organization", "") or ""),
                    "doc_type": str(cfg.get("doc_type", "") or "News Item"),
                    "tags_csv": str(cfg.get("tags_csv", "") or "trade-association"),
                    "source_format": "html",
                    "discovery_source": "html_listing",
                    "listing_page": base_url,
                }
            )
            if len(docs) >= max_items:
                break
        return docs

    def discover_documents(
        self,
        source_key: str,
        base_url: str = "",
        max_pages: int = 1,
        include_rss: bool = True,
    ) -> List[Dict[str, Any]]:
        cfg = TRADE_ASSOCIATION_SOURCES.get(source_key)
        if not cfg:
            raise ValueError(f"Unsupported trade association source: {source_key}")
        target = str(base_url or cfg.get("default_url", "") or "").strip()
        max_items = max(1, int(max_pages or 1)) * 25
        debug: Dict[str, Any] = {
            "source_key": source_key,
            "source_index_url": target,
            "rss_attempts": [],
            "html_attempted": False,
            "errors": [],
        }
        docs: List[Dict[str, Any]] = []
        if include_rss:
            for feed_url in cfg.get("rss_candidates", []) or []:
                try:
                    found = self._discover_from_rss(source_key, cfg, str(feed_url), max_items=max_items)
                    debug["rss_attempts"].append({"feed_url": feed_url, "count": len(found)})
                    docs.extend(found)
                    if len(docs) >= max_items:
                        break
                except Exception as exc:
                    debug["rss_attempts"].append({"feed_url": feed_url, "error": str(exc)})
        if len(docs) < max_items and target:
            try:
                debug["html_attempted"] = True
                docs.extend(self._discover_from_html(source_key, cfg, target, max_items=max_items - len(docs)))
            except Exception as exc:
                debug["errors"].append(str(exc))

        dedup: Dict[str, Dict[str, Any]] = {}
        for doc in docs:
            key = _url_key(doc.get("url", ""))
            if key and key not in dedup:
                dedup[key] = doc
        self.last_discovery_debug = {**debug, "discovered_count": len(dedup)}
        return list(dedup.values())[:max_items]

    def extract_document(
        self,
        entry_or_url: Any,
        fallback_title: str = "",
        fallback_date: str = "",
        fallback_description: str = "",
        fallback_source_name: str = "",
    ) -> Dict[str, Any]:
        entry = entry_or_url if isinstance(entry_or_url, dict) else {"url": str(entry_or_url or "")}
        url = str(entry.get("url", "") or "").strip()
        title = str(entry.get("title", "") or fallback_title or "").strip() or _title_from_url(url)
        date_text = str(entry.get("date", "") or fallback_date or "").strip()
        description = str(entry.get("description", "") or fallback_description or "").strip()
        source_name = str(entry.get("source_label", "") or fallback_source_name or entry.get("organization", "") or "").strip()
        if not url:
            return {"success": False, "error": "No URL supplied for trade association extraction.", "data": {}}

        try:
            response = self._fetch(url, timeout=90)
            soup = BeautifulSoup(response.text, "html.parser")
            page_title = (
                _normalize_space((soup.select_one("h1") or soup.select_one("meta[property='og:title']") or {}).get_text(" ", strip=True))
                if soup.select_one("h1")
                else ""
            )
            if not page_title:
                og = soup.select_one("meta[property='og:title'], meta[name='twitter:title']")
                page_title = _normalize_space(og.get("content", "") if og else "")
            title = page_title or title
            time_node = soup.select_one("time[datetime], time")
            date_text = _date_to_display(
                (time_node.get("datetime") if time_node and time_node.has_attr("datetime") else "")
                or (time_node.get_text(" ", strip=True) if time_node else "")
                or date_text
            )
            text = _best_article_text(soup)
            if not text:
                text = description
            return {
                "success": True,
                "data": {
                    "url": response.url or url,
                    "title": title,
                    "date": date_text,
                    "description": description,
                    "full_text": text,
                    "word_count": len(text.split()),
                    "source_name": source_name,
                    "source_format": "html",
                    "extraction_mode": "html",
                },
            }
        except Exception as exc:
            text = "\n".join(
                part
                for part in [
                    title,
                    f"Source: {source_name}",
                    f"Date: {date_text}",
                    f"URL: {url}",
                    description,
                    f"Extraction note: detail page fetch failed: {exc}",
                ]
                if str(part or "").strip()
            )
            return {
                "success": True,
                "data": {
                    "url": url,
                    "title": title,
                    "date": date_text,
                    "description": description,
                    "full_text": text,
                    "word_count": len(text.split()),
                    "source_name": source_name,
                    "source_format": "snippet",
                    "extraction_mode": "metadata_fallback",
                },
            }
