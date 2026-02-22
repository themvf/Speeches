#!/usr/bin/env python3
"""
Federal Reserve speeches and testimony scraper.

Discovers annual listing-page links for speeches/testimony and extracts
full text from detail pages.
"""

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup


FED_SPEECH_TESTIMONY_URL = "https://www.federalreserve.gov/newsevents/speeches-testimony.htm"
FED_FEED_SPEECHES_AND_TESTIMONY = "https://www.federalreserve.gov/feeds/speeches_and_testimony.xml"


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
    for fmt in (
        "%A, %B %d, %Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y-%m-%d",
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


def _is_fed_detail_url(url: str) -> bool:
    lower = str(url or "").lower()
    if not ("/newsevents/speech/" in lower or "/newsevents/testimony/" in lower):
        return False
    if not lower.endswith(".htm"):
        return False
    # Exclude annual listing pages.
    if re.search(r"/newsevents/speech/\d{4}-speeches\.htm$", lower):
        return False
    if re.search(r"/newsevents/speech/\d{4}speech\.htm$", lower):
        return False
    if re.search(r"/newsevents/testimony/\d{4}-testimony\.htm$", lower):
        return False
    if re.search(r"/newsevents/testimony/\d{4}testimony\.htm$", lower):
        return False
    return True


def _speaker_from_feed_title(title: str) -> str:
    text = _normalize_space(title)
    if not text:
        return ""
    if "," not in text:
        return ""
    speaker = text.split(",", 1)[0].strip()
    if len(speaker) > 80:
        return ""
    return speaker


class FederalReserveSpeechTestimonyScraper:
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

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_delay_seconds:
            time.sleep(self.min_delay_seconds - elapsed)
        self._last_request_ts = time.time()

    def _fetch_html(self, url: str, timeout: int = 60) -> requests.Response:
        target = str(url or "").strip()
        if not target:
            raise ValueError("URL is required")
        self._rate_limit()
        response = self.session.get(target, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        return response

    @staticmethod
    def _dedupe_urls(urls: List[str]) -> List[str]:
        out = []
        seen = set()
        for raw in urls:
            url = str(raw or "").strip()
            if not url:
                continue
            if url in seen:
                continue
            seen.add(url)
            out.append(url)
        return out

    def _year_listing_candidates(self, year: int, kind: str, base_url: str = FED_SPEECH_TESTIMONY_URL) -> List[str]:
        root = "https://www.federalreserve.gov"
        year = int(year)
        kind = str(kind or "").strip().lower()

        # Keep the user-provided root URL visible in debug by deriving one candidate from it.
        parsed_base = urlparse(str(base_url or FED_SPEECH_TESTIMONY_URL))
        base_root = f"{parsed_base.scheme or 'https'}://{parsed_base.netloc or 'www.federalreserve.gov'}"

        if kind == "speech":
            candidates = [
                f"{root}/newsevents/speech/{year}-speeches.htm",
                f"{base_root}/newsevents/speech/{year}-speeches.htm",
                f"{root}/newsevents/{year}-speeches.htm",
                f"{root}/newsevents/speech/{year}speech.htm",
                f"{root}/newsevents/{year}speech.htm",
            ]
            return self._dedupe_urls(candidates)

        candidates = [
            f"{root}/newsevents/testimony/{year}-testimony.htm",
            f"{base_root}/newsevents/testimony/{year}-testimony.htm",
            f"{root}/newsevents/{year}-testimony.htm",
            f"{root}/newsevents/testimony/{year}testimony.htm",
            f"{root}/newsevents/{year}testimony.htm",
        ]
        return self._dedupe_urls(candidates)

    def _discover_from_listing_page(
        self,
        page_url: str,
        default_kind: str = "",
        year: int = 0,
    ) -> List[Dict[str, str]]:
        response = self._fetch_html(page_url, timeout=60)
        soup = BeautifulSoup(response.text, "html.parser")
        rows = soup.select("div.row.eventlist div.row")
        if not rows:
            return []

        out: List[Dict[str, str]] = []
        seen = set()
        for row in rows:
            detail_url = ""
            title_text = ""
            for anchor in row.select("a[href]"):
                href = anchor.get("href", "")
                full_url = _url_without_query(urljoin("https://www.federalreserve.gov", href))
                if not _is_fed_detail_url(full_url):
                    continue
                detail_url = full_url
                title_text = _normalize_space(anchor.get_text(" ", strip=True))
                if title_text:
                    break
            if not detail_url:
                continue

            key = _url_key(detail_url)
            if not key or key in seen:
                continue
            seen.add(key)

            date_el = row.find("time")
            date_text = _normalize_space(date_el.get_text(" ", strip=True) if date_el else "")
            date_value = _date_to_display(date_text)

            speaker_el = row.select_one("p.news__speaker, p.speaker")
            speaker = _normalize_space(speaker_el.get_text(" ", strip=True) if speaker_el else "")

            location = ""
            for p in row.find_all("p"):
                p_text = _normalize_space(p.get_text(" ", strip=True))
                if not p_text:
                    continue
                if p.find("a", href=True):
                    continue
                if p.get("class") and ("news__speaker" in p.get("class") or "speaker" in p.get("class")):
                    continue
                if location:
                    break
                location = p_text

            kind = "Speech"
            lower_url = detail_url.lower()
            if "/newsevents/testimony/" in lower_url:
                kind = "Testimony"
            elif str(default_kind).lower() == "testimony":
                kind = "Testimony"

            out.append(
                {
                    "url": detail_url,
                    "title": title_text or "Federal Reserve Speech/Testimony",
                    "date": date_value,
                    "speaker": speaker,
                    "location": location,
                    "doc_type": kind,
                    "source_format": "html",
                    "listing_page": str(page_url or ""),
                    "year": str(year or ""),
                }
            )

        return out

    def _discover_from_feed(self, feed_url: str = FED_FEED_SPEECHES_AND_TESTIMONY) -> List[Dict[str, str]]:
        self._rate_limit()
        response = self.session.get(feed_url, timeout=45, allow_redirects=True)
        response.raise_for_status()

        # Parse bytes directly to avoid charset mismatch on UTF-8 BOM feeds.
        root = ET.fromstring(response.content)
        items = [node for node in root.iter() if node.tag.lower().endswith("item")]
        out = []
        seen = set()
        for item in items:
            payload = {}
            for child in list(item):
                tag = str(child.tag).split("}", 1)[-1].lower()
                payload[tag] = _normalize_space(child.text or "")

            link = _url_without_query(payload.get("link", ""))
            if not _is_fed_detail_url(link):
                continue
            key = _url_key(link)
            if not key or key in seen:
                continue
            seen.add(key)

            title = payload.get("title", "")
            date_value = _date_to_display(payload.get("pubdate", ""))
            kind = "Testimony" if "/newsevents/testimony/" in link.lower() else "Speech"
            speaker = _speaker_from_feed_title(title)

            out.append(
                {
                    "url": link,
                    "title": title or "Federal Reserve Speech/Testimony",
                    "date": date_value,
                    "speaker": speaker,
                    "location": "",
                    "doc_type": kind,
                    "source_format": "xml_rss",
                    "listing_page": str(feed_url or ""),
                    "year": "",
                }
            )
        return out

    def discover_documents(
        self,
        base_url: str = FED_SPEECH_TESTIMONY_URL,
        max_pages: int = 3,
        fallback_to_feed: bool = True,
    ) -> List[Dict[str, str]]:
        max_pages = max(1, int(max_pages or 1))
        current_year = datetime.utcnow().year
        target_count = max(20, min(3000, max_pages * 30))

        out: List[Dict[str, str]] = []
        seen = set()
        listing_added = 0
        feed_added = 0

        debug: Dict[str, Any] = {
            "base_url": str(base_url or FED_SPEECH_TESTIMONY_URL),
            "max_pages_requested": max_pages,
            "target_count": target_count,
            "fallback_to_feed": bool(fallback_to_feed),
            "years_scanned": [],
            "pages": [],
            "listing_added": 0,
            "feed_added": 0,
            "feed_supplement_used": False,
            "feed_supplement_error": "",
            "total_unique": 0,
            "stop_reason": "",
        }

        for offset in range(max_pages):
            year = current_year - offset
            debug["years_scanned"].append(int(year))

            for kind in ("speech", "testimony"):
                page_debug: Dict[str, Any] = {
                    "year": int(year),
                    "kind": kind,
                    "page_url": "",
                    "attempts": 0,
                    "error_type": "",
                    "error_status": 0,
                    "error_message": "",
                    "returned_items": 0,
                    "unique_added": 0,
                }

                discovered: List[Dict[str, str]] = []
                for candidate_url in self._year_listing_candidates(year, kind, base_url=base_url):
                    page_debug["attempts"] += 1
                    page_debug["page_url"] = candidate_url
                    try:
                        discovered = self._discover_from_listing_page(
                            candidate_url,
                            default_kind=kind,
                            year=year,
                        )
                        if discovered:
                            break
                    except requests.HTTPError as e:
                        page_debug["error_type"] = "HTTPError"
                        page_debug["error_status"] = int(
                            getattr(getattr(e, "response", None), "status_code", 0) or 0
                        )
                        page_debug["error_message"] = str(e)
                        continue
                    except Exception as e:
                        page_debug["error_type"] = type(e).__name__
                        page_debug["error_message"] = str(e)
                        continue

                page_debug["returned_items"] = len(discovered)
                for item in discovered:
                    key = _url_key(item.get("url", ""))
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    out.append(item)
                    page_debug["unique_added"] += 1
                    listing_added += 1

                debug["pages"].append(page_debug)

        if fallback_to_feed and len(out) < target_count:
            debug["feed_supplement_used"] = True
            try:
                feed_items = self._discover_from_feed()
                for item in feed_items:
                    key = _url_key(item.get("url", ""))
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    out.append(item)
                    feed_added += 1
            except Exception as e:
                debug["feed_supplement_error"] = str(e)

        def _sort_key(item: Dict[str, str]):
            return _parse_date_text(item.get("date", "")) or datetime.min

        out.sort(key=_sort_key, reverse=True)

        debug["listing_added"] = int(listing_added)
        debug["feed_added"] = int(feed_added)
        debug["total_unique"] = int(len(out))
        if not out:
            debug["stop_reason"] = "no_results"
        elif len(out) < target_count and fallback_to_feed:
            debug["stop_reason"] = "completed_below_target"
        else:
            debug["stop_reason"] = "completed"
        self.last_discovery_debug = debug
        return out

    def extract_document(
        self,
        url: str,
        fallback_title: str = "",
        fallback_date: str = "",
        fallback_speaker: str = "",
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

        article = soup.select_one("div#article") or soup.find("article") or soup.find("main") or soup
        heading = article.select_one("div.heading") if article else None
        if heading is not None:
            for node in heading.select("ul.list-unstyled, a.watchLive"):
                node.decompose()

        date_text = ""
        date_node = article.select_one("p.article__time, time") if article else None
        if date_node is not None:
            date_text = _normalize_space(date_node.get_text(" ", strip=True))

        title = ""
        title_node = article.select_one("h3.title") if article else None
        if title_node is not None:
            title = _normalize_space(title_node.get_text(" ", strip=True))
        if not title:
            title_tag = _normalize_space(soup.title.get_text(" ", strip=True) if soup.title else "")
            title = re.sub(r"\s*-\s*Federal Reserve Board\s*$", "", title_tag, flags=re.IGNORECASE).strip()
        if not title:
            title = _normalize_space(fallback_title) or "Federal Reserve Speech/Testimony"

        speaker = ""
        speaker_node = article.select_one("p.speaker, p.news__speaker") if article else None
        if speaker_node is not None:
            speaker = _normalize_space(speaker_node.get_text(" ", strip=True))
        if not speaker:
            speaker = _normalize_space(fallback_speaker)

        location = ""
        location_node = article.select_one("p.location") if article else None
        if location_node is not None:
            location = _normalize_space(location_node.get_text(" ", strip=True))

        full_text = _clean_multiline(article.get_text("\n")) if article is not None else ""
        if not full_text:
            full_text = _clean_multiline(soup.get_text("\n"))

        date_value = _date_to_display(date_text or fallback_date)
        if not date_value:
            date_value = _extract_first_date(full_text)

        doc_type = "Testimony" if "/newsevents/testimony/" in final_url.lower() else "Speech"

        return {
            "success": True,
            "data": {
                "url": final_url,
                "title": title,
                "date": date_value,
                "speaker": speaker,
                "location": location,
                "doc_type": doc_type,
                "full_text": full_text,
                "word_count": len(full_text.split()),
                "source_format": "html",
            },
        }
