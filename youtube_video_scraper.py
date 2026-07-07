#!/usr/bin/env python3
"""YouTube video transcript scraper for public policy source channels."""

from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote, urlparse

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import CouldNotRetrieveTranscript, IpBlocked, RequestBlocked
from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig

from webshare_proxy import should_retry_with_webshare, webshare_rotating_proxies


YOUTUBE_RSS_NS = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}
YOUTUBE_WATCH_URL = "https://www.youtube.com/watch?v={video_id}"
SEC_YOUTUBE_DEFAULT_URL = "https://www.youtube.com/user/SECViews"


def _normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _parse_datetime(value: Any) -> Optional[datetime]:
    text = _normalize_space(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(text)
        if parsed is not None:
            return parsed.replace(tzinfo=None)
    except Exception:
        pass
    return None


def _date_display(value: Any) -> str:
    parsed = _parse_datetime(value)
    return parsed.strftime("%B %d, %Y") if parsed else _normalize_space(value)


def _video_id_from_url(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if re.fullmatch(r"[\w-]{11}", raw):
        return raw
    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        return parsed.path.strip("/").split("/", 1)[0]
    qs = parse_qs(parsed.query)
    if qs.get("v"):
        return str(qs["v"][0] or "").strip()
    parts = [part for part in parsed.path.split("/") if part]
    for marker in ("shorts", "embed", "live"):
        if marker in parts:
            idx = parts.index(marker)
            if len(parts) > idx + 1:
                return parts[idx + 1]
    return ""


class YouTubeVideoScraper:
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

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_delay_seconds:
            time.sleep(self.min_delay_seconds - elapsed)
        self._last_request_ts = time.time()

    def _fetch(self, url: str, timeout: int = 30) -> requests.Response:
        target = str(url or "").strip()
        if not target:
            raise ValueError("URL is required")
        self._rate_limit()
        response = self.session.get(target, timeout=timeout, allow_redirects=True)
        if should_retry_with_webshare(response.status_code):
            proxies = webshare_rotating_proxies()
            if proxies:
                response = self.session.get(target, timeout=timeout, allow_redirects=True, proxies=proxies)
        response.raise_for_status()
        return response

    def resolve_channel_id(self, channel_ref: str) -> str:
        ref = str(channel_ref or "").strip() or SEC_YOUTUBE_DEFAULT_URL
        if re.fullmatch(r"UC[\w-]+", ref):
            return ref
        if "feeds/videos.xml" in ref:
            channel_id = str(parse_qs(urlparse(ref).query).get("channel_id", [""])[0] or "").strip()
            if channel_id:
                return channel_id
        parsed = urlparse(ref)
        parts = [part for part in parsed.path.split("/") if part]
        if "channel" in parts:
            idx = parts.index("channel")
            if len(parts) > idx + 1 and parts[idx + 1].startswith("UC"):
                return parts[idx + 1]
        url = ref if ref.startswith("http") else f"https://www.youtube.com/{ref if ref.startswith('@') else '@' + ref.lstrip('@')}"
        response = self._fetch(url, timeout=30)
        match = re.search(r'"externalId":"(UC[\w-]+)"', response.text)
        if not match:
            raise RuntimeError(f"Could not resolve YouTube channel id from {ref}")
        return match.group(1)

    def fetch_rss_entries(self, channel_id: str, max_items: int = 25) -> List[Dict[str, Any]]:
        feed_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        response = self._fetch(feed_url, timeout=30)
        root = ET.fromstring(response.text)
        out: List[Dict[str, Any]] = []
        for entry in root.findall("atom:entry", YOUTUBE_RSS_NS):
            video_id_node = entry.find("yt:videoId", YOUTUBE_RSS_NS)
            title_node = entry.find("atom:title", YOUTUBE_RSS_NS)
            published_node = entry.find("atom:published", YOUTUBE_RSS_NS)
            updated_node = entry.find("atom:updated", YOUTUBE_RSS_NS)
            if video_id_node is None or title_node is None:
                continue
            video_id = _normalize_space(video_id_node.text)
            if not video_id:
                continue
            published = _normalize_space(published_node.text if published_node is not None else "")
            out.append(
                {
                    "video_id": video_id,
                    "title": _normalize_space(title_node.text),
                    "url": YOUTUBE_WATCH_URL.format(video_id=video_id),
                    "date": _date_display(published),
                    "published_at": published,
                    "updated_at": _normalize_space(updated_node.text if updated_node is not None else ""),
                    "channel_id": channel_id,
                    "discovery_source": "youtube_channel_rss",
                }
            )
            if len(out) >= max_items:
                break
        return out

    def discover_video(self, video_ref: str) -> Dict[str, Any]:
        video_id = _video_id_from_url(video_ref)
        if not video_id:
            raise ValueError(f"Could not parse YouTube video id from {video_ref}")

        watch_url = YOUTUBE_WATCH_URL.format(video_id=video_id)
        title = ""
        published = ""
        channel_id = ""

        try:
            oembed_url = f"https://www.youtube.com/oembed?url={quote(watch_url, safe='')}&format=json"
            payload = json.loads(self._fetch(oembed_url, timeout=20).text)
            title = _normalize_space(payload.get("title"))
        except Exception:
            pass

        try:
            html = self._fetch(watch_url, timeout=30).text
            date_match = re.search(r'"datePublished"\s*:\s*"([^"]+)"', html)
            channel_match = re.search(r'"channelId"\s*:\s*"(UC[\w-]+)"', html)
            title_match = re.search(r'"title"\s*:\s*"([^"]+)"', html)
            if date_match:
                published = _normalize_space(date_match.group(1))
            if channel_match:
                channel_id = _normalize_space(channel_match.group(1))
            if not title and title_match:
                title = _normalize_space(title_match.group(1))
        except Exception:
            pass

        return {
            "video_id": video_id,
            "title": title or video_id,
            "url": watch_url,
            "date": _date_display(published),
            "published_at": published,
            "updated_at": "",
            "channel_id": channel_id,
            "discovery_source": "youtube_direct_video",
        }

    def discover_documents(self, channel_ref: str = "", max_pages: int = 1, limit: int = 25) -> List[Dict[str, Any]]:
        direct_video_id = _video_id_from_url(channel_ref)
        if direct_video_id:
            entry = self.discover_video(channel_ref)
            self.last_discovery_debug = {
                "channel_ref": channel_ref,
                "video_id": direct_video_id,
                "discovery_source": "youtube_direct_video",
                "discovered_count": 1,
            }
            return [entry]

        channel_id = self.resolve_channel_id(channel_ref or SEC_YOUTUBE_DEFAULT_URL)
        max_items = max(1, min(50, int(limit or 25), int(max_pages or 1) * 15))
        entries = self.fetch_rss_entries(channel_id, max_items=max_items)
        self.last_discovery_debug = {
            "channel_ref": channel_ref or SEC_YOUTUBE_DEFAULT_URL,
            "channel_id": channel_id,
            "feed_url": f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}",
            "max_items": max_items,
            "discovered_count": len(entries),
        }
        return entries

    def _transcript_proxy_config(self) -> Any:
        username = str(os.getenv("WEBSHARE_PROXY_USERNAME") or "").strip()
        password = str(os.getenv("WEBSHARE_PROXY_PASSWORD") or "").strip()
        if username and password:
            return WebshareProxyConfig(proxy_username=username.removesuffix("-rotate"), proxy_password=password)
        proxy_url = str(os.getenv("YOUTUBE_PROXY_URL") or "").strip()
        if proxy_url:
            return GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)
        return None

    def fetch_transcript(self, video_id: str, attempts: int = 3) -> str:
        last_error: Optional[BaseException] = None
        api = YouTubeTranscriptApi(proxy_config=self._transcript_proxy_config())
        for attempt in range(max(1, int(attempts))):
            try:
                transcript = api.fetch(video_id)
                return " ".join(_normalize_space(snippet.text) for snippet in transcript if _normalize_space(snippet.text))
            except (IpBlocked, RequestBlocked, CouldNotRetrieveTranscript):
                raise
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt < attempts - 1:
                    time.sleep(2 * (attempt + 1))
        if last_error:
            raise last_error
        raise RuntimeError(f"Could not fetch transcript for YouTube video {video_id}")

    def extract_document(self, video_url_or_id: str, fallback_title: str = "", fallback_date: str = "") -> Dict[str, Any]:
        video_id = _video_id_from_url(video_url_or_id)
        if not video_id:
            raise ValueError(f"Could not parse YouTube video id from {video_url_or_id}")
        transcript = self.fetch_transcript(video_id)
        if len(transcript.split()) < 25:
            raise RuntimeError(f"YouTube transcript is too short for {video_id}")
        return {
            "data": {
                "video_id": video_id,
                "url": YOUTUBE_WATCH_URL.format(video_id=video_id),
                "title": _normalize_space(fallback_title) or video_id,
                "date": _date_display(fallback_date),
                "published_at": _normalize_space(fallback_date),
                "full_text": transcript,
            }
        }
