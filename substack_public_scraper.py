#!/usr/bin/env python3
"""Direct public Substack search and post extraction connector."""

from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote, unquote, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

try:
    from curl_cffi import requests as curl_requests
except ImportError:  # pragma: no cover - requests remains a supported local fallback.
    curl_requests = None


SUBSTACK_SEARCH_URL = "https://substack.com/api/v1/post/search"
DEFAULT_KEYWORDS = ["securities", "financial industry", "decentralized finance"]
CORE_FEEDS: List[Dict[str, str]] = [
    {
        "label": "Capitol Account",
        "feed_url": "https://www.capitolaccountdc.com/feed",
        "tags_csv": "capitol-account,financial-regulation,washington-policy",
    },
    {
        "label": "FinRegRag",
        "feed_url": "https://www.finregrag.com/feed",
        "tags_csv": "finregrag,financial-regulation,policy-commentary",
    },
    {
        "label": "Bank Reg Blog",
        "feed_url": "https://bankregblog.substack.com/feed",
        "tags_csv": "bank-reg-blog,bank-regulation,financial-policy",
    },
    {
        "label": "The Public Interest by Better Markets",
        "feed_url": "https://bettermarkets.substack.com/feed",
        "tags_csv": "better-markets,market-structure,financial-stability",
    },
    {
        "label": "DeFi Education Fund",
        "feed_url": "https://defieducationfund.substack.com/feed",
        "tags_csv": "defi-education-fund,defi,crypto-policy",
    },
    {
        "label": "Trustless Policy",
        "feed_url": "https://trustlesspolicy.substack.com/feed",
        "tags_csv": "trustless-policy,crypto-policy,market-structure",
    },
    {
        "label": "Fintech Business Weekly",
        "feed_url": "https://fintechbusinessweekly.substack.com/feed",
        "tags_csv": "fintech-business-weekly,fintech,banking",
    },
    {
        "label": "The Dig",
        "feed_url": "https://thedig.substack.com/feed",
        "tags_csv": "the-dig,accounting,audit,governance",
    },
]
ADDITIONAL_SUBSTACK_FEEDS: List[Dict[str, str]] = [
    {
        "label": "The Bear Cave",
        "feed_url": "https://thebearcave.substack.com/feed",
        "tags_csv": "the-bear-cave,short-research,corporate-misconduct,equities",
    },
    {
        "label": "Klement on Investing",
        "feed_url": "https://klementoninvesting.substack.com/feed",
        "tags_csv": "klement-on-investing,financial-markets,behavioral-finance,macro",
    },
    {
        "label": "Doomberg",
        "feed_url": "https://newsletter.doomberg.com/feed",
        "tags_csv": "doomberg,financial-markets,energy,macro",
    },
    {
        "label": "Net Interest",
        "feed_url": "https://www.netinterest.co/feed",
        "tags_csv": "net-interest,financial-markets,banking,asset-management",
    },
    {
        "label": "The Overshoot",
        "feed_url": "https://theovershoot.co/feed",
        "tags_csv": "the-overshoot,macro,economic-policy,financial-markets",
    },
    {
        "label": "Apricitas Economics",
        "feed_url": "https://www.apricitas.io/feed",
        "tags_csv": "apricitas,economics,macro,financial-markets",
    },
    {
        "label": "The Macro Compass",
        "feed_url": "https://themacrocompass.substack.com/feed",
        "tags_csv": "macro-compass,macro,financial-markets,rates",
    },
    {
        "label": "Payments Wrap Up",
        "feed_url": "https://www.paymentswrapup.com/feed",
        "tags_csv": "payments-wrap-up,payments,fintech,banking",
    },
    {
        "label": "Canadian Fintech",
        "feed_url": "https://canadianfintech.substack.com/feed",
        "tags_csv": "canadian-fintech,fintech,open-banking,financial-policy",
    },
    {
        "label": "Fintech Is Easy",
        "feed_url": "https://fintechiseasy.com/feed",
        "tags_csv": "fintech-is-easy,fintech,payments,banking",
    },
    {
        "label": "Web3 vs. the Law",
        "feed_url": "https://davidlopezkurtz.substack.com/feed",
        "tags_csv": "web3-law,crypto-policy,securities-regulation,digital-assets",
    },
    {
        "label": "Venture in Security",
        "feed_url": "https://ventureinsecurity.net/feed",
        "tags_csv": "venture-in-security,cybersecurity,security-market,security-leadership",
    },
    {
        "label": "Resilient Cyber",
        "feed_url": "https://www.resilientcyber.io/feed",
        "tags_csv": "resilient-cyber,cybersecurity,cyber-policy,security-risk",
    },
    {
        "label": "The Cybersecurity Pulse",
        "feed_url": "https://www.cybersecuritypulse.net/feed",
        "tags_csv": "cybersecurity-pulse,cybersecurity,threats,security-operations",
    },
    {
        "label": "Cybersecurity Chronicles",
        "feed_url": "https://cybersecuritychronicles.substack.com/feed",
        "tags_csv": "cybersecurity-chronicles,cyber-risk,governance,security-policy",
    },
    {
        "label": "CyberMaterial",
        "feed_url": "https://www.cybermaterial.com/feed",
        "tags_csv": "cybermaterial,cybersecurity,threat-intelligence,security-news",
    },
    {
        "label": "ToxSec",
        "feed_url": "https://www.toxsec.com/feed",
        "tags_csv": "toxsec,cybersecurity,ai-security,security-research",
    },
    {
        "label": "TechLetters",
        "feed_url": "https://techletters.substack.com/feed",
        "tags_csv": "techletters,cyber-policy,privacy,technology-policy",
    },
    {
        "label": "Imperva Weekly Threat Intelligence",
        "feed_url": "https://imperva.substack.com/feed",
        "tags_csv": "imperva,threat-intelligence,cybersecurity,security-operations",
    },
    {
        "label": "The Attack Surface",
        "feed_url": "https://attacksurfaceai.substack.com/feed",
        "tags_csv": "attack-surface,ai-security,cybersecurity,threat-intelligence",
    },
    {
        "label": "Import AI",
        "feed_url": "https://importai.substack.com/feed",
        "tags_csv": "import-ai,artificial-intelligence,ai-policy,frontier-ai",
    },
    {
        "label": "One Useful Thing",
        "feed_url": "https://www.oneusefulthing.org/feed",
        "tags_csv": "one-useful-thing,artificial-intelligence,generative-ai,ai-adoption",
    },
    {
        "label": "Interconnects",
        "feed_url": "https://www.interconnects.ai/feed",
        "tags_csv": "interconnects,artificial-intelligence,open-models,ai-research",
    },
    {
        "label": "AI as Normal Technology",
        "feed_url": "https://www.normaltech.ai/feed",
        "tags_csv": "normal-technology,artificial-intelligence,ai-policy,technology-policy",
    },
    {
        "label": "a16z AI Policy Brief",
        "feed_url": "https://a16zpolicy.substack.com/feed",
        "tags_csv": "a16z-ai-policy,ai-policy,technology-policy,venture-capital",
    },
    {
        "label": "DC/ai Decoded",
        "feed_url": "https://www.dcaidecoded.com/feed",
        "tags_csv": "dc-ai-decoded,ai-policy,federal-policy,artificial-intelligence",
    },
    {
        "label": "Asia AI Policy Monitor",
        "feed_url": "https://asiaaipolicymonitor.substack.com/feed",
        "tags_csv": "asia-ai-policy,ai-policy,technology-policy,artificial-intelligence",
    },
]
DEFAULT_FEEDS: List[Dict[str, str]] = [*CORE_FEEDS, *ADDITIONAL_SUBSTACK_FEEDS]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "")
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _looks_like_proxy_tunnel_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "connect tunnel failed" in text
        or "proxy error" in text
        or "proxyerror" in text
    )


def _looks_like_curl_tls_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "tls connect error" in text
        or "openssl_internal" in text
        or "invalid library" in text
    )


def _normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [_normalize_space(item) for item in value if _normalize_space(item)]


def _contains_html_markup(value: Any) -> bool:
    return bool(re.search(r"<[A-Za-z][^>]*>", str(value or "")))


def _post_authors(post: Dict[str, Any]) -> List[str]:
    bylines = post.get("publishedBylines")
    if not isinstance(bylines, list):
        return []
    return [
        _normalize_space(item.get("name", ""))
        for item in bylines
        if isinstance(item, dict) and _normalize_space(item.get("name", ""))
    ]


def _post_tags(post: Dict[str, Any]) -> List[str]:
    tags = post.get("postTags")
    if not isinstance(tags, list):
        return []
    return [
        _normalize_space(item.get("name", ""))
        for item in tags
        if isinstance(item, dict) and _normalize_space(item.get("name", ""))
    ]


def _html_to_text(value: Any) -> str:
    text = str(value or "")
    if not _contains_html_markup(text):
        return _normalize_space(text)
    soup = BeautifulSoup(text, "html.parser")
    for node in soup.select("script, style, noscript"):
        node.decompose()
    blocks = [
        _normalize_space(node.get_text(" ", strip=True))
        for node in soup.select("h1, h2, h3, p, li, blockquote")
    ]
    blocks = [block for block in blocks if block]
    if blocks:
        return "\n\n".join(blocks)
    return _normalize_space(soup.get_text(" ", strip=True))


def _xml_local_name(tag: str) -> str:
    raw = str(tag or "")
    return raw.rsplit("}", 1)[-1] if "}" in raw else raw


def _strip_html(value: Any) -> str:
    text = str(value or "")
    if not _contains_html_markup(text):
        return _normalize_space(text)
    soup = BeautifulSoup(text, "html.parser")
    return _normalize_space(soup.get_text(" ", strip=True))


def _substack_slug_from_url(url: Any) -> str:
    parsed = urlparse(str(url or "").strip())
    segments = [segment for segment in parsed.path.split("/") if segment]
    if not segments:
        return ""
    if "p" in segments:
        idx = segments.index("p")
        if idx + 1 < len(segments):
            return _normalize_space(segments[idx + 1])
    return _normalize_space(segments[-1])


def _feed_config(value: Any) -> Dict[str, str]:
    if isinstance(value, dict):
        feed_url = _normalize_space(value.get("feed_url") or value.get("url") or "")
        return {
            "label": _normalize_space(value.get("label", "")),
            "feed_url": feed_url,
            "tags_csv": _normalize_space(value.get("tags_csv", "")),
        }
    feed_url = _normalize_space(value)
    return {"label": "", "feed_url": feed_url, "tags_csv": ""}


def _normalize_proxy_url(value: Any) -> Tuple[str, str]:
    raw = _normalize_space(value)
    if not raw:
        return "", ""

    for _ in range(2):
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'", "`"}:
            raw = _normalize_space(raw[1:-1])

    if "=" in raw and "://" in raw:
        name, maybe_value = raw.split("=", 1)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:_PROXY_URL|_PROXY|PROXY_URL|PROXY)", name.strip()):
            raw = _normalize_space(maybe_value)
            if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'", "`"}:
                raw = _normalize_space(raw[1:-1])

    candidate = raw
    if "://" not in candidate:
        parts = candidate.split(":")
        if len(parts) >= 4 and parts[1].isdigit():
            host = parts[0].strip()
            port = parts[1].strip()
            username = parts[2].strip()
            password = ":".join(parts[3:]).strip()
            if host and username:
                candidate = (
                    f"http://{quote(username, safe='')}:"
                    f"{quote(password, safe='')}@{host}:{port}"
                )
        else:
            candidate = f"http://{candidate}"

    parsed = urlparse(candidate)
    if parsed.scheme.lower() not in {"http", "https", "socks5", "socks5h"}:
        return "", "Proxy URL must use http, https, socks5, or socks5h."
    try:
        port = parsed.port
    except ValueError:
        return "", "Proxy URL must include a numeric port between 0 and 65535."
    if not parsed.hostname or port is None:
        return "", "Proxy URL must include a host and numeric port."

    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    auth = ""
    if parsed.username is not None or parsed.password is not None:
        auth = (
            f"{quote(unquote(parsed.username or ''), safe='')}:"
            f"{quote(unquote(parsed.password or ''), safe='')}@"
        )
    normalized = urlunparse((parsed.scheme.lower(), f"{auth}{host}:{port}", "", "", "", ""))
    return normalized, ""


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    elif hasattr(response, "dict"):
        payload = response.dict()
    else:
        payload = response
    if isinstance(payload, dict):
        for item in payload.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("text"):
                    return str(content["text"])
    return ""


def _chat_completion_text(response: Any) -> str:
    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    elif hasattr(response, "dict"):
        payload = response.dict()
    else:
        payload = response
    if isinstance(payload, dict):
        choices = payload.get("choices", [])
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            content = message.get("content", "") if isinstance(message, dict) else ""
            if isinstance(content, str):
                return content
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        if isinstance(content, str):
            return content
    return ""


def _json_object(value: str) -> Dict[str, Any]:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _parse_date_for_sort(value: Any) -> datetime:
    text = _normalize_space(value)
    if not text:
        return datetime.min.replace(tzinfo=timezone.utc)
    candidates = [text]
    if text.endswith("Z"):
        candidates.append(f"{text[:-1]}+00:00")
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
    try:
        parsed = parsedate_to_datetime(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)


def _date_sort_key(item: Dict[str, Any]) -> Tuple[datetime, str]:
    return (
        _parse_date_for_sort(item.get("date") or item.get("published_at")),
        _normalize_space(item.get("url", "")),
    )


class SubstackPublicScraper:
    def __init__(self, min_delay_seconds: float = 0.25) -> None:
        self._uses_curl_cffi = curl_requests is not None
        self.session = (
            curl_requests.Session(impersonate="chrome")
            if self._uses_curl_cffi
            else requests.Session()
        )
        self.proxy_url, self.proxy_config_error = _normalize_proxy_url(
            os.getenv("SUBSTACK_PROXY_URL", "")
            or os.getenv("RESIDENTIAL_PROXY_URL", "")
            or os.getenv("APIFY_PROXY_URL", "")
        )
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "Accept": "application/json,text/plain,*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://substack.com/",
            }
        )
        self.min_delay_seconds = max(0.0, float(min_delay_seconds))
        self.request_timeout_seconds = max(
            5.0, _env_float("SUBSTACK_REQUEST_TIMEOUT_SECONDS", 15.0)
        )
        self.direct_proxy_fallback = _env_bool("SUBSTACK_DIRECT_PROXY_FALLBACK", True)
        self._last_request_ts = 0.0
        self.last_discovery_debug: Dict[str, Any] = {}

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_delay_seconds:
            time.sleep(self.min_delay_seconds - elapsed)
        self._last_request_ts = time.time()

    def _validate_proxy_config(self) -> None:
        if self.proxy_config_error:
            raise RuntimeError(self.proxy_config_error)

    def _get_with_proxy_fallback(self, url: str, **request_options: Any) -> Any:
        try:
            return self.session.get(url, **request_options)
        except Exception as exc:
            if self._uses_curl_cffi and _looks_like_curl_tls_error(exc):
                return self._get_with_requests_fallback(url, **request_options)
            if not (
                self.proxy_url
                and self.direct_proxy_fallback
                and _looks_like_proxy_tunnel_error(exc)
            ):
                raise
            direct_options = dict(request_options)
            direct_options.pop("proxy", None)
            direct_options.pop("proxies", None)
            return self.session.get(url, **direct_options)

    def _get_with_requests_fallback(self, url: str, **request_options: Any) -> Any:
        plain_options = dict(request_options)
        proxy = plain_options.pop("proxy", None)
        if proxy and "proxies" not in plain_options:
            plain_options["proxies"] = {"http": proxy, "https": proxy}
        fallback_session = requests.Session()
        fallback_session.headers.update(dict(getattr(self.session, "headers", {}) or {}))
        try:
            return fallback_session.get(url, **plain_options)
        except Exception as exc:
            if not (
                self.direct_proxy_fallback
                and plain_options.get("proxies")
                and _looks_like_proxy_tunnel_error(exc)
            ):
                raise
            direct_options = dict(plain_options)
            direct_options.pop("proxies", None)
            return fallback_session.get(url, **direct_options)

    def _get_json(
        self, url: str, *, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        self._validate_proxy_config()
        self._rate_limit()
        request_options: Dict[str, Any] = {
            "params": params,
            "timeout": self.request_timeout_seconds,
        }
        if self.proxy_url:
            if self._uses_curl_cffi:
                request_options["proxy"] = self.proxy_url
            else:
                request_options["proxies"] = {
                    "http": self.proxy_url,
                    "https": self.proxy_url,
                }
        response = self._get_with_proxy_fallback(url, **request_options)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Substack returned a non-object response for {url}")
        return payload

    def _warm_search_session(self, keyword: str) -> None:
        self._validate_proxy_config()
        self._rate_limit()
        request_options: Dict[str, Any] = {
            "params": {"searching": "all_posts"},
            "timeout": self.request_timeout_seconds,
            "allow_redirects": True,
        }
        if self.proxy_url:
            if self._uses_curl_cffi:
                request_options["proxy"] = self.proxy_url
            else:
                request_options["proxies"] = {
                    "http": self.proxy_url,
                    "https": self.proxy_url,
                }
        response = self._get_with_proxy_fallback(
            f"https://substack.com/search/{quote(keyword, safe='')}", **request_options
        )
        response.raise_for_status()

    def _get_text(self, url: str) -> str:
        self._validate_proxy_config()
        self._rate_limit()
        request_options: Dict[str, Any] = {
            "timeout": self.request_timeout_seconds,
            "allow_redirects": True,
        }
        if self.proxy_url:
            if self._uses_curl_cffi:
                request_options["proxy"] = self.proxy_url
            else:
                request_options["proxies"] = {
                    "http": self.proxy_url,
                    "https": self.proxy_url,
                }
        response = self._get_with_proxy_fallback(url, **request_options)
        response.raise_for_status()
        return str(response.text or "")

    def discover_feed_documents(
        self,
        *,
        feeds: Optional[Iterable[Any]] = None,
        max_items_per_feed: int = 20,
    ) -> List[Dict[str, Any]]:
        feed_configs = []
        seen_feeds = set()
        for raw in feeds or DEFAULT_FEEDS:
            cfg = _feed_config(raw)
            feed_url = cfg["feed_url"]
            if not feed_url or feed_url.lower() in seen_feeds:
                continue
            seen_feeds.add(feed_url.lower())
            feed_configs.append(cfg)

        limit = max(1, int(max_items_per_feed or 20))
        discovered: Dict[str, Dict[str, Any]] = {}
        debug: Dict[str, Any] = {
            "mode": "curated_substack_feeds",
            "feeds": feed_configs,
            "max_items_per_feed": limit,
            "proxy_configured": bool(self.proxy_url),
            "proxy_config_error": self.proxy_config_error,
            "requests": [],
            "errors": [],
        }

        for cfg in feed_configs:
            feed_url = cfg["feed_url"]
            try:
                raw_xml = self._get_text(feed_url)
                root = ET.fromstring(raw_xml)
            except Exception as exc:
                debug["errors"].append(f"{feed_url}: {exc}")
                continue

            channel_title = cfg["label"]
            channel = root.find("channel")
            if channel is not None:
                title_node = channel.find("title")
                if title_node is not None and _normalize_space(title_node.text):
                    channel_title = channel_title or _normalize_space(title_node.text)

            items = root.findall(".//item")
            if not items:
                items = root.findall(".//{http://www.w3.org/2005/Atom}entry")
            debug["requests"].append({"feed_url": feed_url, "result_count": len(items)})

            for position, item in enumerate(items[:limit], 1):
                entry: Dict[str, Any] = {}
                categories: List[str] = []
                authors: List[str] = []
                for child in item:
                    local = _xml_local_name(child.tag)
                    text = _normalize_space(child.text or "")
                    if local == "title" and not entry.get("title"):
                        entry["title"] = text
                    elif local == "link" and not entry.get("url"):
                        entry["url"] = text or _normalize_space(child.get("href", ""))
                    elif local in {"guid", "id"} and not entry.get("guid"):
                        entry["guid"] = text
                    elif local in {"pubDate", "published", "updated"} and not entry.get(
                        "date"
                    ):
                        entry["date"] = text
                    elif local in {
                        "description",
                        "summary",
                        "encoded",
                    } and not entry.get("summary"):
                        entry["summary"] = _strip_html(child.text or "")
                    elif local in {"creator", "name"} and text:
                        authors.append(text)
                    elif local == "author" and text:
                        authors.append(text)
                    elif local == "category" and text:
                        categories.append(text)

                    if local == "link" and not entry.get("url"):
                        href = _normalize_space(child.get("href", ""))
                        if href:
                            entry["url"] = href

                url = _normalize_space(entry.get("url", ""))
                title = _normalize_space(entry.get("title", ""))
                slug = _substack_slug_from_url(url)
                if not url or not title or not slug:
                    continue

                key = url.lower().rstrip("/")
                discovered[key] = {
                    "url": url,
                    "title": title,
                    "date": _normalize_space(entry.get("date", "")),
                    "slug": slug,
                    "substack_post_id": _normalize_space(entry.get("guid", "")),
                    "publication_id": "",
                    "publication_name": channel_title,
                    "authors": authors,
                    "summary": _normalize_space(entry.get("summary", "")),
                    "preview_text": _normalize_space(entry.get("summary", "")),
                    "post_tags": categories,
                    "post_type": "newsletter",
                    "audience": "",
                    "free_unlock_required": False,
                    "wordcount": 0,
                    "reaction_count": 0,
                    "comment_count": 0,
                    "matched_keywords": [],
                    "search_position": position,
                    "feed_url": feed_url,
                    "feed_tags": [
                        tag.strip()
                        for tag in str(cfg.get("tags_csv", "") or "").split(",")
                        if tag.strip()
                    ],
                    "discovery_mode": "feed",
                    "discovery_modes": ["feed"],
                }

        results = list(discovered.values())
        results.sort(key=_date_sort_key, reverse=True)
        debug["items_found"] = len(results)
        self.last_discovery_debug = debug
        return results

    def discover_documents(
        self,
        *,
        keywords: Optional[Iterable[str]] = None,
        max_pages: int = 1,
        feeds: Optional[Iterable[Any]] = None,
        include_feeds: bool = False,
        max_items_per_feed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        search_terms = []
        seen_terms = set()
        for raw in keywords or DEFAULT_KEYWORDS:
            term = _normalize_space(raw)
            if not term or term.lower() in seen_terms:
                continue
            seen_terms.add(term.lower())
            search_terms.append(term)

        pages_per_keyword = max(1, int(max_pages or 1))
        discovered: Dict[str, Dict[str, Any]] = {}
        debug: Dict[str, Any] = {
            "mode": "public_substack_api",
            "search_url": SUBSTACK_SEARCH_URL,
            "keywords": search_terms,
            "pages_per_keyword": pages_per_keyword,
            "proxy_configured": bool(self.proxy_url),
            "proxy_config_error": self.proxy_config_error,
            "requests": [],
            "errors": [],
        }

        for keyword in search_terms:
            feed_session_id = ""
            try:
                self._warm_search_session(keyword)
            except Exception as exc:
                debug["errors"].append(f"{keyword} session warmup: {exc}")
            for page in range(pages_per_keyword):
                params: Dict[str, Any] = {
                    "query": keyword,
                    "page": page,
                    "numberFocused": 3,
                    "includePlatformResults": "true",
                    "filter": "all",
                    "dateRange": "all",
                }
                if feed_session_id:
                    params["feedSessionId"] = feed_session_id
                try:
                    payload = self._get_json(SUBSTACK_SEARCH_URL, params=params)
                except Exception as exc:
                    debug["errors"].append(f"{keyword} page {page}: {exc}")
                    break

                feed_session_id = (
                    _normalize_space(payload.get("feedSessionId", ""))
                    or feed_session_id
                )
                posts = (
                    payload.get("results")
                    if isinstance(payload.get("results"), list)
                    else []
                )
                publications = (
                    payload.get("publications")
                    if isinstance(payload.get("publications"), list)
                    else []
                )
                publication_names = {
                    str(pub.get("id")): _normalize_space(pub.get("name", ""))
                    for pub in publications
                    if isinstance(pub, dict) and pub.get("id") is not None
                }
                debug["requests"].append(
                    {
                        "keyword": keyword,
                        "page": page,
                        "result_count": len(posts),
                        "more": bool(payload.get("more")),
                    }
                )

                for position, post in enumerate(posts, 1):
                    if not isinstance(post, dict):
                        continue
                    url = _normalize_space(post.get("canonical_url", ""))
                    slug = _normalize_space(post.get("slug", ""))
                    post_id = _normalize_space(post.get("id", ""))
                    key = post_id or url.lower()
                    if not key or not url or not slug:
                        continue
                    existing = discovered.get(key)
                    if existing is not None:
                        matched = existing.setdefault("matched_keywords", [])
                        if keyword not in matched:
                            matched.append(keyword)
                        continue
                    discovered[key] = {
                        "url": url,
                        "title": _normalize_space(post.get("title", ""))
                        or "Substack post",
                        "date": _normalize_space(post.get("post_date", "")),
                        "slug": slug,
                        "substack_post_id": post.get("id"),
                        "publication_id": post.get("publication_id"),
                        "publication_name": publication_names.get(
                            str(post.get("publication_id")), ""
                        ),
                        "authors": _post_authors(post),
                        "summary": _normalize_space(
                            post.get("subtitle") or post.get("description") or ""
                        ),
                        "preview_text": _normalize_space(
                            post.get("truncated_body_text", "")
                        ),
                        "post_tags": _post_tags(post),
                        "post_type": _normalize_space(post.get("type", "newsletter")),
                        "audience": _normalize_space(post.get("audience", "")),
                        "free_unlock_required": bool(
                            post.get("free_unlock_required", False)
                        ),
                        "wordcount": int(post.get("wordcount", 0) or 0),
                        "reaction_count": int(post.get("reaction_count", 0) or 0),
                        "comment_count": int(post.get("comment_count", 0) or 0),
                        "matched_keywords": [keyword],
                        "search_position": position,
                        "discovery_mode": "search",
                        "discovery_modes": ["search"],
                    }
                if not payload.get("more"):
                    break

        if include_feeds:
            feed_results = self.discover_feed_documents(
                feeds=feeds,
                max_items_per_feed=max_items_per_feed
                or max(10, pages_per_keyword * 10),
            )
            feed_debug = dict(self.last_discovery_debug)
            for feed_entry in feed_results:
                url_key = (
                    _normalize_space(feed_entry.get("url", "")).lower().rstrip("/")
                )
                post_key = _normalize_space(feed_entry.get("substack_post_id", ""))
                existing = discovered.get(post_key) if post_key else None
                if existing is None:
                    existing = next(
                        (
                            item
                            for item in discovered.values()
                            if _normalize_space(item.get("url", "")).lower().rstrip("/")
                            == url_key
                        ),
                        None,
                    )
                if existing is None:
                    discovered[post_key or url_key] = feed_entry
                    continue

                existing["feed_url"] = _normalize_space(feed_entry.get("feed_url", ""))
                existing["feed_tags"] = _string_list(feed_entry.get("feed_tags"))
                existing["publication_name"] = _normalize_space(
                    existing.get("publication_name", "")
                ) or _normalize_space(feed_entry.get("publication_name", ""))
                modes = existing.setdefault("discovery_modes", [])
                if "feed" not in modes:
                    modes.append("feed")
                existing["discovery_mode"] = "search+feed"

            debug["feed_discovery"] = feed_debug
            debug["include_feeds"] = True

        results = list(discovered.values())
        results.sort(key=_date_sort_key, reverse=True)
        debug["items_found"] = len(results)
        self.last_discovery_debug = debug
        return results

    def extract_document(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        url = _normalize_space(entry.get("url", ""))
        slug = _normalize_space(entry.get("slug", "")) or _substack_slug_from_url(url)
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or not slug:
            raise ValueError("A valid Substack canonical URL and slug are required")
        detail_url = (
            f"{parsed.scheme}://{parsed.netloc}/api/v1/posts/{quote(slug, safe='')}"
        )
        post = self._get_json(detail_url)

        audience = _normalize_space(post.get("audience") or entry.get("audience", ""))
        free_unlock_required = bool(
            post.get("free_unlock_required", entry.get("free_unlock_required", False))
        )
        is_public = audience == "everyone" and not free_unlock_required
        body_html = str(post.get("body_html", "") or "") if is_public else ""
        full_text = _html_to_text(body_html) if body_html else ""
        preview = _normalize_space(
            post.get("truncated_body_text") or entry.get("preview_text", "")
        )
        if not full_text:
            full_text = preview

        return {
            "success": True,
            "data": {
                "url": _normalize_space(post.get("canonical_url") or url),
                "title": _normalize_space(post.get("title") or entry.get("title", "")),
                "date": _normalize_space(
                    post.get("post_date") or entry.get("date", "")
                ),
                "authors": _post_authors(post) or _string_list(entry.get("authors")),
                "publication_name": _normalize_space(entry.get("publication_name", "")),
                "summary": _normalize_space(
                    post.get("subtitle")
                    or post.get("description")
                    or entry.get("summary", "")
                ),
                "full_text": full_text,
                "preview_text": preview,
                "post_tags": _post_tags(post) or _string_list(entry.get("post_tags")),
                "post_type": _normalize_space(
                    post.get("type") or entry.get("post_type", "newsletter")
                ),
                "audience": audience,
                "free_unlock_required": free_unlock_required,
                "access_limited": not is_public,
                "wordcount": int(post.get("wordcount", entry.get("wordcount", 0)) or 0),
                "reaction_count": int(
                    post.get("reaction_count", entry.get("reaction_count", 0)) or 0
                ),
                "comment_count": int(
                    post.get("comment_count", entry.get("comment_count", 0)) or 0
                ),
                "detail_url": detail_url,
            },
        }

    def filter_institutional_finance(
        self,
        entries: List[Dict[str, Any]],
        *,
        client: Any,
        model: str = "deepseek-v4-flash",
        provider: str = "deepseek",
        exclusion_threshold: float = 0.8,
        batch_size: int = 20,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if client is None:
            raise RuntimeError("A hosted model client is required for Substack relevance filtering")
        provider = _normalize_space(provider).lower() or "deepseek"
        decisions: Dict[str, Dict[str, Any]] = {}
        size = max(1, int(batch_size or 20))

        for start in range(0, len(entries), size):
            batch = entries[start : start + size]
            candidates = [
                {
                    "post_id": str(item.get("substack_post_id") or item.get("url", "")),
                    "matched_keywords": item.get("matched_keywords", []),
                    "title": item.get("title", ""),
                    "subtitle": item.get("summary", ""),
                    "preview": item.get("preview_text", ""),
                    "tags": item.get("post_tags", []),
                    "publication": item.get("publication_name", ""),
                }
                for item in batch
            ]
            instruction = (
                "Classify Substack search results for an institutional financial-policy news feed. "
                'Return raw JSON only as {"decisions":[{"post_id":string,"classification":string,'
                '"confidence":number,"reason":string}]}. classification must be one of '
                "institutional_finance, personal_finance, ambiguous. Institutional finance includes securities "
                "regulation, capital markets, banking, payments, asset management, financial institutions, "
                "financial technology, decentralized finance, market structure, and financial policy. Personal "
                "finance includes household budgeting, debt payoff, credit repair, mortgages for individuals, "
                "retirement advice, personal investing tips, financial-freedom coaching, and consumer product "
                "promotion. Use personal_finance only when the content is primarily advice or promotion for an "
                "individual consumer. Use ambiguous when evidence is insufficient. Include every post_id exactly once."
            )
            if provider == "deepseek":
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": json.dumps({"candidates": candidates}, ensure_ascii=True)},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                parsed = _json_object(_chat_completion_text(response))
            else:
                response = client.responses.create(
                    model=model,
                    instructions=instruction,
                    input=json.dumps({"candidates": candidates}, ensure_ascii=True),
                )
                parsed = _json_object(_response_text(response))
            raw_decisions = (
                parsed.get("decisions")
                if isinstance(parsed.get("decisions"), list)
                else []
            )
            for decision in raw_decisions:
                if not isinstance(decision, dict):
                    continue
                post_id = _normalize_space(decision.get("post_id", ""))
                classification = _normalize_space(
                    decision.get("classification", "")
                ).lower()
                if classification not in {
                    "institutional_finance",
                    "personal_finance",
                    "ambiguous",
                }:
                    classification = "ambiguous"
                try:
                    confidence = max(
                        0.0, min(1.0, float(decision.get("confidence", 0.0) or 0.0))
                    )
                except (TypeError, ValueError):
                    confidence = 0.0
                if post_id:
                    decisions[post_id] = {
                        "classification": classification,
                        "confidence": confidence,
                        "reason": _normalize_space(decision.get("reason", ""))[:500],
                    }

        included: List[Dict[str, Any]] = []
        excluded: List[Dict[str, Any]] = []
        for entry in entries:
            post_id = str(entry.get("substack_post_id") or entry.get("url", ""))
            decision = decisions.get(
                post_id,
                {
                    "classification": "ambiguous",
                    "confidence": 0.0,
                    "reason": "No model decision returned.",
                },
            )
            enriched = dict(entry)
            enriched["relevance_classification"] = decision["classification"]
            enriched["relevance_confidence"] = decision["confidence"]
            enriched["relevance_reason"] = decision["reason"]
            if (
                decision["classification"] == "personal_finance"
                and decision["confidence"] >= exclusion_threshold
            ):
                excluded.append(enriched)
            else:
                included.append(enriched)
        return included, excluded


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
