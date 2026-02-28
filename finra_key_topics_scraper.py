#!/usr/bin/env python3
"""
FINRA Key Topics scraper.

Discovers FINRA Key Topic hub pages and extracts sectioned topic content plus
linked resource relationships for use as a taxonomy layer in the corpus.
"""

import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


FINRA_KEY_TOPICS_URL = "https://www.finra.org/rules-guidance/key-topics"


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clean_multiline(text: str) -> str:
    lines = []
    for raw in str(text or "").splitlines():
        line = _normalize_space(raw)
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _url_key(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}"


def _topic_slug_from_url(url: str) -> str:
    path = str(urlparse(str(url or "")).path or "").strip("/")
    if not path:
        return ""
    return path.split("/")[-1].strip().lower()


def _dedupe_link_entries(entries: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    seen = set()
    for item in entries:
        if not isinstance(item, dict):
            continue
        url = _url_key(item.get("url", ""))
        title = _normalize_space(item.get("title", ""))
        if not url:
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append(
            {
                "title": title,
                "url": url,
            }
        )
    return out


def _looks_like_topic_link(url: str) -> bool:
    target = _url_key(url)
    if not target.startswith("https://www.finra.org/"):
        return False
    path = urlparse(target).path.rstrip("/")
    if not path or path == "/rules-guidance/key-topics":
        return False
    if "/sites/default/files/" in path:
        return False
    return True


class FINRAKeyTopicsScraper:
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

    def discover_documents(self, index_url: str = FINRA_KEY_TOPICS_URL) -> List[Dict[str, str]]:
        response = self._fetch(index_url, timeout=45)
        final_url = str(getattr(response, "url", index_url) or index_url)
        soup = BeautifulSoup(response.text, "html.parser")
        article = soup.find("article") or soup.find("main") or soup

        out = []
        seen = set()
        for anchor in article.select("a[href]"):
            href = str(anchor.get("href", "") or "").strip()
            if not href or href.startswith("#") or href.lower().startswith(("javascript:", "mailto:")):
                continue
            absolute_url = urljoin(final_url, href)
            if not _looks_like_topic_link(absolute_url):
                continue

            title = _normalize_space(anchor.get_text(" ", strip=True))
            if len(title) < 3:
                continue

            key = _url_key(absolute_url)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "url": key,
                    "title": title,
                    "topic_name": title,
                    "topic_slug": _topic_slug_from_url(key),
                    "source_format": "html",
                    "discovery_source": "index",
                }
            )

        out.sort(key=lambda item: str(item.get("topic_name", "") or "").lower())
        return out

    @staticmethod
    def _extract_tab_target(tab) -> str:
        if tab is None:
            return ""
        for attr in ("data-bs-target", "data-target", "href"):
            value = str(tab.get(attr, "") or "").strip()
            if value.startswith("#") and len(value) > 1:
                return value[1:]
        tab_id = str(tab.get("id", "") or "").strip()
        if tab_id.endswith("-tab"):
            return tab_id[:-4]
        return ""

    @staticmethod
    def _extract_links(block, base_url: str) -> List[Dict[str, str]]:
        entries = []
        if block is None:
            return entries
        for anchor in block.select("a[href]"):
            href = str(anchor.get("href", "") or "").strip()
            if not href or href.startswith("#") or href.lower().startswith(("javascript:", "mailto:")):
                continue
            absolute_url = urljoin(base_url, href)
            title = _normalize_space(anchor.get_text(" ", strip=True))
            if not title:
                title = absolute_url.rsplit("/", 1)[-1]
            entries.append({"title": title, "url": absolute_url})
        return _dedupe_link_entries(entries)

    @staticmethod
    def _classify_link(section_name: str, entry: Dict[str, str]) -> Optional[str]:
        section_blob = _normalize_space(section_name).lower()
        title = _normalize_space(entry.get("title", "")).lower()
        url = _url_key(entry.get("url", ""))
        blob = f"{section_blob} {title} {url}"

        if "/rules-guidance/notices/" in url or "regulatory notice" in blob:
            return "linked_notices"
        if "/rules-guidance/rulebooks/" in url or section_blob == "rules":
            return "linked_rules"
        if "/investors/" in url or "investor education" in section_blob or "investor insights" in blob:
            return "linked_investor_education"
        if "/media-center/newsreleases/" in url or "news release" in blob:
            return "linked_news"
        if "/rules-guidance/guidance/" in url or section_blob == "guidance":
            return "linked_guidance"
        return "linked_resources"

    def _extract_sections(self, article, final_url: str) -> Dict[str, Any]:
        tab_block = article.select_one(".block-content-quicker_tabs") or article
        sections = []

        tabs = tab_block.select(".nav-tabs button, .nav-tabs a")
        for tab in tabs:
            label = _normalize_space(tab.get_text(" ", strip=True))
            pane_id = self._extract_tab_target(tab)
            if not label or not pane_id:
                continue
            pane = tab_block.select_one(f".tab-pane#{pane_id}")
            if pane is None:
                continue
            sections.append(
                {
                    "name": label,
                    "text": _clean_multiline(pane.get_text("\n")),
                    "links": self._extract_links(pane, final_url),
                }
            )

        if not sections:
            overview_text = _clean_multiline(article.get_text("\n"))
            sections.append(
                {
                    "name": "Overview",
                    "text": overview_text,
                    "links": self._extract_links(article, final_url),
                }
            )

        linked_buckets = {
            "linked_notices": [],
            "linked_guidance": [],
            "linked_rules": [],
            "linked_news": [],
            "linked_investor_education": [],
            "linked_resources": [],
        }
        for section in sections:
            for entry in section.get("links", []):
                bucket_name = self._classify_link(section.get("name", ""), entry)
                if bucket_name:
                    linked_buckets[bucket_name].append(entry)

        for key, entries in list(linked_buckets.items()):
            linked_buckets[key] = _dedupe_link_entries(entries)

        return {
            "sections": sections,
            "section_names": [s.get("name", "") for s in sections if s.get("name", "")],
            **linked_buckets,
        }

    @staticmethod
    def _extract_ogc_contacts(article) -> List[str]:
        heading = None
        for candidate in article.find_all(["h2", "h3"]):
            text = _normalize_space(candidate.get_text(" ", strip=True)).lower()
            if text in {"contact ogc", "ogc staff contacts:"} or text.startswith("contact ogc"):
                heading = candidate
                break
        if heading is None:
            return []

        chunks = []
        node = heading
        while True:
            node = node.find_next_sibling()
            if node is None or getattr(node, "name", None) == "h2":
                break
            text = _normalize_space(node.get_text(" ", strip=True))
            if text:
                chunks.append(text)

        if not chunks:
            return []

        contacts = []
        seen = set()
        for chunk in chunks:
            if chunk.lower().startswith("finra's office of general counsel"):
                continue
            for piece in re.split(r"\s{2,}|\s*\|\s*", chunk):
                label = _normalize_space(piece)
                if not label:
                    continue
                if label.lower().startswith("ogc staff contacts"):
                    label = _normalize_space(re.sub(r"^ogc staff contacts:\s*", "", label, flags=re.IGNORECASE))
                    if not label:
                        continue
                key = label.lower()
                if key in seen:
                    continue
                seen.add(key)
                contacts.append(label)
        return contacts[:12]

    def extract_document(self, url: str, fallback_title: str = "") -> Dict[str, Any]:
        response = self._fetch(url, timeout=60)
        final_url = str(getattr(response, "url", url) or url)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup.find_all(["script", "style", "noscript"]):
            tag.decompose()

        main_content = soup.find("main") or soup
        article = main_content.find("article") or main_content
        heading = article.find("h1") or main_content.find("h1") or soup.find("h1")
        topic_name = _normalize_space(heading.get_text(" ", strip=True) if heading else "")
        if not topic_name:
            topic_name = str(fallback_title or "").strip() or "FINRA Key Topic"
        topic_slug = _topic_slug_from_url(final_url)

        sections_payload = self._extract_sections(main_content, final_url)
        section_names = sections_payload.get("section_names", [])
        ogc_contacts = self._extract_ogc_contacts(main_content)

        section_links = {}
        for section in sections_payload.get("sections", []):
            name = str(section.get("name", "") or "").strip()
            if not name:
                continue
            section_links[name] = section.get("links", [])

        overview_text = ""
        for section in sections_payload.get("sections", []):
            if str(section.get("name", "")).strip().lower() == "overview":
                overview_text = str(section.get("text", "") or "").strip()
                break
        if not overview_text and sections_payload.get("sections"):
            overview_text = str(sections_payload["sections"][0].get("text", "") or "").strip()

        header_lines = [
            f"Topic Name: {topic_name}",
            f"Topic Slug: {topic_slug}" if topic_slug else "",
            f"Source URL: {final_url}",
            "Section Names: " + ", ".join(section_names) if section_names else "",
            "OGC Contacts: " + "; ".join(ogc_contacts) if ogc_contacts else "",
        ]
        header_lines = [line for line in header_lines if line]

        body_parts = []
        for section in sections_payload.get("sections", []):
            section_name = str(section.get("name", "") or "").strip()
            section_text = str(section.get("text", "") or "").strip()
            section_lines = [f"[{section_name}]"] if section_name else []
            if section_text:
                section_lines.append(section_text)
            links = section.get("links", [])
            if links:
                section_lines.append("Linked Resources:")
                for entry in links[:40]:
                    title = _normalize_space(entry.get("title", ""))
                    link_url = _url_key(entry.get("url", ""))
                    section_lines.append(f"- {title}: {link_url}")
            if section_lines:
                body_parts.append("\n".join(section_lines).strip())

        full_text = "\n".join(header_lines).strip()
        if body_parts:
            full_text = f"{full_text}\n\n" + "\n\n".join(body_parts).strip()

        return {
            "success": True,
            "data": {
                "url": _url_key(final_url),
                "title": topic_name,
                "topic_name": topic_name,
                "topic_slug": topic_slug,
                "section_names": section_names,
                "section_links": section_links,
                "overview_text": overview_text,
                "ogc_contacts": ogc_contacts,
                "linked_notices": [item.get("url", "") for item in sections_payload.get("linked_notices", [])],
                "linked_guidance": [item.get("url", "") for item in sections_payload.get("linked_guidance", [])],
                "linked_rules": [item.get("url", "") for item in sections_payload.get("linked_rules", [])],
                "linked_news": [item.get("url", "") for item in sections_payload.get("linked_news", [])],
                "linked_investor_education": [
                    item.get("url", "") for item in sections_payload.get("linked_investor_education", [])
                ],
                "linked_resources": [item.get("url", "") for item in sections_payload.get("linked_resources", [])],
                "linked_urls": [
                    item.get("url", "")
                    for key in [
                        "linked_notices",
                        "linked_guidance",
                        "linked_rules",
                        "linked_news",
                        "linked_investor_education",
                        "linked_resources",
                    ]
                    for item in sections_payload.get(key, [])
                ],
                "full_text": full_text.strip(),
                "word_count": len(full_text.split()),
                "source_format": "html",
            },
        }
