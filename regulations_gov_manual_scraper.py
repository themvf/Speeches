#!/usr/bin/env python3
"""
Manual Regulations.gov scraper for curated rule and comment URLs.

Supports public docket/document/comment pages and direct downloads.regulations.gov
file URLs. The scraper prefers API-backed metadata when an ID can be resolved, then
falls back to public download URLs and HTML text extraction.
"""

from __future__ import annotations

import io
import re
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


REGULATIONS_API_BASE = "https://api.regulations.gov/v4"
REGULATIONS_PUBLIC_BASE = "https://www.regulations.gov"
REGULATIONS_DOWNLOAD_BASE = "https://downloads.regulations.gov"


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
    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%B %d, %Y",
        "%b %d, %Y",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        normalized = text.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).replace(tzinfo=None)
    except Exception:
        return None


def _date_to_display(value: Any) -> str:
    parsed = _parse_date_text(value)
    if parsed is None:
        return str(value or "").strip()
    return parsed.strftime("%B %d, %Y")


def _unique_strings(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _is_url(value: Any) -> bool:
    text = str(value or "").strip()
    return text.startswith("http://") or text.startswith("https://")


def _is_pdf_url(url: Any) -> bool:
    return str(url or "").strip().lower().endswith(".pdf")


def _public_docket_url(docket_id: Any) -> str:
    docket = str(docket_id or "").strip()
    return f"{REGULATIONS_PUBLIC_BASE}/docket/{docket}" if docket else ""


def _public_document_url(document_id: Any) -> str:
    doc_id = str(document_id or "").strip()
    return f"{REGULATIONS_PUBLIC_BASE}/document/{doc_id}" if doc_id else ""


def _public_comment_url(comment_id: Any) -> str:
    item_id = str(comment_id or "").strip()
    return f"{REGULATIONS_PUBLIC_BASE}/comment/{item_id}" if item_id else ""


def _download_content_url(record_id: Any, ext: str) -> str:
    item_id = str(record_id or "").strip()
    suffix = str(ext or "").strip().lstrip(".").lower()
    if not item_id or not suffix:
        return ""
    return f"{REGULATIONS_DOWNLOAD_BASE}/{item_id}/content.{suffix}"


def _download_attachment_url(comment_id: Any, index: int, ext: str) -> str:
    item_id = str(comment_id or "").strip()
    suffix = str(ext or "").strip().lstrip(".").lower()
    if not item_id or not suffix or index <= 0:
        return ""
    return f"{REGULATIONS_DOWNLOAD_BASE}/{item_id}/attachment_{index}.{suffix}"


def _dict_get_ci(mapping: Dict[str, Any], *keys: str) -> Any:
    if not isinstance(mapping, dict):
        return ""
    lowered = {str(k).lower(): v for k, v in mapping.items()}
    for key in keys:
        value = lowered.get(str(key).lower())
        if value not in (None, "", [], {}):
            return value
    return ""


def _parse_regulations_url(url: Any) -> Dict[str, str]:
    target = str(url or "").strip()
    parsed = urlparse(target)
    parts = [part for part in parsed.path.split("/") if part]
    info = {
        "url": target,
        "kind": "unknown",
        "docket_id": "",
        "document_id": "",
        "comment_id": "",
        "record_id": "",
        "download_kind": "",
        "download_filename": "",
    }
    host = (parsed.netloc or "").lower()
    if "downloads.regulations.gov" in host:
        info["kind"] = "download"
        if parts:
            info["record_id"] = parts[0]
        if len(parts) > 1:
            info["download_filename"] = parts[1]
            filename = parts[1].lower()
            if filename.startswith("attachment_"):
                info["download_kind"] = "attachment"
            elif filename.startswith("content."):
                info["download_kind"] = "content"
        return info

    if "regulations.gov" not in host or len(parts) < 2:
        return info

    prefix = parts[0].lower()
    identifier = parts[1]
    if prefix == "docket":
        info["kind"] = "docket"
        info["docket_id"] = identifier
    elif prefix == "document":
        info["kind"] = "document"
        info["document_id"] = identifier
        info["record_id"] = identifier
    elif prefix == "comment":
        info["kind"] = "comment"
        info["comment_id"] = identifier
        info["record_id"] = identifier
    return info


def _extract_urls_from_value(value: Any) -> List[str]:
    found: List[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, str) and _is_url(child):
                found.append(child.strip())
            elif str(key).lower() == "fileformats" and isinstance(child, list):
                for item in child:
                    if _is_url(item):
                        found.append(str(item).strip())
            else:
                found.extend(_extract_urls_from_value(child))
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            found.extend(_extract_urls_from_value(child))
    elif isinstance(value, str) and _is_url(value):
        found.append(value.strip())
    return _unique_strings(found)


def _extract_text_from_html(html_text: str) -> str:
    soup = BeautifulSoup(str(html_text or ""), "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    root = (
        soup.select_one("main")
        or soup.select_one("article")
        or soup.select_one("#main-content")
        or soup.body
        or soup
    )
    text = root.get_text("\n")
    return _clean_multiline(text)


def _extract_pdf_text(blob: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError(f"PDF extraction requires pypdf: {e}") from e

    reader = PdfReader(io.BytesIO(blob))
    pages: List[str] = []
    for page in reader.pages:
        page_text = _clean_multiline(page.extract_text() or "")
        if page_text:
            pages.append(page_text)
    return "\n\n".join(pages).strip()


def _candidate_attachment_urls(comment_id: str) -> List[str]:
    candidates: List[str] = []
    for index in range(1, 5):
        for ext in ("pdf", "txt", "htm", "html"):
            candidates.append(_download_attachment_url(comment_id, index, ext))
    return _unique_strings(candidates)


def _summarize_text(text: Any, max_chars: int = 1200) -> str:
    cleaned = _clean_multiline(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _download_priority(url: Any) -> tuple:
    target = str(url or "").strip().lower()
    if not target:
        return (99, "")
    if "downloads.regulations.gov" in target and "/attachment_" in target and target.endswith(".pdf"):
        return (0, target)
    if "downloads.regulations.gov" in target and target.endswith(".pdf"):
        return (1, target)
    if "downloads.regulations.gov" in target and "/attachment_" in target:
        return (2, target)
    if "downloads.regulations.gov" in target:
        return (3, target)
    if target.endswith(".pdf"):
        return (4, target)
    if target.endswith(".txt"):
        return (5, target)
    if target.endswith((".htm", ".html")):
        return (6, target)
    if "regulations.gov" in target:
        return (8, target)
    return (7, target)


def _select_primary_document(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _score(item: Dict[str, Any]) -> tuple:
        attrs = item.get("attributes", {}) if isinstance(item, dict) else {}
        doc_type = _normalize_space(
            _dict_get_ci(attrs, "documentType", "type", "category")
        ).lower()
        title = _normalize_space(_dict_get_ci(attrs, "title", "objectTitle")).lower()
        posted_date = _parse_date_text(_dict_get_ci(attrs, "postedDate", "lastModifiedDate"))

        kind_score = 10
        if "final rule" in doc_type or doc_type == "rule":
            kind_score = 100
        elif "proposed rule" in doc_type:
            kind_score = 95
        elif "rule" in doc_type:
            kind_score = 90
        elif "notice" in doc_type:
            kind_score = 80
        elif "request for information" in title:
            kind_score = 75
        elif "supporting" in doc_type:
            kind_score = 20

        return (kind_score, posted_date or datetime.min, title)

    if not documents:
        return {}
    return max(documents, key=_score)


class RegulationsGovManualScraper:
    def __init__(self, api_key: str = "DEMO_KEY", min_delay_seconds: float = 0.25):
        self.api_key = str(api_key or "").strip() or "DEMO_KEY"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "X-Api-Key": self.api_key,
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

    def _fetch_json(self, path_or_url: str, timeout: int = 60) -> Dict[str, Any]:
        target = str(path_or_url or "").strip()
        if not target:
            return {}
        if not target.startswith("http://") and not target.startswith("https://"):
            target = f"{REGULATIONS_API_BASE.rstrip('/')}/{target.lstrip('/')}"
        response = self._fetch(target, timeout=timeout)
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def _try_fetch_json(self, path_or_url: str, warnings: List[str]) -> Dict[str, Any]:
        try:
            return self._fetch_json(path_or_url)
        except Exception as e:
            warnings.append(f"API metadata unavailable for {path_or_url}: {e}")
            return {}

    def _download_best_text(self, urls: Iterable[str], warnings: List[str]) -> Dict[str, Any]:
        candidates = sorted(_unique_strings(urls), key=_download_priority)
        for candidate in candidates:
            try:
                response = self._fetch(candidate, timeout=90)
                final_url = str(getattr(response, "url", candidate) or candidate).strip()
                content_type = str(response.headers.get("Content-Type", "") or "").lower()
                source_format = ""
                text = ""
                if _is_pdf_url(final_url) or "application/pdf" in content_type:
                    source_format = "pdf"
                    text = _extract_pdf_text(response.content)
                elif final_url.lower().endswith((".htm", ".html")) or "text/html" in content_type:
                    source_format = "html"
                    text = _extract_text_from_html(response.text)
                elif final_url.lower().endswith(".txt") or "text/plain" in content_type:
                    source_format = "txt"
                    text = _clean_multiline(response.text)
                else:
                    source_format = "txt"
                    text = _clean_multiline(response.text)

                if text:
                    return {
                        "text": text,
                        "resolved_content_url": final_url,
                        "source_format": source_format,
                    }
                warnings.append(f"No extractable text found at {final_url}")
            except Exception as e:
                warnings.append(f"Failed to fetch {candidate}: {e}")
        return {"text": "", "resolved_content_url": "", "source_format": ""}

    def _extract_public_page_text(self, url: str, warnings: List[str]) -> str:
        target = str(url or "").strip()
        if not target:
            return ""
        try:
            response = self._fetch(target, timeout=60)
            return _extract_text_from_html(response.text)
        except Exception as e:
            warnings.append(f"HTML fallback failed for {target}: {e}")
            return ""

    @staticmethod
    def _payload_data(payload: Dict[str, Any]) -> Dict[str, Any]:
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _payload_attrs(payload: Dict[str, Any]) -> Dict[str, Any]:
        attrs = RegulationsGovManualScraper._payload_data(payload).get("attributes", {})
        return attrs if isinstance(attrs, dict) else {}

    @staticmethod
    def _payload_included(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        included = payload.get("included", []) if isinstance(payload, dict) else []
        return [item for item in included if isinstance(item, dict)]

    @staticmethod
    def _relation_id(data: Dict[str, Any], name: str) -> str:
        rels = data.get("relationships", {}) if isinstance(data, dict) else {}
        if not isinstance(rels, dict):
            return ""
        rel = rels.get(name, {})
        if not isinstance(rel, dict):
            return ""
        rel_data = rel.get("data", {})
        if isinstance(rel_data, dict):
            return str(rel_data.get("id", "") or "").strip()
        if isinstance(rel_data, list) and rel_data:
            first = rel_data[0]
            if isinstance(first, dict):
                return str(first.get("id", "") or "").strip()
        return ""

    def _build_rule_text(
        self,
        title: str,
        agency: str,
        docket_id: str,
        document_id: str,
        docket_url: str,
        document_url: str,
        posted_date: str,
        document_type: str,
        summary: str,
        body_text: str,
        resolved_content_url: str,
        warnings: List[str],
    ) -> str:
        lines = []
        if docket_id:
            lines.append(f"Docket ID: {docket_id}")
        if document_id:
            lines.append(f"Document ID: {document_id}")
        if title:
            lines.append(f"Title: {title}")
        if agency:
            lines.append(f"Agency: {agency}")
        if document_type:
            lines.append(f"Document Type: {document_type}")
        if posted_date:
            lines.append(f"Posted Date: {posted_date}")
        if docket_url:
            lines.append(f"Docket URL: {docket_url}")
        if document_url:
            lines.append(f"Document URL: {document_url}")
        if resolved_content_url:
            lines.append(f"Resolved Content URL: {resolved_content_url}")

        sections = []
        if summary:
            sections.append("Summary\n" + summary)
        if body_text:
            sections.append("Document Text\n" + body_text)
        if warnings:
            sections.append("Extraction Notes\n" + "\n".join(f"- {item}" for item in warnings[:8]))
        return "\n".join(lines).strip() + ("\n\n" + "\n\n".join(sections) if sections else "")

    def _build_comment_text(
        self,
        title: str,
        agency: str,
        docket_id: str,
        comment_id: str,
        commenter_name: str,
        commenter_org: str,
        posted_date: str,
        rule_url: str,
        comment_page_url: str,
        resolved_content_url: str,
        summary: str,
        body_text: str,
        warnings: List[str],
    ) -> str:
        lines = []
        if docket_id:
            lines.append(f"Docket ID: {docket_id}")
        if comment_id:
            lines.append(f"Comment ID: {comment_id}")
        if title:
            lines.append(f"Title: {title}")
        if commenter_name:
            lines.append(f"Commenter: {commenter_name}")
        if commenter_org:
            lines.append(f"Commenter Organization: {commenter_org}")
        if agency:
            lines.append(f"Agency: {agency}")
        if posted_date:
            lines.append(f"Posted Date: {posted_date}")
        if rule_url:
            lines.append(f"Rule URL: {rule_url}")
        if comment_page_url:
            lines.append(f"Comment Page URL: {comment_page_url}")
        if resolved_content_url:
            lines.append(f"Resolved Content URL: {resolved_content_url}")

        sections = []
        if summary:
            sections.append("Comment Summary\n" + summary)
        if body_text:
            sections.append("Comment Text\n" + body_text)
        if warnings:
            sections.append("Extraction Notes\n" + "\n".join(f"- {item}" for item in warnings[:8]))
        return "\n".join(lines).strip() + ("\n\n" + "\n\n".join(sections) if sections else "")

    def extract_rule(self, url: str) -> Dict[str, Any]:
        input_url = str(url or "").strip()
        parsed = _parse_regulations_url(input_url)
        warnings: List[str] = []

        docket_id = parsed.get("docket_id", "")
        document_id = parsed.get("document_id", "") or parsed.get("record_id", "")
        docket_payload: Dict[str, Any] = {}
        document_payload: Dict[str, Any] = {}

        if parsed.get("kind") == "docket" and docket_id:
            docket_payload = self._try_fetch_json(f"dockets/{docket_id}", warnings)
            docs_payload = self._try_fetch_json(
                f"documents?filter[docketId]={docket_id}&sort=-postedDate&page[size]=250",
                warnings,
            )
            documents = docs_payload.get("data", []) if isinstance(docs_payload, dict) else []
            primary_document = _select_primary_document(documents if isinstance(documents, list) else [])
            document_id = str(primary_document.get("id", "") or "").strip() or document_id
        elif document_id:
            document_payload = self._try_fetch_json(f"documents/{document_id}?include=attachments", warnings)

        if document_id and not document_payload:
            document_payload = self._try_fetch_json(f"documents/{document_id}?include=attachments", warnings)

        docket_data = self._payload_data(docket_payload)
        docket_attrs = self._payload_attrs(docket_payload)
        document_data = self._payload_data(document_payload)
        document_attrs = self._payload_attrs(document_payload)

        if not docket_id:
            docket_id = str(
                _dict_get_ci(
                    document_attrs,
                    "docketId",
                    "docketIdValue",
                )
                or self._relation_id(document_data, "docket")
                or _dict_get_ci(docket_attrs, "docketId")
                or parsed.get("docket_id", "")
            ).strip()

        title = _normalize_space(
            _dict_get_ci(document_attrs, "title", "objectTitle", "displayTitle")
            or _dict_get_ci(docket_attrs, "title", "objectTitle", "displayTitle")
        )
        agency = _normalize_space(
            _dict_get_ci(
                document_attrs,
                "agencyName",
                "agency",
                "agencyAcronym",
            )
            or _dict_get_ci(docket_attrs, "agencyName", "agency", "agencyAcronym")
        )
        posted_date = _date_to_display(
            _dict_get_ci(document_attrs, "postedDate", "lastModifiedDate")
            or _dict_get_ci(docket_attrs, "lastModifiedDate", "openedDate")
        )
        document_type = _normalize_space(
            _dict_get_ci(document_attrs, "documentType", "type", "category")
        )
        summary = _summarize_text(
            _dict_get_ci(
                document_attrs,
                "summary",
                "abstract",
                "summaryText",
                "rin",
            )
            or _dict_get_ci(docket_attrs, "summary", "abstract")
        )

        docket_url = _public_docket_url(docket_id) if docket_id else ""
        document_url = _public_document_url(document_id) if document_id else ""

        candidate_urls = []
        if parsed.get("kind") == "download":
            candidate_urls.append(input_url)
        candidate_urls.extend(_extract_urls_from_value(document_payload))
        if document_id:
            candidate_urls.extend(
                [
                    _download_content_url(document_id, "pdf"),
                    _download_content_url(document_id, "htm"),
                    _download_content_url(document_id, "html"),
                    _download_content_url(document_id, "txt"),
                ]
            )

        extracted = self._download_best_text(candidate_urls, warnings)
        body_text = str(extracted.get("text", "") or "").strip()
        resolved_content_url = str(extracted.get("resolved_content_url", "") or "").strip()
        source_format = str(extracted.get("source_format", "") or "").strip()

        if not body_text:
            public_target = document_url or input_url
            body_text = self._extract_public_page_text(public_target, warnings)
            if body_text and not source_format:
                source_format = "html"

        full_text = self._build_rule_text(
            title=title or (docket_id or document_id or "Regulatory Docket"),
            agency=agency,
            docket_id=docket_id,
            document_id=document_id,
            docket_url=docket_url,
            document_url=document_url,
            posted_date=posted_date,
            document_type=document_type,
            summary=summary,
            body_text=body_text,
            resolved_content_url=resolved_content_url,
            warnings=warnings,
        )

        return {
            "success": bool(full_text.strip()),
            "data": {
                "input_url": input_url,
                "url": document_url or docket_url or input_url,
                "docket_url": docket_url or input_url,
                "document_url": document_url,
                "resolved_content_url": resolved_content_url,
                "pdf_url": resolved_content_url if source_format == "pdf" else "",
                "title": title or docket_id or document_id or "Regulatory Docket",
                "agency": agency or "Regulations.gov",
                "date": posted_date,
                "docket_id": docket_id,
                "document_id": document_id,
                "document_type": document_type,
                "summary": summary,
                "full_text": full_text,
                "source_format": source_format or ("pdf" if _is_pdf_url(input_url) else "html"),
                "extraction_mode": (
                    "document_download"
                    if resolved_content_url
                    else ("document_page_html" if body_text else "metadata_only")
                ),
                "warnings": warnings,
            },
        }

    def extract_comment(self, url: str, rule_url: str = "") -> Dict[str, Any]:
        input_url = str(url or "").strip()
        parsed = _parse_regulations_url(input_url)
        warnings: List[str] = []

        comment_id = parsed.get("comment_id", "") or parsed.get("record_id", "")
        comment_payload: Dict[str, Any] = {}
        if comment_id:
            comment_payload = self._try_fetch_json(f"comments/{comment_id}?include=attachments", warnings)

        comment_data = self._payload_data(comment_payload)
        comment_attrs = self._payload_attrs(comment_payload)

        docket_id = str(
            _dict_get_ci(comment_attrs, "docketId", "docketIdValue")
            or self._relation_id(comment_data, "docket")
            or parsed.get("docket_id", "")
        ).strip()
        comment_page_url = _public_comment_url(comment_id) if comment_id else input_url

        title = _normalize_space(
            _dict_get_ci(comment_attrs, "title", "objectTitle", "displayTitle")
        )
        posted_date = _date_to_display(
            _dict_get_ci(comment_attrs, "postedDate", "lastModifiedDate")
        )
        agency = _normalize_space(
            _dict_get_ci(comment_attrs, "agencyName", "agency", "agencyAcronym")
        )

        first_name = _normalize_space(_dict_get_ci(comment_attrs, "firstName"))
        last_name = _normalize_space(_dict_get_ci(comment_attrs, "lastName"))
        commenter_name = _normalize_space(
            " ".join(part for part in [first_name, last_name] if part)
            or _dict_get_ci(comment_attrs, "submitterName", "contactName", "authorName")
        )
        commenter_org = _normalize_space(
            _dict_get_ci(comment_attrs, "organization", "organizationName")
        )
        inline_comment = _clean_multiline(
            _dict_get_ci(comment_attrs, "comment", "commentText", "commentBody")
        )

        attachment_urls = _extract_urls_from_value(comment_payload)
        if comment_id:
            attachment_urls.extend(_candidate_attachment_urls(comment_id))
            attachment_urls.extend(
                [
                    _download_content_url(comment_id, "pdf"),
                    _download_content_url(comment_id, "htm"),
                    _download_content_url(comment_id, "html"),
                    _download_content_url(comment_id, "txt"),
                ]
            )
        if parsed.get("kind") == "download":
            attachment_urls.insert(0, input_url)
        attachment_urls = _unique_strings(attachment_urls)

        extracted = self._download_best_text(attachment_urls, warnings)
        attachment_text = str(extracted.get("text", "") or "").strip()
        resolved_content_url = str(extracted.get("resolved_content_url", "") or "").strip()
        source_format = str(extracted.get("source_format", "") or "").strip()

        body_text = inline_comment
        extraction_mode = "comment_inline"
        if attachment_text and (
            len(attachment_text.split()) >= max(80, len(inline_comment.split()) + 20)
            or not inline_comment
            or parsed.get("download_kind") == "attachment"
        ):
            body_text = attachment_text
            extraction_mode = "comment_attachment"
        elif not body_text:
            body_text = self._extract_public_page_text(comment_page_url, warnings)
            if body_text:
                extraction_mode = "comment_page_html"
                if not source_format:
                    source_format = "html"

        if not title:
            if commenter_name or commenter_org:
                title = f"Comment from {commenter_name or commenter_org}"
            else:
                title = comment_id or "Public Comment"

        summary = _summarize_text(inline_comment or attachment_text)
        full_text = self._build_comment_text(
            title=title,
            agency=agency,
            docket_id=docket_id,
            comment_id=comment_id,
            commenter_name=commenter_name,
            commenter_org=commenter_org,
            posted_date=posted_date,
            rule_url=rule_url,
            comment_page_url=comment_page_url,
            resolved_content_url=resolved_content_url,
            summary=summary,
            body_text=body_text,
            warnings=warnings,
        )

        return {
            "success": bool(full_text.strip()),
            "data": {
                "input_url": input_url,
                "url": comment_page_url or input_url,
                "rule_url": str(rule_url or "").strip(),
                "comment_page_url": comment_page_url,
                "resolved_content_url": resolved_content_url,
                "pdf_url": resolved_content_url if source_format == "pdf" else "",
                "title": title,
                "agency": agency or "Regulations.gov",
                "date": posted_date,
                "docket_id": docket_id,
                "comment_id": comment_id,
                "commenter_name": commenter_name,
                "commenter_org": commenter_org,
                "summary": summary,
                "full_text": full_text,
                "source_format": source_format or ("pdf" if _is_pdf_url(input_url) else "html"),
                "extraction_mode": extraction_mode,
                "attachment_urls": attachment_urls[:10],
                "warnings": warnings,
            },
        }
