#!/usr/bin/env python3
"""Shared helpers for building enrichment candidates from the merged corpus."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional


def normalize_org_label(value: Any) -> str:
    label = str(value).strip() if value is not None else ""
    return label or "SEC"


def org_key_from_label(label: Any) -> str:
    cleaned = "".join(ch.lower() if str(ch).isalnum() else "_" for ch in str(label))
    cleaned = cleaned.strip("_")
    return cleaned or "sec"


def speech_org_label(speech: Dict[str, Any]) -> str:
    metadata = speech.get("metadata", {}) if isinstance(speech, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    return normalize_org_label(metadata.get("organization") or metadata.get("org") or "SEC")


def speech_org_key(speech: Dict[str, Any]) -> str:
    return org_key_from_label(speech_org_label(speech))


def infer_source_kind(metadata: Dict[str, Any]) -> str:
    if not isinstance(metadata, dict):
        return "document"
    source_kind = str(metadata.get("source_kind", "") or "").strip().lower()
    if source_kind:
        return source_kind
    url = str(metadata.get("url", "") or "").lower()
    if "/rules-regulations/public-comments/" in url or "/comments/" in url:
        return "sec_rule_comment"
    if "/rules-regulations/" in url and "release" in str(metadata.get("doc_type", "") or "").strip().lower():
        return "sec_rule_release"
    if "/newsroom/speeches-statements/" in url:
        return "sec_speech"
    doc_type = str(metadata.get("doc_type", "") or "").strip().lower()
    if doc_type == "regulatory notice":
        return "finra_regulatory_notice"
    if doc_type == "key topic":
        return "finra_key_topic"
    if "/trading-markets-frequently-asked-questions/" in url or source_kind == "sec_tm_faq":
        return "sec_tm_faq"
    if "/enforcement-litigation/litigation-releases/" in url or source_kind == "sec_enforcement_litigation":
        return "sec_enforcement_litigation"
    if ("/usao-" in url or "/usao/" in url) and "/pr/" in url:
        return "doj_usao_press_release"
    if "/crs-product/" in url:
        return "congress_crs_product"
    if doc_type in {"speech", "statement", "remarks"}:
        return "sec_speech"
    return "document"


def coerce_int(value: Any, default: int = 0, min_value: Optional[int] = 0) -> int:
    try:
        if isinstance(value, bool):
            num = int(value)
        elif isinstance(value, (int, float)):
            num = int(value)
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                num = int(default)
            else:
                match = re.search(r"-?\d+", text.replace(",", ""))
                num = int(match.group(0)) if match else int(default)
        elif isinstance(value, (list, tuple, set)):
            num = len(value)
        else:
            num = int(default)
    except Exception:
        num = int(default)

    if min_value is not None:
        try:
            num = max(int(min_value), num)
        except Exception:
            pass
    return num


def corpus_doc_id(speech: Dict[str, Any]) -> str:
    metadata = speech.get("metadata", {}) if isinstance(speech, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    existing = str(metadata.get("document_id", "") or "").strip()
    if existing:
        return existing
    stable = "|".join(
        [
            speech_org_key(speech),
            str(metadata.get("url", "") or ""),
            str(metadata.get("title", "") or ""),
            str(metadata.get("speaker", "") or ""),
            str(metadata.get("date", "") or ""),
        ]
    )
    if not stable.strip("|"):
        content = speech.get("content", {}) if isinstance(speech, dict) else {}
        if not isinstance(content, dict):
            content = {}
        stable = str(content.get("full_text", "") or "")[:1000]
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]


def build_enrichment_candidates(
    knowledge_data: Dict[str, Any],
    org_key: Optional[str] = None,
    include_full_text: bool = True,
    allowed_doc_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    dedup: Dict[str, Dict[str, Any]] = {}
    speeches = knowledge_data.get("speeches", []) if isinstance(knowledge_data, dict) else []
    if not isinstance(speeches, list):
        speeches = []
    allowed_ids = {
        str(doc_id).strip()
        for doc_id in (allowed_doc_ids or [])
        if str(doc_id).strip()
    }

    for speech in speeches:
        if not isinstance(speech, dict):
            continue
        candidate_org_key = speech_org_key(speech)
        if org_key and org_key != "__all__" and candidate_org_key != org_key:
            continue

        metadata = speech.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        doc_id = corpus_doc_id(speech)
        if allowed_ids and doc_id not in allowed_ids:
            continue

        content = speech.get("content", {})
        if not isinstance(content, dict):
            content = {}
        raw_text = content.get("full_text", "")
        text = raw_text if isinstance(raw_text, str) else str(raw_text or "")
        if not text.strip():
            continue

        dedup[doc_id] = {
            "doc_id": doc_id,
            "organization": speech_org_label(speech),
            "org_key": candidate_org_key,
            "title": str(metadata.get("title", "") or "").strip(),
            "speaker": str(metadata.get("speaker", "") or "").strip(),
            "date": str(metadata.get("date", "") or "").strip(),
            "url": str(metadata.get("url", "") or "").strip(),
            "doc_type": str(metadata.get("doc_type", "Speech") or "Speech").strip(),
            "source_kind": infer_source_kind(metadata),
            "release_no": str(metadata.get("release_no", "") or "").strip(),
            "word_count": coerce_int(metadata.get("word_count", 0), default=0, min_value=0),
        }
        if include_full_text:
            dedup[doc_id]["full_text"] = text.strip()

    return list(dedup.values())
