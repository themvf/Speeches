from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

NOTICE_SOURCE_KINDS = {"finra_regulatory_notice", "regulations_gov_rule", "sec_rule_release"}
COMMENT_SOURCE_KINDS = {"finra_comment_letter", "regulations_gov_comment", "sec_rule_comment"}
POSITION_BUCKETS = ["supportive", "neutral", "opposed", "mixed", "unclear"]
ENRICHED_STATUSES = {"enriched", "fallback_enriched", "reviewed"}
TOP_TOPIC_LIMIT = 5
NOISE_TOPIC_KEYS = {
    "",
    "sec",
    "finra",
    "regulations gov",
    "rule",
    "rules",
    "rule release",
    "rule releases",
    "rulemaking",
    "rulemakings",
    "public comment",
    "public comments",
    "comment",
    "comments",
    "comment letter",
    "comment letters",
    "notice",
    "notices",
}
CANONICAL_TOPIC_RULES = [
    {
        "label": "Climate Disclosure",
        "exact": ["climate disclosure", "climate disclosures", "climate risk disclosure", "climate related disclosure"],
        "includes": ["climate disclosure", "climate-related", "climate risk"],
    },
    {
        "label": "Artificial Intelligence",
        "exact": ["ai", "artificial intelligence", "machine learning", "generative ai"],
        "includes": ["artificial intelligence", "machine learning", "generative ai"],
    },
    {
        "label": "Anti-Money Laundering",
        "exact": ["aml", "anti money laundering", "anti-money laundering", "bank secrecy act"],
        "includes": ["anti money laundering", "bank secrecy act"],
    },
    {
        "label": "Digital Assets",
        "exact": [
            "crypto",
            "cryptocurrency",
            "cryptocurrencies",
            "digital asset",
            "digital assets",
            "crypto asset",
            "crypto assets",
            "cryptoasset",
            "cryptoassets",
            "stablecoin",
            "stablecoins",
            "tokenization",
            "tokenized securities",
        ],
        "includes": ["digital asset", "crypto asset", "cryptocurrency", "stablecoin", "tokenization"],
    },
    {
        "label": "Cybersecurity",
        "exact": ["cybersecurity", "cyber security", "cyber risk", "cyber risks"],
        "includes": ["cybersecurity", "cyber security", "cyber risk"],
    },
    {"label": "Best Execution", "exact": ["best execution"], "includes": ["best execution"]},
    {
        "label": "Safeguarding Rule",
        "exact": ["safeguarding", "safeguarding rule", "custody rule"],
        "includes": ["safeguarding rule", "custody rule"],
    },
    {
        "label": "Custody",
        "exact": ["custody", "qualified custodian", "qualified custodians"],
        "includes": ["qualified custodian", "custody"],
    },
    {
        "label": "Conflicts of Interest",
        "exact": ["conflict of interest", "conflicts of interest"],
        "includes": ["conflict of interest"],
    },
    {
        "label": "Recordkeeping",
        "exact": ["recordkeeping", "record keeping", "books and records"],
        "includes": ["recordkeeping", "record keeping", "books and records"],
    },
    {
        "label": "Reporting",
        "exact": ["reporting", "report", "reporting requirements", "periodic reporting"],
        "includes": ["reporting"],
    },
    {
        "label": "Disclosure",
        "exact": ["disclosure", "disclosures", "disclosure requirements"],
        "includes": ["disclosure"],
    },
    {
        "label": "Data Privacy",
        "exact": ["privacy", "data privacy", "consumer data", "data portability", "data security"],
        "includes": ["data privacy", "consumer data", "data portability"],
    },
]
DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%B %d, %Y",
    "%b %d, %Y",
    "%m/%d/%Y",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _split_csv(value: Any) -> List[str]:
    return [_normalize_text(item) for item in str(value or "").split(",") if _normalize_text(item)]


def _dedup_list(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        value = _normalize_text(item)
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _parse_datetime(value: Any) -> Optional[datetime]:
    text = _normalize_text(value)
    if not text:
        return None
    iso_candidate = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(iso_candidate)
    except Exception:
        pass
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
    return None


def _sortable_timestamp(value: Any) -> float:
    parsed = _parse_datetime(value)
    if parsed is None:
        return float("-inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _resolved_source_kind(metadata: Dict[str, Any]) -> str:
    explicit = _normalize_text(metadata.get("source_kind", "")).lower()
    if explicit:
        return explicit

    source_family = _normalize_text(metadata.get("source_family", "")).lower()
    doc_type = _normalize_text(metadata.get("doc_type", "")).lower()
    organization = _normalize_text(metadata.get("organization", "")).lower()
    urls = [
        metadata.get("url", ""),
        metadata.get("rule_url", ""),
        metadata.get("notice_url", ""),
        metadata.get("comments_url", ""),
        metadata.get("comment_url", ""),
        metadata.get("comment_page_url", ""),
        metadata.get("docket_url", ""),
        metadata.get("document_url", ""),
        metadata.get("source_index_url", ""),
        metadata.get("resolved_content_url", ""),
    ]
    normalized_urls = [_normalize_text(item).lower() for item in urls if _normalize_text(item)]

    def _has_url(fragment: str) -> bool:
        return any(fragment in item for item in normalized_urls)

    if _has_url("/rules-regulations/public-comments/") or _has_url("/comments/") or (
        source_family == "sec_rule" and doc_type == "public comment"
    ):
        return "sec_rule_comment"

    if source_family == "sec_rule" or (
        _has_url("/rules-regulations/") and ("rule" in doc_type or "release" in doc_type or organization == "sec")
    ):
        return "sec_rule_release"

    if source_family == "regulations_gov" and doc_type == "public comment":
        return "regulations_gov_comment"
    if source_family == "regulations_gov":
        return "regulations_gov_rule"

    if "finra_comment" in source_family or (organization == "finra" and doc_type == "public comment"):
        return "finra_comment_letter"
    if "finra" in source_family or organization == "finra":
        return "finra_regulatory_notice"

    return explicit


def _source_family(metadata: Dict[str, Any]) -> str:
    explicit = _normalize_text(metadata.get("source_family", "")).lower()
    if explicit == "regulations_gov":
        return "regulations_gov"
    if explicit in {"sec_rule", "sec"}:
        return "sec"
    source_kind = _normalize_text(metadata.get("source_kind", "")).lower()
    if source_kind.startswith("regulations_gov_"):
        return "regulations_gov"
    if source_kind.startswith("sec_rule_"):
        return "sec"
    if explicit.startswith("finra") or source_kind.startswith("finra_") or _normalize_text(metadata.get("organization", "")).lower() == "finra":
        return "finra"
    if _normalize_text(metadata.get("organization", "")).lower() == "sec":
        return "sec"
    return explicit or source_kind or "document"


def _source_family_label(family: str) -> str:
    if family == "regulations_gov":
        return "Regulations.gov"
    if family == "sec":
        return "SEC"
    if family == "finra":
        return "FINRA"
    return " ".join(part.capitalize() for part in family.split("_") if part)


def _group_type_label(metadata: Dict[str, Any], family: str) -> str:
    if family == "regulations_gov":
        return "Rulemaking Docket"
    if family == "sec":
        return _normalize_text(metadata.get("rule_type") or metadata.get("doc_type") or "Rule Release") or "Rule Release"
    if family == "finra":
        return _normalize_text(metadata.get("notice_type") or "Regulatory Notice") or "Regulatory Notice"
    return _normalize_text(metadata.get("doc_type") or "Document") or "Document"


def _group_identifier_label(family: str) -> str:
    if family == "regulations_gov":
        return "Docket"
    if family == "sec":
        return "File Number"
    return "Notice"


def _sec_file_number(metadata: Dict[str, Any]) -> str:
    return _normalize_text(metadata.get("file_number") or metadata.get("notice_number") or "").upper()


def _sec_rule_url(metadata: Dict[str, Any]) -> str:
    return _normalize_text(metadata.get("rule_url") or metadata.get("url") or metadata.get("notice_url") or metadata.get("source_index_url") or "")


def _sec_rule_title(metadata: Dict[str, Any]) -> str:
    return _normalize_text(metadata.get("notice_title") or metadata.get("title") or "")


def _extract_regulations_docket_id(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return ""

    match = re.search(r"\b([A-Z][A-Z0-9]*-\d{4}-\d{4})\b", text, flags=re.IGNORECASE)
    if match and match.group(1):
        return _normalize_text(match.group(1)).upper()

    try:
        parsed = urlparse(text)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            prefix = parts[0].lower()
            identifier = _normalize_text(parts[1])
            if prefix == "docket" and identifier:
                return identifier.upper()
            nested = re.match(r"^([A-Z][A-Z0-9]*-\d{4}-\d{4})-\d+$", identifier, flags=re.IGNORECASE)
            if nested and nested.group(1):
                return _normalize_text(nested.group(1)).upper()
    except Exception:
        return ""

    return ""


def _regulations_docket_id(metadata: Dict[str, Any]) -> str:
    candidates = [
        metadata.get("docket_id"),
        metadata.get("rule_url"),
        metadata.get("docket_url"),
        metadata.get("document_url"),
        metadata.get("comment_page_url"),
        metadata.get("comment_url"),
        metadata.get("url"),
        metadata.get("input_url"),
        metadata.get("comment_id"),
        metadata.get("document_id"),
    ]
    for candidate in candidates:
        docket_id = _extract_regulations_docket_id(candidate)
        if docket_id:
            return docket_id
    return ""


def _regulations_group_url(metadata: Dict[str, Any]) -> str:
    return _normalize_text(
        metadata.get("docket_url")
        or metadata.get("rule_url")
        or metadata.get("document_url")
        or metadata.get("url")
        or metadata.get("comment_page_url")
        or metadata.get("comment_url")
        or ""
    )


def _regulations_group_title(metadata: Dict[str, Any]) -> str:
    docket_id = _regulations_docket_id(metadata)
    source_kind = _normalize_text(metadata.get("source_kind") or "")
    metadata_title = _normalize_text(metadata.get("title") or metadata.get("notice_title") or "")
    if source_kind == "regulations_gov_rule" and metadata_title:
        return metadata_title
    if docket_id:
        return docket_id
    return metadata_title or "Rulemaking Docket"


def _notice_group_key(metadata: Dict[str, Any]) -> str:
    family = _source_family(metadata)
    if family == "regulations_gov":
        docket_id = _regulations_docket_id(metadata)
        if docket_id:
            return f"regulations_gov:docket:{docket_id.lower()}"
        rule_url = _regulations_group_url(metadata)
        if rule_url:
            return f"regulations_gov:url:{rule_url.lower()}"

    if family == "sec":
        file_number = _sec_file_number(metadata)
        if file_number:
            return f"sec:file:{file_number.lower()}"
        rule_url = _sec_rule_url(metadata)
        if rule_url:
            return f"sec:url:{rule_url.lower()}"

    notice_number = _normalize_text(metadata.get("notice_number") or "")
    if notice_number:
        return f"finra:notice:{notice_number.lower()}"

    notice_url = _normalize_text(
        metadata.get("notice_url") or metadata.get("source_notice_url") or metadata.get("source_index_url") or metadata.get("url") or ""
    )
    if notice_url:
        return f"finra:url:{notice_url.lower()}"

    title = _normalize_text(metadata.get("notice_title") or metadata.get("title") or "")
    return f"{family}:title:{title.lower()}"


def _group_identifier(metadata: Dict[str, Any], family: str) -> str:
    if family == "regulations_gov":
        return _regulations_docket_id(metadata)
    if family == "sec":
        return _sec_file_number(metadata)
    return _normalize_text(metadata.get("notice_number") or "")


def _summary_for(record: Dict[str, Any], entry: Optional[Dict[str, Any]]) -> str:
    enrichment = entry.get("enrichment", {}) if isinstance(entry, dict) else {}
    enriched = _normalize_text(enrichment.get("summary") if isinstance(enrichment, dict) else "")
    if enriched:
        return enriched

    metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
    metadata_summary = _normalize_text(metadata.get("summary") or "")
    if metadata_summary:
        return metadata_summary[:420]

    content = record.get("content", {}) if isinstance(record.get("content", {}), dict) else {}
    raw = str(content.get("full_text", "") or "")
    blocks = [_normalize_text(item) for item in re.split(r"\n\s*\n", raw) if _normalize_text(item)]
    for block in blocks:
        looks_like_header = bool(re.search(r"Source URL:", block, flags=re.IGNORECASE)) and bool(
            re.search(
                r"(Notice Number:|Published Date:|Commenter:|Professional Affiliation:|Date:|Title:|Docket ID:|Comment ID:|File Number:|Release Numbers?:|Rule Type:|Rule Title:|Rule URL:|Comments URL:|Comment URL:|Letter Type:)",
                block,
                flags=re.IGNORECASE,
            )
        )
        if not looks_like_header:
            return block[:420]

    return _normalize_text(metadata.get("title") or "")[:420]


def _enrichment_status(entry: Optional[Dict[str, Any]]) -> str:
    return _normalize_text((entry or {}).get("status", "not_enriched")) or "not_enriched"


def _review_decision(entry: Optional[Dict[str, Any]]) -> str:
    review = entry.get("review", {}) if isinstance(entry, dict) else {}
    return _normalize_text(review.get("decision", "pending")) or "pending"


def _stance_label(entry: Optional[Dict[str, Any]]) -> str:
    enrichment = entry.get("enrichment", {}) if isinstance(entry, dict) else {}
    stance = enrichment.get("stance", {}) if isinstance(enrichment, dict) else {}
    if not isinstance(stance, dict):
        return ""
    return _normalize_text(stance.get("label") or "")


def _comment_position(entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    enrichment = entry.get("enrichment", {}) if isinstance(entry, dict) else {}
    raw = enrichment.get("comment_position", {}) if isinstance(enrichment, dict) else {}
    if isinstance(raw, dict):
        label = _normalize_text(raw.get("label") or "").lower()
        try:
            confidence = float(raw.get("confidence") or 0)
        except Exception:
            confidence = 0.0
        rationale = _normalize_text(raw.get("rationale") or "")
        if label:
            return {
                "label": label,
                "confidence": max(0.0, min(1.0, confidence)),
                "rationale": rationale,
            }

    legacy = _stance_label(entry).lower()
    if legacy == "supportive":
        return {
            "label": "supportive",
            "confidence": 0.42,
            "rationale": "Derived from legacy enrichment stance while comment-position enrichment is unavailable.",
        }
    if legacy == "critical":
        return {
            "label": "opposed",
            "confidence": 0.42,
            "rationale": "Derived from legacy enrichment stance while comment-position enrichment is unavailable.",
        }
    if legacy == "neutral":
        return {
            "label": "neutral",
            "confidence": 0.35,
            "rationale": "Derived from legacy enrichment stance while comment-position enrichment is unavailable.",
        }
    if legacy == "cautious":
        return {
            "label": "unclear",
            "confidence": 0.25,
            "rationale": "Legacy stance is cautious, which does not map cleanly to support or opposition.",
        }
    return {"label": "unclear", "confidence": 0.0, "rationale": ""}


def _build_notice_tags(record: Dict[str, Any], entry: Optional[Dict[str, Any]]) -> List[str]:
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
    metadata_tags = _split_csv(metadata.get("tags") or "")
    enrichment = entry.get("enrichment", {}) if isinstance(entry, dict) else {}
    enrich_tags = []
    if isinstance(enrichment, dict) and isinstance(enrichment.get("tags"), list):
        enrich_tags = [_normalize_text(item) for item in enrichment.get("tags", []) if _normalize_text(item)]
    return _dedup_list(metadata_tags + enrich_tags)


def _build_keywords(entry: Optional[Dict[str, Any]]) -> List[str]:
    enrichment = entry.get("enrichment", {}) if isinstance(entry, dict) else {}
    if not isinstance(enrichment, dict) or not isinstance(enrichment.get("keywords"), list):
        return []
    return _dedup_list([_normalize_text(item) for item in enrichment.get("keywords", []) if _normalize_text(item)])


def _title_case(value: str) -> str:
    parts = []
    for part in value.split(" "):
        if not part:
            continue
        if re.fullmatch(r"[A-Z0-9]{2,}", part):
            parts.append(part)
        else:
            parts.append(part[:1].upper() + part[1:].lower())
    return " ".join(parts)


def _normalized_topic_key(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", _normalize_text(value).lower())).strip()


def _canonical_topic_label(value: Any) -> str:
    key = _normalized_topic_key(value)
    if not key or key in NOISE_TOPIC_KEYS:
        return ""
    if re.match(r"^(?:file|notice)\s+", key):
        return ""
    if re.match(r"^s\d+\s+\d{4}\s+\d{2}$", key):
        return ""

    for rule in CANONICAL_TOPIC_RULES:
        if key in rule.get("exact", []):
            return str(rule["label"])
    for rule in CANONICAL_TOPIC_RULES:
        if any(fragment in key for fragment in rule.get("includes", [])):
            return str(rule["label"])
    return _title_case(key)


def _comment_topics(comment: Dict[str, Any]) -> List[str]:
    keywords = comment.get("keywords", []) if isinstance(comment.get("keywords"), list) else []
    tags = comment.get("tags", []) if isinstance(comment.get("tags"), list) else []
    preferred = keywords if keywords else tags
    out: List[str] = []
    seen = set()
    for raw in preferred:
        label = _canonical_topic_label(raw)
        key = _normalized_topic_key(label)
        if not label or not key or key in seen:
            continue
        seen.add(key)
        out.append(label)
    return out


def _empty_overview() -> Dict[str, Any]:
    return {
        "total_comments": 0,
        "enriched_comments": 0,
        "position_counts": {bucket: 0 for bucket in POSITION_BUCKETS},
        "top_topics": [],
    }


def _build_notice_overview(comments: List[Dict[str, Any]]) -> Dict[str, Any]:
    overview = _empty_overview()
    overview["total_comments"] = len(comments)
    topic_counts: Dict[str, Dict[str, Any]] = {}

    for comment in comments:
        if _normalize_text(comment.get("enrichment_status") or "").lower() in ENRICHED_STATUSES:
            overview["enriched_comments"] += 1

        position = _normalize_text(((comment.get("comment_position") or {}).get("label") if isinstance(comment.get("comment_position"), dict) else "") or "").lower()
        bucket = position if position in {"supportive", "neutral", "opposed", "mixed"} else "unclear"
        overview["position_counts"][bucket] = int(overview["position_counts"].get(bucket, 0) or 0) + 1

        for label in _comment_topics(comment):
            key = _normalized_topic_key(label)
            current = topic_counts.get(key)
            if current:
                current["count"] = int(current.get("count", 0) or 0) + 1
            else:
                topic_counts[key] = {"label": label, "count": 1}

    topics = sorted(topic_counts.values(), key=lambda item: (-int(item.get("count", 0) or 0), str(item.get("label", ""))))
    overview["top_topics"] = [
        {
            "label": str(item.get("label", "")),
            "count": int(item.get("count", 0) or 0),
            "share": (int(item.get("count", 0) or 0) / float(overview["total_comments"])) if overview["total_comments"] else 0.0,
        }
        for item in topics[:TOP_TOPIC_LIMIT]
    ]
    return overview


def _build_base_group(metadata: Dict[str, Any], record: Dict[str, Any], entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    family = _source_family(metadata)
    source_kind = _resolved_source_kind(metadata)
    published_at = _normalize_text(metadata.get("published_date") or metadata.get("date") or "")
    identifier = _group_identifier(metadata, family)
    docket_id = _regulations_docket_id(metadata) if family == "regulations_gov" else _normalize_text(metadata.get("docket_id") or "")
    title = (
        _regulations_group_title(metadata)
        if family == "regulations_gov"
        else (_sec_rule_title(metadata) or "SEC Rule Release")
        if family == "sec"
        else _normalize_text(metadata.get("title") or metadata.get("notice_title") or "") or "Regulatory Notice"
    )
    url = (
        _regulations_group_url(metadata)
        if family == "regulations_gov"
        else _sec_rule_url(metadata)
        if family == "sec"
        else _normalize_text(metadata.get("url") or metadata.get("notice_url") or metadata.get("rule_url") or metadata.get("docket_url") or metadata.get("document_url") or "")
    )
    return {
        "notice_key": _notice_group_key(metadata),
        "source_kind": source_kind,
        "source_family": family,
        "source_family_label": _source_family_label(family),
        "group_type_label": _group_type_label(metadata, family),
        "group_identifier_label": _group_identifier_label(family),
        "group_identifier": identifier,
        "notice_document_id": _normalize_text(metadata.get("document_id") or ""),
        "notice_number": _sec_file_number(metadata) if family == "sec" else _normalize_text(metadata.get("notice_number") or ""),
        "docket_id": docket_id,
        "title": title or ("Rulemaking Docket" if family == "regulations_gov" else "SEC Rule Release" if family == "sec" else "Regulatory Notice"),
        "summary": _summary_for(record, entry),
        "organization": _normalize_text(metadata.get("organization") or "") or ("Regulations.gov" if family == "regulations_gov" else "SEC" if family == "sec" else "FINRA"),
        "url": url,
        "pdf_url": _normalize_text(metadata.get("pdf_url") or ""),
        "published_at": published_at,
        "effective_date": _normalize_text(metadata.get("effective_date") or ""),
        "comment_deadline": _normalize_text(metadata.get("comment_deadline") or ""),
        "tags": _build_notice_tags(record, entry),
        "keywords": _build_keywords(entry),
        "enrichment_status": _enrichment_status(entry),
        "review_decision": _review_decision(entry),
        "comment_count": 0,
        "latest_comment_at": "",
        "overview": _empty_overview(),
        "comment_document_ids": [],
        "_comment_inputs": [],
        "_comment_refs": [],
    }


def _build_fallback_group(metadata: Dict[str, Any], record: Dict[str, Any], entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = _build_base_group(metadata, record, entry)
    family = str(base.get("source_family") or "")
    base["notice_document_id"] = ""
    base["title"] = (
        (_regulations_group_title(metadata) if family == "regulations_gov" else _sec_rule_title(metadata) if family == "sec" else "")
        or str(base.get("title") or "")
        or _normalize_text(metadata.get("notice_title") or metadata.get("title") or "")
        or ("Rulemaking Docket" if family == "regulations_gov" else "SEC Rule Release" if family == "sec" else "Regulatory Notice")
    )
    base["summary"] = ""
    base["organization"] = str(base.get("organization") or "") or ("Regulations.gov" if family == "regulations_gov" else "SEC" if family == "sec" else "FINRA")
    base["url"] = (
        (_regulations_group_url(metadata) if family == "regulations_gov" else _sec_rule_url(metadata) if family == "sec" else "")
        or str(base.get("url") or "")
        or _normalize_text(metadata.get("rule_url") or metadata.get("notice_url") or metadata.get("source_notice_url") or metadata.get("docket_url") or metadata.get("document_url") or "")
    )
    return base


def build_rule_summaries_payload(custom_payload: Dict[str, Any], enrichment_state: Dict[str, Any]) -> Dict[str, Any]:
    documents = custom_payload.get("documents", []) if isinstance(custom_payload, dict) else []
    if not isinstance(documents, list):
        documents = []
    entries = enrichment_state.get("entries", {}) if isinstance(enrichment_state, dict) else {}
    if not isinstance(entries, dict):
        entries = {}

    groups: Dict[str, Dict[str, Any]] = {}

    for record in documents:
        if not isinstance(record, dict):
            continue
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
        source_kind = _resolved_source_kind(metadata)
        if source_kind not in NOTICE_SOURCE_KINDS:
            continue
        doc_id = _normalize_text(metadata.get("document_id") or "")
        entry = entries.get(doc_id) if isinstance(entries.get(doc_id), dict) else None
        group = _build_base_group(metadata, record, entry)
        groups[str(group["notice_key"])] = group

    for record in documents:
        if not isinstance(record, dict):
            continue
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
        source_kind = _resolved_source_kind(metadata)
        if source_kind not in COMMENT_SOURCE_KINDS:
            continue
        doc_id = _normalize_text(metadata.get("document_id") or "")
        if not doc_id:
            continue
        entry = entries.get(doc_id) if isinstance(entries.get(doc_id), dict) else None
        key = _notice_group_key(metadata)
        published_at = _normalize_text(metadata.get("published_date") or metadata.get("date") or "")
        group = groups.get(key) or _build_fallback_group(metadata, record, entry)
        group["_comment_refs"].append((doc_id, published_at))
        group["_comment_inputs"].append(
            {
                "tags": _build_notice_tags(record, entry),
                "keywords": _build_keywords(entry),
                "enrichment_status": _enrichment_status(entry),
                "comment_position": _comment_position(entry),
                "review_decision": _review_decision(entry),
            }
        )
        group["comment_count"] = len(group["_comment_refs"])
        if _sortable_timestamp(published_at) >= _sortable_timestamp(group.get("latest_comment_at") or ""):
            group["latest_comment_at"] = published_at
        if key not in groups:
            groups[key] = group

    ordered_groups: List[Dict[str, Any]] = []
    for group in groups.values():
        refs = list(group.pop("_comment_refs", []))
        comment_inputs = list(group.pop("_comment_inputs", []))
        refs.sort(key=lambda item: (-_sortable_timestamp(item[1]), str(item[0])))
        group["comment_document_ids"] = [doc_id for doc_id, _ in refs]
        group["comment_count"] = len(group["comment_document_ids"])
        group["overview"] = _build_notice_overview(comment_inputs)
        ordered_groups.append(group)

    ordered_groups.sort(
        key=lambda group: (
            -_sortable_timestamp(group.get("published_at") or ""),
            -_sortable_timestamp(group.get("latest_comment_at") or ""),
            str(group.get("group_identifier") or ""),
        )
    )

    totals = {
        "notices": len(ordered_groups),
        "comments": sum(int(group.get("comment_count", 0) or 0) for group in ordered_groups),
        "enriched_comments": sum(int((group.get("overview") or {}).get("enriched_comments", 0) or 0) for group in ordered_groups),
        "pending_review_comments": 0,
    }

    # Recompute pending-review counts directly from source entries.
    pending_review_comments = 0
    for record in documents:
        if not isinstance(record, dict):
            continue
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
        source_kind = _resolved_source_kind(metadata)
        if source_kind not in COMMENT_SOURCE_KINDS:
            continue
        doc_id = _normalize_text(metadata.get("document_id") or "")
        entry = entries.get(doc_id) if isinstance(entries.get(doc_id), dict) else None
        if _enrichment_status(entry) in ENRICHED_STATUSES and _review_decision(entry) not in {"accepted", "edited", "rejected"}:
            pending_review_comments += 1
    totals["pending_review_comments"] = pending_review_comments

    return {
        "version": 1,
        "updated_at": "",
        "generated_at": _utc_now_iso(),
        "custom_documents_updated_at": _normalize_text((custom_payload or {}).get("updated_at") if isinstance(custom_payload, dict) else ""),
        "enrichment_state_updated_at": _normalize_text((enrichment_state or {}).get("updated_at") if isinstance(enrichment_state, dict) else ""),
        "totals": totals,
        "groups": ordered_groups,
    }
