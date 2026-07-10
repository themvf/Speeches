#!/usr/bin/env python3
"""Headless connector extraction pipeline for non-NewsAPI sources."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import run_financial_news_pipeline as core
from source_health import record_source_health


SEC_TM_FAQ_DEFAULT_URL = "https://www.sec.gov/rules-regulations/staff-guidance/trading-markets-frequently-asked-questions"
SEC_LIT_DEFAULT_URL = "https://www.sec.gov/enforcement-litigation/litigation-releases"
SEC_SPEECH_DEFAULT_URL = "https://www.sec.gov/newsroom/speeches-statements"
FINRA_NOTICE_DEFAULT_URL = "https://www.finra.org/rules-guidance/notices"
FINRA_AWC_DEFAULT_URL = "https://www.finra.org/rules-guidance/oversight-enforcement/finra-disciplinary-actions"
DOJ_DEFAULT_URL = "https://www.justice.gov/usao/pressreleases"
FED_DEFAULT_URL = "https://www.federalreserve.gov/newsevents/speeches-testimony.htm"
CFTC_PRESS_RELEASE_DEFAULT_URL = "https://www.cftc.gov/PressRoom/PressReleases"
CFTC_PUBLIC_STATEMENT_DEFAULT_URL = "https://www.cftc.gov/PressRoom/SpeechesTestimony/index.htm"
TREASURY_FEATURED_STORIES_DEFAULT_URL = "https://home.treasury.gov/news/featured-stories"
TREASURY_PRESS_RELEASES_DEFAULT_URL = "https://home.treasury.gov/news/press-releases"
TREASURY_STATEMENTS_REMARKS_DEFAULT_URL = "https://home.treasury.gov/news/press-releases/statements-remarks"
SIFMA_NEWS_DEFAULT_URL = "https://www.sifma.org/news"
CONGRESS_CRS_PRODUCTS_DEFAULT_URL = "https://www.congress.gov/crs-products"
BLOOMBERG_PUBLIC_DEFAULT_URL = ""
SUBSTACK_PUBLIC_DEFAULT_URL = "https://substack.com/api/v1/post/search"
WSJ_DOW_JONES_DEFAULT_URL = "https://feeds.content.dowjones.io/public/rss/WSJcomUSBusinessNews"
HEDGE_FUND_LETTER_DEFAULT_URL = "https://fiscal.ai/fund-letters/"
SEC_YOUTUBE_DEFAULT_URL = "https://www.youtube.com/user/SECViews"
YOUTUBE_DEFAULT_URL = ""

YOUTUBE_CONNECTORS = {
    "sec_youtube_video",
    "youtube_video",
}

BLOOMBERG_CONNECTORS = {
    "bloomberg_public_article",
    "bloomberg_public_latest",
    # Backward-compatible aliases. These now use the public Bloomberg connector,
    # not Apify.
    "bloomberg_apify_article",
    "bloomberg_latest_apify",
}

SECURITIES_MARKET_CONNECTORS = {
    "sec_press_release_rss",
    "sec_administrative_proceeding",
    "sec_trading_suspension",
    "sec_federal_register",
    "sec_pcaob_rulemaking",
    "pcaob_update",
    "msrb_press_release",
}

TRADE_MEDIA_CONNECTORS = {
    "jdsupra_article",
    "investmentnews_article",
    "citywire_article",
    "therecord_media_article",
    "wired_article",
    "tripwire_article",
    "akamai_blog_article",
    "ritholtz_article",
    "ft_portfolios_market_commentary",
    "liberty_street_economics_article",
    "wealth_of_common_sense_article",
    # Crypto media
    "coindesk_article",
    "cointelegraph_article",
    "decrypt_article",
    "the_block_article",
    # Cyber / threat-intel media
    "krebs_on_security_article",
    "the_hacker_news_article",
    "welivesecurity_article",
    "sophos_security_operations_article",
    "flashpoint_blog_article",
    "recorded_future_article",
    "intel471_blog_article",
    "securityweek_article",
    "dark_reading_article",
    # Wire service and Google News aggregators
    "prnewswire_article",
    "google_news_ponzi_investor_fraud_article",
    "google_news_senate_committee_article",
}

TRADE_ASSOCIATION_CONNECTORS = {
    "ici_news_item",
    "isda_news_item",
    "mfa_news_item",
    "fia_news_item",
    "aba_news_item",
    "bpi_news_item",
    "icba_news_item",
    "lsta_news_item",
}

SUPPORTED_CONNECTORS = {
    "sec_speech",
    "sec_tm_faq",
    "sec_rule_comment",
    "sec_enforcement_litigation",
    "finra_regulatory_notice",
    "finra_comment_letter",
    "finra_awc",
    "doj_usao_press_release",
    "federal_reserve_speech_testimony",
    "cisa_cybersecurity_advisory",
    "cftc_press_release",
    "cftc_public_statement_remark",
    "treasury_featured_story",
    "treasury_press_release",
    "treasury_statement_remark",
    "sifma_news_item",
    "congress_crs_product",
    "senate_committee_site",
    "bloomberg_public_article",
    "bloomberg_public_latest",
    "bloomberg_apify_article",
    "bloomberg_latest_apify",
    "substack_public_article",
    *YOUTUBE_CONNECTORS,
    "wsj_dow_jones",
    "reddit_post",
    "hedge_fund_letter",
    *TRADE_MEDIA_CONNECTORS,
    *TRADE_ASSOCIATION_CONNECTORS,
    *SECURITIES_MARKET_CONNECTORS,
}


def _default_base_url(connector: str) -> str:
    if connector == "sec_speech":
        return SEC_SPEECH_DEFAULT_URL
    if connector == "sec_tm_faq":
        return SEC_TM_FAQ_DEFAULT_URL
    if connector == "sec_rule_comment":
        return ""
    if connector == "sec_enforcement_litigation":
        return SEC_LIT_DEFAULT_URL
    if connector == "finra_regulatory_notice":
        return FINRA_NOTICE_DEFAULT_URL
    if connector == "finra_comment_letter":
        return ""
    if connector == "finra_awc":
        return FINRA_AWC_DEFAULT_URL
    if connector == "doj_usao_press_release":
        return DOJ_DEFAULT_URL
    if connector == "federal_reserve_speech_testimony":
        return FED_DEFAULT_URL
    if connector == "cisa_cybersecurity_advisory":
        from cisa_cybersecurity_advisory_scraper import CISA_CYBERSECURITY_ADVISORIES_URL

        return CISA_CYBERSECURITY_ADVISORIES_URL
    if connector == "cftc_press_release":
        return CFTC_PRESS_RELEASE_DEFAULT_URL
    if connector == "cftc_public_statement_remark":
        return CFTC_PUBLIC_STATEMENT_DEFAULT_URL
    if connector == "treasury_featured_story":
        return TREASURY_FEATURED_STORIES_DEFAULT_URL
    if connector == "treasury_press_release":
        return TREASURY_PRESS_RELEASES_DEFAULT_URL
    if connector == "treasury_statement_remark":
        return TREASURY_STATEMENTS_REMARKS_DEFAULT_URL
    if connector == "sifma_news_item":
        return SIFMA_NEWS_DEFAULT_URL
    if connector == "congress_crs_product":
        return CONGRESS_CRS_PRODUCTS_DEFAULT_URL
    if connector == "senate_committee_site":
        from senate_committee_scraper import SENATE_COMMITTEE_DEFAULT_URL

        return SENATE_COMMITTEE_DEFAULT_URL
    if connector in BLOOMBERG_CONNECTORS:
        return BLOOMBERG_PUBLIC_DEFAULT_URL
    if connector == "substack_public_article":
        return SUBSTACK_PUBLIC_DEFAULT_URL
    if connector == "sec_youtube_video":
        return SEC_YOUTUBE_DEFAULT_URL
    if connector == "youtube_video":
        return YOUTUBE_DEFAULT_URL
    if connector == "wsj_dow_jones":
        return WSJ_DOW_JONES_DEFAULT_URL
    if connector == "reddit_post":
        return "https://www.reddit.com/search.json"
    if connector == "hedge_fund_letter":
        return HEDGE_FUND_LETTER_DEFAULT_URL
    if connector in TRADE_MEDIA_CONNECTORS:
        from trade_media_scraper import TRADE_MEDIA_SOURCES

        return str(TRADE_MEDIA_SOURCES.get(connector, {}).get("default_url", "") or "")
    if connector in TRADE_ASSOCIATION_CONNECTORS:
        from trade_association_scraper import TRADE_ASSOCIATION_SOURCES

        return str(TRADE_ASSOCIATION_SOURCES.get(connector, {}).get("default_url", "") or "")
    if connector in SECURITIES_MARKET_CONNECTORS:
        from securities_market_sources_scraper import SECURITIES_MARKET_SOURCES

        return str(SECURITIES_MARKET_SOURCES.get(connector, {}).get("default_url", "") or "")
    return ""


def _normalize_space(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _is_generic_document_title(value: Any) -> bool:
    return _normalize_space(value).lower() in {
        "notice",
        "rule",
        "proposed rule",
        "final rule",
        "interim final rule",
        "presidential document",
        "correction",
    }


def _to_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_filter_terms(value: Any) -> List[str]:
    terms: List[str] = []
    seen = set()
    for raw in re.split(r"[,;\n]+", str(value or "")):
        term = _normalize_space(raw).lower()
        if not term or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _load_current_topic_rules() -> List[Dict[str, Any]]:
    try:
        from neon_feeds import get_topic_rules

        rules = get_topic_rules(only_active=True)
        if isinstance(rules, list) and rules:
            return [rule for rule in rules if isinstance(rule, dict)]
    except Exception:
        pass

    try:
        from neon_feeds import DEFAULT_TOPIC_RULES

        return [dict(rule, active=True) for rule in DEFAULT_TOPIC_RULES if isinstance(rule, dict)]
    except Exception:
        return []


def _topic_rule_terms(rule: Dict[str, Any]) -> List[str]:
    raw_terms = _parse_filter_terms(rule.get("keywords", ""))
    label = _normalize_space(rule.get("label", ""))
    if label:
        raw_terms.append(label.lower())

    terms: List[str] = []
    seen = set()
    for term in raw_terms:
        cleaned = _normalize_space(term).lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        terms.append(cleaned)
    return terms


def _topic_rules_to_search_terms(
    rules: List[Dict[str, Any]], *, max_terms: Optional[int] = None
) -> List[str]:
    sorted_rules = sorted(rules, key=lambda item: int(item.get("sort_order", 100) or 100))
    if max_terms is not None and max_terms > 0:
        buckets = [_topic_rule_terms(rule) for rule in sorted_rules]
        terms: List[str] = []
        seen = set()
        while len(terms) < max_terms and any(buckets):
            for bucket in buckets:
                while bucket:
                    term = bucket.pop(0)
                    if term in seen:
                        continue
                    seen.add(term)
                    terms.append(term)
                    break
                if len(terms) >= max_terms:
                    break
        return terms

    terms: List[str] = []
    seen = set()
    for rule in sorted_rules:
        for term in _topic_rule_terms(rule):
            if term in seen:
                continue
            seen.add(term)
            terms.append(term)
    return terms


def _substack_topic_search_term_limit() -> int:
    raw = os.getenv("SUBSTACK_TOPIC_SEARCH_TERM_LIMIT", "9")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 9


def _annotate_topic_matches(entry: Dict[str, Any], rules: List[Dict[str, Any]]) -> None:
    if not rules:
        return
    haystack = " ".join(
        str(part or "")
        for part in [
            entry.get("title", ""),
            entry.get("summary", ""),
            entry.get("preview_text", ""),
            entry.get("publication_name", ""),
            " ".join(entry.get("post_tags", []) if isinstance(entry.get("post_tags"), list) else []),
            " ".join(entry.get("matched_keywords", []) if isinstance(entry.get("matched_keywords"), list) else []),
            " ".join(entry.get("feed_tags", []) if isinstance(entry.get("feed_tags"), list) else []),
        ]
    ).lower()
    matched_keys: List[str] = []
    matched_labels: List[str] = []
    matched_terms: List[str] = []
    for rule in rules:
        rule_terms = _topic_rule_terms(rule)
        hits = [term for term in rule_terms if term and term in haystack]
        if not hits:
            continue
        key = _normalize_space(rule.get("topic_key", ""))
        label = _normalize_space(rule.get("label", ""))
        if key:
            matched_keys.append(key)
        if label:
            matched_labels.append(label)
        matched_terms.extend(hits)
    if matched_keys or matched_terms:
        entry["matched_topic_keys"] = matched_keys
        entry["matched_topic_labels"] = matched_labels
        entry["matched_topic_keywords"] = matched_terms[:20]


def _match_filter_terms(parts: List[Any], terms: List[str]) -> List[str]:
    if not terms:
        return []
    haystack = " ".join(str(part or "") for part in parts).lower()
    return [term for term in terms if term in haystack]


def _safe_source_name(url: str, fallback_prefix: str, source_ext: str) -> str:
    parsed = urlparse(str(url or "").strip())
    candidate = parsed.path.rsplit("/", 1)[-1].strip() if parsed.path else ""
    if not candidate:
        candidate = fallback_prefix
    candidate = core._safe_filename(candidate)
    if "." not in candidate:
        candidate += source_ext
    return candidate


def _parse_doc_date(value: Any) -> Any:
    parsed = core._parse_single_date(value)
    if parsed is not None:
        return parsed
    return str(value or "").strip()


def _load_existing_speech_url_keys(storage: Any) -> set[str]:
    keys: set[str] = set()

    if storage is not None:
        try:
            payload = storage.load_speeches()
            for item in payload.get("speeches", []):
                if not isinstance(item, dict):
                    continue
                metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
                key = core._url_match_key(metadata.get("url", ""))
                if key:
                    keys.add(key)
        except Exception as e:
            print(f"WARNING: GCS speech load failed — URL deduplication will be incomplete: {e}", file=sys.stderr)

    local_file = core.DATA_DIR / "all_speeches_final.json"
    if local_file.exists():
        try:
            data = json.loads(local_file.read_text(encoding="utf-8"))
            for item in data.get("speeches", []):
                if not isinstance(item, dict):
                    continue
                metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
                key = core._url_match_key(metadata.get("url", ""))
                if key:
                    keys.add(key)
        except Exception as e:
            print(f"WARNING: Local speech file read failed — URL deduplication will be incomplete: {e}", file=sys.stderr)

    return keys


def _build_existing_custom_map(custom_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in custom_payload.get("documents", []):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
        key = core._url_match_key(metadata.get("url", ""))
        if key:
            out[key] = metadata
    return out


def _build_existing_custom_record_map(custom_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in custom_payload.get("documents", []):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
        key = core._url_match_key(metadata.get("url", ""))
        if key:
            out[key] = item
    return out


def _remove_duplicate_bloomberg_records(custom_payload: Dict[str, Any]) -> int:
    docs_list = custom_payload.get("documents", [])
    if not isinstance(docs_list, list):
        return 0

    groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for idx, item in enumerate(docs_list):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
        if _normalize_space(metadata.get("source_kind", "")).lower() not in {
            "bloomberg_apify_article",
            "bloomberg_public_article",
        }:
            continue
        title_key = _normalize_space(metadata.get("title", "")).lower()
        if title_key:
            groups.setdefault(title_key, []).append((idx, item))

    remove_indexes: set[int] = set()
    for records in groups.values():
        if len(records) <= 1:
            continue

        def score(pair: Tuple[int, Dict[str, Any]]) -> Tuple[int, int]:
            _, record = pair
            metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
            content = record.get("content", {}) if isinstance(record.get("content", {}), dict) else {}
            date_text = _normalize_space(metadata.get("published_date") or metadata.get("date") or "")
            url = _normalize_space(metadata.get("url", "")).lower()
            full_text = str(content.get("full_text", "") or "")
            word_count = len(full_text.split())
            quality = word_count
            if date_text:
                quality += 10000
                parsed = core._parse_date_text(date_text)
                if parsed is not None:
                    date_path = f"/{parsed.year:04d}-{parsed.month:02d}-{parsed.day:02d}/"
                    alt_date_path = f"/{parsed.year:04d}/{parsed.month:02d}/{parsed.day:02d}/"
                    if date_path in url or alt_date_path in url:
                        quality += 5000
            if _normalize_space(metadata.get("summary", "")):
                quality += 1000
            return (quality, -pair[0])

        keep_idx = max(records, key=score)[0]
        for idx, _ in records:
            if idx != keep_idx:
                remove_indexes.add(idx)

    if not remove_indexes:
        return 0
    custom_payload["documents"] = [item for idx, item in enumerate(docs_list) if idx not in remove_indexes]
    return len(remove_indexes)


def _remove_legacy_bloomberg_apify_records(custom_payload: Dict[str, Any]) -> int:
    docs_list = custom_payload.get("documents", [])
    if not isinstance(docs_list, list):
        return 0

    kept: List[Dict[str, Any]] = []
    removed = 0
    for item in docs_list:
        if not isinstance(item, dict):
            kept.append(item)
            continue
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
        if _normalize_space(metadata.get("source_kind", "")).lower() == "bloomberg_apify_article":
            removed += 1
            continue
        kept.append(item)
    if removed:
        custom_payload["documents"] = kept
    return removed


def _remove_invalid_wired_coupon_records(custom_payload: Dict[str, Any]) -> int:
    docs_list = custom_payload.get("documents", [])
    if not isinstance(docs_list, list):
        return 0

    from trade_media_scraper import TRADE_MEDIA_SOURCES, _passes_source_url_filters

    cfg = TRADE_MEDIA_SOURCES.get("wired_article", {})
    title_pattern = re.compile(r"\b(?:promo\s+codes?|coupon(?:s)?|discount\s+codes?)\b", re.IGNORECASE)
    kept: List[Dict[str, Any]] = []
    removed = 0
    for item in docs_list:
        if not isinstance(item, dict):
            kept.append(item)
            continue
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
        if _normalize_space(metadata.get("source_kind", "")).lower() != "wired_article":
            kept.append(item)
            continue

        title = _normalize_space(metadata.get("title", ""))
        url = _normalize_space(metadata.get("url", ""))
        if (url and not _passes_source_url_filters(url, cfg)) or title_pattern.search(title):
            removed += 1
            continue
        kept.append(item)

    if removed:
        custom_payload["documents"] = kept
    return removed


def _repair_existing_finra_notice_metadata(entry: Dict[str, Any], existing_record: Optional[Dict[str, Any]]) -> Optional[str]:
    if not existing_record:
        return None
    metadata = existing_record.get("metadata", {}) if isinstance(existing_record.get("metadata", {}), dict) else {}
    if not metadata:
        return None

    changed = False
    date_text = _normalize_space(entry.get("date", ""))
    if date_text and (
        _normalize_space(metadata.get("published_date", "")) != date_text
        or _normalize_space(metadata.get("date", "")) != date_text
    ):
        metadata["published_date"] = date_text
        metadata["date"] = date_text
        changed = True

    for target_key, entry_key in [
        ("effective_date", "effective_date"),
        ("comment_deadline", "comment_deadline"),
        ("notice_number", "notice_number"),
        ("discovery_source", "discovery_source"),
    ]:
        value = _normalize_space(entry.get(entry_key, ""))
        if value and _normalize_space(metadata.get(target_key, "")) != value:
            metadata[target_key] = value
            changed = True

    if changed:
        return _normalize_space(metadata.get("document_id", ""))
    return None


def _build_short_text_fallback(
    *,
    title: str,
    url: str,
    date_text: str,
    organization: str,
    source_label: str,
    extracted_text: str = "",
) -> str:
    """Create a transparent placeholder body when metadata is useful but body extraction is thin."""
    parts = [
        str(title or "").strip(),
        f"Source: {str(source_label or organization or '').strip()}",
        f"Organization: {str(organization or '').strip()}",
        f"Date: {str(date_text or '').strip()}",
        f"URL: {str(url or '').strip()}",
    ]
    short_text = str(extracted_text or "").strip()
    if short_text:
        parts.append(f"Extracted text snippet: {short_text}")
    parts.append(
        "Note: The source page was discovered successfully, but the article body extraction returned a short result. "
        f"{core.METADATA_FALLBACK_TEXT_MARKER}"
    )
    return "\n".join(part for part in parts if part).strip()


def _build_bloomberg_article_record(
    *,
    entry: Dict[str, Any],
    scraper: Any,
    idx: int,
    base_url: str,
    source_kind: str = "bloomberg_public_article",
) -> Dict[str, Any]:
    src_url = str(entry.get("url", "") or "").strip()
    extraction_error = str(entry.get("extraction_error", "") or "").strip()
    if extraction_error:
        raise RuntimeError(f"Bloomberg connector failed to extract article: {extraction_error}")

    text = str(entry.get("full_text", "") or "").strip()
    summary = str(entry.get("summary", "") or "").strip()
    title = str(entry.get("title", "") or "").strip() or "Bloomberg article"
    date_text = str(entry.get("date", "") or "").strip()
    if len(text.split()) < 50:
        text = _build_short_text_fallback(
            title=title,
            url=src_url,
            date_text=date_text,
            organization="Bloomberg",
            source_label="Bloomberg public feed",
            extracted_text=text or summary,
        )
    doc_date = _parse_doc_date(date_text)
    authors = entry.get("authors") if isinstance(entry.get("authors"), list) else []
    author_text = ", ".join(str(author or "").strip() for author in authors if str(author or "").strip())
    keywords = entry.get("keywords") if isinstance(entry.get("keywords"), list) else []
    keyword_tags = [str(keyword or "").strip() for keyword in keywords if str(keyword or "").strip()]
    tags = ["bloomberg", "financial-news", "public-feed", *keyword_tags[:8]]
    source_name = _safe_source_name(src_url, f"bloomberg-public-{idx}", ".html")

    record = core._create_uploaded_document_record(
        text=text,
        organization="Bloomberg",
        title=title,
        speaker=author_text or "Bloomberg News",
        doc_date=doc_date,
        doc_type="Article",
        source_url=src_url,
        source_filename=source_name,
        source_ext=".html",
        source_local_path="",
        source_gcs_path="",
        tags_csv=",".join(tags),
        source_kind=source_kind,
    )
    metadata = record.setdefault("metadata", {})
    metadata["source_family"] = source_kind
    metadata["source_index_url"] = base_url
    metadata["published_date"] = date_text
    metadata["summary"] = summary
    metadata["source_name"] = str(entry.get("source", "") or "Bloomberg").strip()
    metadata["authors"] = authors
    metadata["keywords"] = keyword_tags
    metadata["extraction_mode"] = str(entry.get("extraction_mode", "") or "").strip()
    metadata["access_limited"] = bool(entry.get("access_limited", False))
    metadata["connector_mode"] = "public"
    raw_item = entry.get("raw_item")
    if isinstance(raw_item, dict):
        metadata["raw_item_keys"] = sorted(str(key) for key in raw_item.keys())[:50]
    discovery_raw = entry.get("discovery_raw_item")
    if isinstance(discovery_raw, dict):
        metadata["discovery_raw_keys"] = sorted(str(key) for key in discovery_raw.keys())[:50]
    return record


def _status_for_entry(
    connector: str,
    entry: Dict[str, Any],
    existing_meta: Optional[Dict[str, Any]],
    existing_speech_keys: set[str],
) -> str:
    key = core._url_match_key(entry.get("url", ""))
    if not existing_meta:
        return "existing_in_speeches" if key and key in existing_speech_keys else "new"

    if connector == "sec_tm_faq":
        existing_updated = _normalize_space(
            existing_meta.get("last_reviewed_or_updated")
            or existing_meta.get("updated_date")
            or ""
        )
        incoming_updated = _normalize_space(entry.get("updated_date", ""))
        if incoming_updated and existing_updated and incoming_updated != existing_updated:
            return "update_available"
        return "existing"

    if connector == "sec_speech":
        existing_date = _normalize_space(existing_meta.get("published_date") or existing_meta.get("date") or "")
        incoming_date = _normalize_space(entry.get("date", ""))
        existing_speaker = _normalize_space(existing_meta.get("listing_speaker") or existing_meta.get("speaker") or "")
        incoming_speaker = _normalize_space(entry.get("speaker", ""))
        if (incoming_date and existing_date and incoming_date != existing_date) or (
            incoming_speaker and existing_speaker and incoming_speaker != existing_speaker
        ):
            return "update_available"
        return "existing"

    if connector == "finra_regulatory_notice":
        existing_date = _normalize_space(existing_meta.get("published_date") or existing_meta.get("date") or "")
        incoming_date = _normalize_space(entry.get("date", ""))
        existing_effective = _normalize_space(existing_meta.get("effective_date", ""))
        incoming_effective = _normalize_space(entry.get("effective_date", ""))
        existing_comment = _normalize_space(existing_meta.get("comment_deadline", ""))
        incoming_comment = _normalize_space(entry.get("comment_deadline", ""))
        if (
            (incoming_date and existing_date and incoming_date != existing_date)
            or (incoming_effective and existing_effective and incoming_effective != existing_effective)
            or (incoming_comment and existing_comment and incoming_comment != existing_comment)
        ):
            return "update_available"
        return "existing"

    if connector == "finra_awc":
        existing_case_id = _normalize_space(existing_meta.get("case_id", ""))
        incoming_case_id = _normalize_space(entry.get("case_id", ""))
        existing_date = _normalize_space(existing_meta.get("published_date") or existing_meta.get("date") or "")
        incoming_date = _normalize_space(entry.get("date", ""))
        if (
            (incoming_case_id and existing_case_id and incoming_case_id != existing_case_id)
            or (incoming_date and existing_date and incoming_date != existing_date)
        ):
            return "update_available"
        return "existing"

    if connector == "finra_comment_letter":
        existing_date = _normalize_space(existing_meta.get("published_date") or existing_meta.get("date") or "")
        incoming_date = _normalize_space(entry.get("date", ""))
        existing_commenter = _normalize_space(existing_meta.get("commenter_name") or existing_meta.get("speaker") or "")
        incoming_commenter = _normalize_space(entry.get("commenter_name", ""))
        existing_notice = _normalize_space(existing_meta.get("notice_number", ""))
        incoming_notice = _normalize_space(entry.get("notice_number", ""))
        if (
            (incoming_date and existing_date and incoming_date != existing_date)
            or (incoming_commenter and existing_commenter and incoming_commenter != existing_commenter)
            or (incoming_notice and existing_notice and incoming_notice != existing_notice)
        ):
            return "update_available"
        return "existing"

    if connector == "sec_rule_comment":
        entry_kind = _normalize_space(entry.get("entry_kind", "")).lower()
        existing_date = _normalize_space(existing_meta.get("published_date") or existing_meta.get("date") or "")
        incoming_date = _normalize_space(entry.get("date", ""))
        existing_file = _normalize_space(existing_meta.get("file_number") or existing_meta.get("notice_number") or "").upper()
        incoming_file = _normalize_space(entry.get("file_number") or entry.get("notice_number") or "").upper()
        if entry_kind == "rule":
            existing_type = _normalize_space(existing_meta.get("rule_type") or existing_meta.get("doc_type") or "")
            incoming_type = _normalize_space(entry.get("rule_type", ""))
            existing_effective = _normalize_space(existing_meta.get("effective_date", ""))
            incoming_effective = _normalize_space(entry.get("effective_date", ""))
            if (
                (incoming_date and existing_date and incoming_date != existing_date)
                or (incoming_file and existing_file and incoming_file != existing_file)
                or (incoming_type and existing_type and incoming_type != existing_type)
                or (incoming_effective and existing_effective and incoming_effective != existing_effective)
            ):
                return "update_available"
            return "existing"

        existing_commenter = _normalize_space(existing_meta.get("commenter_name") or existing_meta.get("speaker") or "")
        incoming_commenter = _normalize_space(entry.get("commenter_name", ""))
        existing_letter_type = _normalize_space(existing_meta.get("letter_type", ""))
        incoming_letter_type = _normalize_space(entry.get("letter_type", ""))
        if (
            (incoming_date and existing_date and incoming_date != existing_date)
            or (incoming_file and existing_file and incoming_file != existing_file)
            or (incoming_commenter and existing_commenter and incoming_commenter != existing_commenter)
            or (incoming_letter_type and existing_letter_type and incoming_letter_type != existing_letter_type)
        ):
            return "update_available"
        return "existing"

    if connector == "sifma_news_item":
        existing_date = _normalize_space(existing_meta.get("published_date") or existing_meta.get("date") or "")
        incoming_date = _normalize_space(entry.get("date", ""))
        existing_title = _normalize_space(existing_meta.get("title", ""))
        incoming_title = _normalize_space(entry.get("title", ""))
        existing_category = _normalize_space(existing_meta.get("category", ""))
        incoming_category = _normalize_space(entry.get("category", ""))
        existing_doc_type = _normalize_space(existing_meta.get("doc_type", ""))
        incoming_doc_type = _normalize_space(entry.get("doc_type", ""))
        if (
            (incoming_date and existing_date and incoming_date != existing_date)
            or (incoming_title and existing_title and incoming_title != existing_title)
            or (incoming_category and existing_category and incoming_category != existing_category)
            or (incoming_doc_type and existing_doc_type and incoming_doc_type != existing_doc_type)
        ):
            return "update_available"
        return "existing"

    if connector == "congress_crs_product":
        existing_date_raw = existing_meta.get("published_date") or existing_meta.get("date") or ""
        incoming_date_raw = entry.get("date", "")
        existing_date = core._parse_single_date(existing_date_raw)
        incoming_date = core._parse_single_date(incoming_date_raw)
        date_changed = (
            existing_date != incoming_date
            if existing_date is not None and incoming_date is not None
            else _normalize_space(existing_date_raw) != _normalize_space(incoming_date_raw)
        )
        existing_title = _normalize_space(existing_meta.get("title", ""))
        incoming_title = _normalize_space(entry.get("title", ""))
        existing_author = _normalize_space(existing_meta.get("speaker", ""))
        incoming_author = _normalize_space(entry.get("authors", ""))
        existing_doc_type = _normalize_space(existing_meta.get("doc_type", ""))
        incoming_doc_type = _normalize_space(entry.get("doc_type", ""))
        if (
            date_changed
            or (incoming_title and existing_title and incoming_title != existing_title)
            or (incoming_author and existing_author and incoming_author != existing_author)
            or (incoming_doc_type and existing_doc_type and incoming_doc_type != existing_doc_type)
        ):
            return "update_available"
        return "existing"

    if connector in {"treasury_featured_story", "treasury_press_release", "treasury_statement_remark"}:
        existing_date = _normalize_space(existing_meta.get("published_date") or existing_meta.get("date") or "")
        incoming_date = _normalize_space(entry.get("date", ""))
        existing_title = _normalize_space(existing_meta.get("title", ""))
        incoming_title = _normalize_space(entry.get("title", ""))
        existing_speaker = _normalize_space(existing_meta.get("speaker", ""))
        incoming_speaker = _normalize_space(entry.get("speaker", ""))
        existing_doc_type = _normalize_space(existing_meta.get("doc_type", ""))
        incoming_doc_type = _normalize_space(entry.get("doc_type", ""))
        if (
            (incoming_date and existing_date and incoming_date != existing_date)
            or (incoming_title and existing_title and incoming_title != existing_title)
            or (incoming_speaker and existing_speaker and incoming_speaker != existing_speaker)
            or (incoming_doc_type and existing_doc_type and incoming_doc_type != existing_doc_type)
        ):
            return "update_available"
        return "existing"

    if connector == "cisa_cybersecurity_advisory":
        existing_date = _normalize_space(existing_meta.get("published_date") or existing_meta.get("date") or "")
        incoming_date = _normalize_space(entry.get("date", ""))
        existing_title = _normalize_space(existing_meta.get("title", ""))
        incoming_title = _normalize_space(entry.get("title", ""))
        existing_doc_type = _normalize_space(existing_meta.get("doc_type", ""))
        incoming_doc_type = _normalize_space(entry.get("doc_type", ""))
        existing_alert_code = _normalize_space(existing_meta.get("alert_code", ""))
        incoming_alert_code = _normalize_space(entry.get("alert_code", ""))
        if (
            (incoming_date and existing_date and incoming_date != existing_date)
            or (incoming_title and existing_title and incoming_title != existing_title)
            or (incoming_doc_type and existing_doc_type and incoming_doc_type != existing_doc_type)
            or (incoming_alert_code and existing_alert_code and incoming_alert_code != existing_alert_code)
        ):
            return "update_available"
        return "existing"

    if connector in SECURITIES_MARKET_CONNECTORS:
        existing_date = _normalize_space(existing_meta.get("published_date") or existing_meta.get("date") or "")
        incoming_date = _normalize_space(entry.get("date", ""))
        existing_title = _normalize_space(existing_meta.get("title", ""))
        incoming_title = _normalize_space(entry.get("title", ""))
        if (
            (incoming_date and existing_date and incoming_date != existing_date)
            or (
                connector == "sec_federal_register"
                and _is_generic_document_title(existing_title)
                and incoming_title
                and incoming_title != existing_title
            )
        ):
            return "update_available"
        return "existing"

    existing_date = _normalize_space(existing_meta.get("published_date") or existing_meta.get("date") or "")
    incoming_date = _normalize_space(entry.get("date") or entry.get("published_date") or "")
    if incoming_date and existing_date and incoming_date != existing_date:
        return "update_available"
    return "existing"


def _discover_connector(
    connector: str,
    base_url: str,
    max_pages: int,
    include_pdfs: bool,
    include_rss: bool,
    keywords: Optional[List[str]] = None,
    connector_settings: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
    if connector == "sec_speech":
        from speech_analyzer import SECSpeechAnalyzer

        analyzer = SECSpeechAnalyzer()
        docs = analyzer.scraper.discover_speech_urls(base_url=base_url, max_pages=max_pages)
        return analyzer, docs, {}

    if connector == "sec_tm_faq":
        from sec_tm_faq_scraper import TradingMarketsFAQScraper

        scraper = TradingMarketsFAQScraper()
        docs = scraper.discover_documents(index_url=base_url, include_pdfs=include_pdfs)
        return scraper, docs, {}

    if connector == "sec_rule_comment":
        from sec_rule_comments_scraper import SECRuleCommentsScraper

        scraper = SECRuleCommentsScraper()
        docs = scraper.discover_documents(rule_url=base_url, include_pdfs=include_pdfs)
        return scraper, docs, {}

    if connector == "sec_enforcement_litigation":
        from sec_enforcement_litigation_scraper import SECEnforcementLitigationScraper

        scraper = SECEnforcementLitigationScraper()
        docs = scraper.discover_documents(base_url=base_url, max_pages=max_pages)
        return scraper, docs, {}

    if connector == "finra_regulatory_notice":
        from finra_regulatory_notice_scraper import FINRARegulatoryNoticeScraper

        scraper = FINRARegulatoryNoticeScraper()
        docs = scraper.discover_documents(base_url=base_url, max_pages=max_pages, include_rss=include_rss)
        return scraper, docs, {}

    if connector == "finra_awc":
        from finra_awc_scraper import FINRAAWCScraper

        scraper = FINRAAWCScraper()
        docs = scraper.discover_documents(base_url=base_url, max_pages=max_pages)
        return scraper, docs, {}

    if connector == "finra_comment_letter":
        from finra_comment_letter_scraper import FINRACommentLetterScraper

        scraper = FINRACommentLetterScraper()
        docs = scraper.discover_documents(notice_url=base_url, include_pdfs=include_pdfs)
        return scraper, docs, {}

    if connector == "doj_usao_press_release":
        from doj_usao_press_release_scraper import DOJUSAOPressReleaseScraper

        scraper = DOJUSAOPressReleaseScraper()
        docs = scraper.discover_documents(base_url=base_url, max_pages=max_pages)
        debug = getattr(scraper, "last_discovery_debug", {})
        return scraper, docs, debug if isinstance(debug, dict) else {}

    if connector == "federal_reserve_speech_testimony":
        from federal_reserve_speech_testimony_scraper import FederalReserveSpeechTestimonyScraper

        scraper = FederalReserveSpeechTestimonyScraper()
        docs = scraper.discover_documents(base_url=base_url, max_pages=max_pages, fallback_to_feed=True)
        debug = getattr(scraper, "last_discovery_debug", {})
        return scraper, docs, debug if isinstance(debug, dict) else {}

    if connector == "cisa_cybersecurity_advisory":
        from cisa_cybersecurity_advisory_scraper import CISACybersecurityAdvisoryScraper

        scraper = CISACybersecurityAdvisoryScraper()
        docs = scraper.discover_documents(base_url=base_url, max_pages=max_pages, include_rss=include_rss)
        debug = getattr(scraper, "last_discovery_debug", {})
        return scraper, docs, debug if isinstance(debug, dict) else {}

    if connector in {"cftc_press_release", "cftc_public_statement_remark"}:
        from cftc_press_room_scraper import CFTCPressRoomScraper

        scraper = CFTCPressRoomScraper()
        docs = scraper.discover_documents(source_key=connector, base_url=base_url, max_pages=max_pages)
        debug = getattr(scraper, "last_discovery_debug", {})
        return scraper, docs, debug if isinstance(debug, dict) else {}

    if connector in {"treasury_featured_story", "treasury_press_release", "treasury_statement_remark"}:
        from treasury_news_scraper import TreasuryNewsScraper

        scraper = TreasuryNewsScraper()
        docs = scraper.discover_documents(source_key=connector, base_url=base_url, max_pages=max_pages)
        debug = getattr(scraper, "last_discovery_debug", {})
        return scraper, docs, debug if isinstance(debug, dict) else {}

    if connector == "sifma_news_item":
        from sifma_news_scraper import SIFMANewsScraper

        scraper = SIFMANewsScraper()
        docs = scraper.discover_documents(base_url=base_url, max_pages=max_pages)
        debug = getattr(scraper, "last_discovery_debug", {})
        return scraper, docs, debug if isinstance(debug, dict) else {}

    if connector == "congress_crs_product":
        from congress_crs_products_scraper import CongressCRSProductsScraper

        scraper = CongressCRSProductsScraper()
        docs = scraper.discover_documents(base_url=base_url, max_pages=max_pages)
        debug = getattr(scraper, "last_discovery_debug", {})
        return scraper, docs, debug if isinstance(debug, dict) else {}

    if connector == "senate_committee_site":
        from senate_committee_scraper import SenateCommitteeScraper

        scraper = SenateCommitteeScraper()
        docs = scraper.discover_documents(base_url=base_url, max_pages=max_pages)
        debug = getattr(scraper, "last_discovery_debug", {})
        return scraper, docs, debug if isinstance(debug, dict) else {}

    if connector in BLOOMBERG_CONNECTORS:
        from bloomberg_public_scraper import BloombergPublicNewsScraper

        scraper = BloombergPublicNewsScraper()
        docs = scraper.discover_documents(base_url=base_url, max_pages=max_pages)
        debug = getattr(scraper, "last_discovery_debug", {})
        return scraper, docs, debug if isinstance(debug, dict) else {}

    if connector == "substack_public_article":
        from substack_public_scraper import SubstackPublicScraper

        scraper = SubstackPublicScraper()
        docs = scraper.discover_documents(
            keywords=keywords,
            max_pages=max_pages,
            include_feeds=include_rss,
        )
        debug = getattr(scraper, "last_discovery_debug", {})
        return scraper, docs, debug if isinstance(debug, dict) else {}

    if connector in YOUTUBE_CONNECTORS:
        from youtube_video_scraper import YouTubeVideoScraper

        scraper = YouTubeVideoScraper()
        docs = scraper.discover_documents(
            channel_ref=base_url,
            max_pages=max_pages,
            limit=max(1, int(max_pages or 1) * 15),
        )
        debug = getattr(scraper, "last_discovery_debug", {})
        return scraper, docs, debug if isinstance(debug, dict) else {}

    if connector in TRADE_MEDIA_CONNECTORS:
        from trade_media_scraper import TRADE_MEDIA_SOURCES, TradeMediaScraper

        scraper = TradeMediaScraper()
        source_cfg = TRADE_MEDIA_SOURCES.get(connector, {})
        docs = scraper.discover_documents(
            source_key=connector,
            base_url=base_url or str(source_cfg.get("default_url", "") or ""),
            max_pages=max_pages,
            include_rss=include_rss,
            search_query=str(source_cfg.get("default_search_query", "") or ""),
        )
        debug = getattr(scraper, "last_discovery_debug", {})
        return scraper, docs, debug if isinstance(debug, dict) else {}

    if connector in TRADE_ASSOCIATION_CONNECTORS:
        from trade_association_scraper import TRADE_ASSOCIATION_SOURCES, TradeAssociationScraper

        scraper = TradeAssociationScraper()
        source_cfg = TRADE_ASSOCIATION_SOURCES.get(connector, {})
        docs = scraper.discover_documents(
            source_key=connector,
            base_url=base_url or str(source_cfg.get("default_url", "") or ""),
            max_pages=max_pages,
            include_rss=include_rss,
        )
        debug = getattr(scraper, "last_discovery_debug", {})
        return scraper, docs, debug if isinstance(debug, dict) else {}

    if connector == "wsj_dow_jones":
        from wsj_rss_scraper import WSJRssScraper

        scraper = WSJRssScraper()
        docs = scraper.discover_documents(feed_url=base_url, max_items=max(1, int(max_pages or 1)) * 25)
        for item in docs:
            item["url"] = str(item.get("source_url", "") or item.get("url", "")).strip()
        debug = getattr(scraper, "last_discovery_debug", {})
        return scraper, docs, debug if isinstance(debug, dict) else {}

    if connector == "reddit_post":
        import os
        from reddit_scraper import DEFAULT_SEARCH_TERMS, DEFAULT_TAGS, REDDIT_SEARCH_URL, RedditScraper

        cfg = connector_settings if isinstance(connector_settings, dict) else {}
        if cfg.get("enabled", True) is False:
            scraper = RedditScraper()
            return scraper, [], {"enabled": False, "reason": "Reddit connector disabled in settings."}

        search_terms = keywords or cfg.get("search_terms") or DEFAULT_SEARCH_TERMS
        subreddits = cfg.get("subreddits") or []
        scraper = RedditScraper(
            search_terms=[str(term) for term in search_terms if str(term or "").strip()],
            subreddits=[str(sub) for sub in subreddits if str(sub or "").strip()],
            sort=str(cfg.get("sort", "new") or "new"),
            time_filter=str(cfg.get("time_filter", "week") or "week"),
            limit_per_term=int(cfg.get("limit_per_term", 25) or 25),
            tags_csv=str(cfg.get("tags_csv", DEFAULT_TAGS) or DEFAULT_TAGS),
            client_id=str(cfg.get("client_id") or os.getenv("REDDIT_CLIENT_ID", "") or ""),
            client_secret=str(cfg.get("client_secret") or os.getenv("REDDIT_CLIENT_SECRET", "") or ""),
            user_agent=str(cfg.get("user_agent") or os.getenv("REDDIT_USER_AGENT", "") or "PolicyResearchHub/1.0"),
        )
        posts = scraper.discover_posts()
        records = scraper.build_documents(posts=posts)
        docs: List[Dict[str, Any]] = []
        for record in records:
            metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
            docs.append(
                {
                    "url": str(metadata.get("url", "") or "").strip(),
                    "title": str(metadata.get("title", "") or "").strip(),
                    "date": str(metadata.get("published_date", "") or metadata.get("date", "") or "").strip(),
                    "_record": record,
                }
            )
        return scraper, docs, {
            "enabled": True,
            "search_terms_count": len(search_terms),
            "subreddit_count": len(subreddits),
            "using_praw": bool(getattr(scraper, "using_praw", False)),
            "errors": list(getattr(scraper, "errors", []) or []),
            "source_index_url": REDDIT_SEARCH_URL,
        }

    if connector in SECURITIES_MARKET_CONNECTORS:
        from securities_market_sources_scraper import SecuritiesMarketSourcesScraper

        scraper = SecuritiesMarketSourcesScraper()
        docs = scraper.discover_documents(source_key=connector, base_url=base_url, max_pages=max_pages)
        debug = getattr(scraper, "last_discovery_debug", {})
        return scraper, docs, debug if isinstance(debug, dict) else {}

    if connector == "hedge_fund_letter":
        from hedge_fund_letter_scraper import HedgeFundLetterScraper

        scraper = HedgeFundLetterScraper()
        docs = scraper.discover_documents(base_url=base_url, max_pages=max_pages)
        debug = getattr(scraper, "last_discovery_debug", {})
        return scraper, docs, debug if isinstance(debug, dict) else {}

    raise RuntimeError(f"Unsupported connector: {connector}")


def _extract_record(connector: str, scraper: Any, entry: Dict[str, Any], idx: int, base_url: str) -> Dict[str, Any]:
    if connector == "sec_speech":
        extracted = scraper.extract_speech_for_analysis(
            entry.get("url", ""),
            listing_metadata=entry if isinstance(entry, dict) else None,
        )
        if not extracted.get("success"):
            raise RuntimeError(str(extracted.get("error", "") or "Extraction returned unsuccessful result."))

        data = extracted.get("data", {}) if isinstance(extracted.get("data", {}), dict) else {}
        if not scraper.validate_full_text_extraction(data):
            raise RuntimeError("Extracted text failed SEC speech quality validation.")

        data_meta = data.get("metadata", {}) if isinstance(data.get("metadata", {}), dict) else {}
        data_content = data.get("content", {}) if isinstance(data.get("content", {}), dict) else {}
        text = str(data_content.get("full_text", "") or "").strip()
        if len(text.split()) < 80:
            print("WARNING: Extracted text appears too short; retaining record.", file=sys.stderr)

        src_url = str(data_meta.get("url", "") or entry.get("url", "")).strip()
        source_name = _safe_source_name(src_url, f"sec-speech-{idx}", ".html")
        date_text = str(data_meta.get("date", "") or entry.get("date", "")).strip()
        doc_date = _parse_doc_date(date_text)
        speaker = str(data_meta.get("speaker", "") or entry.get("speaker", "")).strip() or "SEC Speaker"
        doc_type = str(entry.get("type", "") or data_meta.get("speech_type", "") or "Speech").strip() or "Speech"
        doc_type_lower = doc_type.lower()
        if "remarks" in doc_type_lower:
            tags_csv = "sec,remarks,speech,policy"
        elif "statement" in doc_type_lower:
            tags_csv = "sec,statement,policy"
        else:
            tags_csv = "sec,speech,policy"

        record = core._create_uploaded_document_record(
            text=text,
            organization="SEC",
            title=str(data_meta.get("title", "") or entry.get("title", "")).strip(),
            speaker=speaker,
            doc_date=doc_date,
            doc_type=doc_type,
            source_url=src_url,
            source_filename=source_name,
            source_ext=".html",
            source_local_path="",
            source_gcs_path="",
            tags_csv=tags_csv,
            source_kind="sec_speech",
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = "sec_speech"
        metadata["source_index_url"] = base_url
        metadata["published_date"] = date_text
        metadata["listing_speaker"] = str(entry.get("speaker", "") or "").strip()
        metadata["speech_type"] = doc_type
        return record

    if connector == "sec_tm_faq":
        extracted = scraper.extract_document(
            entry.get("url", ""),
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("updated_date") or entry.get("published_date") or "",
        )
        data = extracted.get("data", {})
        text = str(data.get("full_text", "") or "").strip()
        if len(text.split()) < 80:
            print("WARNING: Extracted text appears too short; retaining record.", file=sys.stderr)
        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        src_format = str(data.get("source_format", "") or entry.get("source_format", "html")).lower()
        source_ext = ".pdf" if src_format == "pdf" else ".html"
        source_name = _safe_source_name(src_url, f"tm-faq-{idx}", source_ext)
        doc_date = _parse_doc_date(data.get("date", "") or entry.get("updated_date") or entry.get("published_date") or "")

        record = core._create_uploaded_document_record(
            text=text,
            organization="SEC",
            title=str(data.get("title", "") or entry.get("title", "")).strip(),
            speaker="Division of Trading and Markets",
            doc_date=doc_date,
            doc_type="FAQ",
            source_url=src_url,
            source_filename=source_name,
            source_ext=source_ext,
            source_local_path="",
            source_gcs_path="",
            tags_csv="sec,trading-markets,faq,staff-guidance",
            source_kind="sec_tm_faq",
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = "sec_tm_faq"
        metadata["source_index_url"] = base_url
        metadata["published_date"] = str(entry.get("published_date", "") or "")
        metadata["updated_date"] = str(entry.get("updated_date", "") or "")
        metadata["last_reviewed_or_updated"] = str(data.get("last_reviewed_or_updated", "") or entry.get("updated_date", "") or "")
        return record

    if connector == "sec_rule_comment":
        entry_kind = _normalize_space(entry.get("entry_kind", "")).lower()
        if entry_kind == "rule":
            extracted = scraper.extract_rule(
                entry.get("url", ""),
                fallback_title=entry.get("title", ""),
                fallback_date=entry.get("date", ""),
                fallback_file_number=entry.get("file_number", ""),
                fallback_release_numbers=entry.get("release_numbers", []),
                fallback_rule_type=entry.get("rule_type", ""),
                fallback_comments_url=entry.get("comments_url", ""),
                fallback_pdf_url=entry.get("pdf_url", ""),
                fallback_effective_date=entry.get("effective_date", ""),
                fallback_sec_issue_date=entry.get("sec_issue_date", ""),
                fallback_federal_register_publish_date=entry.get("federal_register_publish_date", ""),
            )
            if not extracted.get("success"):
                raise RuntimeError("Extraction returned unsuccessful result.")
            data = extracted.get("data", {})
            text = str(data.get("full_text", "") or "").strip()
            if len(text.split()) < 80:
                raise RuntimeError("Extracted text appears too short; skipping.")
            src_url = str(data.get("url", "") or entry.get("url", "")).strip()
            source_format = str(data.get("source_format", "") or entry.get("source_format", "html")).strip().lower()
            source_ext = ".pdf" if source_format == "pdf" else ".html"
            date_text = str(data.get("date", "") or entry.get("date", "")).strip()
            rule_type = str(data.get("rule_type", "") or entry.get("rule_type", "") or "Rule Release").strip()
            file_number = str(data.get("file_number", "") or entry.get("file_number", "")).strip().upper()
            tags_csv = "sec,rulemaking,rule-release,public-comment"
            if file_number:
                tags_csv = f"{tags_csv},file-{file_number.lower()}"
            record = core._create_uploaded_document_record(
                text=text,
                organization="SEC",
                title=str(data.get("title", "") or entry.get("title", "")).strip() or "SEC Rule Release",
                speaker="SEC",
                doc_date=_parse_doc_date(date_text),
                doc_type=rule_type or "Rule Release",
                source_url=src_url,
                source_filename=_safe_source_name(src_url, f"sec-rule-release-{idx}", source_ext),
                source_ext=source_ext,
                source_local_path="",
                source_gcs_path="",
                tags_csv=tags_csv,
                source_kind="sec_rule_release",
            )
            metadata = record.setdefault("metadata", {})
            metadata["source_family"] = "sec_rule"
            metadata["source_index_url"] = base_url
            metadata["published_date"] = date_text
            metadata["file_number"] = file_number
            metadata["notice_number"] = file_number
            metadata["release_numbers"] = data.get("release_numbers", []) if isinstance(data.get("release_numbers", []), list) else []
            metadata["rule_type"] = rule_type
            metadata["sec_issue_date"] = str(data.get("sec_issue_date", "") or entry.get("sec_issue_date", "")).strip()
            metadata["effective_date"] = str(data.get("effective_date", "") or entry.get("effective_date", "")).strip()
            metadata["federal_register_publish_date"] = str(
                data.get("federal_register_publish_date", "") or entry.get("federal_register_publish_date", "")
            ).strip()
            metadata["rule_url"] = src_url
            metadata["notice_url"] = src_url
            metadata["comments_url"] = str(data.get("comments_url", "") or entry.get("comments_url", "")).strip()
            metadata["notice_title"] = str(data.get("title", "") or entry.get("title", "")).strip()
            metadata["pdf_url"] = str(data.get("pdf_url", "") or entry.get("pdf_url", "")).strip()
            metadata["source_format"] = source_format
            metadata["discovery_source"] = str(entry.get("discovery_source", "") or "").strip()
            return record

        extracted = scraper.extract_comment(
            entry.get("url", ""),
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("date", ""),
            fallback_commenter_name=entry.get("commenter_name", ""),
            fallback_file_number=entry.get("file_number", "") or entry.get("notice_number", ""),
            fallback_release_numbers=entry.get("release_numbers", []),
            fallback_rule_title=entry.get("rule_title", "") or entry.get("notice_title", ""),
            fallback_rule_url=entry.get("rule_url", "") or base_url,
            fallback_comments_url=entry.get("comments_url", ""),
            fallback_letter_type=entry.get("letter_type", ""),
        )
        if not extracted.get("success"):
            raise RuntimeError("Extraction returned unsuccessful result.")
        data = extracted.get("data", {})
        text = str(data.get("full_text", "") or "").strip()
        if len(text.split()) < 20:
            raise RuntimeError("Extracted text appears too short; skipping.")
        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        source_format = str(data.get("source_format", "") or entry.get("source_format", "html")).strip().lower()
        source_ext = ".pdf" if source_format == "pdf" else ".txt" if source_format == "txt" else ".html"
        date_text = str(data.get("date", "") or entry.get("date", "")).strip()
        commenter_name = str(data.get("commenter_name", "") or entry.get("commenter_name", "")).strip()
        commenter_org = str(data.get("commenter_org", "") or "").strip()
        file_number = str(data.get("file_number", "") or entry.get("file_number", "") or entry.get("notice_number", "")).strip().upper()
        tags_csv = "sec,rulemaking,public-comment"
        if file_number:
            tags_csv = f"{tags_csv},file-{file_number.lower()}"
        record = core._create_uploaded_document_record(
            text=text,
            organization="SEC",
            title=str(data.get("title", "") or entry.get("title", "")).strip() or "Public Comment",
            speaker=commenter_name or commenter_org or "Commenter",
            doc_date=_parse_doc_date(date_text),
            doc_type="Public Comment",
            source_url=src_url,
            source_filename=_safe_source_name(src_url, f"sec-rule-comment-{idx}", source_ext),
            source_ext=source_ext,
            source_local_path="",
            source_gcs_path="",
            tags_csv=tags_csv,
            source_kind="sec_rule_comment",
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = "sec_rule"
        metadata["source_index_url"] = str(data.get("rule_url", "") or entry.get("rule_url", "") or base_url).strip()
        metadata["published_date"] = date_text
        metadata["file_number"] = file_number
        metadata["notice_number"] = file_number
        metadata["release_numbers"] = data.get("release_numbers", []) if isinstance(data.get("release_numbers", []), list) else []
        metadata["rule_url"] = str(data.get("rule_url", "") or entry.get("rule_url", "") or base_url).strip()
        metadata["notice_url"] = metadata["rule_url"]
        metadata["comments_url"] = str(data.get("comments_url", "") or entry.get("comments_url", "")).strip()
        metadata["notice_title"] = str(data.get("rule_title", "") or entry.get("rule_title", "") or entry.get("notice_title", "")).strip()
        metadata["comment_url"] = str(data.get("comment_url", "") or entry.get("comment_url", "") or src_url).strip()
        metadata["pdf_url"] = str(data.get("pdf_url", "") or entry.get("pdf_url", "")).strip()
        metadata["commenter_name"] = commenter_name
        metadata["commenter_org"] = commenter_org
        metadata["letter_type"] = str(data.get("letter_type", "") or entry.get("letter_type", "")).strip()
        metadata["source_format"] = source_format
        metadata["discovery_source"] = str(entry.get("discovery_source", "") or "").strip()
        return record

    if connector == "sec_enforcement_litigation":
        extracted = scraper.extract_document(
            entry.get("url", ""),
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("date", ""),
            fallback_release_no=entry.get("release_no", ""),
        )
        data = extracted.get("data", {})
        text = str(data.get("full_text", "") or "").strip()
        if len(text.split()) < 80:
            print("WARNING: Extracted text appears too short; retaining record.", file=sys.stderr)
        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        source_name = _safe_source_name(src_url, f"litigation-release-{idx}", ".html")
        doc_date = _parse_doc_date(data.get("date", "") or entry.get("date", ""))

        record = core._create_uploaded_document_record(
            text=text,
            organization="SEC",
            title=str(data.get("title", "") or entry.get("title", "")).strip(),
            speaker="SEC Division of Enforcement",
            doc_date=doc_date,
            doc_type="Litigation Release",
            source_url=src_url,
            source_filename=source_name,
            source_ext=".html",
            source_local_path="",
            source_gcs_path="",
            tags_csv="sec,enforcement,litigation-release",
            source_kind="sec_enforcement_litigation",
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = "sec_enforcement_litigation"
        metadata["source_index_url"] = base_url
        metadata["release_no"] = str(data.get("release_no", "") or entry.get("release_no", "")).strip()
        metadata["published_date"] = str(entry.get("date", "") or "")
        inferred = core._infer_enforcement_metadata(
            title=metadata.get("title", ""),
            text=text,
            url=src_url,
            doc_type=metadata.get("doc_type", ""),
            source_kind=metadata.get("source_kind", ""),
            release_no=metadata.get("release_no", ""),
        )
        metadata["action_type"] = inferred.get("action_type", "unknown")
        metadata["forum"] = inferred.get("forum", "unknown")
        metadata["alleged_violations"] = inferred.get("alleged_violations", [])
        metadata["outcome_status"] = inferred.get("outcome_status", "unknown")
        metadata["respondents"] = inferred.get("respondents", [])
        metadata["entities"] = inferred.get("entities", [])
        return record

    if connector == "finra_regulatory_notice":
        extracted = scraper.extract_document(
            entry.get("url", ""),
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("date", ""),
            fallback_notice_number=entry.get("notice_number", ""),
            fallback_effective_date=entry.get("effective_date", ""),
            fallback_comment_deadline=entry.get("comment_deadline", ""),
        )
        data = extracted.get("data", {})
        text = str(data.get("full_text", "") or "").strip()
        if len(text.split()) < 80:
            print("WARNING: Extracted text appears too short; retaining record.", file=sys.stderr)
        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        source_name = _safe_source_name(src_url, f"finra-regulatory-notice-{idx}", ".html")
        doc_date = _parse_doc_date(data.get("date", "") or entry.get("date", ""))

        record = core._create_uploaded_document_record(
            text=text,
            organization="FINRA",
            title=str(data.get("title", "") or entry.get("title", "")).strip(),
            speaker="FINRA",
            doc_date=doc_date,
            doc_type="Regulatory Notice",
            source_url=src_url,
            source_filename=source_name,
            source_ext=".html",
            source_local_path="",
            source_gcs_path="",
            tags_csv="finra,regulatory-notice,rule-guidance,member-supervision",
            source_kind="finra_regulatory_notice",
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = "finra_regulatory_notice"
        metadata["source_index_url"] = base_url
        metadata["notice_type"] = "Regulatory Notice"
        metadata["notice_number"] = str(data.get("notice_number", "") or entry.get("notice_number", "")).strip()
        metadata["published_date"] = str(data.get("date", "") or entry.get("date", "")).strip()
        metadata["effective_date"] = str(data.get("effective_date", "") or entry.get("effective_date", "")).strip()
        metadata["comment_deadline"] = str(data.get("comment_deadline", "") or entry.get("comment_deadline", "")).strip()
        metadata["pdf_url"] = str(data.get("pdf_url", "") or "").strip()
        metadata["discovery_source"] = str(entry.get("discovery_source", "") or "").strip()
        return record

    if connector == "finra_awc":
        extracted = scraper.extract_document(
            entry.get("url", ""),
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("date", ""),
            fallback_case_id=entry.get("case_id", ""),
            fallback_subject=entry.get("subject_text", ""),
            fallback_case_summary=entry.get("case_summary", ""),
            fallback_sanctions=entry.get("sanctions_text", ""),
        )
        data = extracted.get("data", {})
        text = str(data.get("full_text", "") or "").strip()
        if len(text.split()) < 30:
            print("WARNING: Extracted AWC text appears too short; retaining record.", file=sys.stderr)
        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        pdf_url = str(data.get("pdf_url", "") or entry.get("pdf_url", "")).strip()
        source_ext = ".pdf" if pdf_url else ".html"
        source_name = _safe_source_name(src_url, f"finra-awc-{idx}", source_ext)
        doc_date = _parse_doc_date(data.get("date", "") or entry.get("date", ""))
        case_id = str(data.get("case_id", "") or entry.get("case_id", "")).strip()
        subject_text = str(data.get("subject_text", "") or entry.get("subject_text", "")).strip()
        sanctions_text = str(data.get("sanctions_text", "") or entry.get("sanctions_text", "")).strip()

        record = core._create_uploaded_document_record(
            text=text,
            organization="FINRA",
            title=str(data.get("title", "") or entry.get("title", "")).strip(),
            speaker="FINRA",
            doc_date=doc_date,
            doc_type="AWC",
            source_url=src_url,
            source_filename=source_name,
            source_ext=source_ext,
            source_local_path="",
            source_gcs_path="",
            tags_csv="finra,awc,enforcement,disciplinary-action,acceptance-waiver-consent",
            source_kind="finra_awc",
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = "finra_awc"
        metadata["source_index_url"] = base_url
        metadata["published_date"] = str(data.get("date", "") or entry.get("date", "")).strip()
        metadata["case_id"] = case_id
        metadata["subject_text"] = subject_text
        metadata["sanctions_text"] = sanctions_text
        metadata["pdf_url"] = pdf_url
        metadata["discovery_source"] = str(entry.get("discovery_source", "") or "").strip()
        return record

    if connector == "finra_comment_letter":
        extracted = scraper.extract_document(
            entry.get("url", ""),
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("date", ""),
            fallback_commenter_name=entry.get("commenter_name", ""),
            fallback_notice_number=entry.get("notice_number", ""),
            fallback_notice_title=entry.get("notice_title", ""),
            fallback_notice_url=entry.get("notice_url", ""),
        )
        data = extracted.get("data", {})
        text = str(data.get("full_text", "") or "").strip()
        if len(text.split()) < 20:
            print("WARNING: Extracted text appears too short; retaining record.", file=sys.stderr)

        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        source_format = str(data.get("source_format", "") or entry.get("source_format", "html")).strip().lower()
        source_ext = ".pdf" if source_format == "pdf" else ".html"
        source_name = _safe_source_name(src_url, f"finra-comment-letter-{idx}", source_ext)
        doc_date = _parse_doc_date(data.get("date", "") or entry.get("date", ""))
        commenter_name = str(data.get("commenter_name", "") or entry.get("commenter_name", "")).strip()
        commenter_org = str(data.get("commenter_org", "") or "").strip()

        tags = "finra,comment-letter,rule-guidance,public-comment"
        notice_number = str(data.get("notice_number", "") or entry.get("notice_number", "")).strip()
        if notice_number:
            tags = f"{tags},notice-{notice_number.lower()}"

        record = core._create_uploaded_document_record(
            text=text,
            organization="FINRA",
            title=str(data.get("title", "") or entry.get("title", "")).strip() or "Comment Letter",
            speaker=commenter_name or commenter_org or "Commenter",
            doc_date=doc_date,
            doc_type="Comment Letter",
            source_url=src_url,
            source_filename=source_name,
            source_ext=source_ext,
            source_local_path="",
            source_gcs_path="",
            tags_csv=tags,
            source_kind="finra_comment_letter",
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = "finra_comment_letter"
        metadata["source_index_url"] = str(data.get("notice_url", "") or entry.get("comments_url", "") or base_url).strip()
        metadata["notice_number"] = notice_number
        metadata["notice_title"] = str(data.get("notice_title", "") or entry.get("notice_title", "")).strip()
        metadata["notice_url"] = str(data.get("notice_url", "") or entry.get("notice_url", "")).strip()
        metadata["comment_url"] = str(data.get("comment_url", "") or src_url).strip()
        metadata["pdf_url"] = str(data.get("pdf_url", "") or (src_url if source_format == "pdf" else "")).strip()
        metadata["commenter_name"] = commenter_name
        metadata["commenter_org"] = commenter_org
        metadata["published_date"] = str(data.get("date", "") or entry.get("date", "")).strip()
        metadata["discovery_source"] = str(entry.get("discovery_source", "") or "").strip()
        return record

    if connector == "doj_usao_press_release":
        extracted = scraper.extract_document(
            entry.get("url", ""),
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("date", ""),
            fallback_office=entry.get("office", ""),
        )
        data = extracted.get("data", {})
        text = str(data.get("full_text", "") or "").strip()
        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        source_name = _safe_source_name(src_url, f"doj-press-release-{idx}", ".html")
        doc_date = _parse_doc_date(data.get("date", "") or entry.get("date", ""))
        office = str(data.get("office", "") or entry.get("office", "")).strip() or "U.S. Attorney's Office"
        title = str(data.get("title", "") or entry.get("title", "")).strip()
        short_text_fallback = len(text.split()) < 80
        if short_text_fallback:
            text = _build_short_text_fallback(
                title=title,
                url=src_url,
                date_text=str(data.get("date", "") or entry.get("date", "")).strip(),
                organization="DOJ",
                source_label=office,
                extracted_text=text,
            )

        record = core._create_uploaded_document_record(
            text=text,
            organization="DOJ",
            title=title,
            speaker=office,
            doc_date=doc_date,
            doc_type="Press Release",
            source_url=src_url,
            source_filename=source_name,
            source_ext=".html",
            source_local_path="",
            source_gcs_path="",
            tags_csv="doj,usao,press-release",
            source_kind="doj_usao_press_release",
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = "doj_usao_press_release"
        metadata["source_index_url"] = base_url
        metadata["office"] = office
        metadata["published_date"] = str(entry.get("date", "") or "")
        metadata["updated_date"] = str(data.get("updated_date", "") or "")
        if short_text_fallback:
            metadata["extraction_mode"] = "metadata_fallback"
            metadata["extraction_warnings"] = ["body_text_too_short"]
            metadata["body_word_count"] = int(data.get("word_count", 0) or 0)
        return record

    if connector == "federal_reserve_speech_testimony":
        extracted = scraper.extract_document(
            entry.get("url", ""),
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("date", ""),
            fallback_speaker=entry.get("speaker", ""),
        )
        data = extracted.get("data", {})
        text = str(data.get("full_text", "") or "").strip()
        if len(text.split()) < 80:
            print("WARNING: Extracted text appears too short; retaining record.", file=sys.stderr)
        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        source_name = _safe_source_name(src_url, f"federal-reserve-doc-{idx}", ".html")
        doc_date = _parse_doc_date(data.get("date", "") or entry.get("date", ""))
        speaker = str(data.get("speaker", "") or entry.get("speaker", "")).strip() or "Federal Reserve Board"
        doc_type = str(data.get("doc_type", "") or entry.get("doc_type", "")).strip() or "Speech"

        record = core._create_uploaded_document_record(
            text=text,
            organization="Federal Reserve",
            title=str(data.get("title", "") or entry.get("title", "")).strip(),
            speaker=speaker,
            doc_date=doc_date,
            doc_type=doc_type,
            source_url=src_url,
            source_filename=source_name,
            source_ext=".html",
            source_local_path="",
            source_gcs_path="",
            tags_csv="federal-reserve,speech,testimony,monetary-policy",
            source_kind="federal_reserve_speech_testimony",
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = "federal_reserve_speech_testimony"
        metadata["source_index_url"] = base_url
        metadata["published_date"] = str(data.get("date", "") or entry.get("date", "")).strip()
        metadata["location"] = str(data.get("location", "") or entry.get("location", "")).strip()
        return record

    if connector == "cftc_press_release":
        extracted = scraper.extract_document(
            entry.get("url", ""),
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("date", ""),
            fallback_doc_type="Press Release",
        )
        data = extracted.get("data", {})
        text = str(data.get("full_text", "") or "").strip()
        if len(text.split()) < 80:
            print("WARNING: Extracted text appears too short; retaining record.", file=sys.stderr)
        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        source_name = _safe_source_name(src_url, f"cftc-press-release-{idx}", ".html")
        doc_date = _parse_doc_date(data.get("date", "") or entry.get("date", ""))

        record = core._create_uploaded_document_record(
            text=text,
            organization="CFTC",
            title=str(data.get("title", "") or entry.get("title", "")).strip(),
            speaker="CFTC",
            doc_date=doc_date,
            doc_type="Press Release",
            source_url=src_url,
            source_filename=source_name,
            source_ext=".html",
            source_local_path="",
            source_gcs_path="",
            tags_csv="cftc,press-release,commodities-regulation,market-oversight",
            source_kind="cftc_press_release",
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = "cftc_press_release"
        metadata["source_index_url"] = base_url
        metadata["published_date"] = str(data.get("date", "") or entry.get("date", "")).strip()
        return record

    if connector == "cftc_public_statement_remark":
        extracted = scraper.extract_document(
            entry.get("url", ""),
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("date", ""),
            fallback_speaker=entry.get("speaker", ""),
            fallback_doc_type=entry.get("doc_type", ""),
        )
        data = extracted.get("data", {})
        text = str(data.get("full_text", "") or "").strip()
        if len(text.split()) < 80:
            print("WARNING: Extracted text appears too short; retaining record.", file=sys.stderr)
        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        source_name = _safe_source_name(src_url, f"cftc-statement-{idx}", ".html")
        doc_date = _parse_doc_date(data.get("date", "") or entry.get("date", ""))
        speaker = str(data.get("speaker", "") or entry.get("speaker", "")).strip() or "CFTC Official"
        doc_type = str(data.get("doc_type", "") or entry.get("doc_type", "")).strip() or "Statement"
        doc_type_lower = doc_type.lower()
        if "testimony" in doc_type_lower:
            tags_csv = "cftc,testimony,public-statement,market-regulation"
        elif "remark" in doc_type_lower or "speech" in doc_type_lower:
            tags_csv = "cftc,remarks,public-statement,market-regulation"
        else:
            tags_csv = "cftc,statement,public-statement,market-regulation"

        record = core._create_uploaded_document_record(
            text=text,
            organization="CFTC",
            title=str(data.get("title", "") or entry.get("title", "")).strip(),
            speaker=speaker,
            doc_date=doc_date,
            doc_type=doc_type,
            source_url=src_url,
            source_filename=source_name,
            source_ext=".html",
            source_local_path="",
            source_gcs_path="",
            tags_csv=tags_csv,
            source_kind="cftc_public_statement_remark",
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = "cftc_public_statement_remark"
        metadata["source_index_url"] = base_url
        metadata["published_date"] = str(data.get("date", "") or entry.get("date", "")).strip()
        metadata["location"] = str(data.get("location", "") or entry.get("location", "")).strip()
        return record

    if connector in {"treasury_featured_story", "treasury_press_release", "treasury_statement_remark"}:
        extracted = scraper.extract_document(
            entry.get("url", ""),
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("date", ""),
            fallback_speaker=entry.get("speaker", ""),
            fallback_doc_type=entry.get("doc_type", ""),
        )
        data = extracted.get("data", {})
        text = str(data.get("full_text", "") or "").strip()
        if len(text.split()) < 60:
            print("WARNING: Extracted text appears too short; retaining record.", file=sys.stderr)

        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        source_name = _safe_source_name(src_url, f"{connector}-{idx}", ".html")
        doc_date = _parse_doc_date(data.get("date", "") or entry.get("date", ""))
        speaker = str(data.get("speaker", "") or entry.get("speaker", "")).strip() or "Treasury"
        doc_type = str(data.get("doc_type", "") or entry.get("doc_type", "")).strip() or "Document"
        doc_type_lower = doc_type.lower()

        if connector == "treasury_featured_story":
            tags_csv = "treasury,featured-story,department-news,policy"
        elif connector == "treasury_press_release":
            tags_csv = "treasury,press-release,department-news"
        elif "testimony" in doc_type_lower:
            tags_csv = "treasury,testimony,statement,policy"
        elif "readout" in doc_type_lower:
            tags_csv = "treasury,readout,statement,policy"
        elif "remark" in doc_type_lower or "speech" in doc_type_lower:
            tags_csv = "treasury,remarks,statement,policy"
        else:
            tags_csv = "treasury,statement,policy"

        record = core._create_uploaded_document_record(
            text=text,
            organization="Treasury",
            title=str(data.get("title", "") or entry.get("title", "")).strip() or "Treasury Document",
            speaker=speaker,
            doc_date=doc_date,
            doc_type=doc_type,
            source_url=src_url,
            source_filename=source_name,
            source_ext=".html",
            source_local_path="",
            source_gcs_path="",
            tags_csv=tags_csv,
            source_kind=connector,
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = connector
        metadata["source_index_url"] = base_url
        metadata["published_date"] = str(data.get("date", "") or entry.get("date", "")).strip()
        metadata["source_format"] = str(data.get("source_format", "") or entry.get("source_format", "html")).strip()
        metadata["listing_page"] = str(entry.get("listing_page", "") or "").strip()
        return record

    if connector == "cisa_cybersecurity_advisory":
        extracted = scraper.extract_document(
            entry,
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("date", ""),
            fallback_doc_type=entry.get("doc_type", ""),
        )
        if not extracted.get("success"):
            raise RuntimeError(str(extracted.get("error", "CISA advisory extraction failed.")))
        data = extracted.get("data", {})
        text = str(data.get("full_text", "") or "").strip()
        title = str(data.get("title", "") or entry.get("title", "")).strip() or "CISA Advisory"
        date_text = str(data.get("date", "") or entry.get("date", "")).strip()
        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        doc_type = str(data.get("doc_type", "") or entry.get("doc_type", "")).strip() or "Cybersecurity Advisory"
        if len(text.split()) < 40:
            text = _build_short_text_fallback(
                title=title,
                url=src_url,
                date_text=date_text,
                organization="CISA",
                source_label="CISA",
                extracted_text=str(data.get("summary", "") or entry.get("summary", "") or text).strip(),
            )

        source_name = _safe_source_name(src_url, f"cisa-advisory-{idx}", ".html")
        alert_code = str(data.get("alert_code", "") or entry.get("alert_code", "")).strip()
        tags = ["cisa", "cybersecurity", "advisory", "alert", "critical-infrastructure"]
        if alert_code:
            tags.append(alert_code.lower())
        if "ics" in doc_type.lower():
            tags.append("industrial-control-systems")
        if "kev" in doc_type.lower():
            tags.append("known-exploited-vulnerability")

        record = core._create_uploaded_document_record(
            text=text,
            organization="CISA",
            title=title,
            speaker="Cybersecurity and Infrastructure Security Agency",
            doc_date=_parse_doc_date(date_text),
            doc_type=doc_type,
            source_url=src_url,
            source_filename=source_name,
            source_ext=".html",
            source_local_path="",
            source_gcs_path="",
            tags_csv=",".join(tags),
            source_kind="cisa_cybersecurity_advisory",
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = "cisa_cybersecurity_advisory"
        metadata["source_index_url"] = base_url
        metadata["published_date"] = date_text
        metadata["summary"] = str(data.get("summary", "") or entry.get("summary", "")).strip()
        metadata["alert_code"] = alert_code
        metadata["source_format"] = str(data.get("source_format", "") or entry.get("source_format", "html")).strip()
        metadata["listing_page"] = str(entry.get("listing_page", "") or "").strip()
        metadata["extraction_mode"] = str(data.get("extraction_mode", "") or "").strip()
        return record

    if connector == "sifma_news_item":
        extracted = scraper.extract_document(
            entry.get("url", ""),
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("date", ""),
            fallback_category=entry.get("category", ""),
            fallback_doc_type=entry.get("doc_type", ""),
        )
        data = extracted.get("data", {})
        text = str(data.get("full_text", "") or "").strip()
        if len(text.split()) < 40:
            print("WARNING: Extracted text appears too short; retaining record.", file=sys.stderr)

        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        source_name = _safe_source_name(src_url, f"sifma-news-{idx}", ".html")
        doc_date = _parse_doc_date(data.get("date", "") or entry.get("date", ""))
        category = str(data.get("category", "") or entry.get("category", "")).strip()
        doc_type = str(data.get("doc_type", "") or entry.get("doc_type", "")).strip() or "News Item"
        doc_type_lower = doc_type.lower()

        if "press release" in doc_type_lower:
            tags_csv = "sifma,press-release,association-news,capital-markets"
        elif "speech" in doc_type_lower:
            tags_csv = "sifma,speech,association-news,capital-markets"
        elif "podcast" in doc_type_lower:
            tags_csv = "sifma,podcast,association-news,capital-markets"
        elif "blog" in doc_type_lower:
            tags_csv = "sifma,blog,association-news,capital-markets"
        else:
            tags_csv = "sifma,news,association-news,capital-markets"

        record = core._create_uploaded_document_record(
            text=text,
            organization="SIFMA",
            title=str(data.get("title", "") or entry.get("title", "")).strip() or "SIFMA News Item",
            speaker="SIFMA",
            doc_date=doc_date,
            doc_type=doc_type,
            source_url=src_url,
            source_filename=source_name,
            source_ext=".html",
            source_local_path="",
            source_gcs_path="",
            tags_csv=tags_csv,
            source_kind="sifma_news_item",
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = "sifma_news_item"
        metadata["source_index_url"] = base_url
        metadata["published_date"] = str(data.get("date", "") or entry.get("date", "")).strip()
        metadata["category"] = category
        metadata["topics"] = str(entry.get("topics", "") or "").strip()
        metadata["listing_page"] = str(entry.get("listing_page", "") or "").strip()
        metadata["source_format"] = str(data.get("source_format", "") or entry.get("source_format", "html")).strip()
        return record

    if connector == "congress_crs_product":
        extracted = scraper.extract_document(
            entry.get("mirror_url", "") or entry.get("url", ""),
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("date", ""),
            fallback_doc_type=entry.get("doc_type", ""),
            fallback_authors=entry.get("authors", ""),
            fallback_product_number=entry.get("product_number", ""),
            canonical_url_override=entry.get("url", ""),
        )
        data = extracted.get("data", {})
        text = str(data.get("full_text", "") or "").strip()
        if len(text.split()) < 60:
            print("WARNING: Extracted text appears too short; retaining record.", file=sys.stderr)

        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        product_number = str(data.get("product_number", "") or entry.get("product_number", "")).strip().upper()
        source_name = _safe_source_name(src_url, product_number or f"congress-crs-{idx}", ".html")
        doc_date = _parse_doc_date(data.get("date", "") or entry.get("date", ""))
        authors = str(data.get("authors", "") or entry.get("authors", "")).strip()
        doc_type = str(data.get("doc_type", "") or entry.get("doc_type", "")).strip() or "CRS Product"
        topics = data.get("topics", entry.get("topics", []))
        if not isinstance(topics, list):
            topics = [str(topics or "").strip()] if str(topics or "").strip() else []
        topic_tags = []
        seen_topic_tags = set()
        for topic in topics:
            cleaned = re.sub(r"[^a-z0-9]+", "-", str(topic or "").strip().lower()).strip("-")
            if not cleaned or cleaned in seen_topic_tags:
                continue
            seen_topic_tags.add(cleaned)
            topic_tags.append(cleaned)
        tags_csv = ",".join(["crs", "congress", "library-of-congress", *topic_tags])

        record = core._create_uploaded_document_record(
            text=text,
            organization="Congressional Research Service",
            title=str(data.get("title", "") or entry.get("title", "")).strip() or (product_number or "CRS Product"),
            speaker=authors or "Congressional Research Service",
            doc_date=doc_date,
            doc_type=doc_type,
            source_url=src_url,
            source_filename=source_name,
            source_ext=".html",
            source_local_path="",
            source_gcs_path="",
            tags_csv=tags_csv,
            source_kind="congress_crs_product",
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = "congress_crs_product"
        metadata["source_index_url"] = base_url
        metadata["published_date"] = str(data.get("date", "") or entry.get("date", "")).strip()
        metadata["pdf_url"] = str(data.get("pdf_url", "") or entry.get("pdf_url", "")).strip()
        metadata["tags"] = tags_csv
        metadata["source_name"] = "Congress.gov"
        metadata["product_number"] = product_number
        metadata["crs_topics"] = "; ".join(str(topic or "").strip() for topic in topics if str(topic or "").strip())
        return record

    if connector == "senate_committee_site":
        extracted = scraper.extract_document(entry)
        if not extracted.get("success"):
            raise RuntimeError(str(extracted.get("error", "") or "Senate committee site extraction failed."))
        data = extracted.get("data", {}) if isinstance(extracted.get("data", {}), dict) else {}
        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        title = str(data.get("title", "") or entry.get("title", "")).strip() or "Senate Committee Item"
        date_text = str(data.get("date", "") or entry.get("date", "")).strip()
        text = str(data.get("full_text", "") or "").strip()
        source_label = str(entry.get("source_label", "") or "Senate Committee Site").strip()
        organization = str(entry.get("organization", "") or "Senate Committee").strip()
        if len(text.split()) < 40:
            text = _build_short_text_fallback(
                title=title,
                url=src_url,
                date_text=date_text,
                organization=organization,
                source_label=source_label,
                extracted_text=text or str(data.get("summary", "") or entry.get("summary", "")).strip(),
            )
        source_name = _safe_source_name(src_url, f"senate-committee-site-{idx}", ".html")
        tags_csv = str(entry.get("tags_csv", "") or "senate,congress,committee,press-release").strip()

        record = core._create_uploaded_document_record(
            text=text,
            organization=organization,
            title=title,
            speaker=source_label,
            doc_date=_parse_doc_date(date_text),
            doc_type=str(entry.get("doc_type", "") or "Press Release").strip() or "Press Release",
            source_url=src_url,
            source_filename=source_name,
            source_ext=".html",
            source_local_path="",
            source_gcs_path="",
            tags_csv=tags_csv,
            source_kind="senate_committee_site",
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = "senate_committee_site"
        metadata["source_index_url"] = base_url
        metadata["published_date"] = date_text
        metadata["summary"] = str(data.get("summary", "") or entry.get("summary", "")).strip()
        metadata["source_name"] = source_label
        metadata["source_label"] = source_label
        metadata["source_key"] = str(entry.get("source_key", "") or "").strip()
        metadata["source_format"] = "html"
        metadata["listing_page"] = str(entry.get("listing_page", "") or "").strip()
        metadata["extraction_mode"] = str(data.get("extraction_mode", "") or "senate_committee_html").strip()
        metadata["tags"] = tags_csv
        return record

    if connector in BLOOMBERG_CONNECTORS:
        src_url = str(entry.get("url", "") or "").strip()
        if not src_url:
            raise RuntimeError("Bloomberg discovery item did not include a URL.")
        authors = entry.get("authors") if isinstance(entry.get("authors"), list) else []
        author_text = ", ".join(str(author or "").strip() for author in authors if str(author or "").strip())
        extracted = scraper.extract_document(
            src_url,
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("date", ""),
            fallback_summary=entry.get("summary", ""),
            fallback_author=author_text or entry.get("author", ""),
        )
        if not extracted.get("success"):
            raise RuntimeError(str(extracted.get("error", "") or "Bloomberg public extraction failed."))
        data = extracted.get("data", {}) if isinstance(extracted.get("data", {}), dict) else {}
        normalized = {
            "url": data.get("url") or src_url,
            "title": data.get("title") or entry.get("title"),
            "date": data.get("date") or entry.get("date"),
            "authors": authors or ([data.get("author")] if data.get("author") else []),
            "keywords": entry.get("keywords") if isinstance(entry.get("keywords"), list) else [],
            "summary": data.get("summary") or entry.get("summary"),
            "full_text": data.get("full_text", ""),
            "source": data.get("source_name") or entry.get("source") or "Bloomberg",
            "extraction_mode": data.get("extraction_mode", ""),
            "access_limited": data.get("access_limited", False),
            "discovery_raw_item": entry.get("discovery_raw_item"),
            "raw_item": entry.get("raw_item"),
        }
        return _build_bloomberg_article_record(
            entry=normalized,
            scraper=scraper,
            idx=idx,
            base_url=base_url,
            source_kind="bloomberg_public_article",
        )

    if connector in SECURITIES_MARKET_CONNECTORS:
        extracted = scraper.extract_document(entry)
        if not extracted.get("success"):
            raise RuntimeError(str(extracted.get("error", "") or "Securities market source extraction failed."))
        data = extracted.get("data", {}) if isinstance(extracted.get("data", {}), dict) else {}
        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        title = str(data.get("title", "") or entry.get("title", "")).strip() or "Securities Market Source"
        date_text = str(data.get("date", "") or entry.get("date", "")).strip()
        text = str(data.get("full_text", "") or "").strip()
        if len(text.split()) < 40:
            text = _build_short_text_fallback(
                title=title,
                url=src_url,
                date_text=date_text,
                organization=str(entry.get("organization", "") or "").strip() or "Securities Market Source",
                source_label=str(entry.get("source_label", "") or connector).strip(),
                extracted_text=text or str(data.get("summary", "") or entry.get("summary", "")).strip(),
            )
        source_format = str(data.get("source_format", "") or entry.get("source_format", "html")).strip().lower()
        source_ext = ".pdf" if source_format == "pdf" else ".html"
        source_name = _safe_source_name(src_url, f"{connector}-{idx}", source_ext)
        organization = str(entry.get("organization", "") or "").strip() or "Securities Market Source"
        doc_type = str(entry.get("doc_type", "") or "Document").strip() or "Document"
        tags_csv = str(entry.get("tags_csv", "") or "securities-market,official-source").strip()

        record = core._create_uploaded_document_record(
            text=text,
            organization=organization,
            title=title,
            speaker=organization,
            doc_date=_parse_doc_date(date_text),
            doc_type=doc_type,
            source_url=src_url,
            source_filename=source_name,
            source_ext=source_ext,
            source_local_path="",
            source_gcs_path="",
            tags_csv=tags_csv,
            source_kind=connector,
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = connector
        metadata["source_index_url"] = base_url
        metadata["source_label"] = str(entry.get("source_label", "") or "").strip()
        metadata["published_date"] = date_text
        metadata["summary"] = str(data.get("summary", "") or entry.get("summary", "")).strip()
        metadata["source_format"] = source_format
        metadata["listing_page"] = str(entry.get("listing_page", "") or "").strip()
        metadata["extraction_mode"] = str(data.get("extraction_mode", "") or "").strip()
        metadata["pdf_url"] = src_url if source_format == "pdf" else ""
        return record

    if connector == "hedge_fund_letter":
        extracted = scraper.extract_document(entry)
        if not extracted.get("success"):
            raise RuntimeError(str(extracted.get("error", "") or "Hedge fund letter extraction failed."))
        data = extracted.get("data", {}) if isinstance(extracted.get("data", {}), dict) else {}
        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        title = str(data.get("title", "") or entry.get("title", "")).strip() or "Investor Letter"
        date_text = str(data.get("date", "") or entry.get("date", "")).strip()
        text = str(data.get("full_text", "") or "").strip()
        organization = str(entry.get("organization", "") or "Investor Letters").strip()
        source_label = str(data.get("source_label", "") or entry.get("source_label", "") or organization).strip()
        fund_name = str(data.get("fund_name", "") or entry.get("fund_name", "") or "").strip()
        if len(text.split()) < 40:
            text = _build_short_text_fallback(
                title=title,
                url=src_url,
                date_text=date_text,
                organization=organization,
                source_label=source_label,
                extracted_text=text or str(data.get("summary", "") or entry.get("summary", "")).strip(),
            )
        source_format = str(data.get("source_format", "") or entry.get("source_format", "html")).strip().lower()
        source_ext = ".pdf" if source_format == "pdf" else ".html"
        record = core._create_uploaded_document_record(
            text=text,
            organization=organization,
            title=title,
            speaker=fund_name or source_label,
            doc_date=_parse_doc_date(date_text),
            doc_type=str(entry.get("doc_type", "") or "Investor Letter").strip() or "Investor Letter",
            source_url=src_url,
            source_filename=_safe_source_name(src_url, f"hedge-fund-letter-{idx}", source_ext),
            source_ext=source_ext,
            source_local_path="",
            source_gcs_path="",
            tags_csv=str(entry.get("tags_csv", "") or "hedge-fund,investor-letter,market-commentary"),
            source_kind="hedge_fund_letter",
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = "hedge_fund_letter"
        metadata["source_index_url"] = base_url
        metadata["published_date"] = date_text
        metadata["summary"] = str(data.get("summary", "") or entry.get("summary", "")).strip()
        metadata["source_name"] = source_label
        metadata["source_label"] = source_label
        metadata["source_key"] = str(data.get("source_key", "") or entry.get("source_key", "")).strip()
        metadata["source_format"] = source_format
        metadata["listing_page"] = str(entry.get("listing_page", "") or "").strip()
        metadata["fund_name"] = fund_name
        metadata["extraction_mode"] = str(data.get("extraction_mode", "") or "").strip()
        metadata["pdf_url"] = src_url if source_format == "pdf" else ""
        metadata["connector_mode"] = "public"
        return record

    if connector == "substack_public_article":
        extracted = scraper.extract_document(entry)
        if not extracted.get("success"):
            raise RuntimeError(str(extracted.get("error", "") or "Substack public extraction failed."))
        data = extracted.get("data", {}) if isinstance(extracted.get("data", {}), dict) else {}
        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        title = str(data.get("title", "") or entry.get("title", "")).strip() or "Substack post"
        date_text = str(data.get("date", "") or entry.get("date", "")).strip()
        summary = str(data.get("summary", "") or entry.get("summary", "")).strip()
        text = str(data.get("full_text", "") or "").strip()
        discovery_mode = str(entry.get("discovery_mode", "") or "search").strip()
        if len(text.split()) < 50:
            text = _build_short_text_fallback(
                title=title,
                url=src_url,
                date_text=date_text,
                organization="Substack",
                source_label="Substack public feed" if "feed" in discovery_mode else "Substack public search",
                extracted_text=text or summary,
            )
        authors = data.get("authors") if isinstance(data.get("authors"), list) else []
        author_text = ", ".join(str(author or "").strip() for author in authors if str(author or "").strip())
        publication_name = str(data.get("publication_name", "") or entry.get("publication_name", "")).strip()
        post_tags = data.get("post_tags") if isinstance(data.get("post_tags"), list) else []
        matched_keywords = entry.get("matched_keywords") if isinstance(entry.get("matched_keywords"), list) else []
        feed_tags = entry.get("feed_tags") if isinstance(entry.get("feed_tags"), list) else []
        tags = ["substack", "financial-news", "public-feed", discovery_mode, *feed_tags, *matched_keywords, *post_tags[:8]]
        post_type = str(data.get("post_type", "") or entry.get("post_type", "newsletter")).strip().lower()
        doc_type = "Podcast" if post_type == "podcast" else "Article"
        source_name = _safe_source_name(src_url, f"substack-public-{idx}", ".html")
        record = core._create_uploaded_document_record(
            text=text,
            organization="Substack",
            title=title,
            speaker=author_text or publication_name or "Substack Author",
            doc_date=_parse_doc_date(date_text),
            doc_type=doc_type,
            source_url=src_url,
            source_filename=source_name,
            source_ext=".html",
            source_local_path="",
            source_gcs_path="",
            tags_csv=",".join(str(tag).strip() for tag in tags if str(tag).strip()),
            source_kind="substack_public_article",
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = "substack_public_article"
        metadata["source_index_url"] = base_url
        metadata["published_date"] = date_text
        metadata["summary"] = summary
        metadata["publication_name"] = publication_name
        metadata["authors"] = authors
        metadata["post_tags"] = post_tags
        metadata["feed_url"] = str(entry.get("feed_url", "") or "").strip()
        metadata["feed_tags"] = feed_tags
        metadata["discovery_mode"] = discovery_mode
        metadata["discovery_modes"] = entry.get("discovery_modes") if isinstance(entry.get("discovery_modes"), list) else []
        metadata["matched_keywords"] = matched_keywords
        metadata["matched_topic_keys"] = entry.get("matched_topic_keys") if isinstance(entry.get("matched_topic_keys"), list) else []
        metadata["matched_topic_labels"] = entry.get("matched_topic_labels") if isinstance(entry.get("matched_topic_labels"), list) else []
        metadata["matched_topic_keywords"] = entry.get("matched_topic_keywords") if isinstance(entry.get("matched_topic_keywords"), list) else []
        metadata["substack_post_id"] = entry.get("substack_post_id")
        metadata["audience"] = str(data.get("audience", "") or entry.get("audience", "")).strip()
        metadata["access_limited"] = bool(data.get("access_limited", False))
        metadata["reaction_count"] = int(data.get("reaction_count", 0) or 0)
        metadata["comment_count"] = int(data.get("comment_count", 0) or 0)
        metadata["relevance_classification"] = str(entry.get("relevance_classification", "")).strip()
        metadata["relevance_confidence"] = float(entry.get("relevance_confidence", 0.0) or 0.0)
        metadata["relevance_reason"] = str(entry.get("relevance_reason", "")).strip()
        metadata["connector_mode"] = "public"
        return record

    if connector in YOUTUBE_CONNECTORS:
        extracted = scraper.extract_document(
            entry.get("url", "") or entry.get("video_id", ""),
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("published_at", "") or entry.get("date", ""),
        )
        data = extracted.get("data", {}) if isinstance(extracted.get("data", {}), dict) else {}
        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        video_id = str(data.get("video_id", "") or entry.get("video_id", "")).strip()
        is_sec_youtube = connector == "sec_youtube_video"
        source_kind = "sec_youtube_video" if is_sec_youtube else "youtube_video"
        org_label = "SEC" if is_sec_youtube else "YouTube"
        speaker_label = "SEC" if is_sec_youtube else "YouTube"
        fallback_stem = "sec-youtube-video" if is_sec_youtube else "youtube-video"
        title = str(data.get("title", "") or entry.get("title", "")).strip() or (
            "SEC YouTube video" if is_sec_youtube else "YouTube video"
        )
        date_text = str(data.get("date", "") or entry.get("date", "")).strip()
        text = str(data.get("full_text", "") or "").strip()
        if len(text.split()) < 25:
            raise RuntimeError(f"YouTube transcript for {video_id or src_url} is too short.")
        record = core._create_uploaded_document_record(
            text=text,
            organization=org_label,
            title=title,
            speaker=speaker_label,
            doc_date=_parse_doc_date(date_text),
            doc_type="Video Transcript",
            source_url=src_url,
            source_filename=_safe_source_name(src_url or video_id, f"{fallback_stem}-{idx}", ".txt"),
            source_ext=".txt",
            source_local_path="",
            source_gcs_path="",
            tags_csv="sec,youtube,video,transcript,roundtable" if is_sec_youtube else "youtube,video,transcript",
            source_kind=source_kind,
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = source_kind
        metadata["source_index_url"] = base_url
        metadata["published_date"] = date_text
        metadata["published_at"] = str(entry.get("published_at", "") or data.get("published_at", "") or "").strip()
        metadata["youtube_video_id"] = video_id
        metadata["youtube_channel_id"] = str(entry.get("channel_id", "") or "").strip()
        metadata["youtube_url"] = src_url
        metadata["transcript_source"] = "youtube_transcript_api"
        metadata["discovery_source"] = str(entry.get("discovery_source", "") or "youtube_channel_rss").strip()
        metadata["connector_mode"] = "public"
        return record

    if connector in TRADE_MEDIA_CONNECTORS:
        from trade_media_scraper import TRADE_MEDIA_SOURCES

        cfg = TRADE_MEDIA_SOURCES.get(connector, {})
        source_label = str(cfg.get("label", "") or entry.get("source_name", "") or "Trade Media").strip()
        extracted = scraper.extract_document(
            entry.get("url", ""),
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("date", ""),
            fallback_description=entry.get("description", ""),
            fallback_source_name=source_label,
        )
        if not extracted.get("success"):
            raise RuntimeError(str(extracted.get("error", "") or "Trade media extraction failed."))
        data = extracted.get("data", {}) if isinstance(extracted.get("data", {}), dict) else {}
        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        title = str(data.get("title", "") or entry.get("title", "")).strip() or "Trade Media Article"
        date_text = str(data.get("date", "") or entry.get("date", "")).strip()
        text = str(data.get("full_text", "") or "").strip()
        if len(text.split()) < 40:
            text = _build_short_text_fallback(
                title=title,
                url=src_url,
                date_text=date_text,
                organization=str(cfg.get("organization", "") or source_label),
                source_label=source_label,
                extracted_text=text or str(data.get("description", "") or entry.get("description", "")).strip(),
            )
        source_format = str(data.get("source_format", "") or entry.get("source_format", "html")).strip().lower()
        source_ext = ".pdf" if source_format == "pdf" else ".html"
        record = core._create_uploaded_document_record(
            text=text,
            organization=str(cfg.get("organization", "") or source_label),
            title=title,
            speaker=str(data.get("source_name", "") or source_label),
            doc_date=_parse_doc_date(date_text),
            doc_type="Article",
            source_url=src_url,
            source_filename=_safe_source_name(src_url, f"{connector}-{idx}", source_ext),
            source_ext=source_ext,
            source_local_path="",
            source_gcs_path="",
            tags_csv=str(cfg.get("tags_csv", "") or "trade-media,financial-news"),
            source_kind=connector,
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = connector
        metadata["source_index_url"] = base_url
        metadata["published_date"] = date_text
        metadata["description"] = str(data.get("description", "") or entry.get("description", "")).strip()
        metadata["source_name"] = source_label
        metadata["source_format"] = source_format
        metadata["discovery_source"] = str(entry.get("discovery_source", "") or "").strip()
        metadata["listing_page"] = str(entry.get("listing_page", "") or "").strip()
        return record

    if connector in TRADE_ASSOCIATION_CONNECTORS:
        from trade_association_scraper import TRADE_ASSOCIATION_SOURCES

        cfg = TRADE_ASSOCIATION_SOURCES.get(connector, {})
        organization = str(cfg.get("organization", "") or entry.get("organization", "") or "Trade Association").strip()
        source_label = str(cfg.get("label", "") or entry.get("source_label", "") or organization).strip()
        extracted = scraper.extract_document(
            entry,
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("date", ""),
            fallback_description=entry.get("description", ""),
            fallback_source_name=source_label,
        )
        if not extracted.get("success"):
            raise RuntimeError(str(extracted.get("error", "") or "Trade association extraction failed."))
        data = extracted.get("data", {}) if isinstance(extracted.get("data", {}), dict) else {}
        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        title = str(data.get("title", "") or entry.get("title", "")).strip() or "Trade Association Item"
        date_text = str(data.get("date", "") or entry.get("date", "")).strip()
        text = str(data.get("full_text", "") or "").strip()
        if len(text.split()) < 40:
            text = _build_short_text_fallback(
                title=title,
                url=src_url,
                date_text=date_text,
                organization=organization,
                source_label=source_label,
                extracted_text=text or str(data.get("description", "") or entry.get("description", "")).strip(),
            )
        source_format = str(data.get("source_format", "") or entry.get("source_format", "html")).strip().lower()
        source_ext = ".pdf" if source_format == "pdf" else ".html"
        doc_type = str(entry.get("doc_type", "") or cfg.get("doc_type", "") or "News Item").strip() or "News Item"
        record = core._create_uploaded_document_record(
            text=text,
            organization=organization,
            title=title,
            speaker=organization,
            doc_date=_parse_doc_date(date_text),
            doc_type=doc_type,
            source_url=src_url,
            source_filename=_safe_source_name(src_url, f"{connector}-{idx}", source_ext),
            source_ext=source_ext,
            source_local_path="",
            source_gcs_path="",
            tags_csv=str(cfg.get("tags_csv", "") or entry.get("tags_csv", "") or "trade-association"),
            source_kind=connector,
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = connector
        metadata["source_index_url"] = base_url
        metadata["published_date"] = date_text
        metadata["description"] = str(data.get("description", "") or entry.get("description", "")).strip()
        metadata["source_name"] = source_label
        metadata["source_format"] = source_format
        metadata["discovery_source"] = str(entry.get("discovery_source", "") or "").strip()
        metadata["listing_page"] = str(entry.get("listing_page", "") or "").strip()
        metadata["extraction_mode"] = str(data.get("extraction_mode", "") or "").strip()
        return record

    if connector == "wsj_dow_jones":
        extracted = scraper.extract_document(
            entry.get("url", "") or entry.get("source_url", ""),
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("date", ""),
            fallback_description=entry.get("description", ""),
            fallback_author=entry.get("author", ""),
        )
        if not extracted.get("success"):
            raise RuntimeError(str(extracted.get("error", "") or "WSJ/Dow Jones extraction failed."))
        data = extracted.get("data", {}) if isinstance(extracted.get("data", {}), dict) else {}
        src_url = str(data.get("url", "") or entry.get("url", "") or entry.get("source_url", "")).strip()
        title = str(data.get("title", "") or entry.get("title", "")).strip() or "WSJ/Dow Jones Article"
        date_text = str(data.get("date", "") or entry.get("date", "")).strip()
        text = str(data.get("full_text", "") or "").strip()
        if len(text.split()) < 40:
            text = _build_short_text_fallback(
                title=title,
                url=src_url,
                date_text=date_text,
                organization="WSJ / Dow Jones",
                source_label="WSJ / Dow Jones RSS",
                extracted_text=text or str(entry.get("description", "") or "").strip(),
            )
        record = core._create_uploaded_document_record(
            text=text,
            organization="WSJ / Dow Jones",
            title=title,
            speaker=str(data.get("author", "") or entry.get("author", "") or "WSJ / Dow Jones").strip(),
            doc_date=_parse_doc_date(date_text),
            doc_type="Article",
            source_url=src_url,
            source_filename=_safe_source_name(src_url, f"wsj-dow-jones-{idx}", ".html"),
            source_ext=".html",
            source_local_path="",
            source_gcs_path="",
            tags_csv="wsj,dow-jones,business,financial-news",
            source_kind="wsj_dow_jones",
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = "wsj_dow_jones"
        metadata["source_index_url"] = base_url
        metadata["published_date"] = date_text
        metadata["summary"] = str(entry.get("description", "") or "").strip()
        metadata["guid"] = str(entry.get("guid", "") or "").strip()
        metadata["extraction_mode"] = str(data.get("extraction_mode", "") or "").strip()
        metadata["source_format"] = str(data.get("source_format", "") or "html").strip()
        return record

    if connector == "reddit_post":
        record = entry.get("_record")
        if not isinstance(record, dict):
            raise RuntimeError("Reddit discovery item did not include a built document record.")
        return record

    raise RuntimeError(f"Unsupported connector: {connector}")


def _run_connector_extraction(args: argparse.Namespace) -> Dict[str, Any]:
    if args.connector not in SUPPORTED_CONNECTORS:
        raise RuntimeError(f"Unsupported connector '{args.connector}'.")

    secrets_payload = core._load_streamlit_secrets()
    storage, gcs_status = core._get_gcs_storage(secrets_payload)
    if args.require_remote_persistence and storage is None:
        raise RuntimeError(gcs_status)

    base_url = str(args.base_url or "").strip() or _default_base_url(args.connector)
    if not base_url and args.connector not in BLOOMBERG_CONNECTORS:
        raise RuntimeError(f"No base URL configured for connector '{args.connector}'.")

    custom_payload = core._load_custom_documents(storage)
    existing_custom = _build_existing_custom_map(custom_payload)
    existing_custom_records = _build_existing_custom_record_map(custom_payload)
    existing_speech_keys = _load_existing_speech_url_keys(storage)
    topic_rules = _load_current_topic_rules() if args.connector == "substack_public_article" else []
    connector_settings: Optional[Dict[str, Any]] = None
    if args.connector == "reddit_post":
        settings = core._load_news_connector_settings(storage)
        reddit_settings = settings.get("reddit", {}) if isinstance(settings, dict) else {}
        connector_settings = reddit_settings if isinstance(reddit_settings, dict) else {}
    explicit_keywords = _parse_filter_terms(getattr(args, "keywords", ""))
    discovery_keywords = explicit_keywords or (
        _topic_rules_to_search_terms(
            topic_rules,
            max_terms=_substack_topic_search_term_limit(),
        )
        if args.connector == "substack_public_article"
        else []
    )

    scraper, discovered_raw, discovery_debug = _discover_connector(
        connector=args.connector,
        base_url=base_url,
        max_pages=max(1, int(args.max_pages)),
        include_pdfs=bool(args.include_pdfs),
        include_rss=bool(args.include_rss),
        keywords=discovery_keywords or None,
        connector_settings=connector_settings,
    )
    discovered = [item for item in discovered_raw if isinstance(item, dict)]
    if args.connector == "substack_public_article":
        for entry in discovered:
            _annotate_topic_matches(entry, topic_rules)
        discovery_debug["topic_rule_count"] = len(topic_rules)
        discovery_debug["topic_keywords_used"] = discovery_keywords
        discovery_debug["topic_keywords_source"] = "cli" if explicit_keywords else "rss_topic_rules"
        discovery_debug["topic_keywords_limit"] = 0 if explicit_keywords else _substack_topic_search_term_limit()
    if args.connector == "substack_public_article" and not discovered and discovery_debug.get("errors"):
        errors = "; ".join(str(item) for item in discovery_debug.get("errors", [])[:3])
        raise RuntimeError(f"Substack discovery failed: {errors}")
    exclude_terms = _parse_filter_terms(getattr(args, "exclude_terms", ""))
    excluded: List[Dict[str, Any]] = []
    filtered_discovered: List[Dict[str, Any]] = []
    if args.connector == "substack_public_article":
        provider = str(getattr(args, "relevance_provider", "") or "deepseek").strip().lower()
        if provider not in {"deepseek", "openai"}:
            provider = "deepseek"
        model = str(
            getattr(args, "relevance_model", "")
            or ("deepseek-v4-flash" if provider == "deepseek" else "gpt-5-mini")
        ).strip()
        client = core._get_model_client(secrets_payload, provider)
        filtered_discovered, excluded = scraper.filter_institutional_finance(
            discovered,
            client=client,
            model=model,
            provider=provider,
        )
        discovery_debug["relevance_provider"] = provider
        discovery_debug["relevance_model"] = model
        discovery_debug["relevance_included_count"] = len(filtered_discovered)
        discovery_debug["relevance_excluded_count"] = len(excluded)
    elif args.connector == "doj_usao_press_release" and exclude_terms:
        for entry in discovered:
            matched_terms = _match_filter_terms(
                [
                    entry.get("title", ""),
                    entry.get("teaser", ""),
                    entry.get("office", ""),
                    entry.get("url", ""),
                ],
                exclude_terms,
            )
            if matched_terms:
                skipped_entry = dict(entry)
                skipped_entry["exclude_matches"] = matched_terms
                excluded.append(skipped_entry)
            else:
                filtered_discovered.append(entry)
    else:
        filtered_discovered = list(discovered)

    status_counts = {"new": 0, "update_available": 0, "existing": 0, "existing_in_speeches": 0}
    for entry in filtered_discovered:
        key = core._url_match_key(entry.get("url", ""))
        existing_meta = existing_custom.get(key)
        status = _status_for_entry(args.connector, entry, existing_meta, existing_speech_keys)
        entry["ingest_status"] = status
        status_counts[status] = int(status_counts.get(status, 0)) + 1

    if args.selection == "all":
        candidates = list(filtered_discovered)
    else:
        candidates = [
            entry for entry in filtered_discovered if entry.get("ingest_status") in {"new", "update_available"}
        ]

    limit = len(candidates) if args.limit is None else max(0, int(args.limit))
    selected = candidates[:limit] if limit > 0 else []

    saved_new = 0
    saved_updates = 0
    failed: List[Dict[str, Any]] = []
    skipped_blocked: List[Dict[str, Any]] = []
    processed_doc_ids: List[str] = []
    duplicate_records_removed = 0
    legacy_bloomberg_records_removed = 0
    invalid_wired_records_removed = 0

    for idx, entry in enumerate(selected, 1):
        try:
            record = _extract_record(args.connector, scraper, entry, idx, base_url)
            metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
            doc_id = str(metadata.get("document_id", "") or "").strip()
            replaced = core._upsert_custom_document_record(custom_payload, record)
            if replaced:
                saved_updates += 1
            else:
                saved_new += 1
            if doc_id:
                processed_doc_ids.append(doc_id)
        except Exception as exc:
            key = core._url_match_key(entry.get("url", ""))
            repaired_doc_id = (
                _repair_existing_finra_notice_metadata(entry, existing_custom_records.get(key))
                if args.connector == "finra_regulatory_notice"
                else None
            )
            if repaired_doc_id:
                saved_updates += 1
                processed_doc_ids.append(repaired_doc_id)
            elif args.connector == "finra_regulatory_notice" and "403" in str(exc):
                skipped_blocked.append(
                    {
                        "url": str(entry.get("url", "") or ""),
                        "title": str(entry.get("title", "") or ""),
                        "reason": "FINRA blocked detail-page fetch and no existing record was available to repair.",
                    }
                )
            else:
                failed.append(
                    {
                        "url": str(entry.get("url", "") or ""),
                        "title": str(entry.get("title", "") or ""),
                        "error": str(exc),
                    }
                )

    if args.connector in BLOOMBERG_CONNECTORS and (saved_new or saved_updates):
        duplicate_records_removed = _remove_duplicate_bloomberg_records(custom_payload)
        legacy_bloomberg_records_removed = _remove_legacy_bloomberg_apify_records(custom_payload)

    if args.connector == "wired_article":
        invalid_wired_records_removed = _remove_invalid_wired_coupon_records(custom_payload)

    rule_summaries_rebuilt = False
    if not args.dry_run and (saved_new or saved_updates or invalid_wired_records_removed):
        core._save_custom_documents(storage, custom_payload, require_remote=args.require_remote_persistence)
        enrichment_state = core._load_enrichment_state(storage)
        core._rebuild_rule_summaries(
            storage,
            custom_payload=custom_payload,
            enrichment_state=enrichment_state,
            require_remote=args.require_remote_persistence,
        )
        rule_summaries_rebuilt = True

    summary = {
        "mode": "extract",
        "connector": args.connector,
        "ran_at": core._utc_now_iso(),
        "require_remote_persistence": bool(args.require_remote_persistence),
        "remote_persistence": bool(storage is not None),
        "base_url": base_url,
        "selection": args.selection,
        "max_pages": int(args.max_pages),
        "limit": limit,
        "include_pdfs": bool(args.include_pdfs),
        "include_rss": bool(args.include_rss),
        "exclude_terms": exclude_terms,
        "keywords": _parse_filter_terms(getattr(args, "keywords", "")),
        "discovered_count": len(discovered),
        "filtered_count": len(filtered_discovered),
        "excluded_count": len(excluded),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "processed_count": len(processed_doc_ids),
        "saved_new": saved_new,
        "saved_updates": saved_updates,
        "invalid_wired_records_removed": invalid_wired_records_removed,
        "failed_count": len(failed),
        "failed": failed[:25],
        "duplicate_records_removed": duplicate_records_removed,
        "legacy_bloomberg_records_removed": legacy_bloomberg_records_removed,
        "skipped_blocked_count": len(skipped_blocked),
        "skipped_blocked": skipped_blocked[:25],
        "excluded_preview": excluded[:25],
        "status_counts": status_counts,
        "discovery_debug": discovery_debug if isinstance(discovery_debug, dict) else {},
        "dry_run": bool(args.dry_run),
        "rule_summaries_rebuilt": rule_summaries_rebuilt,
    }
    core._write_summary(args.summary_path, summary)
    return summary


def _has_item_failures(summary: Dict[str, Any]) -> bool:
    failed_items = summary.get("failed")
    try:
        count = int(summary.get("failed_count", 0) or 0)
    except (TypeError, ValueError):
        count = 0
    return count > 0 or (isinstance(failed_items, list) and len(failed_items) > 0)


# Failure-rate threshold above which a run with partial success is still
# reported as a failed workflow run. Below this, item-level failures are
# surfaced as a warning instead of failing an otherwise-mostly-successful run
# (e.g. 1 failure out of 30 items no longer turns the whole job red).
_ITEM_FAILURE_RATE_THRESHOLD = 0.5


def _should_fail_for_item_failures(connector: str, summary: Dict[str, Any]) -> bool:
    if not _has_item_failures(summary):
        return False
    try:
        processed_count = int(summary.get("processed_count", 0) or 0)
    except (TypeError, ValueError):
        processed_count = 0
    if processed_count <= 0:
        return True
    try:
        failed_count = int(summary.get("failed_count", 0) or 0)
    except (TypeError, ValueError):
        failed_count = 0
    attempted = processed_count + failed_count
    if attempted <= 0:
        return True
    return (failed_count / attempted) > _ITEM_FAILURE_RATE_THRESHOLD


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Connector extraction pipeline")
    parser.add_argument("--connector", required=True, choices=sorted(SUPPORTED_CONNECTORS))
    parser.add_argument("--base-url", default="")
    parser.add_argument("--selection", choices=["new_or_updated", "all"], default="new_or_updated")
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--include-pdfs", default="")
    parser.add_argument("--include-rss", default="")
    parser.add_argument("--exclude-terms", default="")
    parser.add_argument("--keywords", default="")
    parser.add_argument("--relevance-provider", choices=["openai", "deepseek"], default=os.getenv("RELEVANCE_PROVIDER", "deepseek"))
    parser.add_argument("--relevance-model", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-remote-persistence", action="store_true")
    parser.add_argument("--summary-path", default="")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.base_url == "":
        args.base_url = _default_base_url(args.connector)

    include_pdfs_raw = str(getattr(args, "include_pdfs", "") or "").strip()
    include_rss_raw = str(getattr(args, "include_rss", "") or "").strip()

    if include_pdfs_raw == "":
        args.include_pdfs = args.connector in {"sec_tm_faq", "finra_comment_letter"}
    else:
        args.include_pdfs = _to_bool(include_pdfs_raw)

    if include_rss_raw == "":
        args.include_rss = args.connector in {
            "finra_regulatory_notice",
            "substack_public_article",
            *TRADE_MEDIA_CONNECTORS,
            *TRADE_ASSOCIATION_CONNECTORS,
        }
    else:
        args.include_rss = _to_bool(include_rss_raw)

    try:
        summary = _run_connector_extraction(args)
    except Exception as exc:
        payload = {
            "ok": False,
            "error": str(exc),
            "command": "extract",
            "connector": str(args.connector or ""),
            "ran_at": core._utc_now_iso(),
        }
        core._write_summary(getattr(args, "summary_path", ""), payload)
        record_source_health(payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1

    if _has_item_failures(summary):
        failure_message = f"{summary.get('failed_count', 0)} item-level extraction failure(s)."
        if _should_fail_for_item_failures(args.connector, summary):
            summary["ok"] = False
            summary["error"] = failure_message
            core._write_summary(getattr(args, "summary_path", ""), summary)
            record_source_health(summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 1
        summary["partial_failure"] = True
        summary["warning"] = failure_message

    summary["ok"] = True
    record_source_health(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
