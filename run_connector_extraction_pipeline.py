#!/usr/bin/env python3
"""Headless connector extraction pipeline for non-NewsAPI sources."""

from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import run_financial_news_pipeline as core


SEC_TM_FAQ_DEFAULT_URL = "https://www.sec.gov/rules-regulations/staff-guidance/trading-markets-frequently-asked-questions"
SEC_LIT_DEFAULT_URL = "https://www.sec.gov/enforcement-litigation/litigation-releases"
SEC_SPEECH_DEFAULT_URL = "https://www.sec.gov/newsroom/speeches-statements"
FINRA_NOTICE_DEFAULT_URL = "https://www.finra.org/rules-guidance/notices"
FINRA_TOPIC_DEFAULT_URL = "https://www.finra.org/rules-guidance/key-topics"
DOJ_DEFAULT_URL = "https://www.justice.gov/usao/pressreleases"
FED_DEFAULT_URL = "https://www.federalreserve.gov/newsevents/speeches-testimony.htm"
CFTC_PRESS_RELEASE_DEFAULT_URL = "https://www.cftc.gov/PressRoom/PressReleases"
CFTC_PUBLIC_STATEMENT_DEFAULT_URL = "https://www.cftc.gov/PressRoom/SpeechesTestimony/index.htm"
TREASURY_FEATURED_STORIES_DEFAULT_URL = "https://home.treasury.gov/news/featured-stories"
TREASURY_PRESS_RELEASES_DEFAULT_URL = "https://home.treasury.gov/news/press-releases"
TREASURY_STATEMENTS_REMARKS_DEFAULT_URL = "https://home.treasury.gov/news/press-releases/statements-remarks"
SIFMA_NEWS_DEFAULT_URL = "https://www.sifma.org/news"
CONGRESS_CRS_PRODUCTS_DEFAULT_URL = "https://www.congress.gov/crs-products"

SUPPORTED_CONNECTORS = {
    "sec_speech",
    "sec_tm_faq",
    "sec_enforcement_litigation",
    "finra_regulatory_notice",
    "finra_comment_letter",
    "finra_key_topic",
    "doj_usao_press_release",
    "federal_reserve_speech_testimony",
    "cftc_press_release",
    "cftc_public_statement_remark",
    "treasury_featured_story",
    "treasury_press_release",
    "treasury_statement_remark",
    "sifma_news_item",
    "congress_crs_product",
}


def _default_base_url(connector: str) -> str:
    if connector == "sec_speech":
        return SEC_SPEECH_DEFAULT_URL
    if connector == "sec_tm_faq":
        return SEC_TM_FAQ_DEFAULT_URL
    if connector == "sec_enforcement_litigation":
        return SEC_LIT_DEFAULT_URL
    if connector == "finra_regulatory_notice":
        return FINRA_NOTICE_DEFAULT_URL
    if connector == "finra_comment_letter":
        return ""
    if connector == "finra_key_topic":
        return FINRA_TOPIC_DEFAULT_URL
    if connector == "doj_usao_press_release":
        return DOJ_DEFAULT_URL
    if connector == "federal_reserve_speech_testimony":
        return FED_DEFAULT_URL
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
    return ""


def _normalize_space(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


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
        except Exception:
            pass

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
        except Exception:
            pass

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
        existing_date = _normalize_space(existing_meta.get("published_date") or existing_meta.get("date") or "")
        incoming_date = _normalize_space(entry.get("date", ""))
        existing_title = _normalize_space(existing_meta.get("title", ""))
        incoming_title = _normalize_space(entry.get("title", ""))
        existing_author = _normalize_space(existing_meta.get("speaker", ""))
        incoming_author = _normalize_space(entry.get("authors", ""))
        existing_doc_type = _normalize_space(existing_meta.get("doc_type", ""))
        incoming_doc_type = _normalize_space(entry.get("doc_type", ""))
        if (
            (incoming_date and existing_date and incoming_date != existing_date)
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

    existing_date = _normalize_space(existing_meta.get("published_date") or existing_meta.get("date") or "")
    incoming_date = _normalize_space(entry.get("date") or entry.get("published_date") or "")
    if incoming_date and existing_date and incoming_date != existing_date:
        return "update_available"
    return "existing"


def _discover_connector(connector: str, base_url: str, max_pages: int, include_pdfs: bool, include_rss: bool) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
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

    if connector == "finra_comment_letter":
        from finra_comment_letter_scraper import FINRACommentLetterScraper

        scraper = FINRACommentLetterScraper()
        docs = scraper.discover_documents(notice_url=base_url, include_pdfs=include_pdfs)
        return scraper, docs, {}

    if connector == "finra_key_topic":
        from finra_key_topics_scraper import FINRAKeyTopicsScraper

        scraper = FINRAKeyTopicsScraper()
        docs = scraper.discover_documents(index_url=base_url)
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
            raise RuntimeError("Extracted text appears too short.")

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
            raise RuntimeError("Extracted text appears too short.")
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
            raise RuntimeError("Extracted text appears too short.")
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
            raise RuntimeError("Extracted text appears too short.")
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
            raise RuntimeError("Extracted text appears too short.")

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

    if connector == "finra_key_topic":
        extracted = scraper.extract_document(
            entry.get("url", ""),
            fallback_title=entry.get("topic_name", "") or entry.get("title", ""),
        )
        data = extracted.get("data", {})
        text = str(data.get("full_text", "") or "").strip()
        if len(text.split()) < 20:
            raise RuntimeError("Extracted text appears too short.")
        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        source_name = _safe_source_name(src_url, f"finra-key-topic-{idx}", ".html")

        record = core._create_uploaded_document_record(
            text=text,
            organization="FINRA",
            title=str(data.get("topic_name", "") or entry.get("topic_name", "") or entry.get("title", "")).strip(),
            speaker="FINRA",
            doc_date="",
            doc_type="Key Topic",
            source_url=src_url,
            source_filename=source_name,
            source_ext=".html",
            source_local_path="",
            source_gcs_path="",
            tags_csv="finra,key-topic,rule-guidance,taxonomy",
            source_kind="finra_key_topic",
        )
        metadata = record.setdefault("metadata", {})
        metadata["source_family"] = "finra_key_topic"
        metadata["source_index_url"] = base_url
        metadata["topic_name"] = str(data.get("topic_name", "") or entry.get("topic_name", "")).strip()
        metadata["topic_slug"] = str(data.get("topic_slug", "") or entry.get("topic_slug", "")).strip().lower()
        metadata["section_names"] = data.get("section_names", []) if isinstance(data.get("section_names", []), list) else []
        metadata["overview_text"] = str(data.get("overview_text", "") or "").strip()
        metadata["ogc_contacts"] = data.get("ogc_contacts", []) if isinstance(data.get("ogc_contacts", []), list) else []
        metadata["linked_notices"] = data.get("linked_notices", []) if isinstance(data.get("linked_notices", []), list) else []
        metadata["linked_guidance"] = data.get("linked_guidance", []) if isinstance(data.get("linked_guidance", []), list) else []
        metadata["linked_rules"] = data.get("linked_rules", []) if isinstance(data.get("linked_rules", []), list) else []
        metadata["linked_news"] = data.get("linked_news", []) if isinstance(data.get("linked_news", []), list) else []
        metadata["linked_investor_education"] = data.get("linked_investor_education", []) if isinstance(data.get("linked_investor_education", []), list) else []
        metadata["linked_resources"] = data.get("linked_resources", []) if isinstance(data.get("linked_resources", []), list) else []
        metadata["section_links"] = data.get("section_links", {}) if isinstance(data.get("section_links", {}), dict) else {}
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
        if len(text.split()) < 80:
            raise RuntimeError("Extracted text appears too short.")
        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
        source_name = _safe_source_name(src_url, f"doj-press-release-{idx}", ".html")
        doc_date = _parse_doc_date(data.get("date", "") or entry.get("date", ""))
        office = str(data.get("office", "") or entry.get("office", "")).strip() or "U.S. Attorney's Office"

        record = core._create_uploaded_document_record(
            text=text,
            organization="DOJ",
            title=str(data.get("title", "") or entry.get("title", "")).strip(),
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
            raise RuntimeError("Extracted text appears too short.")
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
            raise RuntimeError("Extracted text appears too short.")
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
            raise RuntimeError("Extracted text appears too short.")
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
            raise RuntimeError("Extracted text appears too short.")

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
            raise RuntimeError("Extracted text appears too short.")

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
            entry.get("url", ""),
            fallback_title=entry.get("title", ""),
            fallback_date=entry.get("date", ""),
            fallback_doc_type=entry.get("doc_type", ""),
            fallback_authors=entry.get("authors", ""),
            fallback_product_number=entry.get("product_number", ""),
        )
        data = extracted.get("data", {})
        text = str(data.get("full_text", "") or "").strip()
        if len(text.split()) < 60:
            raise RuntimeError("Extracted text appears too short.")

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

    raise RuntimeError(f"Unsupported connector: {connector}")


def _run_connector_extraction(args: argparse.Namespace) -> Dict[str, Any]:
    if args.connector not in SUPPORTED_CONNECTORS:
        raise RuntimeError(f"Unsupported connector '{args.connector}'.")

    secrets_payload = core._load_streamlit_secrets()
    storage, gcs_status = core._get_gcs_storage(secrets_payload)
    if args.require_remote_persistence and storage is None:
        raise RuntimeError(gcs_status)

    base_url = str(args.base_url or "").strip() or _default_base_url(args.connector)
    if not base_url:
        raise RuntimeError(f"No base URL configured for connector '{args.connector}'.")

    custom_payload = core._load_custom_documents(storage)
    existing_custom = _build_existing_custom_map(custom_payload)
    existing_speech_keys = _load_existing_speech_url_keys(storage)

    scraper, discovered_raw, discovery_debug = _discover_connector(
        connector=args.connector,
        base_url=base_url,
        max_pages=max(1, int(args.max_pages)),
        include_pdfs=bool(args.include_pdfs),
        include_rss=bool(args.include_rss),
    )
    discovered = [item for item in discovered_raw if isinstance(item, dict)]
    exclude_terms = _parse_filter_terms(getattr(args, "exclude_terms", ""))
    excluded: List[Dict[str, Any]] = []
    filtered_discovered: List[Dict[str, Any]] = []
    if args.connector == "doj_usao_press_release" and exclude_terms:
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
    processed_doc_ids: List[str] = []

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
            failed.append(
                {
                    "url": str(entry.get("url", "") or ""),
                    "title": str(entry.get("title", "") or ""),
                    "error": str(exc),
                }
            )

    if not args.dry_run and (saved_new or saved_updates):
        core._save_custom_documents(storage, custom_payload, require_remote=args.require_remote_persistence)

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
        "discovered_count": len(discovered),
        "filtered_count": len(filtered_discovered),
        "excluded_count": len(excluded),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "processed_count": len(processed_doc_ids),
        "saved_new": saved_new,
        "saved_updates": saved_updates,
        "failed_count": len(failed),
        "failed": failed[:25],
        "excluded_preview": excluded[:25],
        "status_counts": status_counts,
        "discovery_debug": discovery_debug if isinstance(discovery_debug, dict) else {},
        "dry_run": bool(args.dry_run),
    }
    core._write_summary(args.summary_path, summary)
    return summary


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
        args.include_rss = args.connector == "finra_regulatory_notice"
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
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1

    summary["ok"] = True
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
