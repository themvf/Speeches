
#!/usr/bin/env python3
"""Headless financial news ingest and enrichment pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

try:
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None

from gcs_storage import GCSStorage
from newsapi_financial_scraper import NewsAPIFinancialScraper

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
STREAMLIT_SECRETS_PATH = ROOT / ".streamlit" / "secrets.toml"

CUSTOM_DOCS_BLOB_NAME = "custom_documents.json"
CUSTOM_DOCS_LOCAL_PATH = DATA_DIR / "custom_documents.json"
NEWS_CONNECTOR_SETTINGS_BLOB_NAME = "news_connector_settings.json"
NEWS_CONNECTOR_SETTINGS_LOCAL_PATH = DATA_DIR / "news_connector_settings.json"
ENRICHMENT_STATE_BLOB_NAME = "document_enrichment_state.json"
ENRICHMENT_STATE_LOCAL_PATH = DATA_DIR / "document_enrichment_state.json"
ENRICHMENT_PIPELINE_VERSION = "v1"

NEWSAPI_DEFAULT_QUERY = (
    '("securities fraud" OR "wire fraud" OR "market manipulation" OR "insider trading" OR '
    '"ponzi" OR "crypto enforcement" OR "stablecoin regulation" OR "money laundering" OR AML) '
    "AND (SEC OR DOJ OR CFTC OR Treasury OR FinCEN OR Congress)"
)
NEWSAPI_DEFAULT_DOMAINS = (
    "reuters.com,wsj.com,bloomberg.com,ft.com,cnbc.com,apnews.com,marketwatch.com,coindesk.com"
)
NEWSAPI_DEFAULT_TAGS = "news,financial-regulation,fraud,crypto,securities"


def _stderr(message: str):
    print(str(message), file=sys.stderr)


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _coerce_int(value: Any, default: int = 0, min_value: int = 0) -> int:
    try:
        num = int(value)
    except Exception:
        num = int(default)
    return max(int(min_value), num)


def _safe_filename(name: str) -> str:
    raw = str(name or "document").strip()
    raw = raw.replace("\\", "_").replace("/", "_")
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".", " ") else "_" for ch in raw).strip()
    return cleaned or "document"


def _normalize_org_label(value: Any) -> str:
    label = str(value).strip() if value is not None else ""
    return label or "SEC"


def _org_key_from_label(label: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(label))
    cleaned = cleaned.strip("_")
    return cleaned or "sec"


def _url_match_key(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
        scheme = (parsed.scheme or "https").lower()
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/") or "/"
        return f"{scheme}://{netloc}{path}"
    except Exception:
        return raw.rstrip("/")


def _parse_date_text(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).replace(tzinfo=None)
    except Exception:
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
    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%B %d, %Y",
        "%b %d, %Y",
        "%m/%d/%Y",
        "%m/%d/%y",
    ):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=None)
        except ValueError:
            continue
    try:
        parsed = parsedate_to_datetime(text)
        if parsed is not None:
            return parsed.replace(tzinfo=None)
    except Exception:
        pass
    return None


def _parse_single_date(value: Any) -> Optional[date]:
    parsed = _parse_date_text(value)
    return parsed.date() if parsed is not None else None


def _load_streamlit_secrets() -> Dict[str, Any]:
    if tomllib is None or not STREAMLIT_SECRETS_PATH.exists():
        return {}
    try:
        with open(STREAMLIT_SECRETS_PATH, "rb") as f:
            payload = tomllib.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _nested_get(payload: Dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _get_newsapi_api_key(secrets_payload: Dict[str, Any]) -> str:
    return str(
        os.getenv("NEWSAPI_API_KEY", "")
        or _nested_get(secrets_payload, "newsapi", "api_key")
        or ""
    ).strip()


def _get_openai_api_key(secrets_payload: Dict[str, Any]) -> str:
    return str(
        os.getenv("OPENAI_API_KEY", "")
        or _nested_get(secrets_payload, "openai", "api_key")
        or ""
    ).strip()


def _get_gcs_storage(secrets_payload: Dict[str, Any]) -> Tuple[Optional[GCSStorage], str]:
    bucket_name = str(
        os.getenv("GCS_BUCKET_NAME", "")
        or _nested_get(secrets_payload, "gcs", "bucket_name")
        or ""
    ).strip()
    credentials_info = None

    raw_json = str(os.getenv("GCS_CREDENTIALS_JSON", "") or "").strip()
    if raw_json:
        try:
            credentials_info = json.loads(raw_json)
        except Exception as e:
            return None, f"Invalid GCS_CREDENTIALS_JSON: {e}"
    else:
        credentials_path = str(
            os.getenv("GCS_CREDENTIALS_PATH", "")
            or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
            or ""
        ).strip()
        if credentials_path:
            try:
                with open(credentials_path, "r", encoding="utf-8") as f:
                    credentials_info = json.load(f)
            except Exception as e:
                return None, f"Failed to read GCS credentials file: {e}"
        else:
            secret_gcs = _nested_get(secrets_payload, "gcs")
            if isinstance(secret_gcs, dict) and bucket_name:
                credentials_info = {k: v for k, v in secret_gcs.items() if k != "bucket_name"}

    if not bucket_name or not isinstance(credentials_info, dict) or not credentials_info:
        return None, "GCS bucket/credentials not configured"

    try:
        return GCSStorage(bucket_name, credentials_info), ""
    except Exception as e:
        return None, f"Failed to initialize GCS storage: {e}"


def _load_json_store(
    storage: Optional[GCSStorage],
    blob_name: str,
    local_path: Path,
    default_factory,
    normalize_fn,
) -> Dict[str, Any]:
    if storage is not None:
        try:
            blob = storage.bucket.blob(blob_name)
            if blob.exists():
                payload = normalize_fn(json.loads(blob.download_as_text(encoding="utf-8")))
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                return payload
        except Exception as e:
            _stderr(f"Remote load failed for {blob_name}: {e}")
    if local_path.exists():
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                return normalize_fn(json.load(f))
        except Exception:
            pass
    return normalize_fn(default_factory())


def _save_json_store(
    storage: Optional[GCSStorage],
    blob_name: str,
    local_path: Path,
    payload: Dict[str, Any],
    normalize_fn,
    require_remote: bool = False,
) -> None:
    normalized = normalize_fn(payload)
    normalized["updated_at"] = _utc_now_iso()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)

    if storage is None:
        if require_remote:
            raise RuntimeError(f"Remote persistence required for {blob_name}, but GCS is not configured.")
        return
    blob = storage.bucket.blob(blob_name)
    blob.upload_from_string(
        json.dumps(normalized, indent=2, ensure_ascii=False),
        content_type="application/json",
    )


def _empty_custom_docs_payload() -> Dict[str, Any]:
    return {"updated_at": "", "documents": []}


def _normalize_custom_docs_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    docs = payload.get("documents", [])
    if not isinstance(docs, list):
        docs = []
    return {"updated_at": str(payload.get("updated_at", "") or ""), "documents": docs}


def _load_custom_documents(storage: Optional[GCSStorage]) -> Dict[str, Any]:
    return _load_json_store(
        storage=storage,
        blob_name=CUSTOM_DOCS_BLOB_NAME,
        local_path=CUSTOM_DOCS_LOCAL_PATH,
        default_factory=_empty_custom_docs_payload,
        normalize_fn=_normalize_custom_docs_payload,
    )


def _save_custom_documents(
    storage: Optional[GCSStorage],
    payload: Dict[str, Any],
    require_remote: bool = False,
) -> None:
    _save_json_store(
        storage=storage,
        blob_name=CUSTOM_DOCS_BLOB_NAME,
        local_path=CUSTOM_DOCS_LOCAL_PATH,
        payload=payload,
        normalize_fn=_normalize_custom_docs_payload,
        require_remote=require_remote,
    )


def _empty_news_connector_settings() -> Dict[str, Any]:
    return {
        "updated_at": "",
        "query": NEWSAPI_DEFAULT_QUERY,
        "lookback_days": 3,
        "max_pages": 1,
        "page_size": 50,
        "target_count": 100,
        "sort_by": "publishedAt",
        "organization_label": "Financial News",
        "domains": NEWSAPI_DEFAULT_DOMAINS,
        "exclude_domains": "",
        "tags_csv": NEWSAPI_DEFAULT_TAGS,
    }


def _normalize_news_connector_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    base = _empty_news_connector_settings()
    if not isinstance(payload, dict):
        payload = {}

    def _clamp_int(value: Any, default_value: int, min_value: int, max_value: int) -> int:
        try:
            num = int(value)
        except Exception:
            num = int(default_value)
        return max(int(min_value), min(int(max_value), num))

    sort_by = str(payload.get("sort_by", base["sort_by"]) or base["sort_by"]).strip()
    if sort_by not in {"publishedAt", "relevancy", "popularity"}:
        sort_by = base["sort_by"]

    return {
        "updated_at": str(payload.get("updated_at", "") or ""),
        "query": str(payload.get("query", base["query"]) or "").strip() or base["query"],
        "lookback_days": _clamp_int(
            payload.get("lookback_days", base["lookback_days"]),
            base["lookback_days"],
            1,
            30,
        ),
        "max_pages": _clamp_int(payload.get("max_pages", base["max_pages"]), base["max_pages"], 1, 10),
        "page_size": _clamp_int(payload.get("page_size", base["page_size"]), base["page_size"], 10, 100),
        "target_count": _clamp_int(
            payload.get("target_count", base["target_count"]),
            base["target_count"],
            10,
            500,
        ),
        "sort_by": sort_by,
        "organization_label": str(
            payload.get("organization_label", base["organization_label"]) or ""
        ).strip() or base["organization_label"],
        "domains": str(payload.get("domains", base["domains"]) or "").strip(),
        "exclude_domains": str(payload.get("exclude_domains", base["exclude_domains"]) or "").strip(),
        "tags_csv": str(payload.get("tags_csv", base["tags_csv"]) or "").strip() or base["tags_csv"],
    }


def _load_news_connector_settings(storage: Optional[GCSStorage]) -> Dict[str, Any]:
    return _load_json_store(
        storage=storage,
        blob_name=NEWS_CONNECTOR_SETTINGS_BLOB_NAME,
        local_path=NEWS_CONNECTOR_SETTINGS_LOCAL_PATH,
        default_factory=_empty_news_connector_settings,
        normalize_fn=_normalize_news_connector_settings,
    )


def _empty_enrichment_state() -> Dict[str, Any]:
    return {
        "version": 1,
        "pipeline_version": ENRICHMENT_PIPELINE_VERSION,
        "updated_at": "",
        "entries": {},
    }


def _normalize_enrichment_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return _empty_enrichment_state()
    entries = payload.get("entries", {})
    if not isinstance(entries, dict):
        entries = {}
    return {
        "version": int(payload.get("version", 1) or 1),
        "pipeline_version": str(
            payload.get("pipeline_version", ENRICHMENT_PIPELINE_VERSION) or ENRICHMENT_PIPELINE_VERSION
        ),
        "updated_at": str(payload.get("updated_at", "") or ""),
        "entries": entries,
    }


def _load_enrichment_state(storage: Optional[GCSStorage]) -> Dict[str, Any]:
    return _load_json_store(
        storage=storage,
        blob_name=ENRICHMENT_STATE_BLOB_NAME,
        local_path=ENRICHMENT_STATE_LOCAL_PATH,
        default_factory=_empty_enrichment_state,
        normalize_fn=_normalize_enrichment_state,
    )


def _save_enrichment_state(
    storage: Optional[GCSStorage],
    payload: Dict[str, Any],
    require_remote: bool = False,
) -> None:
    _save_json_store(
        storage=storage,
        blob_name=ENRICHMENT_STATE_BLOB_NAME,
        local_path=ENRICHMENT_STATE_LOCAL_PATH,
        payload=payload,
        normalize_fn=_normalize_enrichment_state,
        require_remote=require_remote,
    )

def _create_uploaded_document_record(
    text: str,
    organization: str,
    title: str,
    speaker: str,
    doc_date: Any,
    doc_type: str,
    source_url: str,
    source_filename: str,
    source_ext: str,
    source_local_path: str,
    source_gcs_path: str,
    tags_csv: str,
    source_kind: str = "uploaded",
) -> Dict[str, Any]:
    org_label = _normalize_org_label(organization)
    date_str = doc_date.strftime("%B %d, %Y") if isinstance(doc_date, date) else str(doc_date or "")
    title = str(title or "").strip()
    speaker = str(speaker or "").strip() or "Unknown"
    source_url = str(source_url or "").strip()
    tags = [t.strip() for t in str(tags_csv or "").split(",") if t.strip()]
    stable_seed = "|".join([org_label, title, speaker, date_str, source_filename, str(len(text))])
    doc_id = hashlib.sha256(stable_seed.encode("utf-8")).hexdigest()[:24]
    canonical_url = (
        source_url
        or f"uploaded://{_org_key_from_label(org_label)}/{doc_id}/{_safe_filename(source_filename)}"
    )
    paragraphs = [p.strip() for p in str(text).splitlines() if p.strip()]
    word_count = len(str(text).split())
    return {
        "metadata": {
            "document_id": doc_id,
            "title": title,
            "speaker": speaker,
            "date": date_str,
            "url": canonical_url,
            "word_count": word_count,
            "organization": org_label,
            "doc_type": str(doc_type or "Document"),
            "source_filename": _safe_filename(source_filename),
            "source_format": str(source_ext or "").lstrip(".").lower(),
            "source_local_path": source_local_path,
            "source_gcs_path": source_gcs_path,
            "tags": tags,
            "source_kind": str(source_kind or "uploaded"),
        },
        "content": {
            "full_text": str(text),
            "paragraphs": paragraphs,
            "sentences": [],
        },
        "validation": {
            "completeness_score": 100 if word_count > 0 else 0,
        },
    }


def _upsert_custom_document_record(custom_payload: Dict[str, Any], record: Dict[str, Any]) -> bool:
    docs_list = custom_payload.get("documents", [])
    if not isinstance(docs_list, list):
        docs_list = []
    record_meta = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
    record_doc_id = str(record_meta.get("document_id", "") or "").strip()
    record_url_key = _url_match_key(record_meta.get("url", ""))

    replaced = False
    for idx, existing in enumerate(docs_list):
        if not isinstance(existing, dict):
            continue
        existing_meta = (
            existing.get("metadata", {}) if isinstance(existing.get("metadata", {}), dict) else {}
        )
        existing_doc_id = str(existing_meta.get("document_id", "") or "").strip()
        existing_url_key = _url_match_key(existing_meta.get("url", ""))
        if (record_doc_id and existing_doc_id == record_doc_id) or (
            record_url_key and existing_url_key and existing_url_key == record_url_key
        ):
            docs_list[idx] = record
            replaced = True
            break
    if not replaced:
        docs_list.append(record)
    custom_payload["documents"] = docs_list
    return replaced


def _extract_release_no(text: str = "", url: str = "") -> str:
    blob = f"{text}\n{url}"
    match = re.search(r"\bLR[-\s]?(\d{3,})\b", str(blob or ""), flags=re.IGNORECASE)
    if match:
        return f"LR-{match.group(1)}"
    return ""


def _normalize_enforcement_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    allowed_action_types = {"filing", "settlement", "judgment", "dismissal", "order", "other", "unknown"}
    allowed_forums = {"federal_court", "administrative", "state_court", "unknown"}
    allowed_outcomes = {"pending", "resolved", "partial", "unknown"}
    if not isinstance(payload, dict):
        payload = {}

    release_no = str(payload.get("release_no", "") or "").strip().upper()
    if release_no:
        release_no = release_no.replace(" ", "-")
        if release_no.startswith("LR") and not release_no.startswith("LR-") and len(release_no) > 2:
            release_no = "LR-" + release_no[2:].lstrip("-")
        parsed = _extract_release_no(release_no)
        if parsed:
            release_no = parsed

    action_type = str(payload.get("action_type", "") or "").strip().lower()
    if action_type not in allowed_action_types:
        action_type = "unknown"

    forum = str(payload.get("forum", "") or "").strip().lower()
    if forum not in allowed_forums:
        forum = "unknown"

    outcome_status = str(payload.get("outcome_status", "") or "").strip().lower()
    if outcome_status not in allowed_outcomes:
        outcome_status = "unknown"

    alleged_violations = payload.get("alleged_violations", [])
    if not isinstance(alleged_violations, list):
        alleged_violations = []
    cleaned_violations = []
    seen = set()
    for item in alleged_violations[:12]:
        label = str(item or "").strip()
        key = label.lower()
        if not label or key in seen:
            continue
        seen.add(key)
        cleaned_violations.append(label)

    return {
        "release_no": release_no,
        "action_type": action_type,
        "forum": forum,
        "alleged_violations": cleaned_violations,
        "outcome_status": outcome_status,
    }


def _infer_enforcement_metadata(
    title: str = "",
    text: str = "",
    url: str = "",
    doc_type: str = "",
    source_kind: str = "",
    release_no: str = "",
) -> Dict[str, Any]:
    blob = f"{title}\n{text}\n{url}\n{doc_type}\n{source_kind}".lower()
    release_no_value = str(release_no or "").strip()
    if not release_no_value:
        release_no_value = _extract_release_no(title, url=url) or _extract_release_no(text, url=url)

    action_type = "unknown"
    if any(token in blob for token in ["filed a complaint", "charged", "charges", "complaint alleges"]):
        action_type = "filing"
    elif any(token in blob for token in ["settled", "settlement", "agreed to pay", "consented to"]):
        action_type = "settlement"
    elif any(token in blob for token in ["final judgment", "judgment entered"]):
        action_type = "judgment"
    elif any(token in blob for token in ["dismissed", "dismissal"]):
        action_type = "dismissal"
    elif any(token in blob for token in ["order instituting", "cease-and-desist order", "order"]):
        action_type = "order"

    forum = "unknown"
    if any(token in blob for token in ["u.s. district court", "district court", "federal court"]):
        forum = "federal_court"
    elif any(
        token in blob
        for token in ["administrative proceeding", "administrative law judge", "before the commission"]
    ):
        forum = "administrative"
    elif "state court" in blob:
        forum = "state_court"

    outcome_status = "unknown"
    if any(token in blob for token in ["pending litigation", "alleges", "complaint"]):
        outcome_status = "pending"
    if any(token in blob for token in ["settled", "judgment entered", "resolved", "ordered to pay"]):
        outcome_status = "resolved"
    if any(token in blob for token in ["partial settlement", "partially resolved"]):
        outcome_status = "partial"

    violation_rules = [
        ("securities act", "Securities Act Violations"),
        ("exchange act", "Exchange Act Violations"),
        ("rule 10b-5", "Rule 10b-5 / Antifraud Violations"),
        ("section 17(a)", "Securities Act Section 17(a) Antifraud Violations"),
        ("section 5", "Unregistered Offering Violations"),
        ("books and records", "Books and Records Violations"),
        ("insider trading", "Insider Trading"),
        ("market manipulation", "Market Manipulation"),
        ("offering fraud", "Offering Fraud"),
        ("fcpa", "FCPA Violations"),
    ]
    alleged_violations = []
    seen = set()
    for needle, label in violation_rules:
        if needle in blob and label.lower() not in seen:
            seen.add(label.lower())
            alleged_violations.append(label)

    return _normalize_enforcement_metadata(
        {
            "release_no": release_no_value,
            "action_type": action_type,
            "forum": forum,
            "alleged_violations": alleged_violations,
            "outcome_status": outcome_status,
        }
    )


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    fenced = raw
    if fenced.startswith("```"):
        fenced = fenced.strip("`")
        fenced = fenced.replace("json", "", 1).strip()
    try:
        parsed = json.loads(fenced)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidate = raw[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _normalize_enrichment_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    tags = payload.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip() for t in tags if str(t).strip()][:12]

    keywords = payload.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k).strip() for k in keywords if str(k).strip()][:20]

    entities = payload.get("entities", [])
    normalized_entities = []
    if isinstance(entities, list):
        for item in entities[:30]:
            if isinstance(item, dict):
                name = str(item.get("name", "") or "").strip()
                etype = str(item.get("type", "") or "").strip().upper()
                mentions = _coerce_int(item.get("mentions", 1), default=1, min_value=1)
                if name:
                    normalized_entities.append(
                        {"name": name, "type": etype or "OTHER", "mentions": max(1, mentions)}
                    )

    stance = payload.get("stance", {})
    if not isinstance(stance, dict):
        stance = {"label": str(stance)}
    stance_label = str(stance.get("label", "unclear") or "unclear").strip().lower()
    if stance_label not in {"supportive", "cautious", "critical", "neutral", "unclear"}:
        stance_label = "unclear"
    stance_target = str(stance.get("target", "") or "").strip()

    evidence_spans = payload.get("evidence_spans", [])
    normalized_evidence = []
    if isinstance(evidence_spans, list):
        for item in evidence_spans[:8]:
            if isinstance(item, dict):
                claim = str(item.get("claim", "") or "").strip()
                snippet = str(item.get("snippet", "") or "").strip()
                if claim and snippet:
                    normalized_evidence.append({"claim": claim, "snippet": snippet[:600]})

    try:
        confidence = float(payload.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    enforcement_raw = payload.get("enforcement", {})
    if not isinstance(enforcement_raw, dict):
        enforcement_raw = {}
    for key in ["release_no", "action_type", "forum", "alleged_violations", "outcome_status"]:
        if key in payload and key not in enforcement_raw:
            enforcement_raw[key] = payload.get(key)
    enforcement = _normalize_enforcement_metadata(enforcement_raw)

    summary = str(payload.get("summary", "") or "").strip()[:1200]
    return {
        "summary": summary,
        "tags": tags,
        "keywords": keywords,
        "entities": normalized_entities,
        "stance": {"label": stance_label, "target": stance_target},
        "evidence_spans": normalized_evidence,
        "enforcement": enforcement,
        "confidence": confidence,
    }


def _heuristic_enrichment(doc: Dict[str, Any]) -> Dict[str, Any]:
    text = str(doc.get("full_text", "") or "")
    lower = text.lower()
    topic_rules = {
        "crypto_assets": ["crypto", "token", "digital asset", "blockchain", "stablecoin", "bitcoin", "ether"],
        "enforcement": ["enforcement", "violation", "charges", "penalty", "compliance", "investigation"],
        "disclosure_reporting": ["disclosure", "reporting", "10-k", "8-k", "transparency", "materiality"],
        "market_structure": ["market structure", "exchange", "liquidity", "trading", "order", "broker-dealer"],
        "investor_protection": ["investor", "retail", "protection", "fraud", "harm"],
        "funds_asset_mgmt": ["fund", "asset management", "adviser", "etf", "mutual fund"],
        "ai_technology": ["artificial intelligence", "ai", "machine learning", "automation", "algorithm"],
        "cybersecurity": ["cyber", "security breach", "incident response", "ransomware", "cybersecurity"],
    }
    tags = [tag for tag, needles in topic_rules.items() if any(n in lower for n in needles)]
    if not tags:
        tags = ["general_policy"]

    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", lower)
    cleaned = re.sub(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+){2,}", " ", cleaned)
    words = re.findall(r"[a-z][a-z\-]{3,}", cleaned)
    stop = {
        "that", "this", "with", "from", "have", "will", "their", "there", "which", "about", "would", "should",
        "could", "while", "these", "those", "into", "through", "across", "under", "over", "because", "where",
        "what", "when", "your", "than", "then", "them", "they", "been", "being", "also", "such", "must",
        "commission", "securities", "exchange", "speech", "statement", "sec",
        "https", "http", "www", "newsroom", "speeches", "statements", "ednref", "html", "gov", "us",
    }
    freq: Dict[str, int] = {}
    for word in words:
        if word in stop or len(word) > 28 or any(ch.isdigit() for ch in word) or word.count("-") > 1:
            continue
        freq[word] = freq.get(word, 0) + 1
    keywords = [word for word, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:12]]

    stance_label = "neutral"
    if any(token in lower for token in ["support", "welcome", "encourage", "approve"]):
        stance_label = "supportive"
    if any(token in lower for token in ["risk", "concern", "caution", "guardrail"]):
        stance_label = "cautious"
    if any(token in lower for token in ["oppose", "reject", "critic", "harmful"]):
        stance_label = "critical"

    evidence = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines[:80]:
        low = line.lower()
        if any(token in low for token in ["investor", "risk", "disclosure", "crypto", "enforcement"]):
            evidence.append({"claim": "Potentially relevant policy statement", "snippet": line[:500]})
            if len(evidence) >= 3:
                break

    enforcement = _infer_enforcement_metadata(
        title=doc.get("title", ""),
        text=text,
        url=doc.get("url", ""),
        doc_type=doc.get("doc_type", ""),
        source_kind=doc.get("source_kind", ""),
        release_no=doc.get("release_no", ""),
    )

    return _normalize_enrichment_payload(
        {
            "summary": (lines[0] if lines else str(doc.get("title", "")))[:300],
            "tags": tags,
            "keywords": keywords,
            "entities": [],
            "stance": {"label": stance_label, "target": ""},
            "evidence_spans": evidence,
            "enforcement": enforcement,
            "confidence": 0.35,
        }
    )


def _extract_response_text(response: Any) -> str:
    txt = getattr(response, "output_text", None)
    if txt:
        return txt
    if hasattr(response, "model_dump"):
        response = response.model_dump()
    elif hasattr(response, "dict"):
        response = response.dict()
    if isinstance(response, dict):
        for item in response.get("output", []):
            if item.get("type") == "message":
                for content_item in item.get("content", []):
                    if content_item.get("type") in ("output_text", "text") and content_item.get("text"):
                        return str(content_item.get("text"))
    return "No response text returned."


def _run_enrichment_agent(client: Any, doc: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    text = str(doc.get("full_text", "") or "").strip()
    if len(text) > 90000:
        text = text[:45000] + "\n\n[...TRUNCATED FOR ENRICHMENT...]\n\n" + text[-30000:]
    instruction = (
        "You are an enrichment agent for policy documents. Return ONLY valid JSON with keys: "
        "summary, tags, keywords, entities, stance, evidence_spans, enforcement, confidence. "
        "Use concise tags and keywords. entities must be list of {name,type,mentions}. "
        "stance must be {label,target} where label in [supportive,cautious,critical,neutral,unclear]. "
        "evidence_spans must include verbatim snippets from the document. "
        "enforcement must be {release_no,action_type,forum,alleged_violations,outcome_status} where "
        "action_type in [filing,settlement,judgment,dismissal,order,other,unknown], "
        "forum in [federal_court,administrative,state_court,unknown], and outcome_status in [pending,resolved,partial,unknown]. "
        "For non-enforcement documents, set enforcement fields to unknown/empty."
    )
    prompt = (
        f"Organization: {doc.get('organization', '')}\n"
        f"Title: {doc.get('title', '')}\n"
        f"Speaker: {doc.get('speaker', '')}\n"
        f"Date: {doc.get('date', '')}\n"
        f"Type: {doc.get('doc_type', '')}\n\n"
        f"Source Kind: {doc.get('source_kind', '')}\n"
        f"Release No (if known): {doc.get('release_no', '')}\n\n"
        f"Document Text:\n{text}"
    )

    last_raw = ""
    for attempt in range(1, 3):
        current_instruction = instruction
        if attempt > 1:
            current_instruction += " Respond with raw JSON only. No markdown, no commentary, no code fences."
        response = client.responses.create(model=model_name, instructions=current_instruction, input=prompt)
        raw_text = _extract_response_text(response)
        last_raw = raw_text
        parsed = _extract_first_json_object(raw_text)
        if parsed:
            return _normalize_enrichment_payload(parsed)
    preview = (last_raw or "").replace("\n", " ").strip()[:300]
    raise RuntimeError(f"Model did not return parseable JSON after 2 attempts. Last output: {preview}")


def _compute_reward(enrichment: Dict[str, Any], review_decision: str, status: str = "enriched") -> Dict[str, Any]:
    schema_validity = 1.0 if enrichment.get("tags") and enrichment.get("keywords") else 0.6
    evidence_quality = min(1.0, len(enrichment.get("evidence_spans", [])) / 3.0)
    confidence = max(0.0, min(1.0, float(enrichment.get("confidence", 0.0) or 0.0)))
    review_map = {"accepted": 1.0, "edited": 0.8, "pending": 0.6, "rejected": 0.2}
    review_component = review_map.get(str(review_decision or "pending"), 0.6)
    base_score = (
        (0.35 * schema_validity)
        + (0.30 * evidence_quality)
        + (0.20 * confidence)
        + (0.15 * review_component)
    )
    status_multipliers = {"enriched": 1.0, "reviewed": 1.0, "fallback_enriched": 0.6, "failed": 0.2}
    status_multiplier = status_multipliers.get(str(status or "").strip().lower(), 0.8)
    score = base_score * status_multiplier
    return {
        "score": round(float(score), 4),
        "components": {
            "schema_validity": round(schema_validity, 4),
            "evidence_quality": round(evidence_quality, 4),
            "confidence": round(confidence, 4),
            "review_component": round(review_component, 4),
            "status_multiplier": round(status_multiplier, 4),
        },
    }


def _candidate_chat_models() -> List[str]:
    return ["gpt-5.1", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"]


def _list_project_models(client: Any) -> List[str]:
    listed = client.models.list()
    return sorted({getattr(m, "id", "") for m in getattr(listed, "data", []) if getattr(m, "id", "")})


def _get_accessible_chat_models(client: Any) -> List[str]:
    candidates = _candidate_chat_models()
    try:
        ids = set(_list_project_models(client))
        available = [model for model in candidates if model in ids]
        if available:
            return available
    except Exception:
        pass
    return candidates


def _is_model_access_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "model_not_found" in msg or "does not have access to model" in msg or "access to model" in msg


def _get_openai_client(secrets_payload: Dict[str, Any]) -> Optional[Any]:
    api_key = _get_openai_api_key(secrets_payload)
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        _stderr(f"Failed to initialize OpenAI client: {e}")
        return None


def _load_doc_ids_from_summary(path_text: str) -> List[str]:
    with open(path_text, "r", encoding="utf-8") as f:
        payload = json.load(f)
    doc_ids = payload.get("processed_doc_ids", [])
    if not isinstance(doc_ids, list) or not doc_ids:
        doc_ids = list(payload.get("new_doc_ids", []) or []) + list(payload.get("updated_doc_ids", []) or [])
    out = []
    seen = set()
    for item in doc_ids:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _write_summary(summary_path: Optional[str], payload: Dict[str, Any]) -> None:
    if not summary_path:
        return
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def _run_news_ingest(args: argparse.Namespace) -> Dict[str, Any]:
    secrets_payload = _load_streamlit_secrets()
    storage, gcs_status = _get_gcs_storage(secrets_payload)
    if args.require_remote_persistence and storage is None:
        raise RuntimeError(gcs_status)

    api_key = _get_newsapi_api_key(secrets_payload)
    if not api_key:
        raise RuntimeError("NewsAPI API key is not configured.")

    settings = _load_news_connector_settings(storage)
    overrides = {
        "query": args.query,
        "lookback_days": args.lookback_days,
        "max_pages": args.max_pages,
        "page_size": args.page_size,
        "target_count": args.target_count,
        "sort_by": args.sort_by,
        "organization_label": args.organization_label,
        "domains": args.domains,
        "exclude_domains": args.exclude_domains,
        "tags_csv": args.tags_csv,
    }
    merged_settings = dict(settings)
    for key, value in overrides.items():
        if value is not None and str(value) != "":
            merged_settings[key] = value
    settings = _normalize_news_connector_settings(merged_settings)

    now_utc = datetime.now(UTC).replace(tzinfo=None, microsecond=0)
    from_dt = now_utc - timedelta(days=int(settings["lookback_days"]))
    from_date = from_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    to_date = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    scraper = NewsAPIFinancialScraper(api_key=api_key)
    discovered = scraper.discover_documents(
        query=settings["query"],
        max_pages=int(settings["max_pages"]),
        page_size=int(settings["page_size"]),
        from_date=from_date,
        to_date=to_date,
        sort_by=settings["sort_by"],
        domains=settings["domains"],
        exclude_domains=settings["exclude_domains"],
        target_count=int(settings["target_count"]),
    )

    custom_payload = _load_custom_documents(storage)
    existing_custom = {}
    for item in custom_payload.get("documents", []):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
        existing_custom[_url_match_key(metadata.get("url", ""))] = metadata

    for entry in discovered:
        key = _url_match_key(entry.get("url", ""))
        status = "new"
        existing_meta = existing_custom.get(key)
        if existing_meta:
            existing_published = str(
                existing_meta.get("published_at", "")
                or existing_meta.get("published_date", "")
                or existing_meta.get("date", "")
                or ""
            ).strip()
            incoming_published = str(entry.get("published_at", "") or entry.get("date", "") or "").strip()
            status = (
                "update_available"
                if incoming_published and existing_published and incoming_published != existing_published
                else "existing"
            )
        entry["ingest_status"] = status

    if args.selection == "all":
        candidates = list(discovered)
    else:
        candidates = [entry for entry in discovered if entry.get("ingest_status") in {"new", "update_available"}]

    limit = len(candidates) if args.limit is None else max(0, int(args.limit))
    selected = candidates[:limit] if limit > 0 else []

    saved_new = 0
    saved_updates = 0
    failed = []
    processed_doc_ids: List[str] = []
    new_doc_ids: List[str] = []
    updated_doc_ids: List[str] = []

    for idx, entry in enumerate(selected, 1):
        try:
            extracted = scraper.extract_document(
                entry.get("url", ""),
                fallback_title=entry.get("title", ""),
                fallback_date=entry.get("date", ""),
                fallback_description=entry.get("description", ""),
                fallback_content=entry.get("content_snippet", ""),
                fallback_source_name=entry.get("source_name", ""),
                fallback_author=entry.get("author", ""),
            )
            if not extracted.get("success"):
                raise RuntimeError("Extraction returned unsuccessful result.")
            data = extracted.get("data", {})
            text = str(data.get("full_text", "") or "").strip()
            if len(text.split()) < 30:
                raise RuntimeError("Extracted text appears too short; skipping.")

            src_url = str(data.get("url", "") or entry.get("url", "")).strip()
            source_format = str(data.get("source_format", "") or "html").strip().lower()
            source_ext = ".pdf" if source_format == "pdf" else ".html"
            source_name = urlparse(src_url).path.rsplit("/", 1)[-1].strip()
            if not source_name:
                source_name = f"news-article-{idx}{source_ext}"
            elif "." not in source_name:
                source_name += source_ext

            date_text = str(data.get("date", "") or entry.get("date", "")).strip()
            doc_date_value = _parse_single_date(date_text) or date_text
            source_name_text = str(data.get("source_name", "") or entry.get("source_name", "")).strip()
            author_text = str(data.get("author", "") or entry.get("author", "")).strip()
            speaker_text = source_name_text or author_text or "News Desk"

            record = _create_uploaded_document_record(
                text=text,
                organization=settings["organization_label"],
                title=str(data.get("title", "") or entry.get("title", "")).strip(),
                speaker=speaker_text,
                doc_date=doc_date_value,
                doc_type="News Article",
                source_url=src_url,
                source_filename=source_name,
                source_ext=source_ext,
                source_local_path="",
                source_gcs_path="",
                tags_csv=settings["tags_csv"],
                source_kind="newsapi_article",
            )
            metadata = record.setdefault("metadata", {})
            metadata["source_family"] = "newsapi_article"
            metadata["news_query"] = str(entry.get("query", "") or settings["query"])
            metadata["source_name"] = source_name_text
            metadata["source_id"] = str(entry.get("source_id", "") or "").strip()
            metadata["author"] = author_text
            metadata["published_date"] = str(entry.get("date", "") or "").strip()
            metadata["published_at"] = str(entry.get("published_at", "") or "").strip()
            metadata["description"] = str(data.get("description", "") or entry.get("description", "")).strip()
            metadata["content_snippet"] = str(entry.get("content_snippet", "") or "").strip()
            metadata["newsapi_extraction_mode"] = str(data.get("extraction_mode", "") or "").strip()
            metadata["newsapi_domains"] = settings["domains"]
            metadata["newsapi_exclude_domains"] = settings["exclude_domains"]

            replaced = _upsert_custom_document_record(custom_payload, record)
            doc_id = str(metadata.get("document_id", "") or "").strip()
            if doc_id:
                processed_doc_ids.append(doc_id)
            if replaced:
                saved_updates += 1
                if doc_id:
                    updated_doc_ids.append(doc_id)
            else:
                saved_new += 1
                if doc_id:
                    new_doc_ids.append(doc_id)
        except Exception as e:
            failed.append({"url": entry.get("url", ""), "title": entry.get("title", ""), "error": str(e)})

    if not args.dry_run and (saved_new or saved_updates):
        _save_custom_documents(storage, custom_payload, require_remote=args.require_remote_persistence)

    summary = {
        "mode": "ingest",
        "ran_at": _utc_now_iso(),
        "require_remote_persistence": bool(args.require_remote_persistence),
        "remote_persistence": bool(storage is not None),
        "query": settings["query"],
        "lookback_days": int(settings["lookback_days"]),
        "from_date": from_date,
        "to_date": to_date,
        "selection": args.selection,
        "discovered_count": len(discovered),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "processed_count": len(processed_doc_ids),
        "saved_new": saved_new,
        "saved_updates": saved_updates,
        "processed_doc_ids": processed_doc_ids,
        "new_doc_ids": new_doc_ids,
        "updated_doc_ids": updated_doc_ids,
        "failed_count": len(failed),
        "failed": failed[:25],
        "dry_run": bool(args.dry_run),
    }
    _write_summary(args.summary_path, summary)
    return summary


def _build_news_enrichment_candidates(
    custom_payload: Dict[str, Any],
    source_kind: str,
    doc_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    selected_ids = {str(item).strip() for item in (doc_ids or []) if str(item).strip()}
    docs = []
    for item in custom_payload.get("documents", []):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
        content = item.get("content", {}) if isinstance(item.get("content", {}), dict) else {}
        doc_id = str(metadata.get("document_id", "") or "").strip()
        if not doc_id:
            continue
        if selected_ids and doc_id not in selected_ids:
            continue
        if str(metadata.get("source_kind", "") or "").strip() != source_kind:
            continue
        full_text = str(content.get("full_text", "") or "").strip()
        if not full_text:
            continue
        docs.append(
            {
                "doc_id": doc_id,
                "organization": _normalize_org_label(metadata.get("organization") or metadata.get("org") or ""),
                "org_key": _org_key_from_label(
                    _normalize_org_label(metadata.get("organization") or metadata.get("org") or "")
                ),
                "title": str(metadata.get("title", "") or "").strip(),
                "speaker": str(metadata.get("speaker", "") or "").strip(),
                "date": str(metadata.get("date", "") or "").strip(),
                "url": str(metadata.get("url", "") or "").strip(),
                "doc_type": str(metadata.get("doc_type", "Document") or "Document").strip(),
                "source_kind": str(metadata.get("source_kind", "") or "").strip(),
                "release_no": str(metadata.get("release_no", "") or "").strip(),
                "full_text": full_text,
                "word_count": _coerce_int(metadata.get("word_count", 0), default=0, min_value=0),
            }
        )
    return docs


def _run_news_enrichment(args: argparse.Namespace) -> Dict[str, Any]:
    secrets_payload = _load_streamlit_secrets()
    storage, gcs_status = _get_gcs_storage(secrets_payload)
    if args.require_remote_persistence and storage is None:
        raise RuntimeError(gcs_status)

    custom_payload = _load_custom_documents(storage)
    enrichment_state = _load_enrichment_state(storage)
    entries = enrichment_state.setdefault("entries", {})

    doc_ids: List[str] = []
    if args.doc_ids_from_summary:
        doc_ids.extend(_load_doc_ids_from_summary(args.doc_ids_from_summary))
    if args.doc_id:
        doc_ids.extend([str(item).strip() for item in args.doc_id if str(item).strip()])
    dedup_doc_ids = []
    seen_ids = set()
    for item in doc_ids:
        if item and item not in seen_ids:
            seen_ids.add(item)
            dedup_doc_ids.append(item)

    candidates = _build_news_enrichment_candidates(
        custom_payload=custom_payload,
        source_kind=args.source_kind,
        doc_ids=dedup_doc_ids or None,
    )

    if not dedup_doc_ids and args.mode == "only_missing_or_failed":
        filtered = []
        for candidate in candidates:
            existing = entries.get(candidate["doc_id"], {}) if isinstance(entries.get(candidate["doc_id"], {}), dict) else {}
            status = str(existing.get("status", "") or "")
            if status in {"enriched", "reviewed"}:
                continue
            filtered.append(candidate)
        candidates = filtered

    limit = len(candidates) if args.limit is None else max(0, int(args.limit))
    targets = candidates[:limit] if limit > 0 else []

    client = None if args.heuristic_only else _get_openai_client(secrets_payload)
    accessible_models = _get_accessible_chat_models(client) if client is not None else []
    preferred_model = args.model or (accessible_models[0] if accessible_models else "gpt-5-mini")

    enriched_count = 0
    fallback_count = 0
    used_models: List[str] = []

    for candidate in targets:
        doc_id = candidate["doc_id"]
        existing = entries.get(doc_id, {}) if isinstance(entries.get(doc_id, {}), dict) else {}
        review = existing.get("review", {}) if isinstance(existing.get("review", {}), dict) else {}
        decision = str(review.get("decision", "pending") or "pending")
        notes = str(review.get("notes", "") or "")
        status = "enriched"
        error_msg = ""
        model_used = ""
        try:
            if client is None:
                raise RuntimeError("OpenAI client unavailable; using heuristic enrichment.")
            ordered_models = [preferred_model] + [model for model in accessible_models if model != preferred_model]
            last_error = None
            enrichment = None
            for model_name in ordered_models:
                try:
                    enrichment = _run_enrichment_agent(client, candidate, model_name)
                    model_used = model_name
                    break
                except Exception as e:
                    last_error = e
                    if not _is_model_access_error(e):
                        raise
            if enrichment is None:
                raise last_error or RuntimeError("No model available for enrichment.")
            enriched_count += 1
            if model_used and model_used not in used_models:
                used_models.append(model_used)
        except Exception as e:
            enrichment = _heuristic_enrichment(candidate)
            status = "fallback_enriched"
            error_msg = str(e)
            fallback_count += 1

        reward = _compute_reward(enrichment, decision, status=status)
        entries[doc_id] = {
            "doc_id": doc_id,
            "organization": candidate.get("organization", ""),
            "org_key": candidate.get("org_key", ""),
            "title": candidate.get("title", ""),
            "speaker": candidate.get("speaker", ""),
            "date": candidate.get("date", ""),
            "url": candidate.get("url", ""),
            "doc_type": candidate.get("doc_type", ""),
            "word_count": _coerce_int(candidate.get("word_count", 0), default=0, min_value=0),
            "status": status,
            "error": error_msg,
            "model": model_used or preferred_model,
            "pipeline_version": ENRICHMENT_PIPELINE_VERSION,
            "updated_at": _utc_now_iso(),
            "review": {"decision": decision, "notes": notes},
            "reward": reward,
            "enrichment": enrichment,
        }

    enrichment_state["entries"] = entries
    if not args.dry_run and targets:
        _save_enrichment_state(storage, enrichment_state, require_remote=args.require_remote_persistence)

    summary = {
        "mode": "enrich",
        "ran_at": _utc_now_iso(),
        "require_remote_persistence": bool(args.require_remote_persistence),
        "remote_persistence": bool(storage is not None),
        "source_kind": args.source_kind,
        "mode_selection": args.mode,
        "requested_doc_ids": dedup_doc_ids,
        "candidate_count": len(candidates),
        "selected_count": len(targets),
        "processed_count": len(targets),
        "enriched_count": enriched_count,
        "fallback_enriched_count": fallback_count,
        "used_models": used_models,
        "dry_run": bool(args.dry_run),
    }
    _write_summary(args.summary_path, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Financial news ingest and enrichment pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Discover and ingest financial news articles")
    ingest.add_argument("--query", default=None)
    ingest.add_argument("--lookback-days", type=int, default=None)
    ingest.add_argument("--max-pages", type=int, default=None)
    ingest.add_argument("--page-size", type=int, default=None)
    ingest.add_argument("--target-count", type=int, default=None)
    ingest.add_argument("--sort-by", default=None)
    ingest.add_argument("--organization-label", default=None)
    ingest.add_argument("--domains", default=None)
    ingest.add_argument("--exclude-domains", default=None)
    ingest.add_argument("--tags-csv", default=None)
    ingest.add_argument("--selection", choices=["new_or_updated", "all"], default="new_or_updated")
    ingest.add_argument("--limit", type=int, default=None)
    ingest.add_argument("--dry-run", action="store_true")
    ingest.add_argument("--require-remote-persistence", action="store_true")
    ingest.add_argument("--summary-path", default="")

    enrich = subparsers.add_parser("enrich", help="Enrich ingested financial news articles")
    enrich.add_argument("--source-kind", default="newsapi_article")
    enrich.add_argument("--mode", choices=["all", "only_missing_or_failed"], default="only_missing_or_failed")
    enrich.add_argument("--doc-id", action="append", default=[])
    enrich.add_argument("--doc-ids-from-summary", default="")
    enrich.add_argument("--limit", type=int, default=None)
    enrich.add_argument("--model", default="")
    enrich.add_argument("--heuristic-only", action="store_true")
    enrich.add_argument("--dry-run", action="store_true")
    enrich.add_argument("--require-remote-persistence", action="store_true")
    enrich.add_argument("--summary-path", default="")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.command == "ingest":
            summary = _run_news_ingest(args)
        elif args.command == "enrich":
            summary = _run_news_enrichment(args)
        else:  # pragma: no cover
            parser.error(f"Unknown command: {args.command}")
            return 2
    except Exception as e:
        error_payload = {"ok": False, "error": str(e), "command": args.command, "ran_at": _utc_now_iso()}
        _write_summary(getattr(args, "summary_path", ""), error_payload)
        print(json.dumps(error_payload, indent=2, ensure_ascii=False))
        return 1

    summary["ok"] = True
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
