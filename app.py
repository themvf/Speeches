#!/usr/bin/env python3
"""
SEC Commissioner Speeches Dashboard
Streamlit app for exploring and analyzing SEC Commissioner speeches.
"""

import json
import hashlib
import time
import re
import io
from urllib.parse import urlparse
import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from analysis_pipeline import SpeechAnalysisPipeline
from speaker_utils import extract_speakers, format_speakers, primary_speaker


# --- Page Config ---
st.set_page_config(
    page_title="SEC Speeches Dashboard",
    page_icon="\U0001f4dc",
    layout="wide",
)


CUSTOM_DOCS_BLOB_NAME = "custom_documents.json"
CUSTOM_DOCS_LOCAL_PATH = Path("data/custom_documents.json")
CUSTOM_DOCS_RAW_DIR = Path("data/raw_documents")
ENRICHMENT_STATE_BLOB_NAME = "document_enrichment_state.json"
ENRICHMENT_STATE_LOCAL_PATH = Path("data/document_enrichment_state.json")
ENRICHMENT_PIPELINE_VERSION = "v1"
POLICY_BRIEFS_BLOB_NAME = "policy_delta_briefs.json"
POLICY_BRIEFS_LOCAL_PATH = Path("data/policy_delta_briefs.json")
POLICY_BRIEFING_VERSION = "v1"


# --- GCS helpers ---

def _get_gcs_storage():
    """Return a GCSStorage instance if secrets are configured, else None."""
    try:
        gcs_info = st.secrets["gcs"]
        bucket_name = gcs_info["bucket_name"]
        # Build credentials dict from secrets (exclude bucket_name)
        creds = {k: v for k, v in gcs_info.items() if k != "bucket_name"}
        from gcs_storage import GCSStorage
        return GCSStorage(bucket_name, creds)
    except Exception as e:
        # Store the error so we can display it in the sidebar
        st.session_state["_gcs_error"] = str(e)
        return None


def _get_openai_api_key():
    """Return OpenAI API key from Streamlit secrets, else None."""
    try:
        api_key = st.secrets["openai"]["api_key"]
        api_key = str(api_key).strip()
        if not api_key:
            raise ValueError("openai.api_key is empty")
        return api_key
    except Exception as e:
        st.session_state["_openai_error"] = str(e)
        return None


def _get_openai_client():
    """Create an OpenAI client using secrets-based API key."""
    api_key = _get_openai_api_key()
    if not api_key:
        return None

    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.session_state["_openai_error"] = f"Failed to initialize OpenAI client: {e}"
        return None


def _utc_now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _normalize_org_label(value):
    label = str(value).strip() if value is not None else ""
    return label or "SEC"


def _org_key_from_label(label):
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(label))
    cleaned = cleaned.strip("_")
    return cleaned or "sec"


def _speech_org_label(speech):
    m = speech.get("metadata", {})
    return _normalize_org_label(m.get("organization") or m.get("org") or "SEC")


def _speech_org_key(speech):
    return _org_key_from_label(_speech_org_label(speech))


def _list_org_options(raw_data_obj):
    by_key = {}
    for speech in raw_data_obj.get("speeches", []):
        label = _speech_org_label(speech)
        key = _org_key_from_label(label)
        if key not in by_key:
            by_key[key] = label
    if not by_key:
        by_key["sec"] = "SEC"
    return [{"key": k, "label": by_key[k]} for k in sorted(by_key, key=lambda x: by_key[x].lower())]


def _vector_state_path():
    return Path("data/openai_vector_store_state.json")


def _vector_state_blob_name():
    return "openai_vector_store_state.json"


def _normalize_vector_state(state):
    if not isinstance(state, dict):
        state = {}

    stores = state.get("stores")
    if isinstance(stores, dict):
        state.setdefault("version", 2)
        return state

    # Legacy single-store schema migration.
    migrated = {"version": 2, "stores": {}}
    legacy_id = str(state.get("vector_store_id", "")).strip()
    if legacy_id:
        migrated["stores"]["sec"] = {
            "org_label": "SEC",
            "vector_store_id": legacy_id,
            "docs": state.get("docs", {}),
            "doc_count_indexed": int(state.get("indexed_speeches", 0) or 0),
            "updated_at": state.get("updated_at", ""),
        }
    migrated["updated_at"] = state.get("updated_at", "")
    return migrated


def _load_vector_state_local():
    path = _vector_state_path()
    if not path.exists():
        return {"version": 2, "stores": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return _normalize_vector_state(json.load(f))
    except Exception:
        return {"version": 2, "stores": {}}


def _save_vector_state_local(state):
    state = _normalize_vector_state(state)
    path = _vector_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _load_vector_state():
    state = None
    storage = _get_gcs_storage()
    if storage is not None:
        try:
            blob = storage.bucket.blob(_vector_state_blob_name())
            if blob.exists():
                state = _normalize_vector_state(json.loads(blob.download_as_text(encoding="utf-8")))
                _save_vector_state_local(state)
                return state
        except Exception as e:
            st.session_state["_vector_state_error"] = f"GCS vector-state load failed: {e}"

    state = _load_vector_state_local()
    return _normalize_vector_state(state)


def _save_vector_state(state):
    state = _normalize_vector_state(state)
    _save_vector_state_local(state)

    storage = _get_gcs_storage()
    if storage is None:
        return

    try:
        blob = storage.bucket.blob(_vector_state_blob_name())
        blob.upload_from_string(
            json.dumps(state, indent=2, ensure_ascii=False),
            content_type="application/json",
        )
    except Exception as e:
        st.session_state["_vector_state_error"] = f"GCS vector-state save failed: {e}"


def _empty_custom_docs_payload():
    return {
        "updated_at": "",
        "documents": [],
    }


def _normalize_custom_docs_payload(payload):
    if not isinstance(payload, dict):
        payload = {}
    docs = payload.get("documents", [])
    if not isinstance(docs, list):
        docs = []
    return {
        "updated_at": str(payload.get("updated_at", "") or ""),
        "documents": docs,
    }


def _load_custom_documents_local():
    if not CUSTOM_DOCS_LOCAL_PATH.exists():
        return _empty_custom_docs_payload()
    try:
        with open(CUSTOM_DOCS_LOCAL_PATH, "r", encoding="utf-8") as f:
            return _normalize_custom_docs_payload(json.load(f))
    except Exception:
        return _empty_custom_docs_payload()


def _save_custom_documents_local(payload):
    payload = _normalize_custom_docs_payload(payload)
    CUSTOM_DOCS_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CUSTOM_DOCS_LOCAL_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _load_custom_documents():
    storage = _get_gcs_storage()
    if storage is not None:
        try:
            blob = storage.bucket.blob(CUSTOM_DOCS_BLOB_NAME)
            if blob.exists():
                payload = _normalize_custom_docs_payload(
                    json.loads(blob.download_as_text(encoding="utf-8"))
                )
                _save_custom_documents_local(payload)
                return payload
        except Exception as e:
            st.session_state["_custom_docs_error"] = f"GCS custom-doc load failed: {e}"

    return _load_custom_documents_local()


def _save_custom_documents(payload):
    payload = _normalize_custom_docs_payload(payload)
    payload["updated_at"] = _utc_now_iso()
    _save_custom_documents_local(payload)

    storage = _get_gcs_storage()
    if storage is None:
        return
    try:
        blob = storage.bucket.blob(CUSTOM_DOCS_BLOB_NAME)
        blob.upload_from_string(
            json.dumps(payload, indent=2, ensure_ascii=False),
            content_type="application/json",
        )
    except Exception as e:
        st.session_state["_custom_docs_error"] = f"GCS custom-doc save failed: {e}"


def _safe_filename(name):
    raw = str(name or "document").strip()
    raw = raw.replace("\\", "_").replace("/", "_")
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".", " ") else "_" for ch in raw).strip()
    return cleaned or "document"


def _vector_filename_from_title(org_key, title, doc_id, part_idx=1, total_parts=1):
    base = _safe_filename(title or "")
    if "." in base:
        base = base.rsplit(".", 1)[0]
    base = re.sub(r"\s+", " ", base).strip(" ._-")
    if not base:
        base = f"{org_key}_{doc_id}"
    base = base[:120].strip(" ._-") or f"{org_key}_{doc_id}"
    if total_parts > 1:
        base = f"{base} - part {part_idx} of {total_parts}"
    return f"{base} [{doc_id}].txt"


def _coerce_int(value, default=0, min_value=0):
    """Best-effort integer coercion for noisy data/model outputs."""
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


def _coerce_float(value, default=0.0, min_value=0.0, max_value=1.0):
    try:
        num = float(value)
    except Exception:
        num = float(default)

    if min_value is not None:
        try:
            num = max(float(min_value), num)
        except Exception:
            pass
    if max_value is not None:
        try:
            num = min(float(max_value), num)
        except Exception:
            pass
    return num


def _store_uploaded_source_file(file_bytes, original_filename, doc_id):
    safe_name = _safe_filename(original_filename)
    local_dir = CUSTOM_DOCS_RAW_DIR / doc_id
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / safe_name
    with open(local_path, "wb") as f:
        f.write(file_bytes)

    gcs_path = ""
    storage = _get_gcs_storage()
    if storage is not None:
        try:
            gcs_path = f"documents/raw/{doc_id}/{safe_name}"
            blob = storage.bucket.blob(gcs_path)
            blob.upload_from_string(file_bytes, content_type="application/octet-stream")
        except Exception as e:
            st.session_state["_custom_docs_error"] = f"GCS source-file save failed: {e}"
            gcs_path = ""

    return str(local_path), gcs_path


def _extract_text_from_uploaded_file(uploaded_file):
    file_name = _safe_filename(getattr(uploaded_file, "name", "document"))
    file_ext = Path(file_name).suffix.lower()
    file_bytes = uploaded_file.getvalue()
    warnings = []

    if file_ext == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as e:
            raise RuntimeError(f"PDF parsing requires `pypdf` dependency: {e}")

        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for idx, page in enumerate(reader.pages, 1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages.append(f"[Page {idx}]\n{page_text}")
        text = "\n\n".join(pages).strip()
        if not text:
            warnings.append("No text extracted from PDF. It may be scanned; OCR support is not enabled yet.")
        return text, file_ext, warnings, file_bytes

    if file_ext in (".html", ".htm"):
        decoded = file_bytes.decode("utf-8", errors="replace")
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(decoded, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text("\n")
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        except Exception:
            text = decoded
            warnings.append("HTML sanitizer failed; using raw HTML text fallback.")
        return text.strip(), file_ext, warnings, file_bytes

    decoded = file_bytes.decode("utf-8", errors="replace")
    return decoded.strip(), file_ext, warnings, file_bytes


def _split_text_for_indexing(text, max_chars=40000, overlap_chars=3000):
    text = str(text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def _create_uploaded_document_record(
    text,
    organization,
    title,
    speaker,
    doc_date,
    doc_type,
    source_url,
    source_filename,
    source_ext,
    source_local_path,
    source_gcs_path,
    tags_csv,
    source_kind="uploaded",
):
    org_label = _normalize_org_label(organization)
    date_str = doc_date.strftime("%B %d, %Y") if isinstance(doc_date, date) else str(doc_date or "")
    title = str(title or "").strip()
    speaker = str(speaker or "").strip() or "Unknown"
    source_url = str(source_url or "").strip()
    tags = [t.strip() for t in str(tags_csv or "").split(",") if t.strip()]

    stable_seed = "|".join([org_label, title, speaker, date_str, source_filename, str(len(text))])
    doc_id = hashlib.sha256(stable_seed.encode("utf-8")).hexdigest()[:24]
    canonical_url = source_url or f"uploaded://{_org_key_from_label(org_label)}/{doc_id}/{_safe_filename(source_filename)}"

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


def _custom_docs_as_speeches(payload):
    payload = _normalize_custom_docs_payload(payload)
    docs = []
    for item in payload.get("documents", []):
        if isinstance(item, dict):
            docs.append(item)
    return docs


def _url_match_key(url):
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
        scheme = (parsed.scheme or "https").lower()
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/")
        if not path:
            path = "/"
        return f"{scheme}://{netloc}{path}"
    except Exception:
        return raw.rstrip("/")


def _upsert_custom_document_record(custom_payload, record):
    docs_list = custom_payload.get("documents", [])
    record_meta = record.get("metadata", {})
    record_doc_id = str(record_meta.get("document_id", "") or "").strip()
    record_url_key = _url_match_key(record_meta.get("url", ""))

    replaced = False
    for idx, existing in enumerate(docs_list):
        em = existing.get("metadata", {})
        existing_doc_id = str(em.get("document_id", "") or "").strip()
        existing_url_key = _url_match_key(em.get("url", ""))
        if (
            (record_doc_id and existing_doc_id == record_doc_id)
            or (record_url_key and existing_url_key and existing_url_key == record_url_key)
        ):
            docs_list[idx] = record
            replaced = True
            break
    if not replaced:
        docs_list.append(record)

    custom_payload["documents"] = docs_list
    return replaced


def _build_knowledge_data(sec_raw_data, custom_docs_payload):
    sec_speeches = sec_raw_data.get("speeches", []) if isinstance(sec_raw_data, dict) else []
    custom_speeches = _custom_docs_as_speeches(custom_docs_payload)
    return {"speeches": list(sec_speeches) + list(custom_speeches)}


def _build_knowledge_df(knowledge_data):
    rows = []
    for speech in knowledge_data.get("speeches", []):
        m = speech.get("metadata", {})
        rows.append(
            {
                "organization": _speech_org_label(speech),
                "date": m.get("date", ""),
                "title": m.get("title", ""),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        out["date_parsed"] = pd.NaT
        return out
    out["date_parsed"] = _parse_date_series(out["date"])
    return out


def _infer_source_kind(metadata):
    if not isinstance(metadata, dict):
        return "document"
    source_kind = str(metadata.get("source_kind", "") or "").strip().lower()
    if source_kind:
        return source_kind
    url = str(metadata.get("url", "") or "").lower()
    if "/newsroom/speeches-statements/" in url:
        return "sec_speech"
    if "/trading-markets-frequently-asked-questions/" in url or source_kind == "sec_tm_faq":
        return "sec_tm_faq"
    if "/enforcement-litigation/litigation-releases/" in url or source_kind == "sec_enforcement_litigation":
        return "sec_enforcement_litigation"
    if ("/usao-" in url or "/usao/" in url) and "/pr/" in url:
        return "doj_usao_press_release"
    doc_type = str(metadata.get("doc_type", "") or "").strip().lower()
    if doc_type in {"speech", "statement", "remarks"}:
        return "sec_speech"
    return "document"


def _build_corpus_explorer_df(knowledge_data, enrichment_state):
    entries = enrichment_state.get("entries", {}) if isinstance(enrichment_state, dict) else {}
    if not isinstance(entries, dict):
        entries = {}

    rows = []
    for speech in knowledge_data.get("speeches", []):
        if not isinstance(speech, dict):
            continue
        m = speech.get("metadata", {})
        c = speech.get("content", {})
        doc_id = _corpus_doc_id(speech)

        enrich_entry = entries.get(doc_id, {})
        if not isinstance(enrich_entry, dict):
            enrich_entry = {}
        enrich = enrich_entry.get("enrichment", {})
        if not isinstance(enrich, dict):
            enrich = {}
        review = enrich_entry.get("review", {}) if isinstance(enrich_entry.get("review", {}), dict) else {}
        auto_review = enrich_entry.get("auto_review", {}) if isinstance(enrich_entry.get("auto_review", {}), dict) else {}

        tags = enrich.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tags = [str(t).strip() for t in tags if str(t).strip()]

        keywords = enrich.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(k).strip() for k in keywords if str(k).strip()]

        stance = enrich.get("stance", {})
        if not isinstance(stance, dict):
            stance = {"label": str(stance or ""), "target": ""}
        stance_label = str(stance.get("label", "") or "").strip()
        stance_target = str(stance.get("target", "") or "").strip()
        stance_text = stance_label
        if stance_label and stance_target:
            stance_text = f"{stance_label} ({stance_target})"

        rows.append(
            {
                "doc_id": doc_id,
                "date": str(m.get("date", "") or ""),
                "organization": _speech_org_label(speech),
                "source_kind": _infer_source_kind(m),
                "doc_type": str(m.get("doc_type", "Document") or "Document"),
                "title": str(m.get("title", "") or ""),
                "speaker": str(m.get("speaker", "") or "Unknown"),
                "word_count": _coerce_int(m.get("word_count", 0), default=0, min_value=0),
                "url": str(m.get("url", "") or ""),
                "tags_list": tags,
                "keywords_list": keywords,
                "tags_text": ", ".join(tags),
                "keywords_text": ", ".join(keywords),
                "stance": stance_text,
                "review": str(review.get("decision", "pending") or "pending"),
                "auto_verdict": str(auto_review.get("verdict", "") or ""),
                "status": str(enrich_entry.get("status", "") or ""),
                "full_text": str(c.get("full_text", "") or ""),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        out["date_parsed"] = pd.NaT
        return out
    out["date_parsed"] = _parse_date_series(out["date"])
    return out


def _empty_enrichment_state():
    return {
        "version": 1,
        "pipeline_version": ENRICHMENT_PIPELINE_VERSION,
        "updated_at": "",
        "entries": {},
    }


def _normalize_enrichment_state(state):
    if not isinstance(state, dict):
        return _empty_enrichment_state()
    entries = state.get("entries", {})
    if not isinstance(entries, dict):
        entries = {}
    out = {
        "version": int(state.get("version", 1) or 1),
        "pipeline_version": str(state.get("pipeline_version", ENRICHMENT_PIPELINE_VERSION) or ENRICHMENT_PIPELINE_VERSION),
        "updated_at": str(state.get("updated_at", "") or ""),
        "entries": entries,
    }
    return out


def _load_enrichment_state_local():
    if not ENRICHMENT_STATE_LOCAL_PATH.exists():
        return _empty_enrichment_state()
    try:
        with open(ENRICHMENT_STATE_LOCAL_PATH, "r", encoding="utf-8") as f:
            return _normalize_enrichment_state(json.load(f))
    except Exception:
        return _empty_enrichment_state()


def _save_enrichment_state_local(state):
    state = _normalize_enrichment_state(state)
    ENRICHMENT_STATE_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ENRICHMENT_STATE_LOCAL_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _load_enrichment_state():
    storage = _get_gcs_storage()
    if storage is not None:
        try:
            blob = storage.bucket.blob(ENRICHMENT_STATE_BLOB_NAME)
            if blob.exists():
                state = _normalize_enrichment_state(json.loads(blob.download_as_text(encoding="utf-8")))
                _save_enrichment_state_local(state)
                return state
        except Exception as e:
            st.session_state["_enrichment_error"] = f"GCS enrichment-state load failed: {e}"
    return _load_enrichment_state_local()


def _save_enrichment_state(state):
    state = _normalize_enrichment_state(state)
    state["updated_at"] = _utc_now_iso()
    _save_enrichment_state_local(state)

    storage = _get_gcs_storage()
    if storage is None:
        return
    try:
        blob = storage.bucket.blob(ENRICHMENT_STATE_BLOB_NAME)
        blob.upload_from_string(
            json.dumps(state, indent=2, ensure_ascii=False),
            content_type="application/json",
        )
    except Exception as e:
        st.session_state["_enrichment_error"] = f"GCS enrichment-state save failed: {e}"


def _empty_policy_briefs_payload():
    return {
        "version": POLICY_BRIEFING_VERSION,
        "updated_at": "",
        "briefs": [],
    }


def _normalize_policy_briefs_payload(payload):
    if not isinstance(payload, dict):
        payload = {}
    briefs = payload.get("briefs", [])
    if not isinstance(briefs, list):
        briefs = []
    return {
        "version": str(payload.get("version", POLICY_BRIEFING_VERSION) or POLICY_BRIEFING_VERSION),
        "updated_at": str(payload.get("updated_at", "") or ""),
        "briefs": briefs,
    }


def _load_policy_briefs_local():
    if not POLICY_BRIEFS_LOCAL_PATH.exists():
        return _empty_policy_briefs_payload()
    try:
        with open(POLICY_BRIEFS_LOCAL_PATH, "r", encoding="utf-8") as f:
            return _normalize_policy_briefs_payload(json.load(f))
    except Exception:
        return _empty_policy_briefs_payload()


def _save_policy_briefs_local(payload):
    payload = _normalize_policy_briefs_payload(payload)
    POLICY_BRIEFS_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(POLICY_BRIEFS_LOCAL_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _load_policy_briefs():
    storage = _get_gcs_storage()
    if storage is not None:
        try:
            blob = storage.bucket.blob(POLICY_BRIEFS_BLOB_NAME)
            if blob.exists():
                payload = _normalize_policy_briefs_payload(
                    json.loads(blob.download_as_text(encoding="utf-8"))
                )
                _save_policy_briefs_local(payload)
                return payload
        except Exception as e:
            st.session_state["_policy_briefs_error"] = f"GCS policy-brief load failed: {e}"
    return _load_policy_briefs_local()


def _save_policy_briefs(payload):
    payload = _normalize_policy_briefs_payload(payload)
    payload["updated_at"] = _utc_now_iso()
    _save_policy_briefs_local(payload)

    storage = _get_gcs_storage()
    if storage is None:
        return
    try:
        blob = storage.bucket.blob(POLICY_BRIEFS_BLOB_NAME)
        blob.upload_from_string(
            json.dumps(payload, indent=2, ensure_ascii=False),
            content_type="application/json",
        )
    except Exception as e:
        st.session_state["_policy_briefs_error"] = f"GCS policy-brief save failed: {e}"


def _upsert_policy_delta_brief(payload, record):
    briefs = payload.get("briefs", [])
    record_doc_id = str(record.get("doc_id", "") or "").strip()
    record_org = str(record.get("org_key", "") or "").strip()
    replaced = False
    for idx, existing in enumerate(briefs):
        if not isinstance(existing, dict):
            continue
        existing_doc_id = str(existing.get("doc_id", "") or "").strip()
        existing_org = str(existing.get("org_key", "") or "").strip()
        if existing_doc_id == record_doc_id and existing_org == record_org:
            briefs[idx] = record
            replaced = True
            break
    if not replaced:
        briefs.append(record)
    payload["briefs"] = briefs
    return replaced


_POLICY_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "were", "have", "has", "had", "not", "are",
    "but", "you", "your", "our", "its", "their", "they", "about", "into", "there", "than", "then", "also",
    "because", "while", "would", "could", "should", "under", "after", "before", "over", "such", "more",
    "most", "some", "many", "much", "each", "other", "than", "been", "being", "will", "shall", "may",
    "can", "all", "any", "sec", "commission", "speech", "statement", "release", "litigation", "faq",
}


def _policy_tokenize(text, max_tokens=600):
    tokens = re.findall(r"[a-z0-9][a-z0-9_\-]{2,}", str(text or "").lower())
    out = []
    for token in tokens:
        if token in _POLICY_STOPWORDS:
            continue
        if token.isdigit():
            continue
        out.append(token)
        if len(out) >= max_tokens:
            break
    return set(out)


def _overlap_score(left, right):
    if not left or not right:
        return 0.0
    inter = len(left.intersection(right))
    denom = len(left.union(right))
    if denom <= 0:
        return 0.0
    return inter / denom


def _build_policy_doc_rows(knowledge_data, enrichment_state, org_key=None, org_keys=None):
    entries = enrichment_state.get("entries", {}) if isinstance(enrichment_state, dict) else {}
    if not isinstance(entries, dict):
        entries = {}
    allowed_org_keys = set()
    if isinstance(org_keys, (list, tuple, set)):
        allowed_org_keys = {str(k).strip() for k in org_keys if str(k).strip()}
    elif org_key and str(org_key).strip() and str(org_key).strip() != "__all__":
        allowed_org_keys = {str(org_key).strip()}

    rows = []
    for speech in knowledge_data.get("speeches", []):
        if not isinstance(speech, dict):
            continue
        doc_org_key = _speech_org_key(speech)
        if allowed_org_keys and doc_org_key not in allowed_org_keys:
            continue

        content = speech.get("content", {}) if isinstance(speech.get("content", {}), dict) else {}
        full_text = str(content.get("full_text", "") or "").strip()
        if not full_text:
            continue
        metadata = speech.get("metadata", {}) if isinstance(speech.get("metadata", {}), dict) else {}
        doc_id = _corpus_doc_id(speech)
        enrich_entry = entries.get(doc_id, {}) if isinstance(entries.get(doc_id, {}), dict) else {}
        enrich = enrich_entry.get("enrichment", {}) if isinstance(enrich_entry.get("enrichment", {}), dict) else {}

        tags = enrich.get("tags", [])
        if not isinstance(tags, list):
            tags = metadata.get("tags", []) if isinstance(metadata.get("tags", []), list) else []
        tags = [str(t).strip() for t in tags if str(t).strip()]

        keywords = enrich.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(k).strip() for k in keywords if str(k).strip()]

        stance = enrich.get("stance", {})
        if isinstance(stance, dict):
            stance_label = str(stance.get("label", "") or "").strip()
        else:
            stance_label = str(stance or "").strip()

        date_text = str(metadata.get("date", "") or "").strip()
        date_parsed = _parse_single_date(date_text)

        rows.append(
            {
                "doc_id": doc_id,
                "organization": _speech_org_label(speech),
                "org_key": doc_org_key,
                "date": date_text,
                "date_parsed": date_parsed,
                "title": str(metadata.get("title", "") or "").strip(),
                "speaker": str(metadata.get("speaker", "") or "").strip(),
                "url": str(metadata.get("url", "") or "").strip(),
                "doc_type": str(metadata.get("doc_type", "Document") or "Document").strip(),
                "source_kind": _infer_source_kind(metadata),
                "word_count": _coerce_int(metadata.get("word_count", 0), default=0, min_value=0),
                "tags": tags,
                "keywords": keywords,
                "stance_label": stance_label,
                "full_text": full_text,
            }
        )
    return rows


def _score_policy_similarity(source_doc, prior_doc):
    source_text_tokens = _policy_tokenize(
        f"{source_doc.get('title', '')}\n{source_doc.get('full_text', '')[:9000]}",
        max_tokens=700,
    )
    prior_text_tokens = _policy_tokenize(
        f"{prior_doc.get('title', '')}\n{prior_doc.get('full_text', '')[:6000]}",
        max_tokens=500,
    )
    source_topic_tokens = _policy_tokenize(
        " ".join(source_doc.get("tags", [])) + " " + " ".join(source_doc.get("keywords", [])),
        max_tokens=180,
    )
    prior_topic_tokens = _policy_tokenize(
        " ".join(prior_doc.get("tags", [])) + " " + " ".join(prior_doc.get("keywords", [])),
        max_tokens=180,
    )
    source_title_tokens = _policy_tokenize(source_doc.get("title", ""), max_tokens=80)
    prior_title_tokens = _policy_tokenize(prior_doc.get("title", ""), max_tokens=80)

    text_overlap = _overlap_score(source_text_tokens, prior_text_tokens)
    topic_overlap = _overlap_score(source_topic_tokens, prior_topic_tokens)
    title_overlap = _overlap_score(source_title_tokens, prior_title_tokens)

    recency = 0.0
    source_date = source_doc.get("date_parsed")
    prior_date = prior_doc.get("date_parsed")
    if pd.notna(source_date) and pd.notna(prior_date):
        day_delta = (source_date - prior_date).days
        if day_delta > 0:
            recency = 1.0 / (1.0 + (day_delta / 365.0))

    score = (0.6 * text_overlap) + (0.2 * topic_overlap) + (0.1 * title_overlap) + (0.1 * recency)
    return _coerce_float(score, default=0.0, min_value=0.0, max_value=1.0), {
        "text_overlap": round(text_overlap, 4),
        "topic_overlap": round(topic_overlap, 4),
        "title_overlap": round(title_overlap, 4),
        "recency": round(recency, 4),
    }


def _select_prior_docs_for_policy_delta(
    knowledge_data,
    enrichment_state,
    source_doc_id,
    org_key=None,
    source_org_keys=None,
    compare_org_keys=None,
    compare_source_kinds=None,
    lookback_days=730,
    max_candidates=20,
):
    all_docs = _build_policy_doc_rows(knowledge_data, enrichment_state)
    source = next((d for d in all_docs if str(d.get("doc_id", "")) == str(source_doc_id)), None)
    if not source:
        return None, []
    source_org_scope = set()
    if isinstance(source_org_keys, (list, tuple, set)):
        source_org_scope = {str(k).strip() for k in source_org_keys if str(k).strip()}
    elif org_key and str(org_key).strip() and str(org_key).strip() != "__all__":
        source_org_scope = {str(org_key).strip()}
    if source_org_scope and str(source.get("org_key", "") or "").strip() not in source_org_scope:
        return None, []

    prior_scope_orgs = set()
    if isinstance(compare_org_keys, (list, tuple, set)):
        prior_scope_orgs = {str(k).strip() for k in compare_org_keys if str(k).strip()}
    elif source_org_scope:
        prior_scope_orgs = set(source_org_scope)

    docs = _build_policy_doc_rows(knowledge_data, enrichment_state, org_keys=prior_scope_orgs or None)
    compare_source_kinds_set = set()
    if isinstance(compare_source_kinds, (list, tuple, set)):
        compare_source_kinds_set = {str(k).strip() for k in compare_source_kinds if str(k).strip()}

    lookback_days = _coerce_int(lookback_days, default=730, min_value=1)
    max_candidates = _coerce_int(max_candidates, default=20, min_value=1)
    source_date = source.get("date_parsed")

    scored = []
    for candidate in docs:
        if candidate.get("doc_id") == source.get("doc_id"):
            continue
        if compare_source_kinds_set and str(candidate.get("source_kind", "") or "").strip() not in compare_source_kinds_set:
            continue

        cand_date = candidate.get("date_parsed")
        if pd.notna(source_date) and pd.notna(cand_date):
            if cand_date >= source_date:
                continue
            if (source_date - cand_date).days > lookback_days:
                continue

        sim_score, components = _score_policy_similarity(source, candidate)
        if sim_score < 0.02:
            continue
        scored.append(
            {
                **candidate,
                "similarity_score": sim_score,
                "similarity_components": components,
            }
        )

    if not scored:
        fallback = [d for d in docs if d.get("doc_id") != source.get("doc_id")]
        if pd.notna(source_date):
            fallback = [
                d for d in fallback
                if pd.isna(d.get("date_parsed")) or d.get("date_parsed") < source_date
            ]
        fallback = sorted(
            fallback,
            key=lambda x: x.get("date_parsed") if pd.notna(x.get("date_parsed")) else pd.Timestamp.min,
            reverse=True,
        )
        scored = [{**d, "similarity_score": 0.0, "similarity_components": {}} for d in fallback[:max_candidates]]
    else:
        scored = sorted(
            scored,
            key=lambda x: (
                x.get("similarity_score", 0.0),
                x.get("date_parsed") if pd.notna(x.get("date_parsed")) else pd.Timestamp.min,
            ),
            reverse=True,
        )[:max_candidates]
    return source, scored


def _normalize_policy_delta_brief_obj(payload):
    if not isinstance(payload, dict):
        payload = {}

    allowed_positions = {"continuity", "mixed_shift", "meaningful_shift", "novel_position"}
    allowed_intensity = {"low", "medium", "high"}
    allowed_direction = {
        "more_supportive", "more_cautious", "more_critical", "unchanged", "mixed", "unclear"
    }
    allowed_labels = {"builds_upon", "expansion", "narrowing", "shift", "contradiction", "novel"}

    classifications = payload.get("classifications", [])
    if not isinstance(classifications, list):
        classifications = []
    normalized_classifications = []
    for row in classifications[:12]:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label", "") or "").strip().lower()
        if label not in allowed_labels:
            continue
        evidence = row.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []
        evidence_out = []
        for ev in evidence[:6]:
            if not isinstance(ev, dict):
                continue
            evidence_out.append(
                {
                    "source": str(ev.get("source", "") or "").strip(),
                    "doc_title": str(ev.get("doc_title", "") or "").strip(),
                    "quote": str(ev.get("quote", "") or "").strip()[:400],
                    "date": str(ev.get("date", "") or "").strip(),
                }
            )
        normalized_classifications.append(
            {
                "label": label,
                "confidence": _coerce_float(row.get("confidence", 0.0), default=0.0, min_value=0.0, max_value=1.0),
                "description": str(row.get("description", "") or "").strip(),
                "evidence": evidence_out,
            }
        )

    overall_position = str(payload.get("overall_position", "") or "").strip().lower()
    if overall_position not in allowed_positions:
        overall_position = "mixed_shift"
    change_intensity = str(payload.get("change_intensity", "") or "").strip().lower()
    if change_intensity not in allowed_intensity:
        change_intensity = "medium"
    stance_direction = str(payload.get("stance_direction", "") or "").strip().lower()
    if stance_direction not in allowed_direction:
        stance_direction = "unclear"

    new_elements = payload.get("new_elements", [])
    if not isinstance(new_elements, list):
        new_elements = []
    continued_elements = payload.get("continued_elements", [])
    if not isinstance(continued_elements, list):
        continued_elements = []
    changed_elements = payload.get("changed_elements", [])
    if not isinstance(changed_elements, list):
        changed_elements = []
    legal_risk_points = payload.get("legal_risk_points", [])
    if not isinstance(legal_risk_points, list):
        legal_risk_points = []

    return {
        "overall_position": overall_position,
        "change_intensity": change_intensity,
        "stance_direction": stance_direction,
        "continuity_score": _coerce_float(payload.get("continuity_score", 0.5), default=0.5, min_value=0.0, max_value=1.0),
        "novelty_score": _coerce_float(payload.get("novelty_score", 0.5), default=0.5, min_value=0.0, max_value=1.0),
        "confidence": _coerce_float(payload.get("confidence", 0.6), default=0.6, min_value=0.0, max_value=1.0),
        "executive_summary": str(payload.get("executive_summary", "") or "").strip(),
        "new_elements": [str(x).strip() for x in new_elements[:8] if str(x).strip()],
        "continued_elements": [str(x).strip() for x in continued_elements[:8] if str(x).strip()],
        "changed_elements": [str(x).strip() for x in changed_elements[:8] if str(x).strip()],
        "legal_risk_points": [str(x).strip() for x in legal_risk_points[:10] if str(x).strip()],
        "classifications": normalized_classifications,
    }


def _heuristic_policy_delta_brief(source_doc, prior_docs):
    if not prior_docs:
        return {
            "overall_position": "novel_position",
            "change_intensity": "high",
            "stance_direction": "unclear",
            "continuity_score": 0.15,
            "novelty_score": 0.85,
            "confidence": 0.55,
            "executive_summary": "No closely related prior documents were found in the selected lookback window.",
            "new_elements": ["New document appears to introduce topics not strongly matched in prior corpus."],
            "continued_elements": [],
            "changed_elements": ["Insufficient evidence for continuity; classify as potentially novel."],
            "legal_risk_points": [],
            "classifications": [
                {
                    "label": "novel",
                    "confidence": 0.55,
                    "description": "No high-similarity prior document was identified.",
                    "evidence": [],
                }
            ],
        }

    top = prior_docs[0]
    top_score = _coerce_float(top.get("similarity_score", 0.0), default=0.0, min_value=0.0, max_value=1.0)
    source_stance = str(source_doc.get("stance_label", "") or "").strip().lower()
    prior_stance = str(top.get("stance_label", "") or "").strip().lower()
    stance_shift = bool(source_stance and prior_stance and source_stance != prior_stance)

    if top_score >= 0.5 and not stance_shift:
        overall_position = "continuity"
        change_intensity = "low"
        continuity = 0.78
        novelty = 0.32
        label = "builds_upon"
    elif top_score >= 0.32:
        overall_position = "mixed_shift"
        change_intensity = "medium"
        continuity = 0.58
        novelty = 0.52
        label = "expansion" if not stance_shift else "shift"
    else:
        overall_position = "meaningful_shift"
        change_intensity = "high"
        continuity = 0.33
        novelty = 0.73
        label = "shift"

    changed = []
    if stance_shift:
        changed.append(
            f"Stance label shifted from `{top.get('stance_label', 'unknown')}` to `{source_doc.get('stance_label', 'unknown')}`."
        )
    elif top_score < 0.5:
        changed.append("Topic overlap with the nearest prior document is limited.")

    return {
        "overall_position": overall_position,
        "change_intensity": change_intensity,
        "stance_direction": "mixed" if stance_shift else "unchanged",
        "continuity_score": continuity,
        "novelty_score": novelty,
        "confidence": 0.62,
        "executive_summary": f"Heuristic comparison against top prior doc `{top.get('title', '')}` (score {top_score:.2f}).",
        "new_elements": ["Review detailed claim-level differences in the generated classifications."],
        "continued_elements": [f"Top prior match: `{top.get('title', '')}` ({top.get('date', '')})."],
        "changed_elements": changed,
        "legal_risk_points": [],
        "classifications": [
            {
                "label": label,
                "confidence": 0.62,
                "description": "Heuristic baseline classification using corpus similarity and stance comparison.",
                "evidence": [],
            }
        ],
    }


def _run_policy_delta_brief_llm(client, source_doc, prior_docs, model_name):
    source_excerpt = str(source_doc.get("full_text", "") or "")[:12000]
    prior_sections = []
    for idx, p in enumerate(prior_docs[:18], 1):
        prior_excerpt = str(p.get("full_text", "") or "")[:1800]
        prior_sections.append(
            "\n".join(
                [
                    f"[{idx}]",
                    f"Title: {p.get('title', '')}",
                    f"Date: {p.get('date', '')}",
                    f"Type: {p.get('doc_type', '')}",
                    f"Source Kind: {p.get('source_kind', '')}",
                    f"Similarity Score: {p.get('similarity_score', 0.0):.3f}",
                    f"Tags: {', '.join(p.get('tags', [])[:12])}",
                    f"Keywords: {', '.join(p.get('keywords', [])[:16])}",
                    f"Stance: {p.get('stance_label', '')}",
                    f"Excerpt:\n{prior_excerpt}",
                ]
            )
        )
    priors_text = "\n\n".join(prior_sections)

    system_prompt = (
        "You are a policy-change analyst. Compare one new source document against prior corpus documents and "
        "classify whether the new position is continuity, expansion, narrowing, shift, contradiction, or novel. "
        "Return strict JSON only."
    )
    user_prompt = (
        "Output JSON with keys:\n"
        "- overall_position: continuity|mixed_shift|meaningful_shift|novel_position\n"
        "- change_intensity: low|medium|high\n"
        "- stance_direction: more_supportive|more_cautious|more_critical|unchanged|mixed|unclear\n"
        "- continuity_score: number 0..1\n"
        "- novelty_score: number 0..1\n"
        "- confidence: number 0..1\n"
        "- executive_summary: string (max 120 words)\n"
        "- new_elements: string[]\n"
        "- continued_elements: string[]\n"
        "- changed_elements: string[]\n"
        "- legal_risk_points: string[]\n"
        "- classifications: [{label, confidence, description, evidence:[{source,new_or_prior,doc_title,quote,date}]}]\n"
        "\nRules:\n"
        "- Use only provided text.\n"
        "- Evidence quotes must be short and concrete.\n"
        "- If unsure, lower confidence rather than hallucinating.\n\n"
        f"New Document Title: {source_doc.get('title', '')}\n"
        f"New Document Date: {source_doc.get('date', '')}\n"
        f"New Document Type: {source_doc.get('doc_type', '')}\n"
        f"New Document Source Kind: {source_doc.get('source_kind', '')}\n"
        f"New Document Tags: {', '.join(source_doc.get('tags', [])[:12])}\n"
        f"New Document Keywords: {', '.join(source_doc.get('keywords', [])[:20])}\n"
        f"New Document Stance: {source_doc.get('stance_label', '')}\n"
        f"New Document Excerpt:\n{source_excerpt}\n\n"
        f"Prior Documents:\n{priors_text}\n"
    )

    response = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=2200,
    )
    raw_text = str(getattr(response, "output_text", "") or "").strip()
    if not raw_text:
        raw_text = str(_normalize_obj(response).get("output_text", "") or "").strip()
    parsed = _extract_first_json_object(raw_text)
    if not parsed:
        raise RuntimeError("Policy delta briefing returned non-JSON output.")
    return _normalize_policy_delta_brief_obj(parsed)


def _generate_policy_delta_brief(
    client,
    knowledge_data,
    enrichment_state,
    source_doc_id,
    org_key=None,
    source_org_keys=None,
    compare_org_keys=None,
    compare_source_kinds=None,
    model_name="gpt-4o-mini",
    lookback_days=730,
    max_candidates=20,
):
    source_doc, prior_docs = _select_prior_docs_for_policy_delta(
        knowledge_data=knowledge_data,
        enrichment_state=enrichment_state,
        source_doc_id=source_doc_id,
        org_key=org_key,
        source_org_keys=source_org_keys,
        compare_org_keys=compare_org_keys,
        compare_source_kinds=compare_source_kinds,
        lookback_days=lookback_days,
        max_candidates=max_candidates,
    )
    if not source_doc:
        raise ValueError("Selected source document was not found in the scoped corpus.")

    error = ""
    engine = "heuristic"
    if client is not None:
        try:
            brief = _run_policy_delta_brief_llm(client, source_doc, prior_docs, model_name)
            engine = "llm"
        except Exception as e:
            error = str(e)
            brief = _heuristic_policy_delta_brief(source_doc, prior_docs)
            engine = "heuristic_fallback"
    else:
        brief = _heuristic_policy_delta_brief(source_doc, prior_docs)

    brief_id = hashlib.sha256(
        (
            f"{org_key or source_doc.get('org_key', '')}|{source_doc_id}|policy_delta|"
            f"{','.join(sorted([str(k).strip() for k in (compare_org_keys or []) if str(k).strip()]))}|"
            f"{','.join(sorted([str(k).strip() for k in (compare_source_kinds or []) if str(k).strip()]))}|"
            f"{POLICY_BRIEFING_VERSION}"
        ).encode("utf-8")
    ).hexdigest()[:24]

    source_snapshot = {
        "doc_id": source_doc.get("doc_id", ""),
        "title": source_doc.get("title", ""),
        "date": source_doc.get("date", ""),
        "organization": source_doc.get("organization", ""),
        "doc_type": source_doc.get("doc_type", ""),
        "source_kind": source_doc.get("source_kind", ""),
        "speaker": source_doc.get("speaker", ""),
        "url": source_doc.get("url", ""),
        "tags": source_doc.get("tags", []),
        "keywords": source_doc.get("keywords", []),
        "stance_label": source_doc.get("stance_label", ""),
        "word_count": source_doc.get("word_count", 0),
    }

    prior_snapshot = []
    for p in prior_docs:
        prior_snapshot.append(
            {
                "doc_id": p.get("doc_id", ""),
                "title": p.get("title", ""),
                "date": p.get("date", ""),
                "doc_type": p.get("doc_type", ""),
                "source_kind": p.get("source_kind", ""),
                "speaker": p.get("speaker", ""),
                "url": p.get("url", ""),
                "similarity_score": round(_coerce_float(p.get("similarity_score", 0.0), default=0.0), 4),
                "similarity_components": p.get("similarity_components", {}),
            }
        )

    return {
        "brief_id": brief_id,
        "doc_id": str(source_doc_id or "").strip(),
        "org_key": str(source_doc.get("org_key", "") or org_key or "").strip(),
        "org_label": str(source_doc.get("organization", "") or "").strip(),
        "generated_at": _utc_now_iso(),
        "model": str(model_name or "").strip(),
        "engine": engine,
        "status": "generated",
        "error": error,
        "comparison": {
            "method": "local_similarity_v1",
            "lookback_days": _coerce_int(lookback_days, default=730, min_value=1),
            "candidate_count": len(prior_snapshot),
            "max_candidates": _coerce_int(max_candidates, default=20, min_value=1),
            "source_org_keys": [str(k).strip() for k in (source_org_keys or []) if str(k).strip()],
            "compare_org_keys": [str(k).strip() for k in (compare_org_keys or []) if str(k).strip()],
            "compare_source_kinds": [str(k).strip() for k in (compare_source_kinds or []) if str(k).strip()],
        },
        "source_doc": source_snapshot,
        "prior_docs": prior_snapshot,
        "brief": brief,
    }


def _format_policy_brief_context_block(brief):
    source = brief.get("source_doc", {}) if isinstance(brief.get("source_doc", {}), dict) else {}
    detail = brief.get("brief", {}) if isinstance(brief.get("brief", {}), dict) else {}
    lines = [
        f"Title: {source.get('title', '')}",
        f"Organization: {source.get('organization', '')}",
        f"Date: {source.get('date', '')}",
        f"Type: {source.get('doc_type', '')}",
        f"Source Kind: {source.get('source_kind', '')}",
        f"Overall Position: {detail.get('overall_position', '')}",
        f"Change Intensity: {detail.get('change_intensity', '')}",
        f"Stance Direction: {detail.get('stance_direction', '')}",
        (
            f"Novelty/Continuity/Confidence: "
            f"{_coerce_float(detail.get('novelty_score', 0.0), default=0.0):.2f} / "
            f"{_coerce_float(detail.get('continuity_score', 0.0), default=0.0):.2f} / "
            f"{_coerce_float(detail.get('confidence', 0.0), default=0.0):.2f}"
        ),
        f"Executive Summary: {str(detail.get('executive_summary', '') or '').strip()}",
    ]
    for label, key in [
        ("New Elements", "new_elements"),
        ("Continued Elements", "continued_elements"),
        ("Changed Elements", "changed_elements"),
        ("Legal Risk Points", "legal_risk_points"),
    ]:
        vals = detail.get(key, [])
        if isinstance(vals, list) and vals:
            lines.append(f"{label}: " + " | ".join(str(v).strip() for v in vals[:6] if str(v).strip()))
    return "\n".join(lines)


def _policy_brief_org_key(brief):
    if not isinstance(brief, dict):
        return ""
    source = brief.get("source_doc", {}) if isinstance(brief.get("source_doc", {}), dict) else {}
    org_key = (
        brief.get("org_key", "")
        or source.get("org_key", "")
        or _org_key_from_label(source.get("organization", ""))
    )
    return str(org_key or "").strip()


def _select_policy_brief_context(
    briefs_payload,
    question,
    org_keys=None,
    source_kinds=None,
    limit=5,
):
    brief_rows = briefs_payload.get("briefs", []) if isinstance(briefs_payload, dict) else []
    if not isinstance(brief_rows, list):
        return [], "", []

    org_scope = set()
    if isinstance(org_keys, (list, tuple, set)):
        org_scope = {str(k).strip() for k in org_keys if str(k).strip()}
    source_scope = set()
    if isinstance(source_kinds, (list, tuple, set)):
        source_scope = {str(k).strip() for k in source_kinds if str(k).strip()}

    q_tokens = _policy_tokenize(question, max_tokens=120)
    limit = _coerce_int(limit, default=5, min_value=1)
    scored = []
    for brief in brief_rows:
        if not isinstance(brief, dict):
            continue
        source = brief.get("source_doc", {}) if isinstance(brief.get("source_doc", {}), dict) else {}
        brief_org_key = str(brief.get("org_key", "") or source.get("org_key", "") or _org_key_from_label(source.get("organization", ""))).strip()
        if org_scope and brief_org_key not in org_scope:
            continue
        brief_source_kind = str(source.get("source_kind", "") or "").strip()
        if source_scope and brief_source_kind not in source_scope:
            continue

        detail = brief.get("brief", {}) if isinstance(brief.get("brief", {}), dict) else {}
        blob = "\n".join(
            [
                str(source.get("title", "") or ""),
                str(source.get("doc_type", "") or ""),
                str(source.get("source_kind", "") or ""),
                str(detail.get("executive_summary", "") or ""),
                " ".join(detail.get("new_elements", []) if isinstance(detail.get("new_elements", []), list) else []),
                " ".join(detail.get("changed_elements", []) if isinstance(detail.get("changed_elements", []), list) else []),
                " ".join(detail.get("legal_risk_points", []) if isinstance(detail.get("legal_risk_points", []), list) else []),
            ]
        )
        b_tokens = _policy_tokenize(blob, max_tokens=240)
        overlap = len(q_tokens.intersection(b_tokens)) if q_tokens and b_tokens else 0
        conf = _coerce_float(detail.get("confidence", 0.0), default=0.0, min_value=0.0, max_value=1.0)
        recency_bonus = 0.0
        parsed = _parse_single_date(source.get("date", ""))
        if pd.notna(parsed):
            days_old = max(0, (pd.Timestamp(date.today()) - parsed).days)
            recency_bonus = 1.0 / (1.0 + (days_old / 365.0))
        score = (2.0 * overlap) + (0.8 * conf) + (0.4 * recency_bonus)
        scored.append((score, brief))

    scored = sorted(scored, key=lambda x: x[0], reverse=True)[:limit]
    selected = [row[1] for row in scored]

    context_blocks = []
    sources = []
    for idx, brief in enumerate(selected, 1):
        source = brief.get("source_doc", {}) if isinstance(brief.get("source_doc", {}), dict) else {}
        detail = brief.get("brief", {}) if isinstance(brief.get("brief", {}), dict) else {}
        block = _format_policy_brief_context_block(brief)
        context_blocks.append(f"[Brief {idx}]\n{block}")
        sources.append(
            {
                "filename": source.get("title", f"brief_{idx}"),
                "score": float(scored[idx - 1][0]),
                "file_id": brief.get("brief_id", ""),
                "snippet": str(detail.get("executive_summary", "") or "")[:300],
            }
        )

    return selected, "\n\n".join(context_blocks), sources


def _ask_policy_brief_chat(client, question, model_name, context_text, instructions_text=None):
    system_text = (
        "You are a policy research assistant. Answer using only the provided policy delta briefing context. "
        "If context is insufficient, say so and ask for a narrower scope."
    )
    if instructions_text:
        system_text = f"{system_text}\n\n{instructions_text}"

    user_text = (
        f"Question:\n{question}\n\n"
        f"Policy Delta Briefing Context:\n{context_text}\n\n"
        "Use this context only."
    )
    response = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        max_output_tokens=1600,
    )
    return {"answer": _extract_response_text(response), "results": []}


def _ask_policy_brief_chat_with_fallback(client, question, preferred_model, model_pool, context_text, instructions_text=None):
    ordered = [preferred_model] + [m for m in model_pool if m != preferred_model]
    last_error = None
    for idx, model_name in enumerate(ordered):
        try:
            result = _ask_policy_brief_chat(
                client=client,
                question=question,
                model_name=model_name,
                context_text=context_text,
                instructions_text=instructions_text,
            )
            return {"result": result, "used_model": model_name, "fallback_used": idx > 0}
        except Exception as e:
            last_error = e
            if not _is_model_access_error(e):
                raise
            continue
    if last_error:
        raise last_error
    raise RuntimeError("No model available for policy-brief chat request.")


def _corpus_doc_id(speech):
    m = speech.get("metadata", {})
    existing = str(m.get("document_id", "") or "").strip()
    if existing:
        return existing
    stable = "|".join(
        [
            _speech_org_key(speech),
            str(m.get("url", "") or ""),
            str(m.get("title", "") or ""),
            str(m.get("speaker", "") or ""),
            str(m.get("date", "") or ""),
        ]
    )
    if not stable.strip("|"):
        text = str(speech.get("content", {}).get("full_text", "") or "")
        stable = text[:1000]
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]


def _build_enrichment_candidates(knowledge_data, org_key=None):
    dedup = {}
    for speech in knowledge_data.get("speeches", []):
        key = _speech_org_key(speech)
        if org_key and org_key != "__all__" and key != org_key:
            continue
        text = str(speech.get("content", {}).get("full_text", "") or "").strip()
        if not text:
            continue
        m = speech.get("metadata", {})
        doc_id = _corpus_doc_id(speech)
        dedup[doc_id] = {
            "doc_id": doc_id,
            "organization": _speech_org_label(speech),
            "org_key": key,
            "title": str(m.get("title", "") or "").strip(),
            "speaker": str(m.get("speaker", "") or "").strip(),
            "date": str(m.get("date", "") or "").strip(),
            "url": str(m.get("url", "") or "").strip(),
            "doc_type": str(m.get("doc_type", "Speech") or "Speech").strip(),
            "full_text": text,
            "word_count": _coerce_int(m.get("word_count", 0), default=0, min_value=0),
        }
    return list(dedup.values())


def _extract_first_json_object(text):
    raw = str(text or "").strip()
    if not raw:
        return {}

    # Direct parse.
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Strip fenced markdown.
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


def _normalize_enrichment_payload(payload):
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
                        {
                            "name": name,
                            "type": etype or "OTHER",
                            "mentions": max(1, mentions),
                        }
                    )

    stance = payload.get("stance", {})
    if not isinstance(stance, dict):
        stance = {"label": str(stance)}
    stance_label = str(stance.get("label", "unclear") or "unclear").strip().lower()
    allowed_stance = {"supportive", "cautious", "critical", "neutral", "unclear"}
    if stance_label not in allowed_stance:
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
                    normalized_evidence.append(
                        {
                            "claim": claim,
                            "snippet": snippet[:600],
                        }
                    )

    try:
        confidence = float(payload.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    summary = str(payload.get("summary", "") or "").strip()[:1200]

    return {
        "summary": summary,
        "tags": tags,
        "keywords": keywords,
        "entities": normalized_entities,
        "stance": {"label": stance_label, "target": stance_target},
        "evidence_spans": normalized_evidence,
        "confidence": confidence,
    }


def _heuristic_enrichment(doc):
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
    freq = {}
    for w in words:
        if (
            w in stop
            or len(w) > 28
            or any(ch.isdigit() for ch in w)
            or w.count("-") > 1
        ):
            continue
        freq[w] = freq.get(w, 0) + 1
    keywords = [w for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:12]]

    stance_label = "neutral"
    if any(t in lower for t in ["support", "welcome", "encourage", "approve"]):
        stance_label = "supportive"
    if any(t in lower for t in ["risk", "concern", "caution", "guardrail"]):
        stance_label = "cautious"
    if any(t in lower for t in ["oppose", "reject", "critic", "harmful"]):
        stance_label = "critical"

    evidence = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines[:80]:
        low = line.lower()
        if any(t in low for t in ["investor", "risk", "disclosure", "crypto", "enforcement"]):
            evidence.append({"claim": "Potentially relevant policy statement", "snippet": line[:500]})
            if len(evidence) >= 3:
                break

    return _normalize_enrichment_payload(
        {
            "summary": (lines[0] if lines else str(doc.get("title", "")))[:300],
            "tags": tags,
            "keywords": keywords,
            "entities": [],
            "stance": {"label": stance_label, "target": ""},
            "evidence_spans": evidence,
            "confidence": 0.35,
        }
    )


def _run_enrichment_agent(client, doc, model_name):
    text = str(doc.get("full_text", "") or "").strip()
    if len(text) > 90000:
        text = text[:45000] + "\n\n[...TRUNCATED FOR ENRICHMENT...]\n\n" + text[-30000:]

    base_instruction = (
        "You are an enrichment agent for policy documents. "
        "Return ONLY valid JSON with keys: "
        "summary, tags, keywords, entities, stance, evidence_spans, confidence. "
        "Use concise tags and keywords. "
        "entities must be list of {name,type,mentions}. "
        "stance must be {label,target} where label in [supportive,cautious,critical,neutral,unclear]. "
        "evidence_spans must include verbatim snippets from the document."
    )
    prompt = (
        f"Organization: {doc.get('organization', '')}\n"
        f"Title: {doc.get('title', '')}\n"
        f"Speaker: {doc.get('speaker', '')}\n"
        f"Date: {doc.get('date', '')}\n"
        f"Type: {doc.get('doc_type', '')}\n\n"
        f"Document Text:\n{text}"
    )

    last_raw = ""
    for attempt in range(1, 3):
        instruction = base_instruction
        if attempt > 1:
            instruction += " Respond with raw JSON only. No markdown, no commentary, no code fences."

        response = client.responses.create(
            model=model_name,
            instructions=instruction,
            input=prompt,
        )
        raw_text = _extract_response_text(response)
        last_raw = raw_text
        parsed = _extract_first_json_object(raw_text)
        if parsed:
            return _normalize_enrichment_payload(parsed)

    preview = (last_raw or "").replace("\n", " ").strip()[:300]
    raise RuntimeError(f"Model did not return parseable JSON after 2 attempts. Last output: {preview}")


def _compute_reward(enrichment, review_decision, status="enriched"):
    schema_validity = 1.0 if enrichment.get("tags") and enrichment.get("keywords") else 0.6
    evidence_quality = min(1.0, len(enrichment.get("evidence_spans", [])) / 3.0)
    confidence = float(enrichment.get("confidence", 0.0) or 0.0)
    confidence = max(0.0, min(1.0, confidence))

    review_map = {
        "accepted": 1.0,
        "edited": 0.8,
        "pending": 0.6,
        "rejected": 0.2,
    }
    review_component = review_map.get(str(review_decision or "pending"), 0.6)

    base_score = (
        0.35 * schema_validity
        + 0.30 * evidence_quality
        + 0.20 * confidence
        + 0.15 * review_component
    )
    status_multipliers = {
        "enriched": 1.0,
        "reviewed": 1.0,
        "fallback_enriched": 0.6,
        "failed": 0.2,
    }
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


def _select_enrichment_targets(candidates, enrichment_state, mode):
    entries = enrichment_state.get("entries", {})
    selected = []
    for doc in candidates:
        existing = entries.get(doc["doc_id"], {})
        status = str(existing.get("status", "") or "")
        if mode == "only_missing_or_failed":
            # Retry items that are missing, failed, or fallback_enriched.
            if status in ("enriched", "reviewed"):
                continue
        elif mode == "only_pending_review":
            review = existing.get("review", {})
            decision = str(review.get("decision", "pending") or "pending")
            if decision != "pending":
                continue
        selected.append(doc)
    return selected


def _run_enrichment_batch(client, candidates, enrichment_state, model_name, mode, limit, progress_callback=None):
    entries = enrichment_state.setdefault("entries", {})
    targets = _select_enrichment_targets(candidates, enrichment_state, mode)
    if limit and limit > 0:
        targets = targets[:limit]

    total = len(targets)
    processed = 0

    for doc in targets:
        if progress_callback is not None:
            progress_callback(processed, total, f"Enriching {doc.get('title', doc['doc_id'])}")

        doc_id = doc["doc_id"]
        review = entries.get(doc_id, {}).get("review", {})
        decision = str(review.get("decision", "pending") or "pending")
        notes = str(review.get("notes", "") or "")

        status = "enriched"
        error_msg = ""
        try:
            enrichment = _run_enrichment_agent(client, doc, model_name)
        except Exception as e:
            enrichment = _heuristic_enrichment(doc)
            status = "fallback_enriched"
            error_msg = str(e)

        reward = _compute_reward(enrichment, decision, status=status)
        entries[doc_id] = {
            "doc_id": doc_id,
            "organization": doc.get("organization", ""),
            "org_key": doc.get("org_key", ""),
            "title": doc.get("title", ""),
            "speaker": doc.get("speaker", ""),
            "date": doc.get("date", ""),
            "url": doc.get("url", ""),
            "doc_type": doc.get("doc_type", ""),
            "word_count": _coerce_int(doc.get("word_count", 0), default=0, min_value=0),
            "status": status,
            "error": error_msg,
            "model": model_name,
            "pipeline_version": ENRICHMENT_PIPELINE_VERSION,
            "updated_at": _utc_now_iso(),
            "enrichment": enrichment,
            "review": {
                "decision": decision,
                "notes": notes,
                "reviewed_at": str(review.get("reviewed_at", "") or ""),
            },
            "reward": reward,
        }
        processed += 1

        # checkpoint every 10 docs
        if processed % 10 == 0:
            _save_enrichment_state(enrichment_state)

    if progress_callback is not None:
        progress_callback(total, total, "Enrichment run complete")

    _save_enrichment_state(enrichment_state)
    return {
        "processed": processed,
        "total_selected": total,
    }


def _normalize_auto_review_payload(payload):
    if not isinstance(payload, dict):
        payload = {}

    verdict = str(payload.get("verdict", "human_followup") or "human_followup").strip().lower()
    if verdict not in {"approved", "human_followup"}:
        verdict = "human_followup"

    rationale = str(payload.get("rationale", "") or "").strip()[:900]

    issues = payload.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    issues = [str(x).strip() for x in issues if str(x).strip()][:12]

    try:
        confidence = float(payload.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "verdict": verdict,
        "rationale": rationale,
        "issues": issues,
        "confidence": confidence,
    }


def _heuristic_auto_review(entry):
    enrichment = entry.get("enrichment", {}) if isinstance(entry, dict) else {}
    status = str((entry or {}).get("status", "") or "").strip().lower()
    summary = str(enrichment.get("summary", "") or "").strip().lower()
    tags = enrichment.get("tags", []) if isinstance(enrichment, dict) else []
    keywords = enrichment.get("keywords", []) if isinstance(enrichment, dict) else []
    evidence = enrichment.get("evidence_spans", []) if isinstance(enrichment, dict) else []

    issues = []
    if status == "fallback_enriched":
        issues.append("Used fallback enrichment due to prior LLM failure.")
    if not summary:
        issues.append("Summary is empty.")
    if summary.startswith("## more in this section"):
        issues.append("Summary appears to be page boilerplate.")
    if len(tags) < 2:
        issues.append("Too few tags.")
    if len(keywords) < 4:
        issues.append("Too few keywords.")
    if len(evidence) < 1:
        issues.append("No evidence spans.")

    noisy_terms = {"https", "http", "www", "newsroom", "speeches", "statements", "ednref", "secgov", "html"}
    if any(str(k).strip().lower() in noisy_terms for k in keywords[:12]):
        issues.append("Keywords include URL/navigation noise.")

    verdict = "approved" if not issues else "human_followup"
    confidence = 0.78 if verdict == "approved" else 0.35
    rationale = (
        "Enrichment appears complete and consistent."
        if verdict == "approved"
        else "Enrichment needs human review due to quality/risk signals."
    )
    return _normalize_auto_review_payload(
        {
            "verdict": verdict,
            "rationale": rationale,
            "issues": issues,
            "confidence": confidence,
        }
    )


def _run_auto_review_agent(client, entry, model_name, source_text):
    enrichment_obj = entry.get("enrichment", {}) if isinstance(entry, dict) else {}
    source_text = str(source_text or "").strip()
    if len(source_text) > 22000:
        source_text = source_text[:14000] + "\n\n[...TRUNCATED...]\n\n" + source_text[-6000:]

    instructions = (
        "You are a quality-control reviewer for policy-document enrichment output. "
        "Decide if the enrichment can be auto-approved or should be flagged for human follow-up. "
        "Return ONLY valid JSON with keys: verdict, rationale, issues, confidence. "
        "verdict must be one of: approved, human_followup. "
        "Use human_followup whenever output quality is doubtful or contains boilerplate/noise."
    )
    prompt = (
        f"Document metadata:\n"
        f"- Title: {entry.get('title', '')}\n"
        f"- Organization: {entry.get('organization', '')}\n"
        f"- Date: {entry.get('date', '')}\n"
        f"- Speaker: {entry.get('speaker', '')}\n"
        f"- Status: {entry.get('status', '')}\n"
        f"- Prior Error: {entry.get('error', '')}\n\n"
        f"Enrichment JSON:\n{json.dumps(enrichment_obj, ensure_ascii=False)}\n\n"
        f"Source text sample:\n{source_text}"
    )
    response = client.responses.create(
        model=model_name,
        instructions=instructions,
        input=prompt,
    )
    parsed = _extract_first_json_object(_extract_response_text(response))
    if not parsed:
        raise RuntimeError("Auto-review model did not return parseable JSON.")
    return _normalize_auto_review_payload(parsed)


def _select_auto_review_targets(scoped_entries, mode):
    selected = []
    eligible_status = {"enriched", "fallback_enriched", "reviewed"}
    for entry in scoped_entries:
        status = str(entry.get("status", "") or "").strip().lower()
        if status not in eligible_status:
            continue
        decision = str(entry.get("review", {}).get("decision", "pending") or "pending").strip().lower()
        if mode == "only_pending" and decision != "pending":
            continue
        selected.append(entry)
    return selected


def _run_auto_review_batch(
    client,
    scoped_entries,
    candidate_map,
    enrichment_state,
    model_name,
    mode,
    limit,
    progress_callback=None,
):
    entries = enrichment_state.setdefault("entries", {})
    targets = _select_auto_review_targets(scoped_entries, mode)
    if limit and limit > 0:
        targets = targets[:limit]

    total = len(targets)
    processed = 0
    approved = 0
    flagged = 0
    heuristic_fallbacks = 0

    for item in targets:
        if progress_callback is not None:
            progress_callback(processed, total, f"Reviewing {item.get('title', item.get('doc_id', 'document'))}")

        doc_id = str(item.get("doc_id", "") or "").strip()
        if not doc_id:
            continue
        current = entries.get(doc_id, item)
        source_text = str((candidate_map.get(doc_id, {}) or {}).get("full_text", "") or "")

        auto_engine = "llm"
        try:
            auto_review = _run_auto_review_agent(client, current, model_name, source_text)
        except Exception as e:
            auto_review = _heuristic_auto_review(current)
            auto_review["issues"] = [f"LLM review failed: {e}"] + list(auto_review.get("issues", []))
            auto_engine = "heuristic"
            heuristic_fallbacks += 1

        verdict = auto_review.get("verdict", "human_followup")
        decision = "accepted" if verdict == "approved" else "pending"
        if decision == "accepted":
            approved += 1
        else:
            flagged += 1

        issue_text = "; ".join(auto_review.get("issues", [])[:6])
        confidence = float(auto_review.get("confidence", 0.0) or 0.0)
        review_notes = (
            f"[AUTO_REVIEW] verdict={verdict}; confidence={confidence:.2f}; engine={auto_engine}\n"
            f"rationale: {auto_review.get('rationale', '')}"
        )
        if issue_text:
            review_notes += f"\nissues: {issue_text}"

        reviewed_at = _utc_now_iso()
        current["review"] = {
            "decision": decision,
            "notes": review_notes,
            "reviewed_at": reviewed_at,
        }
        current["auto_review"] = {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "rationale": auto_review.get("rationale", ""),
            "issues": list(auto_review.get("issues", [])),
            "engine": auto_engine,
            "model": model_name,
            "reviewed_at": reviewed_at,
        }
        if decision == "accepted" and str(current.get("status", "") or "") in {"enriched", "fallback_enriched", "reviewed"}:
            current["status"] = "reviewed"
        current["reward"] = _compute_reward(
            current.get("enrichment", {}),
            decision,
            status=current.get("status", ""),
        )
        current["updated_at"] = reviewed_at
        entries[doc_id] = current

        processed += 1
        if processed % 10 == 0:
            _save_enrichment_state(enrichment_state)

    if progress_callback is not None:
        progress_callback(total, total, "Auto review run complete")

    _save_enrichment_state(enrichment_state)
    return {
        "processed": processed,
        "total_selected": total,
        "approved": approved,
        "flagged": flagged,
        "heuristic_fallbacks": heuristic_fallbacks,
    }


def _bulk_accept_reviews(enrichment_state, scoped_entries, only_pending=False):
    entries = enrichment_state.setdefault("entries", {})
    accepted = 0
    for item in scoped_entries:
        doc_id = str(item.get("doc_id", "") or "").strip()
        if not doc_id:
            continue

        current = entries.get(doc_id, item)
        status = str(current.get("status", "") or "").strip().lower()
        if status not in {"enriched", "fallback_enriched", "reviewed"}:
            continue

        current_decision = str(current.get("review", {}).get("decision", "pending") or "pending").strip().lower()
        if only_pending and current_decision != "pending":
            continue

        prior_notes = str(current.get("review", {}).get("notes", "") or "").strip()
        marker = "[BULK_AUTO_ACCEPT]"
        if marker not in prior_notes:
            notes = f"{marker} Accepted from bulk action.\n{prior_notes}".strip()
        else:
            notes = prior_notes

        reviewed_at = _utc_now_iso()
        current["review"] = {
            "decision": "accepted",
            "notes": notes,
            "reviewed_at": reviewed_at,
        }
        if status in {"enriched", "fallback_enriched"}:
            current["status"] = "reviewed"
        current["reward"] = _compute_reward(
            current.get("enrichment", {}),
            "accepted",
            status=current.get("status", ""),
        )
        current["updated_at"] = reviewed_at
        entries[doc_id] = current
        accepted += 1

    _save_enrichment_state(enrichment_state)
    return accepted


def _update_review_decision(enrichment_state, doc_id, decision, notes):
    entries = enrichment_state.setdefault("entries", {})
    item = entries.get(doc_id)
    if not item:
        return False
    decision = str(decision or "pending").strip().lower()
    if decision not in {"pending", "accepted", "rejected", "edited"}:
        decision = "pending"
    item["review"] = {
        "decision": decision,
        "notes": str(notes or ""),
        "reviewed_at": _utc_now_iso(),
    }
    item["reward"] = _compute_reward(
        item.get("enrichment", {}),
        decision,
        status=item.get("status", ""),
    )
    if decision in {"accepted", "edited"} and item.get("status") in {"enriched", "fallback_enriched"}:
        item["status"] = "reviewed"
    entries[doc_id] = item
    _save_enrichment_state(enrichment_state)
    return True


def _build_org_documents(raw_data_obj, org_key, org_label, enrichment_entries=None):
    """Build doc records keyed by deterministic doc_id for one organization."""
    if not isinstance(enrichment_entries, dict):
        enrichment_entries = {}

    docs = {}
    for speech in raw_data_obj.get("speeches", []):
        if _speech_org_key(speech) != org_key:
            continue

        m = speech.get("metadata", {})
        c = speech.get("content", {})
        text = str(c.get("full_text", "") or "").strip()
        if not text:
            continue

        title = str(m.get("title", "") or "").strip()
        speaker = str(m.get("speaker", "") or "").strip()
        speech_date = str(m.get("date", "") or "").strip()
        url = str(m.get("url", "") or "").strip()
        word_count = m.get("word_count", 0)
        doc_type = str(m.get("doc_type", "Speech") or "Speech").strip()
        source_filename = str(m.get("source_filename", "") or "").strip()

        stable_seed = url or "|".join([title, speaker, speech_date])
        if not stable_seed.strip("|"):
            stable_seed = text[:500]
        text_chunks = _split_text_for_indexing(
            text,
            max_chars=40000,
            overlap_chars=3000,
        )
        total_parts = len(text_chunks)
        legacy_doc_id = hashlib.sha256(f"{org_key}|{stable_seed}".encode("utf-8")).hexdigest()[:24]
        enrich_entry = enrichment_entries.get(legacy_doc_id, {})
        if not isinstance(enrich_entry, dict):
            enrich_entry = {}
        enrich_obj = enrich_entry.get("enrichment", {})
        if not isinstance(enrich_obj, dict):
            enrich_obj = {}
        enrich_tags = enrich_obj.get("tags", [])
        if not isinstance(enrich_tags, list):
            enrich_tags = []
        enrich_keywords = enrich_obj.get("keywords", [])
        if not isinstance(enrich_keywords, list):
            enrich_keywords = []
        enrich_tags_text = ", ".join(str(t).strip() for t in enrich_tags if str(t).strip())
        enrich_keywords_text = ", ".join(str(k).strip() for k in enrich_keywords if str(k).strip())
        enrich_stance = enrich_obj.get("stance", {})
        if not isinstance(enrich_stance, dict):
            enrich_stance = {"label": str(enrich_stance or "").strip(), "target": ""}
        stance_label = str(enrich_stance.get("label", "") or "").strip()
        stance_target = str(enrich_stance.get("target", "") or "").strip()
        stance_text = stance_label
        if stance_label and stance_target:
            stance_text = f"{stance_label} ({stance_target})"
        enrich_review = str(enrich_entry.get("review", {}).get("decision", "") or "").strip().lower()

        for part_idx, chunk_text in enumerate(text_chunks, 1):
            if total_parts == 1:
                doc_id = legacy_doc_id
            else:
                chunk_seed = f"{stable_seed}|part|{part_idx}|{total_parts}"
                doc_id = hashlib.sha256(f"{org_key}|{chunk_seed}".encode("utf-8")).hexdigest()[:24]
            vector_filename = _vector_filename_from_title(
                org_key,
                title,
                doc_id,
                part_idx=part_idx,
                total_parts=total_parts,
            )
            chunk_header = f"{part_idx}/{total_parts}" if total_parts > 1 else "1/1"
            header_lines = [
                f"Organization: {org_label}",
                f"Doc ID: {doc_id}",
                f"Title: {title}",
                f"Speaker: {speaker}",
                f"Date: {speech_date}",
                f"Document Type: {doc_type}",
                f"URL: {url}",
                f"Source File: {source_filename}",
                f"Word Count: {word_count}",
                f"Chunk: {chunk_header}",
                f"Vector File Name: {vector_filename}",
            ]
            if enrich_tags_text:
                header_lines.append(f"Enrichment Tags: {enrich_tags_text}")
            if enrich_keywords_text:
                header_lines.append(f"Enrichment Keywords: {enrich_keywords_text}")
            if stance_text:
                header_lines.append(f"Enrichment Stance: {stance_text}")
            if enrich_review:
                header_lines.append(f"Enrichment Review: {enrich_review}")

            rendered = "\n".join(header_lines) + f"\n\n{chunk_text}\n"
            content_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()

            docs[doc_id] = {
                "doc_id": doc_id,
                "title": title,
                "speaker": speaker,
                "date": speech_date,
                "url": url,
                "word_count": word_count,
                "filename": vector_filename,
                "rendered_text": rendered,
                "content_hash": content_hash,
            }

    return docs


def _plan_doc_sync(indexed_docs, current_docs):
    indexed_ids = set(indexed_docs.keys())
    current_ids = set(current_docs.keys())

    add_ids = sorted(current_ids - indexed_ids)
    remove_ids = sorted(indexed_ids - current_ids)
    update_ids = sorted(
        doc_id
        for doc_id in (current_ids & indexed_ids)
        if (indexed_docs.get(doc_id, {}).get("content_hash") or "") != (current_docs.get(doc_id, {}).get("content_hash") or "")
    )
    unchanged_ids = sorted((current_ids & indexed_ids) - set(update_ids))

    return add_ids, update_ids, remove_ids, unchanged_ids


def _chat_doc_path(org_key, doc_id, filename=""):
    base = Path("data/chat_docs") / org_key
    base.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(filename)
    if safe_name:
        if not safe_name.lower().endswith(".txt"):
            safe_name = f"{safe_name}.txt"
        return base / safe_name
    return base / f"{doc_id}.txt"


def _write_chat_doc_file(org_key, doc):
    path = _chat_doc_path(
        org_key,
        doc["doc_id"],
        filename=doc.get("filename", ""),
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc["rendered_text"])
    return path


def _extract_file_ref(upload_obj):
    data = _normalize_obj(upload_obj)
    vector_store_file_id = data.get("id") or ""
    file_id = data.get("file_id") or vector_store_file_id or ""
    return {
        "file_id": file_id,
        "vector_store_file_id": vector_store_file_id,
    }


def _upload_doc_to_vector_store(client, vector_store_id, org_key, doc):
    def _is_retryable_upload_error(exc):
        msg = str(exc).lower()
        retry_signals = [
            "rate limit",
            "timeout",
            "timed out",
            "temporar",
            "try again",
            "connection",
            "502",
            "503",
            "504",
        ]
        return any(token in msg for token in retry_signals)

    file_path = _write_chat_doc_file(org_key, doc)
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            with open(file_path, "rb") as f:
                uploaded = client.vector_stores.files.upload_and_poll(
                    vector_store_id=vector_store_id,
                    file=f,
                )
            file_ref = _extract_file_ref(uploaded)
            if not (file_ref.get("file_id") or file_ref.get("vector_store_file_id")):
                raise RuntimeError("Vector-store upload did not return a file ID.")
            return file_ref
        except Exception as e:
            if attempt >= max_attempts or not _is_retryable_upload_error(e):
                raise
            time.sleep(min(8, 2 ** attempt))


def _delete_indexed_file(client, vector_store_id, indexed_entry):
    file_id = str(indexed_entry.get("file_id", "") or "").strip()
    vs_file_id = str(indexed_entry.get("vector_store_file_id", "") or "").strip()

    if file_id:
        client.vector_stores.files.delete(file_id, vector_store_id=vector_store_id)
        return
    if vs_file_id:
        client.vector_stores.files.delete(vs_file_id, vector_store_id=vector_store_id)


def _get_org_vector_state(state, org_key, org_label):
    stores = state.setdefault("stores", {})
    org_state = stores.get(org_key, {})
    if not isinstance(org_state, dict):
        org_state = {}
    org_state.setdefault("org_label", org_label)
    org_state.setdefault("vector_store_id", "")
    org_state.setdefault("docs", {})
    return org_state


def _get_vector_store_file_counts(client, vector_store_id):
    """Return normalized vector-store file counts from OpenAI."""
    try:
        store = client.vector_stores.retrieve(vector_store_id)
        data = _normalize_obj(store)
        counts = data.get("file_counts", {}) if isinstance(data, dict) else {}
        if not isinstance(counts, dict):
            counts = {}
        return {
            "total": int(counts.get("total", 0) or 0),
            "completed": int(counts.get("completed", 0) or 0),
            "in_progress": int(counts.get("in_progress", 0) or 0),
            "failed": int(counts.get("failed", 0) or 0),
            "cancelled": int(counts.get("cancelled", 0) or 0),
        }
    except Exception:
        return {
            "total": 0,
            "completed": 0,
            "in_progress": 0,
            "failed": 0,
            "cancelled": 0,
        }


def _count_vector_store_files_deep(client, vector_store_id):
    """Deep count by paginating vector_store.files.list (slower but authoritative)."""
    status_counts = {
        "total": 0,
        "completed": 0,
        "in_progress": 0,
        "failed": 0,
        "cancelled": 0,
    }
    first_page = client.vector_stores.files.list(vector_store_id=vector_store_id, limit=100, order="asc")
    for page in first_page.iter_pages():
        for item in getattr(page, "data", []):
            status_counts["total"] += 1
            status = str(getattr(item, "status", "") or "").strip().lower()
            if status in status_counts:
                status_counts[status] += 1
    return status_counts


def _list_project_vector_stores(client, limit=50):
    rows = []
    response = client.vector_stores.list(limit=limit, order="desc")
    for item in getattr(response, "data", []):
        item_dict = _normalize_obj(item)
        file_counts = item_dict.get("file_counts", {}) if isinstance(item_dict, dict) else {}
        if not isinstance(file_counts, dict):
            file_counts = {}
        rows.append(
            {
                "id": item_dict.get("id", ""),
                "name": item_dict.get("name", ""),
                "status": item_dict.get("status", ""),
                "total": int(file_counts.get("total", 0) or 0),
                "completed": int(file_counts.get("completed", 0) or 0),
                "in_progress": int(file_counts.get("in_progress", 0) or 0),
                "failed": int(file_counts.get("failed", 0) or 0),
            }
        )
    return rows


def _verify_active_vector_store(client, vector_store_id):
    if not vector_store_id:
        return {"ok": False, "message": "No active vector store ID selected."}
    try:
        item = client.vector_stores.retrieve(vector_store_id)
        item_dict = _normalize_obj(item)
        file_counts = item_dict.get("file_counts", {}) if isinstance(item_dict, dict) else {}
        return {
            "ok": True,
            "id": item_dict.get("id", ""),
            "name": item_dict.get("name", ""),
            "status": item_dict.get("status", ""),
            "file_counts": file_counts if isinstance(file_counts, dict) else {},
        }
    except Exception as e:
        return {"ok": False, "message": str(e)}


def _ensure_org_vector_store(client, org_state, org_label, force_rebuild=False):
    existing_id = str(org_state.get("vector_store_id", "") or "").strip()

    if existing_id and not force_rebuild:
        try:
            client.vector_stores.retrieve(existing_id)
            return existing_id, False, ""
        except Exception:
            pass

    vector_store = client.vector_stores.create(
        name=f"{org_label} Speeches ({datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC)"
    )
    return vector_store.id, True, existing_id


def _sync_org_vector_store(client, raw_data_obj, org_key, org_label, force_rebuild=False, progress_callback=None):
    """Incrementally sync an org-specific vector store and return a sync report."""
    def _emit_progress(done, total, message):
        if progress_callback is None:
            return
        try:
            progress_callback(done, total, message)
        except Exception:
            pass

    def _display_name(title, doc_id):
        text = str(title or doc_id or "document").strip()
        return text[:80] + ("..." if len(text) > 80 else "")

    state = _load_vector_state()
    stores = state.setdefault("stores", {})
    org_state = _get_org_vector_state(state, org_key, org_label)
    enrichment_state = _load_enrichment_state()
    enrichment_entries = enrichment_state.get("entries", {}) if isinstance(enrichment_state, dict) else {}
    if not isinstance(enrichment_entries, dict):
        enrichment_entries = {}

    current_docs = _build_org_documents(
        raw_data_obj,
        org_key,
        org_label,
        enrichment_entries=enrichment_entries,
    )
    indexed_docs = org_state.get("docs", {})
    if not isinstance(indexed_docs, dict):
        indexed_docs = {}

    existing_store_id = str(org_state.get("vector_store_id", "") or "").strip()
    remote_counts = {}
    if existing_store_id:
        remote_counts = _get_vector_store_file_counts(client, existing_store_id)

    # If we only have legacy aggregate counts (no per-doc manifest), rebuild once
    # so future syncs are truly incremental without duplicate legacy corpus files.
    legacy_without_manifest = (
        not force_rebuild
        and bool(existing_store_id)
        and not indexed_docs
        and (
            int(org_state.get("doc_count_indexed", 0) or 0) > 0
            or int(remote_counts.get("total", 0) or 0) > 0
        )
    )
    if legacy_without_manifest:
        force_rebuild = True

    add_ids, update_ids, remove_ids, unchanged_ids = _plan_doc_sync(indexed_docs, current_docs)
    vector_store_id, created_new_store, replaced_store_id = _ensure_org_vector_store(
        client,
        org_state,
        org_label,
        force_rebuild=force_rebuild,
    )
    sync_mode = "rebuild" if (force_rebuild or created_new_store) else "incremental"

    if force_rebuild or created_new_store:
        add_ids = sorted(current_docs.keys())
        update_ids = []
        remove_ids = []
        unchanged_ids = []
        indexed_docs = {}

    delete_targets = [] if (force_rebuild or created_new_store) else (remove_ids + update_ids)
    upload_targets = add_ids + update_ids
    total_ops = len(delete_targets) + len(upload_targets)
    completed_ops = 0
    _emit_progress(completed_ops, total_ops, "Preparing index sync")

    if not add_ids and not update_ids and not remove_ids and not force_rebuild and not created_new_store:
        _emit_progress(0, 0, "Knowledge index already up to date")
        org_state.update(
            {
                "org_label": org_label,
                "vector_store_id": vector_store_id,
                "docs": indexed_docs,
                "doc_count_indexed": len(indexed_docs),
                "updated_at": _utc_now_iso(),
                "last_sync": {
                    "planned_add": 0,
                    "planned_update": 0,
                    "planned_remove": 0,
                    "uploaded": 0,
                    "deleted": 0,
                    "failed_count": 0,
                    "failed": [],
                    "sync_mode": "noop",
                    "status": "completed",
                },
            }
        )
        stores[org_key] = org_state
        state["version"] = 2
        state["updated_at"] = _utc_now_iso()
        _save_vector_state(state)
        return {
            "vector_store_id": vector_store_id,
            "rebuilt": False,
            "created_new_store": False,
            "up_to_date": True,
            "stats": {"add": 0, "update": 0, "remove": 0, "uploaded": 0, "deleted": 0},
            "failed": [],
            "doc_count": len(indexed_docs),
        }

    # Persist the active target store before long-running operations start.
    # This prevents reboots/interruption from pointing back to an old store.
    if created_new_store or force_rebuild or vector_store_id != existing_store_id:
        org_state.update(
            {
                "org_label": org_label,
                "vector_store_id": vector_store_id,
                "docs": indexed_docs,
                "doc_count_indexed": len(indexed_docs),
                "updated_at": _utc_now_iso(),
                "last_sync": {
                    "planned_add": len(add_ids),
                    "planned_update": len(update_ids),
                    "planned_remove": len(remove_ids),
                    "uploaded": 0,
                    "deleted": 0,
                    "failed_count": 0,
                    "failed": [],
                    "sync_mode": sync_mode,
                    "status": "in_progress",
                },
            }
        )
        stores[org_key] = org_state
        state["version"] = 2
        state["updated_at"] = _utc_now_iso()
        _save_vector_state(state)

    failed = []
    deleted_count = 0
    uploaded_count = 0

    next_docs = {}
    for doc_id in unchanged_ids:
        entry = indexed_docs.get(doc_id, {})
        if entry:
            next_docs[doc_id] = entry

    checkpoint_interval = 100

    def _checkpoint_state(force=False):
        if not force:
            if completed_ops == 0:
                return
            if completed_ops % checkpoint_interval != 0:
                return

        org_state.update(
            {
                "org_label": org_label,
                "vector_store_id": vector_store_id,
                "docs": next_docs,
                "doc_count_indexed": len(next_docs),
                "updated_at": _utc_now_iso(),
                "last_sync": {
                    "planned_add": len(add_ids),
                    "planned_update": len(update_ids),
                    "planned_remove": len(remove_ids),
                    "uploaded": uploaded_count,
                    "deleted": deleted_count,
                    "failed_count": len(failed),
                    "failed": failed[:10],
                    "sync_mode": sync_mode,
                    "status": "in_progress",
                },
            }
        )
        stores[org_key] = org_state
        state["version"] = 2
        state["updated_at"] = _utc_now_iso()
        _save_vector_state(state)

    # Remove stale files before uploading updates when reusing the same store.
    if not (force_rebuild or created_new_store):
        for doc_id in delete_targets:
            entry = indexed_docs.get(doc_id, {})
            display = _display_name(entry.get("title", ""), doc_id)
            _emit_progress(completed_ops, total_ops, f"Deleting {display}")
            try:
                _delete_indexed_file(client, vector_store_id, entry)
                deleted_count += 1
            except Exception as e:
                failed.append(
                    {
                        "doc_id": doc_id,
                        "title": entry.get("title", ""),
                        "stage": "delete",
                        "error": str(e),
                    }
                )
            completed_ops += 1
            _emit_progress(completed_ops, total_ops, f"Processed delete for {display}")
            _checkpoint_state()

    upload_items = []
    for doc_id in upload_targets:
        doc = current_docs.get(doc_id)
        if not doc:
            completed_ops += 1
            _emit_progress(completed_ops, total_ops, f"Skipped missing doc {doc_id}")
            continue
        upload_items.append((doc_id, doc))

    if upload_items:
        max_workers = min(6, len(upload_items))
        if max_workers == 1:
            for doc_id, doc in upload_items:
                display = _display_name(doc.get("title", ""), doc_id)
                _emit_progress(completed_ops, total_ops, f"Uploading {display}")
                try:
                    file_ref = _upload_doc_to_vector_store(client, vector_store_id, org_key, doc)
                    next_docs[doc_id] = {
                        "doc_id": doc_id,
                        "title": doc.get("title", ""),
                        "speaker": doc.get("speaker", ""),
                        "date": doc.get("date", ""),
                        "url": doc.get("url", ""),
                        "word_count": doc.get("word_count", 0),
                        "filename": doc.get("filename", ""),
                        "content_hash": doc.get("content_hash", ""),
                        "file_id": file_ref.get("file_id", ""),
                        "vector_store_file_id": file_ref.get("vector_store_file_id", ""),
                        "indexed_at": _utc_now_iso(),
                    }
                    uploaded_count += 1
                except Exception as e:
                    failed.append(
                        {
                            "doc_id": doc_id,
                            "title": doc.get("title", ""),
                            "stage": "upload",
                            "error": str(e),
                        }
                    )
                completed_ops += 1
                _emit_progress(completed_ops, total_ops, f"Processed upload for {display}")
                _checkpoint_state()
        else:
            _emit_progress(
                completed_ops,
                total_ops,
                f"Uploading {len(upload_items)} documents with {max_workers} workers",
            )
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_map = {
                    pool.submit(
                        _upload_doc_to_vector_store,
                        client,
                        vector_store_id,
                        org_key,
                        doc,
                    ): (doc_id, doc)
                    for doc_id, doc in upload_items
                }
                for future in as_completed(future_map):
                    doc_id, doc = future_map[future]
                    display = _display_name(doc.get("title", ""), doc_id)
                    try:
                        file_ref = future.result()
                        next_docs[doc_id] = {
                            "doc_id": doc_id,
                            "title": doc.get("title", ""),
                            "speaker": doc.get("speaker", ""),
                            "date": doc.get("date", ""),
                            "url": doc.get("url", ""),
                            "word_count": doc.get("word_count", 0),
                            "filename": doc.get("filename", ""),
                            "content_hash": doc.get("content_hash", ""),
                            "file_id": file_ref.get("file_id", ""),
                            "vector_store_file_id": file_ref.get("vector_store_file_id", ""),
                            "indexed_at": _utc_now_iso(),
                        }
                        uploaded_count += 1
                    except Exception as e:
                        failed.append(
                            {
                                "doc_id": doc_id,
                                "title": doc.get("title", ""),
                                "stage": "upload",
                                "error": str(e),
                            }
                        )
                    completed_ops += 1
                    _emit_progress(completed_ops, total_ops, f"Processed upload for {display}")
                    _checkpoint_state()

    org_state.update(
        {
            "org_label": org_label,
            "vector_store_id": vector_store_id,
            "docs": next_docs,
            "doc_count_indexed": len(next_docs),
            "updated_at": _utc_now_iso(),
            "last_sync": {
                "planned_add": len(add_ids),
                "planned_update": len(update_ids),
                "planned_remove": len(remove_ids),
                "uploaded": uploaded_count,
                "deleted": deleted_count,
                "failed_count": len(failed),
                "failed": failed,
                "sync_mode": sync_mode,
                "status": "completed",
            },
        }
    )
    stores[org_key] = org_state
    state["version"] = 2
    state["updated_at"] = _utc_now_iso()
    _save_vector_state(state)

    old_store_deleted = False
    old_store_delete_error = ""
    if force_rebuild and replaced_store_id and replaced_store_id != vector_store_id:
        try:
            client.vector_stores.delete(replaced_store_id)
            old_store_deleted = True
        except Exception as e:
            old_store_delete_error = str(e)

    _emit_progress(total_ops, total_ops, "Index sync complete")

    return {
        "vector_store_id": vector_store_id,
        "rebuilt": bool(force_rebuild or created_new_store),
        "created_new_store": created_new_store,
        "up_to_date": False,
        "all_uploads_failed": bool(upload_targets) and uploaded_count == 0,
        "stats": {
            "add": len(add_ids),
            "update": len(update_ids),
            "remove": len(remove_ids),
            "uploaded": uploaded_count,
            "deleted": deleted_count,
        },
        "failed": failed,
        "doc_count": len(next_docs),
        "old_store_deleted": old_store_deleted,
        "old_store_delete_error": old_store_delete_error,
    }


def _get_org_index_status(raw_data_obj, org_key, org_label):
    state = _load_vector_state()
    org_state = _get_org_vector_state(state, org_key, org_label)
    indexed_docs = org_state.get("docs", {})
    if not isinstance(indexed_docs, dict):
        indexed_docs = {}
    enrichment_state = _load_enrichment_state()
    enrichment_entries = enrichment_state.get("entries", {}) if isinstance(enrichment_state, dict) else {}
    if not isinstance(enrichment_entries, dict):
        enrichment_entries = {}
    current_docs = _build_org_documents(
        raw_data_obj,
        org_key,
        org_label,
        enrichment_entries=enrichment_entries,
    )
    add_ids, update_ids, remove_ids, _ = _plan_doc_sync(indexed_docs, current_docs)
    return {
        "vector_store_id": str(org_state.get("vector_store_id", "") or "").strip(),
        "indexed_docs": len(indexed_docs),
        "current_docs": len(current_docs),
        "pending_add": len(add_ids),
        "pending_update": len(update_ids),
        "pending_remove": len(remove_ids),
        "last_sync": org_state.get("last_sync", {}),
    }


def _normalize_obj(obj):
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return {}


def _has_explicit_timeframe(question):
    text = str(question or "").strip().lower()
    if not text:
        return False

    if re.search(r"\b(19|20)\d{2}\b", text):
        return True
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text):
        return True

    month_names = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    ]
    if any(m in text for m in month_names):
        return True

    if re.search(r"\blast\s+\d+\s+(day|days|week|weeks|month|months|year|years)\b", text):
        return True
    if re.search(r"\bpast\s+\d+\s+(day|days|week|weeks|month|months|year|years)\b", text):
        return True

    explicit_markers = [
        "since ",
        "between ",
        "from ",
        "as of ",
        "before ",
        "after ",
        "during ",
        "in 20",
        "in 19",
        "q1 ",
        "q2 ",
        "q3 ",
        "q4 ",
    ]
    return any(m in text for m in explicit_markers)


def _needs_temporal_clarification(question):
    text = str(question or "").strip().lower()
    if not text:
        return False

    ambiguous_terms = [
        "recent",
        "latest",
        "current",
        "currently",
        "now",
        "today",
        "at present",
        "these days",
    ]
    if not any(term in text for term in ambiguous_terms):
        return False

    return not _has_explicit_timeframe(text)


def _latest_date_for_org(df, org_key):
    if df.empty or "date_parsed" not in df.columns:
        return None

    if "organization" in df.columns:
        org_mask = df["organization"].fillna("").apply(lambda x: _org_key_from_label(x) == org_key)
        org_df = df[org_mask]
        if not org_df.empty:
            latest = org_df["date_parsed"].max()
            if pd.notna(latest):
                return latest

    latest = df["date_parsed"].max()
    return latest if pd.notna(latest) else None


def _build_temporal_clarification_message(df, org_key, org_label):
    latest = _latest_date_for_org(df, org_key)
    latest_txt = latest.strftime("%B %d, %Y") if latest is not None else "the latest indexed date"

    return (
        "Your question uses a time term like `recent/current`, but no date window was provided.\n\n"
        f"Please specify a timeframe so I can answer accurately for {org_label}. "
        f"Current indexed coverage appears to run through {latest_txt}.\n\n"
        "Examples:\n"
        "- last 90 days\n"
        "- last 12 months\n"
        "- since January 1, 2025\n"
        "- between January 1, 2025 and November 30, 2025"
    )


def _build_agent_instructions(df, org_key, org_label):
    latest = _latest_date_for_org(df, org_key)
    latest_txt = latest.strftime("%B %d, %Y") if latest is not None else "unknown"
    today_txt = date.today().strftime("%B %d, %Y")
    return (
        "You are a retrieval-grounded assistant for speech analysis. "
        "Use only the retrieved speech content. "
        "If the user uses ambiguous temporal terms (such as recent, latest, current, now, today) "
        "without a concrete date range, ask one concise clarification question first and do not assume a window. "
        f"Today's date is {today_txt}. "
        f"Latest indexed speech date for {org_label} is {latest_txt}."
    )


def _extract_response_text(response):
    txt = getattr(response, "output_text", None)
    if txt:
        return txt
    resp_dict = _normalize_obj(response)
    output_items = resp_dict.get("output", [])
    for item in output_items:
        if item.get("type") == "message":
            for content_item in item.get("content", []):
                if content_item.get("type") in ("output_text", "text"):
                    if content_item.get("text"):
                        return content_item.get("text")
    return "No response text returned."


def _extract_file_search_results(response):
    resp_dict = _normalize_obj(response)
    results = []
    for item in resp_dict.get("output", []):
        if item.get("type") == "file_search_call":
            for r in item.get("results", []):
                snippet = r.get("text", "")
                if isinstance(snippet, str):
                    snippet = snippet.strip()
                results.append(
                    {
                        "filename": r.get("filename", ""),
                        "score": r.get("score"),
                        "file_id": r.get("file_id", ""),
                        "snippet": snippet[:300] if snippet else "",
                    }
                )
    return results


def _ask_agent(client, vector_store_ids, question, model_name, instructions_text=None):
    if isinstance(vector_store_ids, str):
        vector_store_ids = [vector_store_ids]
    vector_store_ids = [str(v).strip() for v in (vector_store_ids or []) if str(v).strip()]
    if not vector_store_ids:
        raise RuntimeError("No vector stores provided for retrieval.")

    request_payload = {
        "model": model_name,
        "input": question,
        "tools": [
            {
                "type": "file_search",
                "vector_store_ids": vector_store_ids,
                "max_num_results": 8,
            }
        ],
    }
    if instructions_text:
        request_payload["instructions"] = instructions_text
    try:
        response = client.responses.create(
            **request_payload,
            include=["file_search_call.results"],
        )
    except Exception:
        response = client.responses.create(**request_payload)
    return {
        "answer": _extract_response_text(response),
        "results": _extract_file_search_results(response),
    }


def _candidate_chat_models():
    return [
        "gpt-5.1",
        "gpt-5-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
    ]


def _list_project_models(client):
    """Return visible model IDs for the current API key/project."""
    listed = client.models.list()
    ids = sorted({getattr(m, "id", "") for m in getattr(listed, "data", []) if getattr(m, "id", "")})
    return ids


def _get_accessible_chat_models(client):
    """Return preferred chat models that are available to this project."""
    candidates = _candidate_chat_models()
    try:
        ids = set(_list_project_models(client))
        available = [m for m in candidates if m in ids]
        if available:
            return available
    except Exception:
        pass
    return candidates


def _is_model_access_error(exc):
    msg = str(exc).lower()
    return (
        "model_not_found" in msg
        or "does not have access to model" in msg
        or "access to model" in msg
    )


def _ask_agent_with_fallback(client, vector_store_ids, question, preferred_model, model_pool, instructions_text=None):
    """Try preferred model first, then fallback models on access errors."""
    ordered = [preferred_model] + [m for m in model_pool if m != preferred_model]
    last_error = None
    for idx, model_name in enumerate(ordered):
        try:
            result = _ask_agent(
                client,
                vector_store_ids,
                question,
                model_name,
                instructions_text=instructions_text,
            )
            return {
                "result": result,
                "used_model": model_name,
                "fallback_used": idx > 0,
            }
        except Exception as e:
            last_error = e
            if not _is_model_access_error(e):
                raise
            continue
    if last_error:
        raise last_error
    raise RuntimeError("No model available for chat request.")


def _load_raw_data():
    """Load dataset from GCS (preferred) or local file (fallback)."""
    storage = _get_gcs_storage()
    if storage is not None:
        try:
            data = storage.load_speeches()
            if data.get("speeches"):
                return data, storage
        except Exception:
            pass

    # Fallback to local file
    data_file = Path("data/all_speeches_final.json")
    if not data_file.exists():
        st.error("Dataset not found. Configure GCS secrets or place data/all_speeches_final.json.")
        st.stop()
    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f), None


def _parse_single_date(value):
    """Parse one SEC date string into a Timestamp (or NaT)."""
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()
    if not text:
        return pd.NaT

    # Normalize month abbreviations like "Jan. 30, 2026".
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

    for fmt in ("%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except ValueError:
            continue

    return pd.to_datetime(text, errors="coerce")


def _parse_date_series(series: pd.Series) -> pd.Series:
    """Parse mixed SEC date strings into datetimes for reliable sorting."""
    return series.apply(_parse_single_date)


def _sort_table_by_date(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Sort a table by date descending when a date column is present."""
    if df.empty or date_col not in df.columns:
        return df

    out = df.copy()
    sort_col = "__date_sort"
    out[sort_col] = _parse_date_series(out[date_col])
    out = out.sort_values(by=[sort_col], ascending=False, na_position="last")
    out = out.drop(columns=[sort_col])
    return out


def _explode_speakers(df: pd.DataFrame) -> pd.DataFrame:
    """Expand each speech row into one row per individual speaker."""
    if df.empty or "speaker_list" not in df.columns:
        return pd.DataFrame(columns=["speaker_individual"])

    exploded = df.explode("speaker_list").copy()
    exploded = exploded.rename(columns={"speaker_list": "speaker_individual"})
    exploded["speaker_individual"] = (
        exploded["speaker_individual"].fillna("").astype(str).str.strip()
    )
    exploded = exploded[exploded["speaker_individual"] != ""]
    return exploded


@st.cache_data(ttl=300)
def load_data(_cache_buster=None):
    """Load and cache the speech dataset."""
    raw_data, _ = _load_raw_data()

    rows = []
    for speech in raw_data.get("speeches", []):
        m = speech.get("metadata", {})
        c = speech.get("content", {})
        v = speech.get("validation", {})
        raw_speaker = m.get("speaker", "Unknown")
        speaker_list = extract_speakers(raw_speaker)
        speaker_display = "; ".join(speaker_list) if speaker_list else format_speakers(raw_speaker)
        speaker_primary = primary_speaker(raw_speaker) or speaker_display or "Unknown"
        rows.append({
            "title": m.get("title", ""),
            "speaker": speaker_display or "Unknown",
            "speaker_primary": speaker_primary,
            "speaker_list": speaker_list,
            "organization": _speech_org_label(speech),
            "date": m.get("date", ""),
            "url": m.get("url", ""),
            "word_count": m.get("word_count", 0),
            "full_text": c.get("full_text", ""),
            "paragraph_count": len(c.get("paragraphs", [])),
            "sentence_count": len(c.get("sentences", [])),
            "completeness_score": v.get("completeness_score", 0),
        })

    df = pd.DataFrame(rows)
    if not df.empty and "date" in df.columns:
        df["date_parsed"] = _parse_date_series(df["date"])
        sort_cols = ["date_parsed"]
        sort_asc = [False]
        if "title" in df.columns:
            sort_cols.append("title")
            sort_asc.append(True)
        df = df.sort_values(by=sort_cols, ascending=sort_asc, na_position="last").reset_index(drop=True)
    else:
        df["date_parsed"] = pd.NaT

    return raw_data, df


@st.cache_data
def run_analysis(raw_data_json):
    """Run the analysis pipeline and cache results."""
    pipeline = SpeechAnalysisPipeline()
    pipeline.speeches_data = json.loads(raw_data_json)
    pipeline.create_dataframe()

    sentiment = pipeline.basic_sentiment_analysis()
    topics = pipeline.topic_analysis()
    commissioner = pipeline.commissioner_analysis()

    return sentiment, topics, commissioner


# --- Load Data ---
raw_data, df = load_data()
speaker_df = _explode_speakers(df)
custom_docs_payload = _load_custom_documents()
custom_documents = _custom_docs_as_speeches(custom_docs_payload)
knowledge_data = _build_knowledge_data(raw_data, custom_docs_payload)
knowledge_df = _build_knowledge_df(knowledge_data)

raw_data_json = json.dumps(raw_data)
sentiment_data, topic_data, commissioner_data = run_analysis(raw_data_json)


# --- Sidebar Navigation ---
st.sidebar.title("Policy Research Hub")
section = st.sidebar.radio(
    "Section",
    ["Discussion", "Corpus Explorer", "Analytics", "Admin"],
)

if section == "Discussion":
    page = st.sidebar.radio(
        "Discuss",
        ["Agent Chat", "Policy Delta Briefings"],
    )
elif section == "Corpus Explorer":
    page = "Corpus Explorer"
elif section == "Analytics":
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Sentiment Analysis", "Topic Analysis"],
    )
else:
    page = st.sidebar.radio(
        "Admin",
        ["Extraction", "Document Library", "Enrichment Pipeline"],
    )

st.sidebar.markdown("---")
kb_words = 0
for s in knowledge_data.get("speeches", []):
    kb_words += int(s.get("metadata", {}).get("word_count", 0) or 0)
st.sidebar.markdown(f"**{len(df)} SEC speeches**")
st.sidebar.markdown(f"**{speaker_df['speaker_individual'].nunique()} SEC speakers**")
st.sidebar.markdown(f"**{len(custom_documents)} uploaded docs**")
st.sidebar.markdown(f"**{len(knowledge_data.get('speeches', []))} total corpus docs**")
st.sidebar.markdown(f"**{kb_words:,} corpus words**")

# GCS status indicator — with debug info
_gcs_debug = []
try:
    _gcs_debug.append(f"secrets keys: {list(st.secrets.keys())}")
    _gcs_section = st.secrets.get("gcs", None)
    _gcs_debug.append(f"gcs section: {'found' if _gcs_section else 'missing'}")
    if _gcs_section:
        _gcs_debug.append(f"gcs keys: {list(_gcs_section.keys())}")
except Exception as e:
    _gcs_debug.append(f"secrets error: {e}")

_gcs = _get_gcs_storage()
if _gcs is not None:
    try:
        _gcs.bucket.blob("all_speeches.json").exists()
        st.sidebar.success("GCS: Connected", icon="\u2705")
    except Exception as e:
        st.sidebar.error(f"GCS: Error \u2014 {e}", icon="\u274c")
else:
    gcs_err = st.session_state.get("_gcs_error", "no error captured")
    st.sidebar.error(f"GCS: {gcs_err}", icon="\u274c")
    with st.sidebar.expander("Debug"):
        for line in _gcs_debug:
            st.write(line)

_openai_key = _get_openai_api_key()
if _openai_key is not None:
    st.sidebar.success("OpenAI: Configured", icon="\u2705")
else:
    openai_err = st.session_state.get("_openai_error", "no error captured")
    st.sidebar.error(f"OpenAI: {openai_err}", icon="\u274c")


# =====================================================
# PAGE: Overview
# =====================================================
if page == "Overview":
    st.title("SEC Commissioner Speeches Dashboard")
    st.markdown("Analysis of SEC Commissioner speeches \u2014 sentiment, topics, and trends.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Speeches", len(df))
    col2.metric("Unique Speakers", speaker_df["speaker_individual"].nunique())
    col3.metric("Total Words", f"{df['word_count'].sum():,}")
    col4.metric("Avg Words/Speech", f"{df['word_count'].mean():,.0f}")

    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.subheader("Speeches by Speaker")
        st.bar_chart(speaker_df["speaker_individual"].value_counts())
    with right:
        st.subheader("Word Count by Speaker")
        st.bar_chart(speaker_df.groupby("speaker_individual")["word_count"].sum().sort_values(ascending=False))

    st.markdown("---")

    st.subheader("All Speeches")
    display_df = df[["title", "speaker", "date", "word_count", "completeness_score"]].copy()
    display_df.columns = ["Title", "Speaker", "Date", "Words", "Completeness"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# =====================================================
# PAGE: Sentiment Analysis
# =====================================================
elif page == "Sentiment Analysis":
    st.title("Sentiment Analysis")
    st.markdown("Keyword-based sentiment scoring of speech content.")

    results = sentiment_data["results"]
    summary = sentiment_data["summary"]

    col1, col2, col3 = st.columns(3)
    dist = summary["sentiment_distribution"]
    col1.metric("Positive Speeches", dist["positive"])
    col2.metric("Neutral Speeches", dist["neutral"])
    col3.metric("Negative Speeches", dist["negative"])

    st.markdown("---")

    st.subheader("Sentiment Score by Speech")
    sent_df = pd.DataFrame(results)
    sent_df["short_title"] = sent_df["title"].str[:50] + "..."
    st.bar_chart(sent_df.set_index("short_title")["sentiment_score"])

    st.markdown("---")

    st.subheader("Detailed Sentiment Breakdown")
    detail_df = sent_df[["title", "speaker", "sentiment_score", "positive_words", "negative_words", "regulatory_words"]].copy()
    detail_df.columns = ["Title", "Speaker", "Sentiment Score", "Positive Keywords", "Negative Keywords", "Regulatory Keywords"]
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.subheader("Most Positive Speech")
        if summary["most_positive"]:
            mp = summary["most_positive"]
            st.markdown(f"**{mp['title'][:80]}...**")
            st.markdown(f"Speaker: {mp['speaker']}")
            st.markdown(f"Score: {mp['sentiment_score']:.3f}")
    with right:
        st.subheader("Most Negative Speech")
        if summary["most_negative"]:
            mn = summary["most_negative"]
            st.markdown(f"**{mn['title'][:80]}...**")
            st.markdown(f"Speaker: {mn['speaker']}")
            st.markdown(f"Score: {mn['sentiment_score']:.3f}")


# =====================================================
# PAGE: Topic Analysis
# =====================================================
elif page == "Topic Analysis":
    st.title("Topic Analysis")
    st.markdown("Keyword-based topic categorization across 6 regulatory domains.")

    results = topic_data["results"]
    summary = topic_data["summary"]

    st.subheader("Most Discussed Topics")
    topic_df = pd.DataFrame(summary["most_discussed_topics"])
    if not topic_df.empty:
        st.bar_chart(topic_df.set_index("topic")["total_mentions"])

    st.markdown("---")

    st.subheader("Topic Relevance by Speech")
    heatmap_rows = []
    for r in results:
        row = {"Speech": r["title"][:50] + "...", "Speaker": r["speaker"]}
        for topic, data in r["all_topics"].items():
            row[topic] = round(data["relevance_score"], 2)
        heatmap_rows.append(row)
    st.dataframe(pd.DataFrame(heatmap_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("Speeches by Primary Topic")
    groups = summary.get("speeches_by_primary_topic", {})
    for topic, titles in groups.items():
        with st.expander(f"{topic} ({len(titles)} speeches)"):
            for t in titles:
                st.markdown(f"- {t}")


# =====================================================
# PAGE: Corpus Explorer
# =====================================================
elif page in {"Speech Explorer", "Corpus Explorer"}:
    st.title("Corpus Explorer")
    st.markdown("Explore SEC speeches and uploaded documents with metadata and enrichment filters.")

    enrichment_state = _load_enrichment_state()
    corpus_df = _build_corpus_explorer_df(knowledge_data, enrichment_state)

    if corpus_df.empty:
        st.info("No corpus documents available.")
    else:
        source_options = sorted(corpus_df["source_kind"].fillna("unknown").astype(str).unique().tolist())
        org_options = sorted(corpus_df["organization"].fillna("Unknown").astype(str).unique().tolist())
        type_options = sorted(corpus_df["doc_type"].fillna("Unknown").astype(str).unique().tolist())
        stance_options = sorted([x for x in corpus_df["stance"].fillna("").astype(str).unique().tolist() if x.strip()])
        review_options = sorted(corpus_df["review"].fillna("pending").astype(str).unique().tolist())
        all_tags = sorted(
            {
                tag
                for tags in corpus_df["tags_list"].tolist()
                if isinstance(tags, list)
                for tag in tags
            }
        )

        f1, f2, f3 = st.columns(3)
        with f1:
            selected_sources = st.multiselect("Corpus Source", source_options, default=source_options)
        with f2:
            selected_orgs = st.multiselect("Organization", org_options, default=org_options)
        with f3:
            selected_types = st.multiselect("Document Type", type_options, default=type_options)

        f4, f5, f6 = st.columns(3)
        with f4:
            selected_stances = st.multiselect("Stance", stance_options)
        with f5:
            selected_reviews = st.multiselect("Review Status", review_options, default=review_options)
        with f6:
            selected_tags = st.multiselect("Must Include Tags", all_tags)

        q1, q2 = st.columns(2)
        with q1:
            title_query = st.text_input("Search Title")
        with q2:
            tag_keyword_query = st.text_input("Search Tags/Keywords")

        filtered = corpus_df.copy()
        if selected_sources:
            filtered = filtered[filtered["source_kind"].isin(selected_sources)]
        if selected_orgs:
            filtered = filtered[filtered["organization"].isin(selected_orgs)]
        if selected_types:
            filtered = filtered[filtered["doc_type"].isin(selected_types)]
        if selected_stances:
            filtered = filtered[filtered["stance"].isin(selected_stances)]
        if selected_reviews:
            filtered = filtered[filtered["review"].isin(selected_reviews)]
        if selected_tags:
            filtered = filtered[
                filtered["tags_list"].apply(
                    lambda items: isinstance(items, list) and all(tag in items for tag in selected_tags)
                )
            ]
        if title_query.strip():
            filtered = filtered[filtered["title"].str.contains(title_query.strip(), case=False, na=False)]
        if tag_keyword_query.strip():
            q = tag_keyword_query.strip()
            filtered = filtered[
                filtered["tags_text"].str.contains(q, case=False, na=False)
                | filtered["keywords_text"].str.contains(q, case=False, na=False)
            ]

        filtered = _sort_table_by_date(filtered, date_col="date")
        st.markdown(f"**Showing {len(filtered)} of {len(corpus_df)} corpus documents**")

        st.dataframe(
            filtered[
                [
                    "date",
                    "organization",
                    "source_kind",
                    "doc_type",
                    "title",
                    "speaker",
                    "stance",
                    "review",
                    "auto_verdict",
                    "status",
                    "tags_text",
                    "keywords_text",
                    "word_count",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        if not filtered.empty:
            detail_ids = filtered["doc_id"].astype(str).tolist()
            detail_map = {str(r["doc_id"]): r for _, r in filtered.iterrows()}
            selected_doc_id = st.selectbox(
                "Inspect Document",
                detail_ids,
                format_func=lambda d: f"{detail_map[d]['date']} | {detail_map[d]['title']}",
            )
            detail = detail_map[selected_doc_id]

            st.markdown(
                f"**{detail['title']}**\n\n"
                f"Type: `{detail['doc_type']}` | Source: `{detail['source_kind']}` | "
                f"Organization: `{detail['organization']}` | Date: `{detail['date']}`"
            )
            st.caption(
                f"Speaker: {detail['speaker']} | "
                f"Stance: {detail['stance'] or 'n/a'} | "
                f"Review: {detail['review']} | Auto Verdict: {detail['auto_verdict'] or 'n/a'}"
            )
            if detail["url"]:
                st.markdown(f"[Open Source]({detail['url']})")
            if detail["tags_text"]:
                st.markdown(f"**Tags:** {detail['tags_text']}")
            if detail["keywords_text"]:
                st.markdown(f"**Keywords:** {detail['keywords_text']}")

            full_text = str(detail["full_text"] or "")
            st.markdown("---")
            st.markdown(full_text[:5000] + ("..." if len(full_text) > 5000 else ""))
            if len(full_text) > 5000:
                with st.expander("Show full text"):
                    st.markdown(full_text)


# =====================================================
# PAGE: Agent Chat
# =====================================================
elif page == "Agent Chat":
    st.title("Agent Chat")
    st.markdown("Ask questions about the indexed corpus (SEC speeches + uploaded documents).")

    if _openai_key is None:
        st.error("OpenAI API key is not configured. Add `[openai].api_key` in Streamlit secrets.")
        st.stop()

    client = _get_openai_client()
    if client is None:
        st.error(st.session_state.get("_openai_error", "Failed to initialize OpenAI client."))
        st.stop()

    if "project_model_ids" not in st.session_state:
        try:
            st.session_state["project_model_ids"] = _list_project_models(client)
            st.session_state["project_model_error"] = ""
        except Exception as e:
            st.session_state["project_model_ids"] = []
            st.session_state["project_model_error"] = str(e)

    with st.expander("Model Access (This Project)"):
        if st.button("Refresh Model List"):
            try:
                st.session_state["project_model_ids"] = _list_project_models(client)
                st.session_state["project_model_error"] = ""
            except Exception as e:
                st.session_state["project_model_ids"] = []
                st.session_state["project_model_error"] = str(e)

        model_ids = st.session_state.get("project_model_ids", [])
        model_err = st.session_state.get("project_model_error", "")
        if model_err:
            st.error(f"Could not list models: {model_err}")
        else:
            st.caption(f"Visible models: {len(model_ids)}")
            chat_like = [m for m in model_ids if m.startswith("gpt-") or m.startswith("o")]
            if chat_like:
                st.markdown("**Likely chat-capable models:**")
                for mid in chat_like:
                    st.write(f"- {mid}")
            else:
                st.info("No `gpt-*` or `o*` models were returned for this project key.")

    available_models = _get_accessible_chat_models(client)
    if not available_models:
        available_models = _candidate_chat_models()

    default_model = "gpt-5.1" if "gpt-5.1" in available_models else available_models[0]
    model_name = st.selectbox(
        "Model",
        available_models,
        index=available_models.index(default_model),
    )

    org_options = _list_org_options(knowledge_data)
    org_labels = [o["label"] for o in org_options]
    if "agent_org_key" not in st.session_state:
        st.session_state["agent_org_key"] = org_options[0]["key"]

    default_org_idx = 0
    for idx, o in enumerate(org_options):
        if o["key"] == st.session_state.get("agent_org_key"):
            default_org_idx = idx
            break
    selected_org_label = st.selectbox("Organization", org_labels, index=default_org_idx)
    selected_org = next((o for o in org_options if o["label"] == selected_org_label), org_options[0])
    org_key = selected_org["key"]
    org_label = selected_org["label"]
    st.session_state["agent_org_key"] = org_key

    chat_scope = st.radio(
        "Chat Scope",
        ["Selected Organization", "All Organizations"],
        horizontal=True,
        help="Selected Organization queries one org store. All Organizations queries every indexed org store.",
    )
    discussion_mode = st.radio(
        "Discussion Mode",
        ["Corpus Retrieval", "Policy Briefings", "Corpus + Policy Briefings"],
        horizontal=True,
        help=(
            "Corpus Retrieval uses vector-search over indexed docs. "
            "Policy Briefings uses saved policy-delta reports. "
            "Corpus + Policy Briefings combines both."
        ),
    )

    index_status = _get_org_index_status(knowledge_data, org_key, org_label)
    if "vector_store_ids_by_org" not in st.session_state:
        st.session_state["vector_store_ids_by_org"] = {}

    active_vector_store_id = (
        st.session_state["vector_store_ids_by_org"].get(org_key) or index_status.get("vector_store_id")
    )
    if active_vector_store_id:
        st.session_state["vector_store_ids_by_org"][org_key] = active_vector_store_id

    if active_vector_store_id:
        st.caption(f"Current vector store for {org_label}: `{active_vector_store_id}`")
    else:
        st.caption(f"No vector store indexed yet for {org_label}.")

    state_all = _load_vector_state()
    stores_all = state_all.get("stores", {}) if isinstance(state_all, dict) else {}
    org_label_by_key = {o["key"]: o["label"] for o in org_options}
    all_store_rows = []
    for k, v in stores_all.items():
        if not isinstance(v, dict):
            continue
        vsid = str(v.get("vector_store_id", "") or "").strip()
        if not vsid:
            continue
        label = str(v.get("org_label", "") or org_label_by_key.get(k, k.upper()))
        all_store_rows.append({"key": k, "label": label, "vector_store_id": vsid})

    # Include in-memory mapped IDs from this session.
    for k, vsid in st.session_state["vector_store_ids_by_org"].items():
        vsid = str(vsid or "").strip()
        if not vsid:
            continue
        if not any(r["key"] == k for r in all_store_rows):
            all_store_rows.append(
                {
                    "key": k,
                    "label": org_label_by_key.get(k, k.upper()),
                    "vector_store_id": vsid,
                }
            )

    all_store_rows = sorted(all_store_rows, key=lambda r: r["label"].lower())
    all_vector_store_ids = [r["vector_store_id"] for r in all_store_rows]

    if chat_scope == "All Organizations":
        if all_store_rows:
            st.caption(f"All-organizations chat enabled across {len(all_store_rows)} indexed organization stores.")
        else:
            st.caption("All-organizations chat has no indexed stores yet.")

    brief_org_keys = [org_key]
    if chat_scope == "All Organizations":
        brief_org_keys = [o["key"] for o in org_options]
    briefs_payload = _load_policy_briefs()

    available_brief_kinds = sorted(
        {
            str((b.get("source_doc", {}) if isinstance(b.get("source_doc", {}), dict) else {}).get("source_kind", "") or "document")
            for b in briefs_payload.get("briefs", [])
            if isinstance(b, dict)
            and (not brief_org_keys or _policy_brief_org_key(b) in set(brief_org_keys))
        }
    )
    if not available_brief_kinds:
        available_brief_kinds = ["document"]
    selected_brief_source_kinds = available_brief_kinds
    brief_context_limit = 5
    with st.expander("Policy Briefing Context", expanded=(discussion_mode != "Corpus Retrieval")):
        selected_brief_source_kinds = st.multiselect(
            "Briefing Source Kinds",
            available_brief_kinds,
            default=available_brief_kinds,
            key=f"chat_brief_kinds_{chat_scope}_{org_key}",
        )
        brief_context_limit = st.slider(
            "Briefings To Use Per Answer",
            min_value=1,
            max_value=12,
            value=5,
            key=f"chat_brief_limit_{chat_scope}_{org_key}",
        )
        brief_count_in_scope = sum(
            1
            for b in briefs_payload.get("briefs", [])
            if isinstance(b, dict)
            and _policy_brief_org_key(b) in set(brief_org_keys)
            and (
                not selected_brief_source_kinds
                or str((b.get("source_doc", {}) if isinstance(b.get("source_doc", {}), dict) else {}).get("source_kind", "") or "document")
                in set(selected_brief_source_kinds)
            )
        )
        st.caption(f"{brief_count_in_scope} saved policy briefs currently match this chat scope.")

    if active_vector_store_id:
        with st.expander("Vector Store Diagnostics"):
            diag_state_key = f"vector_diag_{org_key}"
            deep_diag_key = f"vector_diag_deep_{org_key}"
            verify_state_key = f"vector_verify_{org_key}"
            listing_state_key = f"vector_listing_{org_key}"

            if st.button("Refresh Store Counts", key=f"refresh_store_counts_{org_key}"):
                try:
                    st.session_state[diag_state_key] = _get_vector_store_file_counts(
                        client,
                        active_vector_store_id,
                    )
                except Exception as e:
                    st.session_state[diag_state_key] = {"error": str(e)}

            if st.button("Run Deep Count (List All Files)", key=f"deep_store_counts_{org_key}"):
                with st.spinner("Counting files in vector store..."):
                    try:
                        st.session_state[deep_diag_key] = _count_vector_store_files_deep(
                            client,
                            active_vector_store_id,
                        )
                    except Exception as e:
                        st.session_state[deep_diag_key] = {"error": str(e)}

            left_diag, right_diag = st.columns(2)
            with left_diag:
                if st.button("Verify Active Store ID", key=f"verify_store_{org_key}"):
                    st.session_state[verify_state_key] = _verify_active_vector_store(
                        client,
                        active_vector_store_id,
                    )
            with right_diag:
                if st.button("List Project Vector Stores", key=f"list_project_stores_{org_key}"):
                    try:
                        st.session_state[listing_state_key] = _list_project_vector_stores(client, limit=100)
                    except Exception as e:
                        st.session_state[listing_state_key] = [{"error": str(e)}]

            diag_counts = st.session_state.get(diag_state_key)
            if isinstance(diag_counts, dict):
                if diag_counts.get("error"):
                    st.error(f"Store count check failed: {diag_counts['error']}")
                else:
                    st.markdown(
                        f"API counts: total={diag_counts.get('total', 0)}, "
                        f"completed={diag_counts.get('completed', 0)}, "
                        f"in_progress={diag_counts.get('in_progress', 0)}, "
                        f"failed={diag_counts.get('failed', 0)}, "
                        f"cancelled={diag_counts.get('cancelled', 0)}"
                    )

            deep_counts = st.session_state.get(deep_diag_key)
            if isinstance(deep_counts, dict):
                if deep_counts.get("error"):
                    st.error(f"Deep count failed: {deep_counts['error']}")
                else:
                    st.markdown(
                        f"Deep counts: total={deep_counts.get('total', 0)}, "
                        f"completed={deep_counts.get('completed', 0)}, "
                        f"in_progress={deep_counts.get('in_progress', 0)}, "
                        f"failed={deep_counts.get('failed', 0)}, "
                        f"cancelled={deep_counts.get('cancelled', 0)}"
                    )

            verify_result = st.session_state.get(verify_state_key)
            if isinstance(verify_result, dict):
                if verify_result.get("ok"):
                    fc = verify_result.get("file_counts", {})
                    st.success(
                        f"Active store verified: {verify_result.get('id', '')} "
                        f"(status={verify_result.get('status', '')}, "
                        f"completed={int(fc.get('completed', 0) or 0)}, total={int(fc.get('total', 0) or 0)})"
                    )
                elif verify_result.get("message"):
                    st.error(f"Active store verification failed: {verify_result['message']}")

            listing_rows = st.session_state.get(listing_state_key)
            if isinstance(listing_rows, list) and listing_rows:
                if "error" in listing_rows[0]:
                    st.error(f"Project vector-store listing failed: {listing_rows[0].get('error', '')}")
                else:
                    listing_df = pd.DataFrame(listing_rows)
                    if not listing_df.empty:
                        if "id" in listing_df.columns:
                            listing_df["active"] = listing_df["id"].apply(
                                lambda x: "yes" if str(x) == str(active_vector_store_id) else ""
                            )
                        show_cols = [
                            c
                            for c in ["active", "id", "name", "status", "total", "completed", "in_progress", "failed"]
                            if c in listing_df.columns
                        ]
                        st.dataframe(listing_df[show_cols], use_container_width=True, hide_index=True)

    idx_col1, idx_col2, idx_col3, idx_col4 = st.columns(4)
    idx_col1.metric("Corpus Docs", index_status.get("current_docs", 0))
    idx_col2.metric("Indexed Docs", index_status.get("indexed_docs", 0))
    idx_col3.metric(
        "Pending Add/Update",
        index_status.get("pending_add", 0) + index_status.get("pending_update", 0),
    )
    idx_col4.metric("Pending Remove", index_status.get("pending_remove", 0))
    pending_total = (
        index_status.get("pending_add", 0)
        + index_status.get("pending_update", 0)
        + index_status.get("pending_remove", 0)
    )
    if pending_total > 0:
        st.warning(
            f"Knowledge index is out of sync for {org_label} "
            f"({pending_total} pending changes). Answers may reflect older positions until sync completes."
        )

    last_sync = index_status.get("last_sync", {})
    if isinstance(last_sync, dict) and last_sync:
        ls_status = str(last_sync.get("status", "unknown") or "unknown")
        ls_mode = str(last_sync.get("sync_mode", "unknown") or "unknown")
        ls_uploaded = int(last_sync.get("uploaded", 0) or 0)
        ls_deleted = int(last_sync.get("deleted", 0) or 0)
        ls_failed = int(last_sync.get("failed_count", len(last_sync.get("failed", []) or [])) or 0)
        st.caption(
            f"Last sync: status={ls_status}, mode={ls_mode}, "
            f"uploaded={ls_uploaded}, deleted={ls_deleted}, failed={ls_failed}"
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Build/Sync Knowledge Index", type="primary"):
            sync_progress = st.progress(0)
            sync_status = st.empty()

            def _sync_progress_cb(done, total, message):
                if total <= 0:
                    sync_progress.progress(100)
                    sync_status.caption(message)
                    return
                safe_done = max(0, min(done, total))
                pct = int((safe_done * 100) / total)
                sync_progress.progress(max(0, min(pct, 100)))
                sync_status.caption(f"{message} ({safe_done}/{total})")

            with st.spinner(f"Syncing {org_label} knowledge index..."):
                try:
                    report = _sync_org_vector_store(
                        client=client,
                        raw_data_obj=knowledge_data,
                        org_key=org_key,
                        org_label=org_label,
                        force_rebuild=False,
                        progress_callback=_sync_progress_cb,
                    )
                    vector_store_id = report.get("vector_store_id", "")
                    if vector_store_id:
                        st.session_state["vector_store_ids_by_org"][org_key] = vector_store_id

                    if report.get("up_to_date"):
                        st.success("Knowledge index is up to date.")
                    else:
                        stats = report.get("stats", {})
                        planned_add = stats.get("add", 0)
                        planned_update = stats.get("update", 0)
                        planned_remove = stats.get("remove", 0)
                        uploaded = stats.get("uploaded", 0)
                        deleted = stats.get("deleted", 0)
                        failed_count = len(report.get("failed", []))

                        st.success(
                            "Knowledge index sync finished: "
                            f"planned +{planned_add}/~{planned_update}/-{planned_remove}, "
                            f"completed uploads={uploaded}, deletes={deleted}."
                        )
                        if report.get("all_uploads_failed"):
                            st.error(
                                "No uploads were recorded in this run. "
                                "Use Vector Store Diagnostics and check operation failures below."
                            )
                        if report.get("failed"):
                            st.warning(f"{failed_count} document operations failed.")
                            for item in report["failed"][:5]:
                                st.write(f"- {item.get('stage', 'unknown')}: {item.get('title', item.get('doc_id', ''))}")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
    with col2:
        if st.button("Force Rebuild Index"):
            rebuild_progress = st.progress(0)
            rebuild_status = st.empty()

            def _rebuild_progress_cb(done, total, message):
                if total <= 0:
                    rebuild_progress.progress(100)
                    rebuild_status.caption(message)
                    return
                safe_done = max(0, min(done, total))
                pct = int((safe_done * 100) / total)
                rebuild_progress.progress(max(0, min(pct, 100)))
                rebuild_status.caption(f"{message} ({safe_done}/{total})")

            with st.spinner(f"Rebuilding {org_label} knowledge index..."):
                try:
                    report = _sync_org_vector_store(
                        client=client,
                        raw_data_obj=knowledge_data,
                        org_key=org_key,
                        org_label=org_label,
                        force_rebuild=True,
                        progress_callback=_rebuild_progress_cb,
                    )
                    vector_store_id = report.get("vector_store_id", "")
                    if vector_store_id:
                        st.session_state["vector_store_ids_by_org"][org_key] = vector_store_id
                    stats = report.get("stats", {})
                    st.success(
                        "Knowledge index rebuild finished: "
                        f"planned +{stats.get('add', 0)}/~{stats.get('update', 0)}/-{stats.get('remove', 0)}, "
                        f"completed uploads={stats.get('uploaded', 0)}, deletes={stats.get('deleted', 0)}."
                    )
                    if report.get("all_uploads_failed"):
                        st.error(
                            "No uploads were recorded in this rebuild. "
                            "Use Vector Store Diagnostics and check failures below."
                        )
                    if report.get("failed"):
                        st.warning(f"{len(report['failed'])} document operations failed during rebuild.")
                    if report.get("old_store_delete_error"):
                        st.info(f"Previous vector store cleanup skipped: {report['old_store_delete_error']}")
                except Exception as e:
                    st.error(f"Rebuild failed: {e}")

    chat_scope_key = org_key
    chat_scope_label = org_label
    chat_vector_store_ids = [active_vector_store_id] if active_vector_store_id else []
    if chat_scope == "All Organizations":
        chat_scope_key = "__all_organizations__"
        chat_scope_label = "All Organizations"
        chat_vector_store_ids = all_vector_store_ids
    chat_mode_key = (
        str(discussion_mode or "corpus")
        .strip()
        .lower()
        .replace(" + ", "_plus_")
        .replace(" ", "_")
    )
    chat_session_key = f"{chat_scope_key}::{chat_mode_key}"

    if "chat_messages_by_org" not in st.session_state:
        st.session_state["chat_messages_by_org"] = {}
    if chat_session_key not in st.session_state["chat_messages_by_org"]:
        st.session_state["chat_messages_by_org"][chat_session_key] = []
    chat_messages = st.session_state["chat_messages_by_org"][chat_session_key]

    for msg in chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("results"):
                with st.expander("Retrieved Sources"):
                    for r in msg["results"][:8]:
                        score = r.get("score")
                        score_txt = f" (score: {score:.3f})" if isinstance(score, (int, float)) else ""
                        st.markdown(f"- `{r.get('filename', 'unknown')}`{score_txt}")
                        if r.get("snippet"):
                            st.caption(r["snippet"])

    user_prompt = st.chat_input(f"Ask a question about {chat_scope_label} documents...")
    if user_prompt:
        chat_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        need_vector = discussion_mode in {"Corpus Retrieval", "Corpus + Policy Briefings"}
        need_briefs = discussion_mode in {"Policy Briefings", "Corpus + Policy Briefings"}
        if need_vector and not chat_vector_store_ids and not need_briefs:
            err_msg = "Please click **Build/Sync Knowledge Index** before chatting."
            chat_messages.append({"role": "assistant", "content": err_msg})
            with st.chat_message("assistant"):
                st.error(err_msg)
        elif _needs_temporal_clarification(user_prompt):
            clarify_msg = _build_temporal_clarification_message(knowledge_df, chat_scope_key, chat_scope_label)
            chat_messages.append({"role": "assistant", "content": clarify_msg})
            with st.chat_message("assistant"):
                st.info(clarify_msg)
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        brief_sources = []
                        brief_context_text = ""
                        if need_briefs:
                            _, brief_context_text, brief_sources = _select_policy_brief_context(
                                briefs_payload=briefs_payload,
                                question=user_prompt,
                                org_keys=brief_org_keys,
                                source_kinds=selected_brief_source_kinds,
                                limit=brief_context_limit,
                            )

                        instructions_text = _build_agent_instructions(
                            knowledge_df,
                            chat_scope_key,
                            chat_scope_label,
                        )
                        if discussion_mode == "Policy Briefings":
                            if not brief_context_text.strip():
                                answer = (
                                    "No saved policy delta briefings match this scope yet. "
                                    "Generate briefings in **Policy Delta Briefings** first."
                                )
                                sources = []
                                used_model = model_name
                                fallback_used = False
                            else:
                                agent_out = _ask_policy_brief_chat_with_fallback(
                                    client=client,
                                    question=user_prompt,
                                    preferred_model=model_name,
                                    model_pool=available_models,
                                    context_text=brief_context_text,
                                    instructions_text=instructions_text,
                                )
                                result = agent_out.get("result", {})
                                answer = result.get("answer", "No answer returned.")
                                sources = brief_sources
                                used_model = agent_out.get("used_model", model_name)
                                fallback_used = agent_out.get("fallback_used", False)
                        elif discussion_mode == "Corpus + Policy Briefings":
                            if chat_vector_store_ids:
                                question_text = str(user_prompt)
                                if brief_context_text.strip():
                                    question_text += (
                                        "\n\nPolicy Delta Briefing Context (use this with retrieved corpus evidence):\n"
                                        f"{brief_context_text}"
                                    )
                                hybrid_instructions = (
                                    f"{instructions_text} "
                                    "If policy briefing context is provided, treat it as analyst metadata and still "
                                    "ground final claims in retrieved corpus text when available."
                                )
                                agent_out = _ask_agent_with_fallback(
                                    client=client,
                                    vector_store_ids=chat_vector_store_ids,
                                    question=question_text,
                                    preferred_model=model_name,
                                    model_pool=available_models,
                                    instructions_text=hybrid_instructions,
                                )
                                result = agent_out.get("result", {})
                                answer = result.get("answer", "No answer returned.")
                                sources = list(result.get("results", [])) + list(brief_sources)
                                used_model = agent_out.get("used_model", model_name)
                                fallback_used = agent_out.get("fallback_used", False)
                            elif brief_context_text.strip():
                                agent_out = _ask_policy_brief_chat_with_fallback(
                                    client=client,
                                    question=user_prompt,
                                    preferred_model=model_name,
                                    model_pool=available_models,
                                    context_text=brief_context_text,
                                    instructions_text=instructions_text,
                                )
                                result = agent_out.get("result", {})
                                answer = result.get("answer", "No answer returned.")
                                sources = brief_sources
                                used_model = agent_out.get("used_model", model_name)
                                fallback_used = agent_out.get("fallback_used", False)
                            else:
                                answer = (
                                    "No indexed vector store and no saved policy briefings are available in this scope. "
                                    "Build/Sync the index or generate policy briefs first."
                                )
                                sources = []
                                used_model = model_name
                                fallback_used = False
                        else:
                            if not chat_vector_store_ids:
                                answer = "Please click **Build/Sync Knowledge Index** before chatting."
                                sources = []
                                used_model = model_name
                                fallback_used = False
                            else:
                                agent_out = _ask_agent_with_fallback(
                                    client=client,
                                    vector_store_ids=chat_vector_store_ids,
                                    question=user_prompt,
                                    preferred_model=model_name,
                                    model_pool=available_models,
                                    instructions_text=instructions_text,
                                )
                                result = agent_out.get("result", {})
                                answer = result.get("answer", "No answer returned.")
                                sources = result.get("results", [])
                                used_model = agent_out.get("used_model", model_name)
                                fallback_used = agent_out.get("fallback_used", False)
                    except Exception as e:
                        answer = f"Chat request failed: {e}"
                        sources = []
                        used_model = model_name
                        fallback_used = False

                st.markdown(answer)
                if fallback_used and used_model != model_name:
                    st.info(
                        f"Selected model `{model_name}` was unavailable for this project. "
                        f"Used fallback model `{used_model}`."
                    )
                if sources:
                    with st.expander("Retrieved Sources"):
                        for r in sources[:8]:
                            score = r.get("score")
                            score_txt = f" (score: {score:.3f})" if isinstance(score, (int, float)) else ""
                            st.markdown(f"- `{r.get('filename', 'unknown')}`{score_txt}")
                            if r.get("snippet"):
                                st.caption(r["snippet"])

            chat_messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "results": sources,
                }
            )


# =====================================================
# PAGE: Extraction
# =====================================================
elif page == "Extraction":
    st.title("Extraction")
    st.markdown("Run all extraction and ingestion pipelines, including manual document upload.")
    st.caption(
        "Use this for long transcripts (including SEC roundtables) and internal documents. "
        "Very long text is chunked automatically during indexing."
    )

    custom_payload = _load_custom_documents()
    custom_docs = _custom_docs_as_speeches(custom_payload)

    st.subheader("SEC Connector: Speeches")
    st.markdown("Discover and extract SEC speeches by date range.")

    sec_col1, sec_col2 = st.columns(2)
    with sec_col1:
        sec_start_date = st.date_input(
            "SEC Speech Start Date",
            value=date.today() - timedelta(days=30),
            key="ext_sec_speech_start_date",
        )
    with sec_col2:
        sec_end_date = st.date_input(
            "SEC Speech End Date",
            value=date.today(),
            key="ext_sec_speech_end_date",
        )

    sec_discovered_key = "ext_sec_speeches_discovered"
    if sec_discovered_key not in st.session_state:
        st.session_state[sec_discovered_key] = []

    if st.button("Discover SEC Speeches", key="ext_discover_sec_speeches"):
        with st.status("Discovering speeches from SEC.gov...", expanded=True) as status:
            from sec_scraper_free import SECScraper

            scraper = SECScraper()
            days_back_to_start = max(0, (date.today() - sec_start_date).days)
            estimated_pages = max(3, days_back_to_start // 14 + 2)
            max_pages = min(80, estimated_pages)

            st.write(
                f"Scanning up to {max_pages} listing pages "
                "(will stop early when start date is reached)..."
            )
            entries = scraper.discover_speech_urls(
                max_pages=max_pages,
                start_date=sec_start_date,
                end_date=sec_end_date,
            )

            existing_urls = {s.get("metadata", {}).get("url", "") for s in raw_data.get("speeches", [])}
            new_entries = [e for e in entries if e["url"] not in existing_urls]
            already = len(entries) - len(new_entries)

            st.session_state[sec_discovered_key] = new_entries
            status.update(
                label=f"Found {len(new_entries)} new speeches ({already} already extracted)",
                state="complete",
            )

    sec_discovered = st.session_state.get(sec_discovered_key, [])
    if sec_discovered:
        sec_disc_df = _sort_table_by_date(pd.DataFrame(sec_discovered), date_col="date")
        if "speaker" in sec_disc_df.columns:
            sec_disc_df["speaker"] = sec_disc_df["speaker"].apply(format_speakers)
        sec_discovered_sorted = sec_disc_df.to_dict(orient="records")

        st.dataframe(
            sec_disc_df[["date", "title", "speaker", "type"]],
            use_container_width=True,
            hide_index=True,
        )

        if len(sec_discovered_sorted) == 1:
            sec_extract_limit = 1
            st.caption("1 speech found. It will be extracted.")
        else:
            sec_extract_limit = st.slider(
                "SEC Speeches To Extract",
                min_value=1,
                max_value=len(sec_discovered_sorted),
                value=len(sec_discovered_sorted),
                key="ext_sec_extract_limit",
            )

        if st.button("Run SEC Speech Extraction", key="ext_extract_sec_speeches"):
            from speech_analyzer import SECSpeechAnalyzer

            analyzer = SECSpeechAnalyzer()
            progress = st.progress(0, text="Starting SEC speech extraction...")
            extracted = []
            failed = []

            for i, entry in enumerate(sec_discovered_sorted[:sec_extract_limit]):
                progress.progress(
                    (i + 1) / sec_extract_limit,
                    text=f"Extracting {i + 1}/{sec_extract_limit}: {entry['title'][:50]}...",
                )
                result = analyzer.extract_speech_for_analysis(entry["url"], listing_metadata=entry)
                if result["success"] and analyzer.validate_full_text_extraction(result["data"]):
                    extracted.append(result["data"])
                else:
                    failed.append(entry["title"])

            progress.progress(1.0, text="SEC speech extraction complete.")

            if extracted:
                updated_data, _ = _load_raw_data()
                updated_data["speeches"].extend(extracted)
                updated_data["extraction_summary"]["successful_extractions"] = len(updated_data["speeches"])

                storage = _get_gcs_storage()
                if storage is not None:
                    storage.save_speeches(updated_data)
                    st.success(f"Saved {len(extracted)} new speeches to Google Cloud Storage.")
                else:
                    with open("data/all_speeches_final.json", "w", encoding="utf-8") as f:
                        json.dump(updated_data, f, indent=2, ensure_ascii=False)
                    st.success(f"Saved {len(extracted)} new speeches locally.")

                load_data.clear()
                run_analysis.clear()
                st.session_state[sec_discovered_key] = []
                st.info("Refresh the page to see the new speeches in the dashboard.")

            if failed:
                st.warning(f"{len(failed)} speeches failed extraction:")
                for title in failed:
                    st.write(f"- {title}")
    else:
        st.info("Use the SEC speech date range above and click **Discover SEC Speeches**.")

    st.markdown("---")
    st.subheader("Manual Upload")

    with st.form("upload_custom_document", clear_on_submit=True):
        uploaded_file = st.file_uploader(
            "Upload document",
            type=["pdf", "txt", "md", "html", "htm"],
            help="Accepted formats: PDF, plain text, markdown, or HTML.",
        )
        pasted_text = st.text_area(
            "Or paste transcript/document text (optional)",
            height=180,
            placeholder="Paste long transcript text here if you are not uploading a file.",
        )

        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            org_name = st.text_input("Organization", value="Custom Documents")
            title = st.text_input("Title")
            speaker = st.text_input("Speaker/Author", value="Unknown")
            doc_type = st.selectbox(
                "Document Type",
                ["Transcript", "Statement", "News Article", "Report", "Memo", "Letter", "Other"],
            )
        with meta_col2:
            doc_date = st.date_input("Document Date", value=date.today())
            source_url = st.text_input("Source URL (optional)")
            tags_csv = st.text_input("Tags (comma-separated, optional)")

        submit_doc = st.form_submit_button("Save Document")

    if submit_doc:
        if not title.strip():
            st.error("Title is required.")
        else:
            text = ""
            source_ext = ".txt"
            source_name = "pasted_text.txt"
            file_bytes = b""
            warnings = []

            if uploaded_file is not None:
                try:
                    text, source_ext, warnings, file_bytes = _extract_text_from_uploaded_file(uploaded_file)
                    source_name = _safe_filename(uploaded_file.name)
                except Exception as e:
                    st.error(f"Could not parse uploaded file: {e}")
                    text = ""
            elif pasted_text.strip():
                text = pasted_text.strip()
                source_ext = ".txt"
                source_name = "pasted_text.txt"
                file_bytes = text.encode("utf-8")
            else:
                st.error("Upload a file or paste text before saving.")

            if text:
                draft_record = _create_uploaded_document_record(
                    text=text,
                    organization=org_name,
                    title=title,
                    speaker=speaker,
                    doc_date=doc_date,
                    doc_type=doc_type,
                    source_url=source_url,
                    source_filename=source_name,
                    source_ext=source_ext,
                    source_local_path="",
                    source_gcs_path="",
                    tags_csv=tags_csv,
                )
                doc_id = draft_record.get("metadata", {}).get("document_id", "")
                local_path, gcs_path = _store_uploaded_source_file(file_bytes, source_name, doc_id)

                final_record = _create_uploaded_document_record(
                    text=text,
                    organization=org_name,
                    title=title,
                    speaker=speaker,
                    doc_date=doc_date,
                    doc_type=doc_type,
                    source_url=source_url,
                    source_filename=source_name,
                    source_ext=source_ext,
                    source_local_path=local_path,
                    source_gcs_path=gcs_path,
                    tags_csv=tags_csv,
                )
                replaced = _upsert_custom_document_record(custom_payload, final_record)
                _save_custom_documents(custom_payload)

                word_count = int(final_record.get("metadata", {}).get("word_count", 0) or 0)
                action = "Updated" if replaced else "Saved"
                st.success(f"{action} document `{title}` ({word_count:,} words).")
                if gcs_path:
                    st.caption(f"Source file stored in GCS at `{gcs_path}`")
                for warn in warnings:
                    st.warning(warn)

                # Refresh in-memory view for this run.
                custom_payload = _load_custom_documents()
                custom_docs = _custom_docs_as_speeches(custom_payload)

    st.markdown("---")
    st.subheader("SEC Connector: Trading & Markets FAQ")
    st.caption(
        "Discover and ingest SEC Division of Trading and Markets FAQ pages directly into the knowledge base."
    )

    tm_index_default = "https://www.sec.gov/rules-regulations/staff-guidance/trading-markets-frequently-asked-questions"
    tm_index_url = st.text_input(
        "Trading & Markets FAQ Index URL",
        value=tm_index_default,
        key="tm_faq_index_url",
    ).strip() or tm_index_default
    tm_include_pdfs = st.checkbox("Include linked PDF FAQs", value=True, key="tm_faq_include_pdfs")

    discover_col1, discover_col2 = st.columns(2)
    with discover_col1:
        discover_tm = st.button("Discover FAQ Links", key="discover_tm_faq")
    with discover_col2:
        clear_tm = st.button("Clear Discovered Links", key="clear_tm_faq")

    tm_state_key = "tm_faq_discovered"
    if tm_state_key not in st.session_state:
        st.session_state[tm_state_key] = []
    if clear_tm:
        st.session_state[tm_state_key] = []

    if discover_tm:
        try:
            from sec_tm_faq_scraper import TradingMarketsFAQScraper

            with st.spinner("Discovering Trading & Markets FAQ links..."):
                tm_scraper = TradingMarketsFAQScraper()
                discovered = tm_scraper.discover_documents(
                    index_url=tm_index_url,
                    include_pdfs=tm_include_pdfs,
                )

            existing_custom = {}
            for item in custom_docs:
                m = item.get("metadata", {})
                existing_custom[_url_match_key(m.get("url", ""))] = m

            existing_speech_urls = {
                _url_match_key(s.get("metadata", {}).get("url", ""))
                for s in raw_data.get("speeches", [])
            }

            for entry in discovered:
                key = _url_match_key(entry.get("url", ""))
                status = "new"
                existing_meta = existing_custom.get(key)
                if existing_meta:
                    existing_updated = str(
                        existing_meta.get("last_reviewed_or_updated")
                        or existing_meta.get("updated_date")
                        or ""
                    ).strip()
                    incoming_updated = str(entry.get("updated_date", "") or "").strip()
                    if incoming_updated and existing_updated and incoming_updated != existing_updated:
                        status = "update_available"
                    else:
                        status = "existing"
                elif key in existing_speech_urls:
                    status = "existing_in_speeches"
                entry["ingest_status"] = status
                entry["date"] = entry.get("updated_date") or entry.get("published_date") or ""

            st.session_state[tm_state_key] = discovered
            new_count = sum(1 for d in discovered if d.get("ingest_status") in {"new", "update_available"})
            st.success(
                f"Discovered {len(discovered)} Trading & Markets FAQ links "
                f"({new_count} new/update candidates)."
            )
        except Exception as e:
            st.error(f"FAQ discovery failed: {e}")

    tm_discovered = st.session_state.get(tm_state_key, [])
    if tm_discovered:
        tm_df = pd.DataFrame(tm_discovered)
        if "date" not in tm_df.columns:
            tm_df["date"] = tm_df.get("updated_date", "")
        tm_df = _sort_table_by_date(tm_df, date_col="date")
        st.dataframe(
            tm_df[["date", "title", "source_format", "ingest_status", "url"]],
            use_container_width=True,
            hide_index=True,
        )

        ingest_filter = st.selectbox(
            "Ingest Selection",
            ["New/Updates Only", "All Discovered"],
            key="tm_faq_ingest_filter",
        )
        if ingest_filter == "New/Updates Only":
            ingest_candidates = [
                d for d in tm_discovered if d.get("ingest_status") in {"new", "update_available"}
            ]
        else:
            ingest_candidates = list(tm_discovered)

        ingest_count = len(ingest_candidates)
        if ingest_count <= 0:
            st.caption("No FAQ links match the selected ingest filter.")
            ingest_limit = 0
        elif ingest_count == 1:
            ingest_limit = 1
            st.caption("1 FAQ document selected for ingest.")
        else:
            ingest_limit = st.slider(
                "FAQ Documents To Ingest",
                min_value=1,
                max_value=ingest_count,
                value=min(10, ingest_count),
                key="tm_faq_ingest_limit",
            )
        st.caption(f"{ingest_count} FAQ documents currently match this ingest selection.")

        if st.button("Run Trading & Markets FAQ Extraction", disabled=(ingest_limit <= 0), key="ingest_tm_faq"):
            try:
                from sec_tm_faq_scraper import TradingMarketsFAQScraper

                tm_scraper = TradingMarketsFAQScraper()
                progress = st.progress(0, text="Starting FAQ ingest...")
                saved_new = 0
                saved_updates = 0
                failed = []

                selected = ingest_candidates[:ingest_limit]
                for idx, entry in enumerate(selected, 1):
                    progress.progress(
                        idx / ingest_limit,
                        text=f"Ingesting {idx}/{ingest_limit}: {entry.get('title', '')[:80]}",
                    )
                    try:
                        fallback_date = entry.get("updated_date") or entry.get("published_date") or ""
                        extracted = tm_scraper.extract_document(
                            entry.get("url", ""),
                            fallback_title=entry.get("title", ""),
                            fallback_date=fallback_date,
                        )
                        if not extracted.get("success"):
                            raise RuntimeError("Extraction returned unsuccessful result.")
                        data = extracted.get("data", {})
                        text = str(data.get("full_text", "") or "").strip()
                        if len(text.split()) < 80:
                            raise RuntimeError("Extracted text appears too short; skipping.")

                        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
                        src_format = str(data.get("source_format", "") or entry.get("source_format", "html")).lower()
                        source_ext = ".pdf" if src_format == "pdf" else ".html"
                        source_name = urlparse(src_url).path.rsplit("/", 1)[-1].strip()
                        if not source_name:
                            source_name = f"tm-faq-{idx}{source_ext}"
                        elif "." not in source_name:
                            source_name += source_ext

                        date_text = str(data.get("date", "") or fallback_date).strip()
                        parsed_date = _parse_single_date(date_text)
                        if pd.notna(parsed_date):
                            doc_date_value = parsed_date.date()
                        else:
                            doc_date_value = date_text

                        record = _create_uploaded_document_record(
                            text=text,
                            organization="SEC",
                            title=str(data.get("title", "") or entry.get("title", "")).strip(),
                            speaker="Division of Trading and Markets",
                            doc_date=doc_date_value,
                            doc_type="FAQ",
                            source_url=src_url,
                            source_filename=source_name,
                            source_ext=source_ext,
                            source_local_path="",
                            source_gcs_path="",
                            tags_csv="sec,trading-markets,faq,staff-guidance",
                            source_kind="sec_tm_faq",
                        )
                        rm = record.setdefault("metadata", {})
                        rm["source_family"] = "sec_tm_faq"
                        rm["source_index_url"] = tm_index_url
                        rm["published_date"] = str(entry.get("published_date", "") or "")
                        rm["updated_date"] = str(entry.get("updated_date", "") or "")
                        rm["last_reviewed_or_updated"] = str(
                            data.get("last_reviewed_or_updated", "") or entry.get("updated_date", "") or ""
                        )

                        replaced = _upsert_custom_document_record(custom_payload, record)
                        if replaced:
                            saved_updates += 1
                        else:
                            saved_new += 1

                    except Exception as e:
                        failed.append(f"{entry.get('title', 'Untitled')}: {e}")

                progress.progress(1.0, text="FAQ ingest complete.")
                if saved_new or saved_updates:
                    _save_custom_documents(custom_payload)
                    st.success(
                        f"Saved {saved_new} new FAQ docs and updated {saved_updates} existing FAQ docs."
                    )
                    custom_payload = _load_custom_documents()
                    custom_docs = _custom_docs_as_speeches(custom_payload)
                if failed:
                    st.warning(f"{len(failed)} FAQ docs failed ingest.")
                    for msg in failed[:20]:
                        st.write(f"- {msg}")
            except Exception as e:
                st.error(f"FAQ ingest failed: {e}")

    st.markdown("---")
    st.subheader("SEC Connector: Enforcement Litigation Releases")
    st.caption("Discover and ingest SEC Litigation Releases directly into the knowledge base.")

    lit_index_default = "https://www.sec.gov/enforcement-litigation/litigation-releases"
    lit_index_url = st.text_input(
        "Litigation Releases Index URL",
        value=lit_index_default,
        key="sec_lit_index_url",
    ).strip() or lit_index_default
    lit_pages = st.slider(
        "Listing Pages To Scan",
        min_value=1,
        max_value=20,
        value=3,
        key="sec_lit_pages",
    )

    lit_col1, lit_col2 = st.columns(2)
    with lit_col1:
        discover_lit = st.button("Discover Litigation Releases", key="discover_sec_lit")
    with lit_col2:
        clear_lit = st.button("Clear Litigation Results", key="clear_sec_lit")

    lit_state_key = "sec_lit_discovered"
    if lit_state_key not in st.session_state:
        st.session_state[lit_state_key] = []
    if clear_lit:
        st.session_state[lit_state_key] = []

    if discover_lit:
        try:
            from sec_enforcement_litigation_scraper import SECEnforcementLitigationScraper

            with st.spinner("Discovering SEC litigation releases..."):
                lit_scraper = SECEnforcementLitigationScraper()
                lit_discovered = lit_scraper.discover_documents(
                    base_url=lit_index_url,
                    max_pages=lit_pages,
                )

            existing_custom = {}
            for item in custom_docs:
                m = item.get("metadata", {})
                existing_custom[_url_match_key(m.get("url", ""))] = m

            existing_speech_urls = {
                _url_match_key(s.get("metadata", {}).get("url", ""))
                for s in raw_data.get("speeches", [])
            }

            for entry in lit_discovered:
                key = _url_match_key(entry.get("url", ""))
                status = "new"
                if key in existing_custom:
                    status = "existing"
                elif key in existing_speech_urls:
                    status = "existing_in_speeches"
                entry["ingest_status"] = status

            st.session_state[lit_state_key] = lit_discovered
            new_count = sum(1 for d in lit_discovered if d.get("ingest_status") == "new")
            st.success(
                f"Discovered {len(lit_discovered)} litigation releases "
                f"({new_count} new candidates)."
            )
        except Exception as e:
            st.error(f"Litigation release discovery failed: {e}")

    lit_discovered = st.session_state.get(lit_state_key, [])
    if lit_discovered:
        lit_df = pd.DataFrame(lit_discovered)
        lit_df = _sort_table_by_date(lit_df, date_col="date")
        show_cols = [c for c in ["date", "release_no", "title", "ingest_status", "url"] if c in lit_df.columns]
        st.dataframe(
            lit_df[show_cols],
            use_container_width=True,
            hide_index=True,
        )

        lit_filter = st.selectbox(
            "Litigation Ingest Selection",
            ["New Only", "All Discovered"],
            key="sec_lit_ingest_filter",
        )
        if lit_filter == "New Only":
            lit_candidates = [d for d in lit_discovered if d.get("ingest_status") == "new"]
        else:
            lit_candidates = list(lit_discovered)

        lit_count = len(lit_candidates)
        if lit_count <= 0:
            lit_limit = 0
            st.caption("No litigation releases match the selected ingest filter.")
        elif lit_count == 1:
            lit_limit = 1
            st.caption("1 litigation release selected for ingest.")
        else:
            lit_limit = st.slider(
                "Litigation Releases To Ingest",
                min_value=1,
                max_value=lit_count,
                value=min(10, lit_count),
                key="sec_lit_ingest_limit",
            )
        st.caption(f"{lit_count} litigation releases currently match this ingest selection.")

        if st.button("Run Litigation Release Extraction", disabled=(lit_limit <= 0), key="ingest_sec_lit"):
            try:
                from sec_enforcement_litigation_scraper import SECEnforcementLitigationScraper

                lit_scraper = SECEnforcementLitigationScraper()
                progress = st.progress(0, text="Starting litigation ingest...")
                saved_new = 0
                saved_updates = 0
                failed = []

                selected = lit_candidates[:lit_limit]
                for idx, entry in enumerate(selected, 1):
                    progress.progress(
                        idx / lit_limit,
                        text=f"Ingesting {idx}/{lit_limit}: {entry.get('title', '')[:80]}",
                    )
                    try:
                        extracted = lit_scraper.extract_document(
                            entry.get("url", ""),
                            fallback_title=entry.get("title", ""),
                            fallback_date=entry.get("date", ""),
                            fallback_release_no=entry.get("release_no", ""),
                        )
                        if not extracted.get("success"):
                            raise RuntimeError("Extraction returned unsuccessful result.")
                        data = extracted.get("data", {})
                        text = str(data.get("full_text", "") or "").strip()
                        if len(text.split()) < 80:
                            raise RuntimeError("Extracted text appears too short; skipping.")

                        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
                        source_name = str(data.get("release_no", "") or entry.get("release_no", "")).strip()
                        if not source_name:
                            source_name = urlparse(src_url).path.rsplit("/", 1)[-1].strip() or f"litigation-release-{idx}"
                        source_name = f"{source_name}.html" if "." not in source_name else source_name

                        date_text = str(data.get("date", "") or entry.get("date", "")).strip()
                        parsed_date = _parse_single_date(date_text)
                        if pd.notna(parsed_date):
                            doc_date_value = parsed_date.date()
                        else:
                            doc_date_value = date_text

                        record = _create_uploaded_document_record(
                            text=text,
                            organization="SEC",
                            title=str(data.get("title", "") or entry.get("title", "")).strip(),
                            speaker="SEC Division of Enforcement",
                            doc_date=doc_date_value,
                            doc_type="Litigation Release",
                            source_url=src_url,
                            source_filename=source_name,
                            source_ext=".html",
                            source_local_path="",
                            source_gcs_path="",
                            tags_csv="sec,enforcement,litigation-release",
                            source_kind="sec_enforcement_litigation",
                        )
                        rm = record.setdefault("metadata", {})
                        rm["source_family"] = "sec_enforcement_litigation"
                        rm["source_index_url"] = lit_index_url
                        rm["release_no"] = str(data.get("release_no", "") or entry.get("release_no", "")).strip()
                        rm["published_date"] = str(entry.get("date", "") or "")

                        replaced = _upsert_custom_document_record(custom_payload, record)
                        if replaced:
                            saved_updates += 1
                        else:
                            saved_new += 1

                    except Exception as e:
                        failed.append(f"{entry.get('title', 'Untitled')}: {e}")

                progress.progress(1.0, text="Litigation ingest complete.")
                if saved_new or saved_updates:
                    _save_custom_documents(custom_payload)
                    st.success(
                        f"Saved {saved_new} new litigation docs and updated {saved_updates} existing litigation docs."
                    )
                    custom_payload = _load_custom_documents()
                    custom_docs = _custom_docs_as_speeches(custom_payload)
                if failed:
                    st.warning(f"{len(failed)} litigation releases failed ingest.")
                    for msg in failed[:20]:
                        st.write(f"- {msg}")
            except Exception as e:
                st.error(f"Litigation ingest failed: {e}")

    st.markdown("---")
    st.subheader("DOJ Connector: USAO Press Releases")
    st.caption(
        "Discover and ingest Department of Justice U.S. Attorneys' Office press releases into the knowledge base."
    )

    doj_index_default = "https://www.justice.gov/usao/pressreleases"
    doj_index_url = st.text_input(
        "DOJ USAO Press Releases URL",
        value=doj_index_default,
        key="doj_usao_index_url",
    ).strip() or doj_index_default
    doj_pages = st.slider(
        "DOJ Listing Pages To Scan",
        min_value=1,
        max_value=20,
        value=3,
        key="doj_usao_pages",
    )

    doj_col1, doj_col2 = st.columns(2)
    with doj_col1:
        discover_doj = st.button("Discover DOJ Press Releases", key="discover_doj_usao")
    with doj_col2:
        clear_doj = st.button("Clear DOJ Results", key="clear_doj_usao")

    doj_state_key = "doj_usao_discovered"
    doj_debug_key = "doj_usao_discovery_debug"
    if doj_state_key not in st.session_state:
        st.session_state[doj_state_key] = []
    if doj_debug_key not in st.session_state:
        st.session_state[doj_debug_key] = {}
    if clear_doj:
        st.session_state[doj_state_key] = []
        st.session_state[doj_debug_key] = {}

    if discover_doj:
        try:
            from doj_usao_press_release_scraper import DOJUSAOPressReleaseScraper

            with st.spinner("Discovering DOJ USAO press releases..."):
                doj_scraper = DOJUSAOPressReleaseScraper()
                doj_discovered = doj_scraper.discover_documents(
                    base_url=doj_index_url,
                    max_pages=doj_pages,
                )
                debug_payload = getattr(doj_scraper, "last_discovery_debug", {})
                if isinstance(debug_payload, dict):
                    st.session_state[doj_debug_key] = debug_payload

            existing_custom = {}
            for item in custom_docs:
                m = item.get("metadata", {})
                existing_custom[_url_match_key(m.get("url", ""))] = m

            existing_speech_urls = {
                _url_match_key(s.get("metadata", {}).get("url", ""))
                for s in raw_data.get("speeches", [])
            }

            for entry in doj_discovered:
                key = _url_match_key(entry.get("url", ""))
                status = "new"
                existing_meta = existing_custom.get(key)
                if existing_meta:
                    existing_date = str(existing_meta.get("published_date") or existing_meta.get("date") or "").strip()
                    incoming_date = str(entry.get("date", "") or "").strip()
                    if incoming_date and existing_date and incoming_date != existing_date:
                        status = "update_available"
                    else:
                        status = "existing"
                elif key in existing_speech_urls:
                    status = "existing_in_speeches"
                entry["ingest_status"] = status

            st.session_state[doj_state_key] = doj_discovered
            new_count = sum(1 for d in doj_discovered if d.get("ingest_status") in {"new", "update_available"})
            st.success(
                f"Discovered {len(doj_discovered)} DOJ USAO press releases "
                f"({new_count} new/update candidates)."
            )
            try:
                dbg = st.session_state.get(doj_debug_key, {})
                if isinstance(dbg, dict):
                    dbg["ingest_status_counts"] = {
                        "new": sum(1 for d in doj_discovered if d.get("ingest_status") == "new"),
                        "update_available": sum(1 for d in doj_discovered if d.get("ingest_status") == "update_available"),
                        "existing": sum(1 for d in doj_discovered if d.get("ingest_status") == "existing"),
                        "existing_in_speeches": sum(1 for d in doj_discovered if d.get("ingest_status") == "existing_in_speeches"),
                    }
                    st.session_state[doj_debug_key] = dbg
            except Exception:
                pass
        except Exception as e:
            st.session_state[doj_debug_key] = {"error": str(e)}
            st.error(f"DOJ press-release discovery failed: {e}")

    doj_discovered = st.session_state.get(doj_state_key, [])
    doj_debug = st.session_state.get(doj_debug_key, {})
    if isinstance(doj_debug, dict) and doj_debug:
        with st.expander("DOJ Discovery Debug", expanded=False):
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Requested Pages", int(doj_debug.get("max_pages_requested", 0) or 0))
            c2.metric("Pages Logged", len(doj_debug.get("pages", []) if isinstance(doj_debug.get("pages", []), list) else []))
            c3.metric("Listing Added", int(doj_debug.get("listing_added", 0) or 0))
            c4.metric("RSS Added", int(doj_debug.get("rss_added", 0) or 0))
            c5.metric("News Added", int(doj_debug.get("news_added", 0) or 0))
            c6.metric("Sitemap Added", int(doj_debug.get("sitemap_added", 0) or 0))
            st.caption(
                f"Stop reason: `{doj_debug.get('stop_reason', '')}` | "
                f"Pagination blocked: `{doj_debug.get('pagination_blocked', False)}` | "
                f"RSS page0 used: `{doj_debug.get('rss_page0_used', False)}` | "
                f"RSS supplement used: `{doj_debug.get('rss_supplement_used', False)}` | "
                f"News supplement used: `{doj_debug.get('news_supplement_used', False)}` | "
                f"Sitemap supplement used: `{doj_debug.get('sitemap_supplement_used', False)}`"
            )
            page_logs = doj_debug.get("pages", [])
            if isinstance(page_logs, list) and page_logs:
                page_df = pd.DataFrame(page_logs)
                show_cols = [
                    c
                    for c in [
                        "page",
                        "attempts",
                        "returned_items",
                        "unique_added",
                        "error_status",
                        "error_type",
                        "error_message",
                        "page_url",
                    ]
                    if c in page_df.columns
                ]
                st.dataframe(page_df[show_cols], use_container_width=True, hide_index=True)
            st.json(doj_debug)

    if doj_discovered:
        doj_df = pd.DataFrame(doj_discovered)
        doj_df = _sort_table_by_date(doj_df, date_col="date")
        show_cols = [c for c in ["date", "office", "title", "ingest_status", "url"] if c in doj_df.columns]
        st.dataframe(
            doj_df[show_cols],
            use_container_width=True,
            hide_index=True,
        )

        doj_filter = st.selectbox(
            "DOJ Ingest Selection",
            ["New/Updates Only", "All Discovered"],
            key="doj_usao_ingest_filter",
        )
        if doj_filter == "New/Updates Only":
            doj_candidates = [d for d in doj_discovered if d.get("ingest_status") in {"new", "update_available"}]
        else:
            doj_candidates = list(doj_discovered)

        doj_count = len(doj_candidates)
        if doj_count <= 0:
            doj_limit = 0
            st.caption("No DOJ press releases match the selected ingest filter.")
        elif doj_count == 1:
            doj_limit = 1
            st.caption("1 DOJ press release selected for ingest.")
        else:
            doj_limit = st.slider(
                "DOJ Press Releases To Ingest",
                min_value=1,
                max_value=doj_count,
                value=min(10, doj_count),
                key="doj_usao_ingest_limit",
            )
        st.caption(f"{doj_count} DOJ press releases currently match this ingest selection.")

        if st.button("Run DOJ Press Release Extraction", disabled=(doj_limit <= 0), key="ingest_doj_usao"):
            try:
                from doj_usao_press_release_scraper import DOJUSAOPressReleaseScraper

                doj_scraper = DOJUSAOPressReleaseScraper()
                progress = st.progress(0, text="Starting DOJ press-release ingest...")
                saved_new = 0
                saved_updates = 0
                failed = []

                selected = doj_candidates[:doj_limit]
                for idx, entry in enumerate(selected, 1):
                    progress.progress(
                        idx / doj_limit,
                        text=f"Ingesting {idx}/{doj_limit}: {entry.get('title', '')[:80]}",
                    )
                    try:
                        extracted = doj_scraper.extract_document(
                            entry.get("url", ""),
                            fallback_title=entry.get("title", ""),
                            fallback_date=entry.get("date", ""),
                            fallback_office=entry.get("office", ""),
                        )
                        if not extracted.get("success"):
                            raise RuntimeError("Extraction returned unsuccessful result.")

                        data = extracted.get("data", {})
                        text = str(data.get("full_text", "") or "").strip()
                        if len(text.split()) < 80:
                            raise RuntimeError("Extracted text appears too short; skipping.")

                        src_url = str(data.get("url", "") or entry.get("url", "")).strip()
                        source_name = urlparse(src_url).path.rsplit("/", 1)[-1].strip() or f"doj-press-release-{idx}"
                        source_name = f"{source_name}.html" if "." not in source_name else source_name

                        date_text = str(data.get("date", "") or entry.get("date", "")).strip()
                        parsed_date = _parse_single_date(date_text)
                        if pd.notna(parsed_date):
                            doc_date_value = parsed_date.date()
                        else:
                            doc_date_value = date_text

                        office_text = str(data.get("office", "") or entry.get("office", "")).strip()
                        if not office_text:
                            office_text = "U.S. Attorney's Office"

                        record = _create_uploaded_document_record(
                            text=text,
                            organization="DOJ",
                            title=str(data.get("title", "") or entry.get("title", "")).strip(),
                            speaker=office_text,
                            doc_date=doc_date_value,
                            doc_type="Press Release",
                            source_url=src_url,
                            source_filename=source_name,
                            source_ext=".html",
                            source_local_path="",
                            source_gcs_path="",
                            tags_csv="doj,usao,press-release",
                            source_kind="doj_usao_press_release",
                        )
                        rm = record.setdefault("metadata", {})
                        rm["source_family"] = "doj_usao_press_release"
                        rm["source_index_url"] = doj_index_url
                        rm["office"] = office_text
                        rm["published_date"] = str(entry.get("date", "") or "")
                        rm["updated_date"] = str(data.get("updated_date", "") or "")

                        replaced = _upsert_custom_document_record(custom_payload, record)
                        if replaced:
                            saved_updates += 1
                        else:
                            saved_new += 1

                    except Exception as e:
                        failed.append(f"{entry.get('title', 'Untitled')}: {e}")

                progress.progress(1.0, text="DOJ ingest complete.")
                if saved_new or saved_updates:
                    _save_custom_documents(custom_payload)
                    st.success(
                        f"Saved {saved_new} new DOJ docs and updated {saved_updates} existing DOJ docs."
                    )
                    custom_payload = _load_custom_documents()
                    custom_docs = _custom_docs_as_speeches(custom_payload)
                if failed:
                    st.warning(f"{len(failed)} DOJ press releases failed ingest.")
                    for msg in failed[:20]:
                        st.write(f"- {msg}")
            except Exception as e:
                st.error(f"DOJ ingest failed: {e}")

    if custom_docs:
        st.markdown("---")
        st.subheader("Uploaded Documents")
        rows = []
        for item in custom_docs:
            m = item.get("metadata", {})
            rows.append(
                {
                    "date": m.get("date", ""),
                    "title": m.get("title", ""),
                    "organization": m.get("organization", ""),
                    "speaker": m.get("speaker", ""),
                    "type": m.get("doc_type", ""),
                    "words": m.get("word_count", 0),
                    "source_file": m.get("source_filename", ""),
                    "url": m.get("url", ""),
                }
            )
        docs_df = pd.DataFrame(rows)
        docs_df = _sort_table_by_date(docs_df, date_col="date")
        st.dataframe(
            docs_df[["date", "title", "organization", "speaker", "type", "words", "source_file"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No uploaded documents yet. Add a PDF/text/HTML file or paste transcript text above.")


# =====================================================
# PAGE: Enrichment Pipeline
# =====================================================
elif page == "Enrichment Pipeline":
    st.title("Enrichment Pipeline")
    st.markdown(
        "Run the enrichment agent to assign tags, keywords, entities, stance, evidence, and reward scores."
    )

    enrichment_state = _load_enrichment_state()

    def _scoped_entries_from_state(state_obj):
        scoped = []
        for entry in state_obj.get("entries", {}).values():
            if not isinstance(entry, dict):
                continue
            if scope_key != "__all__" and str(entry.get("org_key", "") or "") != scope_key:
                continue
            scoped.append(entry)
        return scoped

    org_options = _list_org_options(knowledge_data)
    org_scope_labels = ["All Organizations"] + [o["label"] for o in org_options]
    selected_scope = st.selectbox("Organization Scope", org_scope_labels, index=0)
    scope_key = "__all__"
    scope_label = "All Organizations"
    if selected_scope != "All Organizations":
        selected_org = next((o for o in org_options if o["label"] == selected_scope), org_options[0])
        scope_key = selected_org["key"]
        scope_label = selected_org["label"]

    scoped_candidates = _build_enrichment_candidates(
        knowledge_data,
        org_key=None if scope_key == "__all__" else scope_key,
    )
    candidate_map = {str(d.get("doc_id", "") or ""): d for d in scoped_candidates}

    scoped_entries = _scoped_entries_from_state(enrichment_state)

    total_scope_docs = len(scoped_candidates)
    enriched_count = sum(1 for e in scoped_entries if str(e.get("status", "")) in {"enriched", "reviewed", "fallback_enriched"})
    failed_count = sum(1 for e in scoped_entries if str(e.get("status", "")) == "failed")
    pending_review = sum(
        1
        for e in scoped_entries
        if str(e.get("review", {}).get("decision", "pending") or "pending") == "pending"
        and str(e.get("status", "")) in {"enriched", "fallback_enriched", "reviewed"}
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Docs In Scope", total_scope_docs)
    m2.metric("Enriched", enriched_count)
    m3.metric("Pending Review", pending_review)
    m4.metric("Failed", failed_count)

    mode_label = st.selectbox(
        "Run Mode",
        [
            "Only Missing/Failed",
            "Only Pending Review",
            "Re-enrich All In Scope",
        ],
    )
    mode_map = {
        "Only Missing/Failed": "only_missing_or_failed",
        "Only Pending Review": "only_pending_review",
        "Re-enrich All In Scope": "re_enrich_all",
    }
    mode_key = mode_map[mode_label]

    targets = _select_enrichment_targets(scoped_candidates, enrichment_state, mode_key)
    target_count = len(targets)
    if target_count <= 0:
        batch_limit = 0
        st.caption("No documents match the selected run mode.")
    elif target_count == 1:
        batch_limit = 1
    else:
        batch_limit = st.slider(
            "Docs To Process This Run",
            min_value=1,
            max_value=target_count,
            value=min(25, target_count),
        )
    st.caption(f"{target_count} documents currently match this run mode.")

    available_models = []
    client = None
    if _openai_key is not None:
        client = _get_openai_client()
        if client is not None:
            available_models = _get_accessible_chat_models(client)
    if not available_models:
        available_models = _candidate_chat_models()
    if not available_models:
        available_models = ["gpt-4o-mini"]

    default_enrich_model = "gpt-4o-mini" if "gpt-4o-mini" in available_models else available_models[0]
    enrich_model = st.selectbox(
        "Enrichment Model",
        available_models,
        index=available_models.index(default_enrich_model),
    )

    if st.button("Run Enrichment Batch", type="primary", disabled=(batch_limit <= 0)):
        if client is None:
            st.error("OpenAI client is not configured. Check API key and model access.")
        else:
            run_progress = st.progress(0)
            run_status = st.empty()

            def _run_progress_cb(done, total, message):
                if total <= 0:
                    run_progress.progress(100)
                    run_status.caption(message)
                    return
                safe_done = max(0, min(done, total))
                pct = int((safe_done * 100) / total)
                run_progress.progress(max(0, min(pct, 100)))
                run_status.caption(f"{message} ({safe_done}/{total})")

            with st.spinner(f"Running enrichment over {batch_limit} docs in {scope_label}..."):
                result = _run_enrichment_batch(
                    client=client,
                    candidates=scoped_candidates,
                    enrichment_state=enrichment_state,
                    model_name=enrich_model,
                    mode=mode_key,
                    limit=batch_limit,
                    progress_callback=_run_progress_cb,
                )
            st.success(
                f"Enrichment run complete. Processed {result.get('processed', 0)} "
                f"of {result.get('total_selected', 0)} matching docs."
            )
            enrichment_state = _load_enrichment_state()
            scoped_entries = _scoped_entries_from_state(enrichment_state)

    st.markdown("---")
    st.subheader("Targeted Re-Enrichment")
    if scoped_candidates:
        def _candidate_sort_key(doc_obj):
            parsed = _parse_single_date(doc_obj.get("date", ""))
            if pd.notna(parsed):
                return parsed
            return pd.Timestamp.min

        target_candidates = sorted(scoped_candidates, key=_candidate_sort_key, reverse=True)
        target_ids = [str(d.get("doc_id", "") or "") for d in target_candidates if str(d.get("doc_id", "") or "").strip()]
        if not target_ids:
            st.caption("No valid document IDs found in scope.")
        else:
            target_doc_id = st.selectbox(
                "Select Document",
                target_ids,
                format_func=lambda d: (
                    f"{candidate_map.get(d, {}).get('date', '')} | "
                    f"{candidate_map.get(d, {}).get('title', d)}"
                ),
                key="target_reenrich_doc_id",
            )
            target_doc = candidate_map.get(target_doc_id, {})
            current_entry = enrichment_state.get("entries", {}).get(target_doc_id, {})
            if target_doc:
                current_status = str(current_entry.get("status", "not_enriched") or "not_enriched")
                current_review = str(current_entry.get("review", {}).get("decision", "pending") or "pending")
                st.caption(f"Current status: {current_status} | review: {current_review}")

            if st.button("Re-Enrich Selected Document", key="reenrich_selected_doc"):
                if client is None:
                    st.error("OpenAI client is not configured. Check API key and model access.")
                elif not target_doc:
                    st.error("Select a document first.")
                else:
                    re_progress = st.progress(0)
                    re_status = st.empty()

                    def _re_progress_cb(done, total, message):
                        if total <= 0:
                            re_progress.progress(100)
                            re_status.caption(message)
                            return
                        safe_done = max(0, min(done, total))
                        pct = int((safe_done * 100) / total)
                        re_progress.progress(max(0, min(pct, 100)))
                        re_status.caption(f"{message} ({safe_done}/{total})")

                    with st.spinner("Re-enriching selected document..."):
                        _run_enrichment_batch(
                            client=client,
                            candidates=[target_doc],
                            enrichment_state=enrichment_state,
                            model_name=enrich_model,
                            mode="re_enrich_all",
                            limit=1,
                            progress_callback=_re_progress_cb,
                        )
                    st.success("Selected document re-enriched.")
                    enrichment_state = _load_enrichment_state()
                    scoped_entries = _scoped_entries_from_state(enrichment_state)
    else:
        st.caption("No documents in scope for targeted re-enrichment.")

    st.markdown("---")
    st.subheader("Auto Review Agent")
    st.caption("Automatically mark enriched docs as approved or flagged for human follow-up.")

    auto_mode_label = st.selectbox(
        "Auto Review Mode",
        [
            "Only Pending Review",
            "All Enriched In Scope",
        ],
    )
    auto_mode_map = {
        "Only Pending Review": "only_pending",
        "All Enriched In Scope": "all_eligible",
    }
    auto_mode_key = auto_mode_map[auto_mode_label]
    auto_targets = _select_auto_review_targets(scoped_entries, auto_mode_key)
    auto_target_count = len(auto_targets)
    if auto_target_count <= 0:
        auto_limit = 0
        st.caption("No documents match the selected auto-review mode.")
    elif auto_target_count == 1:
        auto_limit = 1
    else:
        auto_limit = st.slider(
            "Docs To Auto Review This Run",
            min_value=1,
            max_value=auto_target_count,
            value=min(50, auto_target_count),
        )
    st.caption(f"{auto_target_count} documents currently match auto-review mode.")

    default_auto_model = enrich_model if enrich_model in available_models else available_models[0]
    auto_review_model = st.selectbox(
        "Auto Review Model",
        available_models,
        index=available_models.index(default_auto_model),
        key="auto_review_model",
    )

    if st.button("Run Auto Review Agent", disabled=(auto_limit <= 0)):
        if client is None:
            st.error("OpenAI client is not configured. Check API key and model access.")
        else:
            review_progress = st.progress(0)
            review_status = st.empty()

            def _review_progress_cb(done, total, message):
                if total <= 0:
                    review_progress.progress(100)
                    review_status.caption(message)
                    return
                safe_done = max(0, min(done, total))
                pct = int((safe_done * 100) / total)
                review_progress.progress(max(0, min(pct, 100)))
                review_status.caption(f"{message} ({safe_done}/{total})")

            with st.spinner(f"Running auto review over {auto_limit} docs in {scope_label}..."):
                review_result = _run_auto_review_batch(
                    client=client,
                    scoped_entries=scoped_entries,
                    candidate_map=candidate_map,
                    enrichment_state=enrichment_state,
                    model_name=auto_review_model,
                    mode=auto_mode_key,
                    limit=auto_limit,
                    progress_callback=_review_progress_cb,
                )
            st.success(
                f"Auto review complete. Processed {review_result.get('processed', 0)} "
                f"of {review_result.get('total_selected', 0)} docs; "
                f"approved={review_result.get('approved', 0)}, "
                f"flagged={review_result.get('flagged', 0)}, "
                f"heuristic_fallbacks={review_result.get('heuristic_fallbacks', 0)}."
            )
            enrichment_state = _load_enrichment_state()
            scoped_entries = _scoped_entries_from_state(enrichment_state)

    eligible_for_bulk_accept = sum(
        1
        for e in scoped_entries
        if str(e.get("status", "") or "").strip().lower() in {"enriched", "fallback_enriched", "reviewed"}
    )
    st.caption(f"{eligible_for_bulk_accept} docs in scope are eligible for bulk acceptance.")
    if st.button(
        "Auto Accept All In Scope",
        disabled=(eligible_for_bulk_accept <= 0),
        key="bulk_accept_all_scope",
    ):
        accepted_count = _bulk_accept_reviews(
            enrichment_state,
            scoped_entries,
            only_pending=False,
        )
        st.success(f"Auto-accepted {accepted_count} documents in {scope_label}.")
        enrichment_state = _load_enrichment_state()
        scoped_entries = _scoped_entries_from_state(enrichment_state)

    st.markdown("---")
    st.subheader("Enrichment Results")
    rows = []
    for entry in scoped_entries:
        enrich = entry.get("enrichment", {})
        review = entry.get("review", {})
        reward = entry.get("reward", {})
        auto_review = entry.get("auto_review", {})
        rows.append(
            {
                "doc_id": entry.get("doc_id", ""),
                "date": entry.get("date", ""),
                "title": entry.get("title", ""),
                "organization": entry.get("organization", ""),
                "status": entry.get("status", ""),
                "review": review.get("decision", "pending"),
                "auto_verdict": auto_review.get("verdict", ""),
                "auto_conf": auto_review.get("confidence", ""),
                "reward": reward.get("score", 0.0),
                "tags": ", ".join(enrich.get("tags", [])[:6]),
                "keywords": ", ".join(enrich.get("keywords", [])[:8]),
                "error": str(entry.get("error", "") or ""),
                "updated_at": entry.get("updated_at", ""),
            }
        )

    if rows:
        result_df = pd.DataFrame(rows)
        if "date" in result_df.columns:
            result_df = _sort_table_by_date(result_df, date_col="date")
        st.dataframe(
            result_df[["date", "title", "organization", "status", "review", "auto_verdict", "auto_conf", "reward", "tags", "keywords"]],
            use_container_width=True,
            hide_index=True,
        )

        id_to_entry = {str(e.get("doc_id", "")): e for e in scoped_entries}
        review_ids = [r["doc_id"] for r in rows if r["doc_id"] in id_to_entry]
        if review_ids:
            selected_doc_id = st.selectbox("Review Document", review_ids, format_func=lambda d: id_to_entry[d].get("title", d))
            selected_entry = id_to_entry.get(selected_doc_id, {})
            selected_enrichment = selected_entry.get("enrichment", {})
            selected_review = selected_entry.get("review", {})
            selected_reward = selected_entry.get("reward", {})
            selected_auto_review = selected_entry.get("auto_review", {})

            st.markdown(f"**Title:** {selected_entry.get('title', '')}")
            st.markdown(
                f"**Status:** {selected_entry.get('status', '')} | "
                f"**Review:** {selected_review.get('decision', 'pending')} | "
                f"**Reward:** {selected_reward.get('score', 0.0)}"
            )
            selected_error = str(selected_entry.get("error", "") or "").strip()
            if selected_error:
                with st.expander("Stored LLM/Fallback Error", expanded=False):
                    st.code(selected_error)
            if selected_auto_review:
                try:
                    auto_conf = float(selected_auto_review.get("confidence", 0.0) or 0.0)
                except Exception:
                    auto_conf = 0.0
                st.markdown(
                    f"**Auto Review:** {selected_auto_review.get('verdict', '')} | "
                    f"**Confidence:** {auto_conf:.2f} | "
                    f"**Engine:** {selected_auto_review.get('engine', '')}"
                )
                auto_rationale = str(selected_auto_review.get("rationale", "") or "").strip()
                if auto_rationale:
                    st.caption(auto_rationale)
                auto_issues = selected_auto_review.get("issues", [])
                if isinstance(auto_issues, list) and auto_issues:
                    with st.expander("Auto Review Issues"):
                        for issue in auto_issues:
                            st.markdown(f"- {issue}")
            st.markdown(f"**Summary:** {selected_enrichment.get('summary', '')}")
            st.markdown(f"**Tags:** {', '.join(selected_enrichment.get('tags', []))}")
            st.markdown(f"**Keywords:** {', '.join(selected_enrichment.get('keywords', []))}")
            stance = selected_enrichment.get("stance", {})
            st.markdown(
                f"**Stance:** {stance.get('label', 'unclear')} "
                f"{'(' + stance.get('target', '') + ')' if stance.get('target') else ''}"
            )

            evidence_rows = selected_enrichment.get("evidence_spans", [])
            if evidence_rows:
                with st.expander("Evidence Spans"):
                    for ev in evidence_rows:
                        st.markdown(f"- **{ev.get('claim', '')}**")
                        st.caption(ev.get("snippet", ""))

            review_notes_key = f"review_notes_{selected_doc_id}"
            current_notes = str(selected_review.get("notes", "") or "")
            review_notes = st.text_area(
                "Review Notes",
                value=current_notes,
                key=review_notes_key,
            )
            r1, r2, r3, r4 = st.columns(4)
            if r1.button("Mark Accepted", key=f"accept_{selected_doc_id}"):
                if _update_review_decision(enrichment_state, selected_doc_id, "accepted", review_notes):
                    st.success("Marked accepted.")
            if r2.button("Mark Edited", key=f"edited_{selected_doc_id}"):
                if _update_review_decision(enrichment_state, selected_doc_id, "edited", review_notes):
                    st.success("Marked edited.")
            if r3.button("Mark Rejected", key=f"reject_{selected_doc_id}"):
                if _update_review_decision(enrichment_state, selected_doc_id, "rejected", review_notes):
                    st.success("Marked rejected.")
            if r4.button("Reset Pending", key=f"pending_{selected_doc_id}"):
                if _update_review_decision(enrichment_state, selected_doc_id, "pending", review_notes):
                    st.success("Reset to pending.")
    else:
        st.info("No enrichment entries yet in this scope. Run a batch to generate them.")


# =====================================================
# PAGE: Policy Delta Briefings
# =====================================================
elif page == "Policy Delta Briefings":
    st.title("Policy Delta Briefings")
    st.markdown(
        "Generate a structured briefing that explains how a new document compares to prior corpus positions "
        "(continuity, expansion, narrowing, shift, contradiction, or novel)."
    )
    st.caption(
        f"Stored locally at `{POLICY_BRIEFS_LOCAL_PATH}` and in GCS blob `{POLICY_BRIEFS_BLOB_NAME}` "
        "when GCS is configured."
    )

    enrichment_state = _load_enrichment_state()
    briefs_payload = _load_policy_briefs()

    org_options = _list_org_options(knowledge_data)
    org_label_to_key = {o["label"]: o["key"] for o in org_options}
    org_labels = [o["label"] for o in org_options]
    default_source_orgs = ["SEC"] if "SEC" in org_labels else (org_labels[:1] if org_labels else [])
    selected_source_org_labels = st.multiselect(
        "Source Organizations",
        org_labels,
        default=default_source_orgs,
        help="Choose which organizations are eligible as the new/target document.",
    )
    source_org_keys = [org_label_to_key[l] for l in selected_source_org_labels if l in org_label_to_key]
    default_compare_orgs = selected_source_org_labels or org_labels
    selected_compare_org_labels = st.multiselect(
        "Comparison Organizations",
        org_labels,
        default=default_compare_orgs,
        help="Choose which organizations are searched for prior-position comparison.",
    )
    compare_org_keys = [org_label_to_key[l] for l in selected_compare_org_labels if l in org_label_to_key]

    selected_source_kinds = []
    selected_compare_source_kinds = []

    scoped_docs = _build_policy_doc_rows(knowledge_data, enrichment_state, org_keys=source_org_keys or None)
    scoped_docs = sorted(
        scoped_docs,
        key=lambda d: d.get("date_parsed") if pd.notna(d.get("date_parsed")) else pd.Timestamp.min,
        reverse=True,
    )

    if not scoped_docs:
        st.info("No documents found in this scope.")
    else:
        source_kinds = sorted({str(d.get("source_kind", "") or "document") for d in scoped_docs})
        selected_source_kinds = st.multiselect(
            "Source Document Corpus Types",
            source_kinds,
            default=source_kinds,
        )
        filtered_docs = [
            d for d in scoped_docs
            if (not selected_source_kinds or str(d.get("source_kind", "") or "document") in selected_source_kinds)
        ]
        compare_scope_docs = _build_policy_doc_rows(
            knowledge_data,
            enrichment_state,
            org_keys=compare_org_keys or None,
        )
        compare_source_kinds = sorted(
            {str(d.get("source_kind", "") or "document") for d in compare_scope_docs}
        )
        selected_compare_source_kinds = st.multiselect(
            "Comparison Corpus Types",
            compare_source_kinds,
            default=compare_source_kinds,
        )
        if not filtered_docs:
            st.warning("No documents match the selected source-kind filters.")
        else:
            st.caption(f"{len(filtered_docs)} documents available for policy-delta briefing.")
            source_doc_ids = [d["doc_id"] for d in filtered_docs]
            doc_map = {d["doc_id"]: d for d in filtered_docs}
            selected_source_doc_id = st.selectbox(
                "New Document To Brief",
                source_doc_ids,
                format_func=lambda d: (
                    f"{doc_map.get(d, {}).get('date', '')} | "
                    f"{doc_map.get(d, {}).get('title', '')} "
                    f"[{doc_map.get(d, {}).get('source_kind', '')}]"
                ),
            )
            selected_doc = doc_map.get(selected_source_doc_id, {})

            c1, c2, c3 = st.columns(3)
            with c1:
                lookback_days = st.slider("Prior Window (days)", min_value=30, max_value=3650, value=730, step=30)
            with c2:
                max_candidates = st.slider("Prior Docs To Compare", min_value=5, max_value=40, value=20, step=1)
            with c3:
                min_words = st.slider("Min Words In New Doc", min_value=0, max_value=2000, value=120, step=20)

            if _coerce_int(selected_doc.get("word_count", 0), default=0, min_value=0) < min_words:
                st.warning(
                    f"Selected document has {_coerce_int(selected_doc.get('word_count', 0))} words, below your min threshold ({min_words})."
                )

            available_models = []
            client = None
            if _openai_key is not None:
                client = _get_openai_client()
                if client is not None:
                    available_models = _get_accessible_chat_models(client)
            if not available_models:
                available_models = _candidate_chat_models() or ["gpt-4o-mini"]
            default_model = "gpt-4o-mini" if "gpt-4o-mini" in available_models else available_models[0]
            briefing_model = st.selectbox(
                "Briefing Model",
                available_models,
                index=available_models.index(default_model),
            )

            preview_source, preview_prior_docs = _select_prior_docs_for_policy_delta(
                knowledge_data=knowledge_data,
                enrichment_state=enrichment_state,
                source_doc_id=selected_source_doc_id,
                source_org_keys=source_org_keys,
                compare_org_keys=compare_org_keys,
                compare_source_kinds=selected_compare_source_kinds,
                lookback_days=lookback_days,
                max_candidates=max_candidates,
            )
            if preview_source:
                st.markdown(
                    f"**Selected Source:** `{preview_source.get('title', '')}` | "
                    f"{preview_source.get('date', '')} | `{preview_source.get('source_kind', '')}`"
                )
            if preview_prior_docs:
                preview_rows = []
                for item in preview_prior_docs:
                    preview_rows.append(
                        {
                            "date": item.get("date", ""),
                            "title": item.get("title", ""),
                            "source_kind": item.get("source_kind", ""),
                            "doc_type": item.get("doc_type", ""),
                            "similarity": round(_coerce_float(item.get("similarity_score", 0.0), default=0.0), 3),
                        }
                    )
                st.caption("Comparison set preview")
                st.dataframe(
                    _sort_table_by_date(pd.DataFrame(preview_rows), date_col="date"),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No prior comparison docs found in the current scope and window.")

            if st.button("Generate/Refresh Policy Delta Brief", type="primary"):
                if not selected_doc:
                    st.error("Select a source document.")
                elif _coerce_int(selected_doc.get("word_count", 0), default=0, min_value=0) < min_words:
                    st.error("Selected document is below min word threshold. Lower threshold or choose another document.")
                else:
                    with st.spinner("Generating policy delta briefing..."):
                        record = _generate_policy_delta_brief(
                            client=client,
                            knowledge_data=knowledge_data,
                            enrichment_state=enrichment_state,
                            source_doc_id=selected_source_doc_id,
                            org_key=str(selected_doc.get("org_key", "") or ""),
                            source_org_keys=source_org_keys,
                            compare_org_keys=compare_org_keys,
                            compare_source_kinds=selected_compare_source_kinds,
                            model_name=briefing_model,
                            lookback_days=lookback_days,
                            max_candidates=max_candidates,
                        )
                    replaced = _upsert_policy_delta_brief(briefs_payload, record)
                    _save_policy_briefs(briefs_payload)
                    action = "Updated" if replaced else "Saved"
                    st.success(
                        f"{action} policy delta brief for `{record.get('source_doc', {}).get('title', '')}` "
                        f"(engine: {record.get('engine', '')})."
                    )
                    if str(record.get("error", "") or "").strip():
                        st.warning(f"LLM error captured, used fallback: {record.get('error', '')}")
                    briefs_payload = _load_policy_briefs()

    st.markdown("---")
    st.subheader("Saved Briefings")
    saved_rows = []
    brief_map = {}
    for brief in briefs_payload.get("briefs", []):
        if not isinstance(brief, dict):
            continue
        brief_org_key = str(brief.get("org_key", "") or "").strip()
        if source_org_keys and brief_org_key not in set(source_org_keys):
            continue
        source_doc = brief.get("source_doc", {}) if isinstance(brief.get("source_doc", {}), dict) else {}
        if selected_source_kinds and str(source_doc.get("source_kind", "") or "document") not in selected_source_kinds:
            continue
        brief_obj = brief.get("brief", {}) if isinstance(brief.get("brief", {}), dict) else {}
        brief_id = str(brief.get("brief_id", "") or "")
        if not brief_id:
            continue
        brief_map[brief_id] = brief
        saved_rows.append(
            {
                "brief_id": brief_id,
                "date": source_doc.get("date", ""),
                "title": source_doc.get("title", ""),
                "source_kind": source_doc.get("source_kind", ""),
                "overall_position": brief_obj.get("overall_position", ""),
                "change_intensity": brief_obj.get("change_intensity", ""),
                "novelty": brief_obj.get("novelty_score", 0.0),
                "continuity": brief_obj.get("continuity_score", 0.0),
                "confidence": brief_obj.get("confidence", 0.0),
                "engine": brief.get("engine", ""),
                "model": brief.get("model", ""),
                "generated_at": brief.get("generated_at", ""),
            }
        )

    if saved_rows:
        saved_df = pd.DataFrame(saved_rows)
        saved_df = _sort_table_by_date(saved_df, date_col="date")
        st.dataframe(
            saved_df[[
                "date", "title", "source_kind", "overall_position", "change_intensity",
                "novelty", "continuity", "confidence", "engine", "model", "generated_at"
            ]],
            use_container_width=True,
            hide_index=True,
        )

        selected_brief_id = st.selectbox(
            "Review Brief",
            saved_df["brief_id"].tolist(),
            format_func=lambda b: (
                f"{brief_map.get(b, {}).get('source_doc', {}).get('date', '')} | "
                f"{brief_map.get(b, {}).get('source_doc', {}).get('title', b)}"
            ),
            key="review_policy_delta_brief_id",
        )
        selected_brief = brief_map.get(selected_brief_id, {})
        selected_source = selected_brief.get("source_doc", {}) if isinstance(selected_brief.get("source_doc", {}), dict) else {}
        selected_detail = selected_brief.get("brief", {}) if isinstance(selected_brief.get("brief", {}), dict) else {}
        selected_prior = selected_brief.get("prior_docs", []) if isinstance(selected_brief.get("prior_docs", []), list) else []
        selected_comp = selected_brief.get("comparison", {}) if isinstance(selected_brief.get("comparison", {}), dict) else {}

        st.markdown(f"**Title:** {selected_source.get('title', '')}")
        st.markdown(
            f"**Date:** {selected_source.get('date', '')} | "
            f"**Type:** {selected_source.get('doc_type', '')} | "
            f"**Source:** `{selected_source.get('source_kind', '')}`"
        )
        comp_orgs = ", ".join(selected_comp.get("compare_org_keys", [])[:8]) if isinstance(selected_comp.get("compare_org_keys", []), list) else ""
        comp_types = ", ".join(selected_comp.get("compare_source_kinds", [])[:8]) if isinstance(selected_comp.get("compare_source_kinds", []), list) else ""
        if comp_orgs or comp_types:
            st.caption(
                f"Comparison scope | orgs: {comp_orgs or 'all'} | corpus types: {comp_types or 'all'} | "
                f"lookback_days: {int(selected_comp.get('lookback_days', 0) or 0)}"
            )
        st.markdown(
            f"**Overall Position:** `{selected_detail.get('overall_position', '')}` | "
            f"**Change Intensity:** `{selected_detail.get('change_intensity', '')}` | "
            f"**Confidence:** {float(selected_detail.get('confidence', 0.0) or 0.0):.2f}"
        )
        st.markdown(
            f"**Novelty Score:** {float(selected_detail.get('novelty_score', 0.0) or 0.0):.2f} | "
            f"**Continuity Score:** {float(selected_detail.get('continuity_score', 0.0) or 0.0):.2f} | "
            f"**Stance Direction:** `{selected_detail.get('stance_direction', '')}`"
        )
        st.markdown(f"**Executive Summary:** {selected_detail.get('executive_summary', '')}")

        if selected_detail.get("new_elements", []):
            st.markdown("**New Elements**")
            for line in selected_detail.get("new_elements", []):
                st.markdown(f"- {line}")
        if selected_detail.get("continued_elements", []):
            st.markdown("**Continued Elements**")
            for line in selected_detail.get("continued_elements", []):
                st.markdown(f"- {line}")
        if selected_detail.get("changed_elements", []):
            st.markdown("**Changed Elements**")
            for line in selected_detail.get("changed_elements", []):
                st.markdown(f"- {line}")
        if selected_detail.get("legal_risk_points", []):
            st.markdown("**Legal Risk Points**")
            for line in selected_detail.get("legal_risk_points", []):
                st.markdown(f"- {line}")

        classifications = selected_detail.get("classifications", [])
        if isinstance(classifications, list) and classifications:
            with st.expander("Classifications + Evidence", expanded=False):
                for c in classifications:
                    label = c.get("label", "")
                    conf = _coerce_float(c.get("confidence", 0.0), default=0.0, min_value=0.0, max_value=1.0)
                    st.markdown(f"**{label}** (confidence {conf:.2f})")
                    st.caption(c.get("description", ""))
                    for ev in c.get("evidence", [])[:4]:
                        st.markdown(
                            f"- `{ev.get('source', '')}` | **{ev.get('doc_title', '')}** | {ev.get('date', '')}"
                        )
                        if ev.get("quote"):
                            st.caption(ev.get("quote", ""))

        if selected_prior:
            with st.expander("Compared Prior Documents", expanded=False):
                prior_df = pd.DataFrame(
                    [
                        {
                            "date": p.get("date", ""),
                            "title": p.get("title", ""),
                            "source_kind": p.get("source_kind", ""),
                            "doc_type": p.get("doc_type", ""),
                            "similarity": p.get("similarity_score", 0.0),
                        }
                        for p in selected_prior
                    ]
                )
                prior_df = _sort_table_by_date(prior_df, date_col="date")
                st.dataframe(prior_df, use_container_width=True, hide_index=True)
    else:
        st.info("No policy delta briefs saved yet. Generate one above.")


# =====================================================
# PAGE: Document Library
# =====================================================
elif page == "Document Library":
    st.title("Document Library")
    st.markdown("Browse ingested custom documents and connector outputs.")

    custom_docs = _custom_docs_as_speeches(_load_custom_documents())
    if not custom_docs:
        st.info("No ingested custom documents yet. Use the **Extraction** page to add documents.")
    else:
        rows = []
        for item in custom_docs:
            m = item.get("metadata", {})
            rows.append(
                {
                    "date": m.get("date", ""),
                    "title": m.get("title", ""),
                    "organization": m.get("organization", ""),
                    "speaker": m.get("speaker", ""),
                    "doc_type": m.get("doc_type", ""),
                    "source_kind": m.get("source_kind", ""),
                    "words": m.get("word_count", 0),
                    "source_file": m.get("source_filename", ""),
                    "url": m.get("url", ""),
                }
            )
        docs_df = pd.DataFrame(rows)
        docs_df = _sort_table_by_date(docs_df, date_col="date")

        col1, col2 = st.columns(2)
        with col1:
            org_options = sorted(docs_df["organization"].fillna("").astype(str).unique().tolist())
            selected_orgs = st.multiselect("Organization Filter", org_options, default=org_options)
        with col2:
            kind_options = sorted(docs_df["source_kind"].fillna("").astype(str).unique().tolist())
            selected_kinds = st.multiselect("Source Kind Filter", kind_options, default=kind_options)

        filtered = docs_df.copy()
        if selected_orgs:
            filtered = filtered[filtered["organization"].isin(selected_orgs)]
        if selected_kinds:
            filtered = filtered[filtered["source_kind"].isin(selected_kinds)]

        st.dataframe(
            filtered[["date", "title", "organization", "speaker", "doc_type", "source_kind", "words", "source_file"]],
            use_container_width=True,
            hide_index=True,
        )
