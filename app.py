#!/usr/bin/env python3
"""
SEC Commissioner Speeches Dashboard
Streamlit app for exploring and analyzing SEC Commissioner speeches.
"""

import json
import hashlib
import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from analysis_pipeline import SpeechAnalysisPipeline
from speaker_utils import extract_speakers, format_speakers, primary_speaker


# --- Page Config ---
st.set_page_config(
    page_title="SEC Speeches Dashboard",
    page_icon="\U0001f4dc",
    layout="wide",
)


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


def _build_org_documents(raw_data_obj, org_key, org_label):
    """Build doc records keyed by deterministic doc_id for one organization."""
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

        stable_seed = url or "|".join([title, speaker, speech_date])
        if not stable_seed.strip("|"):
            stable_seed = text[:500]
        doc_id = hashlib.sha256(f"{org_key}|{stable_seed}".encode("utf-8")).hexdigest()[:24]

        rendered = (
            f"Organization: {org_label}\n"
            f"Doc ID: {doc_id}\n"
            f"Title: {title}\n"
            f"Speaker: {speaker}\n"
            f"Date: {speech_date}\n"
            f"URL: {url}\n"
            f"Word Count: {word_count}\n\n"
            f"{text}\n"
        )
        content_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()

        docs[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "speaker": speaker,
            "date": speech_date,
            "url": url,
            "word_count": word_count,
            "filename": f"{org_key}_{doc_id}.txt",
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


def _chat_doc_path(org_key, doc_id):
    base = Path("data/chat_docs") / org_key
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{doc_id}.txt"


def _write_chat_doc_file(org_key, doc):
    path = _chat_doc_path(org_key, doc["doc_id"])
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc["rendered_text"])
    return path


def _extract_file_ref(upload_obj):
    data = _normalize_obj(upload_obj)
    vector_store_file_id = data.get("id") or ""
    file_id = data.get("file_id") or ""
    if not file_id and isinstance(vector_store_file_id, str) and not vector_store_file_id.startswith("vsf_"):
        file_id = vector_store_file_id
    return {
        "file_id": file_id,
        "vector_store_file_id": vector_store_file_id,
    }


def _upload_doc_to_vector_store(client, vector_store_id, org_key, doc):
    file_path = _write_chat_doc_file(org_key, doc)
    with open(file_path, "rb") as f:
        uploaded = client.vector_stores.files.upload_and_poll(
            vector_store_id=vector_store_id,
            file=f,
        )
    file_ref = _extract_file_ref(uploaded)
    if not file_ref.get("file_id"):
        raise RuntimeError("Vector-store upload did not return a file ID.")
    return file_ref


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

    current_docs = _build_org_documents(raw_data_obj, org_key, org_label)
    indexed_docs = org_state.get("docs", {})
    if not isinstance(indexed_docs, dict):
        indexed_docs = {}

    # If we only have legacy aggregate counts (no per-doc manifest), rebuild once
    # so future syncs are truly incremental without duplicate legacy corpus files.
    legacy_without_manifest = (
        not force_rebuild
        and bool(str(org_state.get("vector_store_id", "") or "").strip())
        and not indexed_docs
        and int(org_state.get("doc_count_indexed", 0) or 0) > 0
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
                    "failed": [],
                    "sync_mode": "noop",
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

    failed = []
    deleted_count = 0
    uploaded_count = 0

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

    next_docs = {}
    for doc_id in unchanged_ids:
        entry = indexed_docs.get(doc_id, {})
        if entry:
            next_docs[doc_id] = entry

    for doc_id in upload_targets:
        doc = current_docs.get(doc_id)
        if not doc:
            completed_ops += 1
            _emit_progress(completed_ops, total_ops, f"Skipped missing doc {doc_id}")
            continue
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
                "failed": failed,
                "sync_mode": "rebuild" if (force_rebuild or created_new_store) else "incremental",
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
    current_docs = _build_org_documents(raw_data_obj, org_key, org_label)
    add_ids, update_ids, remove_ids, _ = _plan_doc_sync(indexed_docs, current_docs)
    return {
        "vector_store_id": str(org_state.get("vector_store_id", "") or "").strip(),
        "indexed_docs": len(indexed_docs),
        "current_docs": len(current_docs),
        "pending_add": len(add_ids),
        "pending_update": len(update_ids),
        "pending_remove": len(remove_ids),
    }


def _normalize_obj(obj):
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return {}


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


def _ask_agent(client, vector_store_id, question, model_name):
    request_payload = {
        "model": model_name,
        "input": question,
        "tools": [
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
                "max_num_results": 8,
            }
        ],
    }
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


def _ask_agent_with_fallback(client, vector_store_id, question, preferred_model, model_pool):
    """Try preferred model first, then fallback models on access errors."""
    ordered = [preferred_model] + [m for m in model_pool if m != preferred_model]
    last_error = None
    for idx, model_name in enumerate(ordered):
        try:
            result = _ask_agent(client, vector_store_id, question, model_name)
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

raw_data_json = json.dumps(raw_data)
sentiment_data, topic_data, commissioner_data = run_analysis(raw_data_json)


# --- Sidebar Navigation ---
st.sidebar.title("SEC Speeches")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Sentiment Analysis", "Topic Analysis", "Speech Explorer", "Agent Chat", "Extract Speeches"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{len(df)} speeches loaded**")
st.sidebar.markdown(f"**{speaker_df['speaker_individual'].nunique()} unique speakers**")
st.sidebar.markdown(f"**{df['word_count'].sum():,} total words**")

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
# PAGE: Speech Explorer
# =====================================================
elif page == "Speech Explorer":
    st.title("Speech Explorer")
    st.markdown("Browse speeches with topic context from topic analysis.")

    topic_results = topic_data.get("results", [])
    topic_by_url = {r.get("url", ""): r for r in topic_results if r.get("url")}
    topic_by_title_speaker = {(r.get("title", ""), r.get("speaker", "")): r for r in topic_results}
    topic_names = sorted(topic_data.get("summary", {}).get("topic_distribution", {}).keys())

    col1, col2, col3 = st.columns(3)
    with col1:
        speakers = ["All"] + sorted(speaker_df["speaker_individual"].unique().tolist())
        selected_speaker = st.selectbox("Filter by Speaker", speakers)
    with col2:
        search_term = st.text_input("Search in titles")
    with col3:
        selected_topic = st.selectbox("Filter by Primary Topic", ["All"] + topic_names)

    filtered = df.copy()
    if selected_speaker != "All":
        filtered = filtered[filtered["speaker_list"].apply(lambda s: selected_speaker in s if isinstance(s, list) else False)]
    if search_term:
        filtered = filtered[filtered["title"].str.contains(search_term, case=False, na=False)]
    if selected_topic != "All":
        def _row_has_primary_topic(r):
            entry = topic_by_url.get(r.get("url", ""))
            if not entry:
                entry = topic_by_title_speaker.get((r.get("title", ""), r.get("speaker", "")))
            return bool(
                entry
                and entry.get("top_topics")
                and entry["top_topics"][0].get("topic") == selected_topic
            )

        filtered = filtered[filtered.apply(_row_has_primary_topic, axis=1)]
    filtered = _sort_table_by_date(filtered, date_col="date")

    st.markdown(f"**Showing {len(filtered)} of {len(df)} speeches**")
    st.markdown("---")

    for idx, row in filtered.iterrows():
        topic_entry = topic_by_url.get(row.get("url", ""))
        if not topic_entry:
            topic_entry = topic_by_title_speaker.get((row.get("title", ""), row.get("speaker", "")))
        top_topics = topic_entry.get("top_topics", []) if topic_entry else []
        primary_topic = top_topics[0]["topic"] if top_topics else "N/A"
        primary_score = top_topics[0]["score"] if top_topics else 0.0
        top_topics_text = ", ".join(
            [f"{t['topic']} ({t['score']:.2f})" for t in top_topics]
        ) if top_topics else "N/A"

        with st.expander(f"{row['title']} \u2014 {row['speaker']} ({row['date']})"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Words", f"{row['word_count']:,}")
            col2.metric("Paragraphs", row["paragraph_count"])
            col3.metric("Completeness", f"{row['completeness_score']}%")

            st.markdown(f"**Primary Topic:** {primary_topic} ({primary_score:.2f})")
            st.markdown(f"**Top Topics:** {top_topics_text}")

            if row["url"]:
                st.markdown(f"[View on SEC.gov]({row['url']})")

            st.markdown("---")
            st.markdown(row["full_text"][:5000] + ("..." if len(row["full_text"]) > 5000 else ""))

            if len(row["full_text"]) > 5000:
                if st.button("Show full text", key=f"full_{idx}"):
                    st.markdown(row["full_text"])


# =====================================================
# PAGE: Agent Chat
# =====================================================
elif page == "Agent Chat":
    st.title("Agent Chat")
    st.markdown("Ask questions about the speech corpus using retrieval + reasoning.")

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

    org_options = _list_org_options(raw_data)
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

    index_status = _get_org_index_status(raw_data, org_key, org_label)
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

    idx_col1, idx_col2, idx_col3, idx_col4 = st.columns(4)
    idx_col1.metric("Corpus Docs", index_status.get("current_docs", 0))
    idx_col2.metric("Indexed Docs", index_status.get("indexed_docs", 0))
    idx_col3.metric(
        "Pending Add/Update",
        index_status.get("pending_add", 0) + index_status.get("pending_update", 0),
    )
    idx_col4.metric("Pending Remove", index_status.get("pending_remove", 0))

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
                        raw_data_obj=raw_data,
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
                        st.success(
                            "Knowledge index sync complete: "
                            f"+{stats.get('add', 0)} add, "
                            f"~{stats.get('update', 0)} update, "
                            f"-{stats.get('remove', 0)} remove."
                        )
                        if report.get("failed"):
                            st.warning(f"{len(report['failed'])} document operations failed.")
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
                        raw_data_obj=raw_data,
                        org_key=org_key,
                        org_label=org_label,
                        force_rebuild=True,
                        progress_callback=_rebuild_progress_cb,
                    )
                    vector_store_id = report.get("vector_store_id", "")
                    if vector_store_id:
                        st.session_state["vector_store_ids_by_org"][org_key] = vector_store_id
                    st.success("Knowledge index rebuilt from scratch.")
                    if report.get("failed"):
                        st.warning(f"{len(report['failed'])} document operations failed during rebuild.")
                    if report.get("old_store_delete_error"):
                        st.info(f"Previous vector store cleanup skipped: {report['old_store_delete_error']}")
                except Exception as e:
                    st.error(f"Rebuild failed: {e}")

    if "chat_messages_by_org" not in st.session_state:
        st.session_state["chat_messages_by_org"] = {}
    if org_key not in st.session_state["chat_messages_by_org"]:
        st.session_state["chat_messages_by_org"][org_key] = []
    chat_messages = st.session_state["chat_messages_by_org"][org_key]

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

    user_prompt = st.chat_input(f"Ask a question about {org_label} speeches...")
    if user_prompt:
        chat_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        vector_store_id = st.session_state.get("vector_store_ids_by_org", {}).get(org_key)
        if not vector_store_id:
            err_msg = "Please click **Build/Sync Knowledge Index** before chatting."
            chat_messages.append({"role": "assistant", "content": err_msg})
            with st.chat_message("assistant"):
                st.error(err_msg)
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        agent_out = _ask_agent_with_fallback(
                            client=client,
                            vector_store_id=vector_store_id,
                            question=user_prompt,
                            preferred_model=model_name,
                            model_pool=available_models,
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
# PAGE: Extract Speeches
# =====================================================
elif page == "Extract Speeches":
    st.title("Extract Speeches")
    st.markdown("Discover and extract SEC speeches by date range.")

    # Date range picker
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=date.today() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End date", value=date.today())

    # Track discovered speeches in session state
    if "discovered" not in st.session_state:
        st.session_state.discovered = []

    # --- Discover ---
    if st.button("Discover Speeches"):
        with st.status("Discovering speeches from SEC.gov...", expanded=True) as status:
            from sec_scraper_free import SECScraper
            scraper = SECScraper()
            # Estimate pages by how far back we need to scan from "today" to
            # the start of the requested range (not by range width). The scraper
            # will stop early once it reaches rows older than start_date.
            days_back_to_start = max(0, (date.today() - start_date).days)
            estimated_pages = max(3, days_back_to_start // 14 + 2)
            max_pages = min(80, estimated_pages)

            st.write(
                f"Scanning up to {max_pages} listing pages "
                "(will stop early when start date is reached)..."
            )
            entries = scraper.discover_speech_urls(
                max_pages=max_pages,
                start_date=start_date,
                end_date=end_date,
            )

            # Deduplicate against existing dataset
            existing_urls = {s.get("metadata", {}).get("url", "") for s in raw_data.get("speeches", [])}
            new_entries = [e for e in entries if e["url"] not in existing_urls]
            already = len(entries) - len(new_entries)

            st.session_state.discovered = new_entries
            status.update(
                label=f"Found {len(new_entries)} new speeches ({already} already extracted)",
                state="complete",
            )

    # --- Show discovered speeches ---
    discovered = st.session_state.discovered
    if discovered:
        disc_df = _sort_table_by_date(pd.DataFrame(discovered), date_col="date")
        if "speaker" in disc_df.columns:
            disc_df["speaker"] = disc_df["speaker"].apply(format_speakers)
        discovered_sorted = disc_df.to_dict(orient="records")

        st.subheader(f"{len(discovered_sorted)} new speeches available")
        st.dataframe(
            disc_df[["date", "title", "speaker", "type"]],
            use_container_width=True,
            hide_index=True,
        )

        if len(discovered_sorted) == 1:
            max_extract = 1
            st.caption("1 speech found. It will be extracted.")
        else:
            max_extract = st.slider(
                "Speeches to extract",
                min_value=1,
                max_value=len(discovered_sorted),
                value=len(discovered_sorted),
            )

        if st.button("Extract Speeches"):
            from speech_analyzer import SECSpeechAnalyzer
            analyzer = SECSpeechAnalyzer()

            progress = st.progress(0, text="Starting extraction...")
            extracted = []
            failed = []

            for i, entry in enumerate(discovered_sorted[:max_extract]):
                progress.progress(
                    (i + 1) / max_extract,
                    text=f"Extracting {i + 1}/{max_extract}: {entry['title'][:50]}...",
                )
                result = analyzer.extract_speech_for_analysis(entry["url"], listing_metadata=entry)
                if result["success"] and analyzer.validate_full_text_extraction(result["data"]):
                    extracted.append(result["data"])
                else:
                    failed.append(entry["title"])

            progress.progress(1.0, text="Extraction complete!")

            if extracted:
                # Merge into dataset
                updated_data, _ = _load_raw_data()
                updated_data["speeches"].extend(extracted)
                updated_data["extraction_summary"]["successful_extractions"] = len(updated_data["speeches"])

                # Save to GCS
                storage = _get_gcs_storage()
                if storage is not None:
                    storage.save_speeches(updated_data)
                    st.success(f"Saved {len(extracted)} new speeches to Google Cloud Storage.")
                else:
                    # Fallback: save locally
                    with open("data/all_speeches_final.json", "w", encoding="utf-8") as f:
                        json.dump(updated_data, f, indent=2, ensure_ascii=False)
                    st.success(f"Saved {len(extracted)} new speeches locally.")

                # Clear caches so dashboard reflects new data
                load_data.clear()
                run_analysis.clear()
                st.session_state.discovered = []

                st.info("Refresh the page to see the new speeches in the dashboard.")

            if failed:
                st.warning(f"{len(failed)} speeches failed extraction:")
                for title in failed:
                    st.write(f"- {title}")

    elif st.session_state.get("discovered") is not None and not discovered:
        st.info("Use the date range picker above and click **Discover Speeches** to find speeches to extract.")
