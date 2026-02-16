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


def _build_dataset_signature(raw_data_obj):
    """Build a stable signature for current speech corpus."""
    pieces = []
    for speech in raw_data_obj.get("speeches", []):
        m = speech.get("metadata", {})
        pieces.append(
            "|".join(
                [
                    str(m.get("url", "")),
                    str(m.get("title", "")),
                    str(m.get("speaker", "")),
                    str(m.get("date", "")),
                    str(m.get("word_count", "")),
                ]
            )
        )
    payload = "\n".join(sorted(pieces))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _vector_state_path():
    return Path("data/openai_vector_store_state.json")


def _vector_state_blob_name():
    return "openai_vector_store_state.json"


def _load_vector_state_local():
    path = _vector_state_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_vector_state_local(state):
    path = _vector_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _load_vector_state():
    storage = _get_gcs_storage()
    if storage is not None:
        try:
            blob = storage.bucket.blob(_vector_state_blob_name())
            if blob.exists():
                state = json.loads(blob.download_as_text(encoding="utf-8"))
                _save_vector_state_local(state)
                return state
        except Exception as e:
            st.session_state["_vector_state_error"] = f"GCS vector-state load failed: {e}"

    return _load_vector_state_local()


def _save_vector_state(state):
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


def _render_corpus_file(raw_data_obj):
    """Create a plain-text corpus file for vector-store indexing."""
    out_path = Path("data/speeches_corpus_for_chat.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    for i, speech in enumerate(raw_data_obj.get("speeches", []), 1):
        m = speech.get("metadata", {})
        c = speech.get("content", {})
        text = c.get("full_text", "").strip()
        if not text:
            continue

        entry = (
            f"Speech ID: {i}\n"
            f"Title: {m.get('title', '')}\n"
            f"Speaker: {m.get('speaker', '')}\n"
            f"Date: {m.get('date', '')}\n"
            f"URL: {m.get('url', '')}\n"
            f"Word Count: {m.get('word_count', 0)}\n\n"
            f"{text}\n\n"
            "==============================\n\n"
        )
        chunks.append(entry)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("SEC SPEECH CORPUS\n\n")
        f.write("".join(chunks))

    return out_path


def _ensure_vector_store(client, raw_data_obj, force_rebuild=False):
    """Return (vector_store_id, rebuilt_flag) for the current dataset."""
    signature = _build_dataset_signature(raw_data_obj)
    state = _load_vector_state()
    existing_id = state.get("vector_store_id")

    if existing_id and state.get("dataset_signature") == signature and not force_rebuild:
        try:
            client.vector_stores.retrieve(existing_id)
            return existing_id, False
        except Exception:
            pass

    corpus_file = _render_corpus_file(raw_data_obj)
    vector_store = client.vector_stores.create(
        name=f"SEC Speeches ({datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC)"
    )

    with open(corpus_file, "rb") as f:
        # SDK helper to upload and block until indexing is complete.
        try:
            client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id,
                files=[f],
            )
        except Exception:
            # Fallback for SDK variants that expose files.upload_and_poll.
            client.vector_stores.files.upload_and_poll(
                vector_store_id=vector_store.id,
                file=f,
            )

    _save_vector_state(
        {
            "vector_store_id": vector_store.id,
            "dataset_signature": signature,
            "source_file": str(corpus_file),
            "indexed_speeches": len(raw_data_obj.get("speeches", [])),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
    )
    return vector_store.id, True


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

    vector_state = _load_vector_state()
    active_vector_store_id = st.session_state.get("vector_store_id") or vector_state.get("vector_store_id")
    if active_vector_store_id:
        st.session_state["vector_store_id"] = active_vector_store_id

    if active_vector_store_id:
        st.caption(f"Current vector store: `{active_vector_store_id}`")
    else:
        st.caption("No vector store indexed yet.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Build/Sync Knowledge Index", type="primary"):
            with st.spinner("Building vector index from speeches..."):
                try:
                    vector_store_id, rebuilt = _ensure_vector_store(client, raw_data, force_rebuild=False)
                    st.session_state["vector_store_id"] = vector_store_id
                    if rebuilt:
                        st.success("Knowledge index rebuilt.")
                    else:
                        st.success("Knowledge index is up to date.")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
    with col2:
        if st.button("Force Rebuild Index"):
            with st.spinner("Rebuilding vector index..."):
                try:
                    vector_store_id, _ = _ensure_vector_store(client, raw_data, force_rebuild=True)
                    st.session_state["vector_store_id"] = vector_store_id
                    st.success("Knowledge index rebuilt from scratch.")
                except Exception as e:
                    st.error(f"Rebuild failed: {e}")

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    for msg in st.session_state["chat_messages"]:
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

    user_prompt = st.chat_input("Ask a question about SEC speeches...")
    if user_prompt:
        st.session_state["chat_messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        vector_store_id = st.session_state.get("vector_store_id")
        if not vector_store_id:
            err_msg = "Please click **Build/Sync Knowledge Index** before chatting."
            st.session_state["chat_messages"].append({"role": "assistant", "content": err_msg})
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

            st.session_state["chat_messages"].append(
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
