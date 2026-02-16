#!/usr/bin/env python3
"""
SEC Commissioner Speeches Dashboard
Streamlit app for exploring and analyzing SEC Commissioner speeches.
"""

import json
import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from analysis_pipeline import SpeechAnalysisPipeline


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


@st.cache_data(ttl=300)
def load_data(_cache_buster=None):
    """Load and cache the speech dataset."""
    raw_data, _ = _load_raw_data()

    rows = []
    for speech in raw_data.get("speeches", []):
        m = speech.get("metadata", {})
        c = speech.get("content", {})
        v = speech.get("validation", {})
        rows.append({
            "title": m.get("title", ""),
            "speaker": m.get("speaker", "Unknown"),
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

raw_data_json = json.dumps(raw_data)
sentiment_data, topic_data, commissioner_data = run_analysis(raw_data_json)


# --- Sidebar Navigation ---
st.sidebar.title("SEC Speeches")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Sentiment Analysis", "Topic Analysis", "Speech Explorer", "Extract Speeches"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{len(df)} speeches loaded**")
st.sidebar.markdown(f"**{df['speaker'].nunique()} commissioners**")
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


# =====================================================
# PAGE: Overview
# =====================================================
if page == "Overview":
    st.title("SEC Commissioner Speeches Dashboard")
    st.markdown("Analysis of SEC Commissioner speeches \u2014 sentiment, topics, and trends.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Speeches", len(df))
    col2.metric("Commissioners", df["speaker"].nunique())
    col3.metric("Total Words", f"{df['word_count'].sum():,}")
    col4.metric("Avg Words/Speech", f"{df['word_count'].mean():,.0f}")

    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.subheader("Speeches by Commissioner")
        st.bar_chart(df["speaker"].value_counts())
    with right:
        st.subheader("Word Count by Commissioner")
        st.bar_chart(df.groupby("speaker")["word_count"].sum().sort_values(ascending=False))

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
    st.markdown("Browse and read the full text of extracted speeches.")

    col1, col2 = st.columns(2)
    with col1:
        speakers = ["All"] + sorted(df["speaker"].unique().tolist())
        selected_speaker = st.selectbox("Filter by Commissioner", speakers)
    with col2:
        search_term = st.text_input("Search in titles")

    filtered = df.copy()
    if selected_speaker != "All":
        filtered = filtered[filtered["speaker"] == selected_speaker]
    if search_term:
        filtered = filtered[filtered["title"].str.contains(search_term, case=False, na=False)]
    filtered = _sort_table_by_date(filtered, date_col="date")

    st.markdown(f"**Showing {len(filtered)} of {len(df)} speeches**")
    st.markdown("---")

    for idx, row in filtered.iterrows():
        with st.expander(f"{row['title']} \u2014 {row['speaker']} ({row['date']})"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Words", f"{row['word_count']:,}")
            col2.metric("Paragraphs", row["paragraph_count"])
            col3.metric("Completeness", f"{row['completeness_score']}%")

            if row["url"]:
                st.markdown(f"[View on SEC.gov]({row['url']})")

            st.markdown("---")
            st.markdown(row["full_text"][:5000] + ("..." if len(row["full_text"]) > 5000 else ""))

            if len(row["full_text"]) > 5000:
                if st.button("Show full text", key=f"full_{idx}"):
                    st.markdown(row["full_text"])


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
