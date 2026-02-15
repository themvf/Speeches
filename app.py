#!/usr/bin/env python3
"""
SEC Commissioner Speeches Dashboard
Streamlit app for exploring and analyzing SEC Commissioner speeches.
"""

import json
import streamlit as st
import pandas as pd
from pathlib import Path
from analysis_pipeline import SpeechAnalysisPipeline


# --- Page Config ---
st.set_page_config(
    page_title="SEC Speeches Dashboard",
    page_icon="📜",
    layout="wide",
)


@st.cache_data
def load_data():
    """Load and cache the speech dataset."""
    data_file = Path("data/all_speeches_final.json")
    if not data_file.exists():
        st.error("Dataset not found at data/all_speeches_final.json")
        st.stop()

    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Build DataFrame
    rows = []
    for speech in raw_data["speeches"]:
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
    try:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    except Exception:
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

# Serialize for caching (cache_data needs hashable input)
raw_data_json = json.dumps(raw_data)
sentiment_data, topic_data, commissioner_data = run_analysis(raw_data_json)


# --- Sidebar Navigation ---
st.sidebar.title("SEC Speeches")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Sentiment Analysis", "Topic Analysis", "Speech Explorer"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{len(df)} speeches loaded**")
st.sidebar.markdown(f"**{df['speaker'].nunique()} commissioners**")
st.sidebar.markdown(f"**{df['word_count'].sum():,} total words**")


# =====================================================
# PAGE: Overview
# =====================================================
if page == "Overview":
    st.title("SEC Commissioner Speeches Dashboard")
    st.markdown("Analysis of SEC Commissioner speeches — sentiment, topics, and trends.")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Speeches", len(df))
    col2.metric("Commissioners", df["speaker"].nunique())
    col3.metric("Total Words", f"{df['word_count'].sum():,}")
    col4.metric("Avg Words/Speech", f"{df['word_count'].mean():,.0f}")

    st.markdown("---")

    # Two column layout
    left, right = st.columns(2)

    with left:
        st.subheader("Speeches by Commissioner")
        speaker_counts = df["speaker"].value_counts()
        st.bar_chart(speaker_counts)

    with right:
        st.subheader("Word Count by Commissioner")
        word_counts = df.groupby("speaker")["word_count"].sum().sort_values(ascending=False)
        st.bar_chart(word_counts)

    st.markdown("---")

    # Speech listing
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

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    dist = summary["sentiment_distribution"]
    col1.metric("Positive Speeches", dist["positive"])
    col2.metric("Neutral Speeches", dist["neutral"])
    col3.metric("Negative Speeches", dist["negative"])

    st.markdown("---")

    # Sentiment scores chart
    st.subheader("Sentiment Score by Speech")
    sent_df = pd.DataFrame(results)
    sent_df["short_title"] = sent_df["title"].str[:50] + "..."

    chart_data = sent_df.set_index("short_title")["sentiment_score"]
    st.bar_chart(chart_data)

    st.markdown("---")

    # Detailed table
    st.subheader("Detailed Sentiment Breakdown")
    detail_df = sent_df[["title", "speaker", "sentiment_score", "positive_words", "negative_words", "regulatory_words"]].copy()
    detail_df.columns = ["Title", "Speaker", "Sentiment Score", "Positive Keywords", "Negative Keywords", "Regulatory Keywords"]
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    # Highlights
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

    # Top topics
    st.subheader("Most Discussed Topics")
    topic_df = pd.DataFrame(summary["most_discussed_topics"])
    if not topic_df.empty:
        chart_data = topic_df.set_index("topic")["total_mentions"]
        st.bar_chart(chart_data)

    st.markdown("---")

    # Topic heatmap (as a table with color)
    st.subheader("Topic Relevance by Speech")

    heatmap_rows = []
    for r in results:
        row = {"Speech": r["title"][:50] + "...", "Speaker": r["speaker"]}
        for topic, data in r["all_topics"].items():
            row[topic] = round(data["relevance_score"], 2)
        heatmap_rows.append(row)

    heatmap_df = pd.DataFrame(heatmap_rows)
    st.dataframe(
        heatmap_df.style.background_gradient(
            cmap="YlOrRd",
            subset=[c for c in heatmap_df.columns if c not in ["Speech", "Speaker"]],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # Speeches grouped by primary topic
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

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        speakers = ["All"] + sorted(df["speaker"].unique().tolist())
        selected_speaker = st.selectbox("Filter by Commissioner", speakers)
    with col2:
        search_term = st.text_input("Search in titles")

    # Apply filters
    filtered = df.copy()
    if selected_speaker != "All":
        filtered = filtered[filtered["speaker"] == selected_speaker]
    if search_term:
        filtered = filtered[filtered["title"].str.contains(search_term, case=False, na=False)]

    st.markdown(f"**Showing {len(filtered)} of {len(df)} speeches**")
    st.markdown("---")

    # Display speeches
    for idx, row in filtered.iterrows():
        with st.expander(f"{row['title']} — {row['speaker']} ({row['date']})"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Words", f"{row['word_count']:,}")
            col2.metric("Paragraphs", row["paragraph_count"])
            col3.metric("Completeness", f"{row['completeness_score']}%")

            if row["url"]:
                st.markdown(f"[View on SEC.gov]({row['url']})")

            st.markdown("---")
            # Show full text in a scrollable container
            st.markdown(row["full_text"][:5000] + ("..." if len(row["full_text"]) > 5000 else ""))

            if len(row["full_text"]) > 5000:
                if st.button(f"Show full text", key=f"full_{idx}"):
                    st.markdown(row["full_text"])
