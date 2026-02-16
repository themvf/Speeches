#!/usr/bin/env python3
"""
SEC Speech Analysis Pipeline
Comprehensive analysis tools for extracted commissioner speeches
"""

import json
import pandas as pd
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
from speaker_utils import format_speakers


class SpeechAnalysisPipeline:
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.speeches_data = None
        self.df = None

    def load_analysis_dataset(self, file_path: Optional[str] = None):
        """Load the extracted speeches dataset"""

        if file_path:
            data_file = Path(file_path)
        else:
            # Find the most recent dataset in data/
            data_dir = Path("data")
            json_files = list(data_dir.glob("all_speeches*.json"))

            if not json_files:
                raise FileNotFoundError("No speech dataset found in data/ directory.")

            data_file = max(json_files, key=lambda f: f.stat().st_mtime)

        with open(data_file, "r", encoding="utf-8") as f:
            self.speeches_data = json.load(f)

        self.create_dataframe()
        print(f"Loaded {len(self.speeches_data['speeches'])} speeches for analysis")

    def create_dataframe(self):
        """Create pandas DataFrame from speeches data"""

        if not self.speeches_data:
            raise ValueError("No speeches data loaded")

        rows = []
        for speech in self.speeches_data["speeches"]:
            metadata = speech.get("metadata", {})
            content = speech.get("content", {})
            validation = speech.get("validation", {})

            row = {
                "title": metadata.get("title", ""),
                "speaker": format_speakers(metadata.get("speaker", "")),
                "date": metadata.get("date", ""),
                "url": metadata.get("url", ""),
                "word_count": metadata.get("word_count", 0),
                "full_text": content.get("full_text", ""),
                "paragraph_count": len(content.get("paragraphs", [])),
                "sentence_count": len(content.get("sentences", [])),
                "completeness_score": validation.get("completeness_score", 0),
                "extraction_date": metadata.get("extraction_date", ""),
            }
            rows.append(row)

        self.df = pd.DataFrame(rows)

        try:
            self.df["date_parsed"] = pd.to_datetime(self.df["date"], errors="coerce")
        except Exception:
            self.df["date_parsed"] = None

    def basic_sentiment_analysis(self) -> Dict[str, Any]:
        """Basic sentiment analysis using keyword-based approach"""

        if self.df is None:
            raise ValueError("No dataset loaded")

        positive_words = [
            "innovation", "growth", "opportunity", "progress", "beneficial",
            "effective", "successful", "positive", "advance", "improve",
            "enhance", "strengthen", "support", "encourage", "facilitate",
        ]

        negative_words = [
            "risk", "concern", "problem", "challenge", "difficult", "harmful",
            "violation", "fraud", "manipulation", "abuse", "illegal",
            "penalty", "enforcement", "sanctions", "prohibit", "restrict",
        ]

        neutral_regulatory = [
            "regulation", "compliance", "oversight", "guidance", "framework",
            "rule", "requirement", "standard", "procedure", "review",
        ]

        sentiment_results = []

        for idx, row in self.df.iterrows():
            text = row["full_text"].lower()

            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            neu_count = sum(1 for word in neutral_regulatory if word in text)

            total_sentiment_words = pos_count + neg_count + neu_count

            if total_sentiment_words > 0:
                pos_ratio = pos_count / total_sentiment_words
                neg_ratio = neg_count / total_sentiment_words
                neu_ratio = neu_count / total_sentiment_words
            else:
                pos_ratio = neg_ratio = neu_ratio = 0

            sentiment_score = (pos_count - neg_count) / max(row["word_count"] / 100, 1)
            sentiment_score = max(-1, min(1, sentiment_score))

            sentiment_results.append({
                "title": row["title"],
                "speaker": row["speaker"],
                "sentiment_score": sentiment_score,
                "positive_ratio": pos_ratio,
                "negative_ratio": neg_ratio,
                "neutral_ratio": neu_ratio,
                "positive_words": pos_count,
                "negative_words": neg_count,
                "regulatory_words": neu_count,
            })

        return {
            "analysis_type": "basic_sentiment",
            "results": sentiment_results,
            "summary": self.summarize_sentiment_results(sentiment_results),
        }

    def summarize_sentiment_results(self, sentiment_results: List[Dict]) -> Dict[str, Any]:
        """Summarize sentiment analysis results"""

        scores = [r["sentiment_score"] for r in sentiment_results]

        return {
            "average_sentiment": sum(scores) / len(scores) if scores else 0,
            "most_positive": max(sentiment_results, key=lambda x: x["sentiment_score"]) if sentiment_results else None,
            "most_negative": min(sentiment_results, key=lambda x: x["sentiment_score"]) if sentiment_results else None,
            "sentiment_distribution": {
                "positive": len([s for s in scores if s > 0.1]),
                "neutral": len([s for s in scores if -0.1 <= s <= 0.1]),
                "negative": len([s for s in scores if s < -0.1]),
            },
        }

    def topic_analysis(self) -> Dict[str, Any]:
        """Basic topic analysis using keyword frequency"""

        if self.df is None:
            raise ValueError("No dataset loaded")

        topic_categories = {
            "Digital Assets/Crypto": [
                "digital", "crypto", "cryptocurrency", "bitcoin", "blockchain",
                "token", "defi", "decentralized", "digital asset", "stablecoin",
            ],
            "Enforcement": [
                "enforcement", "violation", "penalty", "sanction", "fraud",
                "investigation", "settlement", "fine", "misconduct", "illegal",
            ],
            "Market Structure": [
                "market", "trading", "exchange", "liquidity", "settlement",
                "clearing", "market maker", "order", "execution", "price",
            ],
            "Innovation/Technology": [
                "innovation", "technology", "artificial intelligence", "ai",
                "fintech", "automation", "algorithm", "data", "analytics",
            ],
            "Investor Protection": [
                "investor", "protection", "disclosure", "transparency",
                "retail", "consumer", "fiduciary", "advisory", "education",
            ],
            "Regulatory Framework": [
                "regulation", "rule", "guidance", "framework", "standard",
                "compliance", "oversight", "supervision", "policy", "law",
            ],
        }

        topic_results = []

        for idx, row in self.df.iterrows():
            text = row["full_text"].lower()

            speech_topics = {}
            for topic, keywords in topic_categories.items():
                keyword_count = sum(1 for keyword in keywords if keyword in text)
                relevance_score = keyword_count / max(row["word_count"] / 100, 1)
                speech_topics[topic] = {
                    "keyword_count": keyword_count,
                    "relevance_score": relevance_score,
                }

            top_topics = sorted(speech_topics.items(), key=lambda x: x[1]["relevance_score"], reverse=True)[:3]

            topic_results.append({
                "title": row["title"],
                "speaker": row["speaker"],
                "url": row.get("url", ""),
                "all_topics": speech_topics,
                "top_topics": [{"topic": t[0], "score": t[1]["relevance_score"]} for t in top_topics],
            })

        return {
            "analysis_type": "topic_analysis",
            "results": topic_results,
            "summary": self.summarize_topic_results(topic_results, topic_categories.keys()),
        }

    def summarize_topic_results(self, topic_results: List[Dict], topic_names) -> Dict[str, Any]:
        """Summarize topic analysis results"""

        topic_totals = {topic: 0 for topic in topic_names}
        topic_counts = {topic: 0 for topic in topic_names}

        for result in topic_results:
            for topic, data in result["all_topics"].items():
                if data["relevance_score"] > 0.1:
                    topic_totals[topic] += data["relevance_score"]
                    topic_counts[topic] += 1

        topic_summary = []
        for topic in topic_names:
            avg_score = topic_totals[topic] / max(topic_counts[topic], 1)
            topic_summary.append({
                "topic": topic,
                "total_mentions": topic_counts[topic],
                "average_relevance": avg_score,
                "total_relevance": topic_totals[topic],
            })

        topic_summary.sort(key=lambda x: x["total_relevance"], reverse=True)

        return {
            "most_discussed_topics": topic_summary[:5],
            "topic_distribution": topic_counts,
            "speeches_by_primary_topic": self.group_speeches_by_primary_topic(topic_results),
        }

    def group_speeches_by_primary_topic(self, topic_results: List[Dict]) -> Dict[str, List[str]]:
        """Group speeches by their primary topic"""

        topic_groups = {}

        for result in topic_results:
            if result["top_topics"]:
                primary_topic = result["top_topics"][0]["topic"]
                if primary_topic not in topic_groups:
                    topic_groups[primary_topic] = []
                topic_groups[primary_topic].append(result["title"])

        return topic_groups

    def commissioner_analysis(self) -> Dict[str, Any]:
        """Analyze patterns by commissioner"""

        if self.df is None:
            raise ValueError("No dataset loaded")

        commissioner_stats = {}

        for speaker in self.df["speaker"].unique():
            if not speaker:
                continue

            speaker_df = self.df[self.df["speaker"] == speaker]

            commissioner_stats[speaker] = {
                "speech_count": len(speaker_df),
                "total_words": int(speaker_df["word_count"].sum()),
                "average_words": float(speaker_df["word_count"].mean()),
                "average_completeness": float(speaker_df["completeness_score"].mean()),
                "speech_titles": speaker_df["title"].tolist()[:5],
            }

        return {
            "analysis_type": "commissioner_analysis",
            "results": commissioner_stats,
            "summary": {
                "most_active": max(commissioner_stats.items(), key=lambda x: x[1]["speech_count"])[0] if commissioner_stats else None,
                "most_verbose": max(commissioner_stats.items(), key=lambda x: x[1]["average_words"])[0] if commissioner_stats else None,
                "total_commissioners": len(commissioner_stats),
            },
        }

    def temporal_analysis(self) -> Dict[str, Any]:
        """Analyze speech patterns over time"""

        if self.df is None or "date_parsed" not in self.df.columns:
            return {"error": "No temporal data available"}

        temporal_df = self.df[self.df["date_parsed"].notna()].copy()

        if len(temporal_df) == 0:
            return {"error": "No valid dates found"}

        return {
            "analysis_type": "temporal_analysis",
            "date_range": {
                "earliest": temporal_df["date_parsed"].min().isoformat() if not temporal_df.empty else None,
                "latest": temporal_df["date_parsed"].max().isoformat() if not temporal_df.empty else None,
            },
            "total_timespan_days": (temporal_df["date_parsed"].max() - temporal_df["date_parsed"].min()).days if len(temporal_df) > 1 else 0,
        }

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""

        sentiment_analysis = self.basic_sentiment_analysis()
        topic_analysis = self.topic_analysis()
        commissioner_analysis = self.commissioner_analysis()
        temporal_analysis = self.temporal_analysis()

        report = {
            "report_metadata": {
                "generation_date": datetime.now().isoformat(),
                "dataset_size": len(self.df) if self.df is not None else 0,
                "total_words": int(self.df["word_count"].sum()) if self.df is not None else 0,
            },
            "dataset_overview": {
                "total_speeches": len(self.df) if self.df is not None else 0,
                "unique_commissioners": int(self.df["speaker"].nunique()) if self.df is not None else 0,
                "date_range": temporal_analysis.get("date_range", {}),
                "average_speech_length": float(self.df["word_count"].mean()) if self.df is not None else 0,
            },
            "sentiment_analysis": sentiment_analysis,
            "topic_analysis": topic_analysis,
            "commissioner_analysis": commissioner_analysis,
            "temporal_analysis": temporal_analysis,
        }

        return report


if __name__ == "__main__":
    print("SEC Speech Analysis Pipeline")
    print("=" * 50)

    pipeline = SpeechAnalysisPipeline()

    try:
        pipeline.load_analysis_dataset()
        report = pipeline.generate_comprehensive_report()

        print(f"\nAnalysis complete!")
        print(f"Dataset loaded: {report['dataset_overview']['total_speeches']} speeches")
        print(f"Total words: {report['dataset_overview']['total_words']:,}")
        print(f"Commissioners: {report['dataset_overview']['unique_commissioners']}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure speech data exists in the data/ directory.")
