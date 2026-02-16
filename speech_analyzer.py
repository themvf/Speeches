#!/usr/bin/env python3
"""
SEC Speech Analyzer - Analysis-Optimized Extraction and Processing
Designed for GenAI, sentiment analysis, topic modeling, and advanced analytics
"""

import json
import re
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from sec_scraper_free import SECScraper


class SECSpeechAnalyzer:
    def __init__(self):
        self.scraper = SECScraper()
        self.setup_analysis_directories()
        self.analysis_schema = self.get_analysis_schema()

    def setup_analysis_directories(self):
        """Create analysis-optimized directory structure"""
        base_dir = Path("sec_analysis")

        directories = [
            "raw_data",
            "processed",
            "analysis",
            "outputs",
            "outputs/visualizations",
        ]

        for directory in directories:
            (base_dir / directory).mkdir(parents=True, exist_ok=True)

    def get_analysis_schema(self):
        """Define analysis-optimized extraction schema"""
        return {
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "speaker": {"type": "string"},
                        "speaker_title": {"type": "string"},
                        "date": {"type": "string"},
                        "venue": {"type": "string"},
                        "event": {"type": "string"},
                        "url": {"type": "string"},
                        "speech_type": {"type": "string"},
                    },
                },
                "content": {
                    "type": "object",
                    "properties": {
                        "full_text": {"type": "string"},
                        "introduction": {"type": "string"},
                        "main_body": {"type": "string"},
                        "conclusion": {"type": "string"},
                        "key_quotes": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "policy_positions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
                "topics": {
                    "type": "object",
                    "properties": {
                        "primary_topics": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "regulatory_themes": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "market_sectors": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
                "entities": {
                    "type": "object",
                    "properties": {
                        "people_mentioned": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "organizations": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "regulations_cited": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "companies_mentioned": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
        }

    def extract_speech_for_analysis(self, url: str, listing_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Extract speech in analysis-optimized format.

        Args:
            url: The speech URL to extract.
            listing_metadata: Optional dict with 'date', 'speaker', 'type' from the
                              SEC listing page, used as fallbacks when content parsing
                              cannot find these fields.
        """
        print(f"Extracting speech for analysis: {url}")

        result = self.scraper.scrape_page(url)

        if result["success"]:
            raw_content = result["data"]["data"]["content"]
            analysis_data = self.parse_speech_content(raw_content, url, listing_metadata)

            return {
                "success": True,
                "data": analysis_data,
                "credits_used": 0,
            }
        else:
            return result

    def parse_speech_content(self, raw_content: str, url: str, listing_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Parse raw scraped content into analysis-ready format"""

        # Extract title - look for main heading
        title = ""
        lines = raw_content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("# ") and len(line) > 10 and not line.startswith("# More"):
                title = line[2:].strip()
                break

        # Fallback: look for patterns
        if not title:
            for line in lines:
                line = line.strip()
                if (
                    len(line) > 20
                    and (
                        "Leadership" in line
                        or "Revolution" in line
                        or "Statement" in line
                        or "Remarks" in line
                    )
                    and not line.startswith("[")
                    and not line.startswith("http")
                ):
                    title = line
                    break

        # Extract speaker - look for commissioner names
        speaker = ""
        commissioner_names = [
            "Paul S. Atkins",
            "Hester M. Peirce",
            "Caroline A. Crenshaw",
            "Mark T. Uyeda",
            "Dave A. Sanchez",
        ]
        for name in commissioner_names:
            if name in raw_content:
                speaker = name
                break

        # Extract date - use re.search to get the full match (not just groups)
        date = ""
        date_patterns = [
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},\s+\d{4}",
            r"\d{1,2}/\d{1,2}/\d{4}",
            r"\d{4}-\d{2}-\d{2}",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, raw_content)
            if match:
                date = match.group(0)
                break

        # If no complete date found, try to extract from URL
        if not date or len(date) < 8:
            url_date_match = re.search(r"-(\d{6})(?:\d{2})?(?:-|$)", url)
            if url_date_match:
                date_str = url_date_match.group(1)
                try:
                    month = int(date_str[:2])
                    day = int(date_str[2:4])
                    year_part = date_str[4:6]
                    year = int("20" + year_part)
                    month_names = [
                        "", "January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December",
                    ]
                    if 1 <= month <= 12:
                        date = f"{month_names[month]} {day}, {year}"
                except Exception:
                    pass

        # Use listing_metadata as primary source for date and speaker when available,
        # since the SEC listing page is more authoritative than parsing content text
        if listing_metadata:
            if listing_metadata.get("date"):
                date = listing_metadata["date"]
            if listing_metadata.get("speaker"):
                speaker = listing_metadata["speaker"]
            if not title and listing_metadata.get("title"):
                title = listing_metadata["title"]

        full_text = self.extract_speech_content_improved(raw_content)

        word_count = len(full_text.split()) if full_text else 0
        completeness_score = self.calculate_completeness(full_text)

        analysis_data = {
            "metadata": {
                "extraction_date": datetime.now().isoformat(),
                "url": url,
                "title": title,
                "speaker": speaker,
                "date": date,
                "word_count": word_count,
                "char_count": len(full_text),
            },
            "content": {
                "full_text": full_text,
                "paragraphs": self.split_into_paragraphs(full_text),
                "sentences": self.split_into_sentences(full_text),
            },
            "structured_data": {},
            "entities": {},
            "analysis": {
                "sentiment": None,
                "topics": None,
                "themes": None,
                "embeddings": None,
            },
            "validation": {
                "extraction_method": "parsed_scraping",
                "completeness_score": completeness_score,
                "analysis_ready": True,
            },
        }

        return analysis_data

    def extract_speech_content_improved(self, raw_content: str) -> str:
        """Improved speech content extraction that preserves footnotes"""

        content_lines = raw_content.split("\n")

        speech_start_markers = [
            "Good morning", "Good afternoon", "Good evening", "Thank you",
            "Ladies and gentlemen", "I am pleased", "It is my pleasure",
            "I want to thank", "Today I", "I'm delighted", "I'm honored",
        ]

        speech_start_idx = 0
        for i, line in enumerate(content_lines):
            line_clean = line.strip()
            if any(marker in line_clean for marker in speech_start_markers) and len(line_clean) > 20:
                speech_start_idx = i
                break

        speech_end_idx = len(content_lines)
        for i, line in enumerate(content_lines):
            if "Last Reviewed or Updated:" in line.strip():
                speech_end_idx = i
                break

        if speech_start_idx < speech_end_idx:
            speech_lines = content_lines[speech_start_idx:speech_end_idx]
            cleaned_lines = []

            for line in speech_lines:
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                if line.startswith("[") and line.endswith("]") and ("http" in line or "www." in line):
                    continue
                if len(line) == 1 or line in ["-", "\u2022", "*", "|"]:
                    continue
                cleaned_lines.append(line)

            full_text = self.reconstruct_speech_text(cleaned_lines)
            return full_text

        return ""

    def reconstruct_speech_text(self, lines: List[str]) -> str:
        """Intelligently reconstruct speech text with proper formatting"""

        if not lines:
            return ""

        reconstructed = []
        current_paragraph = []

        for i, line in enumerate(lines):
            is_header = (line.startswith("**") and line.endswith("**")) or line.isupper()
            is_list_item = line.startswith(("-", "\u2022", "*")) or re.match(r"^\d+\.", line)
            is_footnote = re.match(r"^\[\d+\]", line) or line.startswith("[")

            should_break = (
                is_header
                or is_list_item
                or is_footnote
                or (len(current_paragraph) > 0 and len(line) < 100 and line.endswith("."))
                or (i > 0 and len(lines[i - 1]) < 50 and not lines[i - 1].endswith(","))
            )

            if should_break and current_paragraph:
                reconstructed.append(" ".join(current_paragraph))
                current_paragraph = []

            if is_header or is_list_item or is_footnote:
                reconstructed.append(line)
            else:
                current_paragraph.append(line)

        if current_paragraph:
            reconstructed.append(" ".join(current_paragraph))

        return "\n\n".join(reconstructed)

    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs for analysis"""
        if not text:
            return []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return [p for p in paragraphs if len(p) > 50]

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for analysis"""
        if not text:
            return []
        sentences = re.split(r"[.!?]+\s+", text)
        return [s.strip() for s in sentences if s.strip() and len(s) > 20]

    def calculate_completeness(self, text: str) -> int:
        """Calculate completeness score for analysis readiness"""
        if not text:
            return 0

        word_count = len(text.split())
        has_structure = bool(re.search(r"\n\n", text))
        has_substance = word_count > 800
        has_intro_conclusion = self.has_speech_structure(text)

        score = 0
        if word_count > 3000:
            score += 50
        elif word_count > 2000:
            score += 40
        elif word_count > 1000:
            score += 30
        elif word_count > 800:
            score += 20
        else:
            score += 10

        if has_structure:
            score += 25
        if has_substance:
            score += 15
        if has_intro_conclusion:
            score += 10

        return min(score, 100)

    def has_speech_structure(self, text: str) -> bool:
        """Check if text has typical speech structure"""
        text_lower = text.lower()
        intro_markers = ["thank you", "good morning", "good afternoon", "remarks", "speaking", "today i"]
        has_intro = any(marker in text_lower[:500] for marker in intro_markers)
        conclusion_markers = ["in conclusion", "to conclude", "finally", "thank you again", "let me close"]
        has_conclusion = any(marker in text_lower[-500:] for marker in conclusion_markers)
        return has_intro and has_conclusion

    def validate_full_text_extraction(self, analysis_data: Dict[str, Any]) -> bool:
        """Validate that extraction produced usable text.

        Keep validation permissive so shorter SEC statements/remarks are not
        dropped. Only reject empty or nearly-empty outputs that indicate a
        broken scrape/parse.
        """
        content = analysis_data.get("content", {})
        full_text = content.get("full_text", "")
        word_count = analysis_data.get("metadata", {}).get("word_count", 0)
        completeness_score = analysis_data.get("validation", {}).get("completeness_score", 0)
        stripped = full_text.strip()

        # Hard requirements: content must exist and be more than boilerplate.
        hard_criteria = {
            "not_empty": bool(stripped),
            "min_chars": len(stripped) >= 80,
        }
        is_valid = all(hard_criteria.values())

        if not is_valid:
            print(f"  Validation failed - Word count: {word_count}, Completeness: {completeness_score}%")
            failed_criteria = [k for k, v in hard_criteria.items() if not v]
            print(f"  Failed criteria: {', '.join(failed_criteria)}")
            return False

        # Soft quality signal only (non-blocking).
        if word_count < 300 or completeness_score < 40:
            print(
                "  Validation warning - low quality score accepted "
                f"(Word count: {word_count}, Completeness: {completeness_score}%)"
            )

        return True

    def batch_extract_all_speeches(self, speech_entries: list, max_speeches: int = 50) -> Dict[str, Any]:
        """
        Extract all speeches in batch for comprehensive analysis.

        Args:
            speech_entries: List of speech URLs (str) or dicts with 'url' and
                           optional 'date', 'speaker', 'title', 'type' keys.
            max_speeches: Maximum number of speeches to extract
        """
        print(f"Batch extracting {len(speech_entries)} speeches for analysis")

        extracted_speeches = []
        failed_extractions = []

        for i, entry in enumerate(speech_entries[:max_speeches], 1):
            if isinstance(entry, str):
                url = entry
                listing_metadata = None
            else:
                url = entry["url"]
                listing_metadata = entry

            print(f"\nExtracting speech {i}/{min(len(speech_entries), max_speeches)}")

            result = self.extract_speech_for_analysis(url, listing_metadata=listing_metadata)

            if result["success"]:
                if self.validate_full_text_extraction(result["data"]):
                    extracted_speeches.append(result["data"])
                    print(f"SUCCESS: {result['data']['metadata'].get('title', 'Unknown title')[:50]}...")
                    print(f"Word count: {result['data']['metadata']['word_count']}")
                else:
                    failed_extractions.append({"url": url, "error": "Failed full-text validation"})
                    print("REJECTED: Failed full-text validation")
            else:
                failed_extractions.append({"url": url, "error": result.get("error", "Unknown error")})
                print(f"FAILED: {result.get('error', 'Unknown error')}")

        batch_results = {
            "extraction_summary": {
                "total_speeches_attempted": min(len(speech_entries), max_speeches),
                "successful_extractions": len(extracted_speeches),
                "failed_extractions": len(failed_extractions),
                "extraction_date": datetime.now().isoformat(),
            },
            "speeches": extracted_speeches,
            "failed_urls": failed_extractions,
            "analysis_ready": True,
        }

        self.save_analysis_dataset(batch_results)

        return batch_results

    def save_analysis_dataset(self, batch_results: Dict[str, Any]) -> List[str]:
        """Save extracted speeches in multiple analysis-ready formats"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files_created = []

        try:
            json_file = Path("sec_analysis/raw_data") / f"all_speeches_{timestamp}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(batch_results, f, indent=2, ensure_ascii=False)
            files_created.append(str(json_file))

            if batch_results["speeches"]:
                csv_data = []
                for speech in batch_results["speeches"]:
                    row = {
                        "title": speech["metadata"].get("title", ""),
                        "speaker": speech["metadata"].get("speaker", ""),
                        "date": speech["metadata"].get("date", ""),
                        "word_count": speech["metadata"].get("word_count", 0),
                        "url": speech["metadata"].get("url", ""),
                        "full_text": speech["content"].get("full_text", ""),
                        "paragraph_count": len(speech["content"].get("paragraphs", [])),
                        "sentence_count": len(speech["content"].get("sentences", [])),
                        "completeness_score": speech["validation"].get("completeness_score", 0),
                    }
                    csv_data.append(row)

                df = pd.DataFrame(csv_data)
                csv_file = Path("sec_analysis/raw_data") / f"speeches_dataset_{timestamp}.csv"
                df.to_csv(csv_file, index=False, encoding="utf-8")
                files_created.append(str(csv_file))

            print(f"\nAnalysis dataset saved in {len(files_created)} formats:")
            for file_path in files_created:
                print(f"  - {file_path}")

            return files_created

        except Exception as e:
            print(f"Error saving analysis dataset: {e}")
            return files_created


if __name__ == "__main__":
    print("SEC Speech Analyzer - Analysis-Optimized Extraction")
    print("=" * 60)

    analyzer = SECSpeechAnalyzer()

    print("\nThis system creates analysis-ready data for:")
    print("  - GenAI and sentiment analysis")
    print("  - Topic modeling and theme extraction")
    print("  - Entity recognition and network analysis")
    print("  - Cross-speech comparison and trend analysis")
    print("  - Statistical analysis with pandas/numpy")

    print("\nReady to extract all speeches for comprehensive analysis!")
    print("No API credits needed - using free scraping.")
