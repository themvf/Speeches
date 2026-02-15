#!/usr/bin/env python3
"""
Extract All SEC Commissioner Speeches for Analysis
Uses free scraping (no Firecrawl API needed).
"""

import json
from pathlib import Path
from speech_analyzer import SECSpeechAnalyzer
from sec_scraper_free import SECScraper


def discover_speech_entries(max_pages=5):
    """Discover speech entries (with metadata) from the SEC website"""
    scraper = SECScraper()
    print(f"Discovering speech URLs from SEC.gov (up to {max_pages} pages)...")
    speeches = scraper.discover_speech_urls(max_pages=max_pages)
    print(f"Found {len(speeches)} speech entries")
    return speeches


def check_existing_extractions():
    """Check for existing extracted speeches to avoid duplicates"""
    existing_urls = set()
    data_dir = Path("data")

    if data_dir.exists():
        json_files = list(data_dir.glob("all_speeches*.json"))
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for speech in data.get("speeches", []):
                    url = speech.get("metadata", {}).get("url", "")
                    if url:
                        existing_urls.add(url)
                print(f"Found {len(existing_urls)} existing URLs in {json_file.name}")
            except Exception as e:
                print(f"Error reading {json_file}: {e}")

    return existing_urls


def extract_all_speeches_for_analysis(max_pages=5, max_speeches=50):
    """Main function to extract all speeches for analysis"""

    print("SEC Commissioner Speeches - Complete Analysis Extraction")
    print("=" * 70)
    print("Using free scraping - no API credits needed\n")

    analyzer = SECSpeechAnalyzer()

    # Discover speech entries (with date/speaker metadata from listing page)
    speech_entries = discover_speech_entries(max_pages=max_pages)

    if not speech_entries:
        print("No speech entries found. Cannot proceed.")
        return

    # Check for existing extractions to avoid duplicates
    existing_urls = check_existing_extractions()

    # Filter out already-extracted speeches
    new_entries = [e for e in speech_entries if e["url"] not in existing_urls]
    print(f"\nNew speeches to extract: {len(new_entries)}")
    print(f"Already extracted: {len(existing_urls)}")

    if not new_entries:
        print("No new speeches to extract.")
        return

    # Perform batch extraction (entries include listing metadata for date/speaker fallback)
    print(f"\nStarting batch extraction of up to {max_speeches} speeches...")
    results = analyzer.batch_extract_all_speeches(new_entries, max_speeches=max_speeches)

    # Show results
    print(f"\n{'=' * 70}")
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    summary = results["extraction_summary"]
    print(f"Speeches extracted: {summary['successful_extractions']}")
    print(f"Failed extractions: {summary['failed_extractions']}")

    if results["speeches"]:
        total_words = sum(s["metadata"].get("word_count", 0) for s in results["speeches"])
        print(f"Total words extracted: {total_words:,}")
        print(f"Average words per speech: {total_words // len(results['speeches']):,}")

        print(f"\nExtracted Speeches:")
        for i, speech in enumerate(results["speeches"][:10], 1):
            title = speech["metadata"].get("title", "No title")
            speaker = speech["metadata"].get("speaker", "Unknown")
            words = speech["metadata"].get("word_count", 0)
            print(f"  {i}. {title[:50]}...")
            print(f"     Speaker: {speaker}, Words: {words:,}")

    return results


if __name__ == "__main__":
    results = extract_all_speeches_for_analysis()
