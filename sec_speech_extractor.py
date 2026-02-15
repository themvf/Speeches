#!/usr/bin/env python3
"""
SEC Commissioner Speech Extractor
Extracts complete full-text speeches with preservation of formatting.
Uses free scraping (requests + BeautifulSoup) instead of Firecrawl.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from sec_scraper_free import SECScraper


class SECSpeechExtractor:
    def __init__(self):
        self.scraper = SECScraper()
        self.create_directories()

    def create_directories(self):
        """Create organized directory structure for speeches"""
        base_dir = Path("sec_speeches")

        directories = [
            "full_text",
            "structured_data",
            "by_commissioner",
            "by_topic",
            "by_date",
            "analysis",
            "summaries",
        ]

        for directory in directories:
            (base_dir / directory).mkdir(parents=True, exist_ok=True)

    def extract_full_speech(self, url):
        """
        Extract complete full-text speech from SEC URL.

        Args:
            url (str): URL of the commissioner speech

        Returns:
            dict: Complete speech data with validation metrics
        """
        print(f"\nExtracting full speech from:")
        print(f"URL: {url}")

        result = self.scraper.scrape_page(url, formats=["markdown", "html"])

        if result["success"]:
            print("SUCCESS: Speech extracted!")

            validation = self.validate_speech_completeness(result)

            print(f"Validation Results:")
            print(f"  Word count: {validation['word_count']}")
            print(f"  Likely complete: {validation['likely_complete']}")
            print(f"  Has structure: {validation['has_structure']}")

            result["validation"] = validation
            result["extraction_method"] = "free_scraping"
            result["url"] = url

            return result
        else:
            print(f"FAILED: {result['error']}")
            return result

    def validate_speech_completeness(self, result):
        """Validate that we got the complete speech, not excerpts"""

        full_text = result["data"]["data"].get("markdown", "")

        word_count = len(full_text.split())
        char_count = len(full_text)

        text_lower = full_text.lower()

        has_intro = any(
            phrase in text_lower[:1000]
            for phrase in [
                "thank you", "good morning", "good afternoon", "good evening",
                "pleased to be", "honored to speak", "delighted to join",
            ]
        )

        has_conclusion = any(
            phrase in text_lower[-1000:]
            for phrase in [
                "thank you", "questions", "conclusion", "in closing",
                "let me close", "in summary", "to conclude",
            ]
        )

        has_paragraphs = full_text.count("\n\n") > 5
        has_headings = full_text.count("##") > 0 or full_text.count("#") > 0

        likely_complete = word_count > 500 and (has_intro or has_conclusion) and has_paragraphs

        return {
            "word_count": word_count,
            "char_count": char_count,
            "has_intro": has_intro,
            "has_conclusion": has_conclusion,
            "has_paragraphs": has_paragraphs,
            "has_headings": has_headings,
            "likely_complete": likely_complete,
            "has_structure": has_intro and has_conclusion,
            "completeness_score": self.calculate_completeness_score(
                word_count, has_intro, has_conclusion, has_paragraphs
            ),
        }

    def calculate_completeness_score(self, word_count, has_intro, has_conclusion, has_paragraphs):
        """Calculate completeness score from 0-100"""
        score = 0

        if word_count > 2000:
            score += 40
        elif word_count > 1000:
            score += 30
        elif word_count > 500:
            score += 20
        else:
            score += 10

        if has_intro:
            score += 20
        if has_conclusion:
            score += 20
        if has_paragraphs:
            score += 20

        return min(score, 100)

    def save_full_speech(self, speech_data, url):
        """Save complete speech in multiple formats"""

        url_parts = url.split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{url_parts}_{timestamp}"

        files_saved = []

        try:
            json_path = Path("sec_speeches/structured_data") / f"{base_filename}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(speech_data, f, indent=2, ensure_ascii=False)
            files_saved.append(str(json_path))

            full_text = speech_data["data"]["data"].get("markdown", "")
            metadata = speech_data["data"]["data"].get("metadata", {})
            title = metadata.get("title", "Speech")

            markdown_content = f"""# {title}

**URL**: {url}
**Extraction Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Word Count**: {speech_data["validation"]["word_count"]}
**Completeness Score**: {speech_data["validation"]["completeness_score"]}/100

---

{full_text}

---

*Extracted using free SEC scraper*
"""

            md_path = Path("sec_speeches/full_text") / f"{base_filename}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            files_saved.append(str(md_path))

            print(f"\nSpeech saved in {len(files_saved)} formats:")
            for file_path in files_saved:
                print(f"  - {file_path}")

            return files_saved

        except Exception as e:
            print(f"Error saving speech: {e}")
            return []


if __name__ == "__main__":
    print("SEC Commissioner Speech - Full Text Extraction Test")
    print("=" * 60)

    extractor = SECSpeechExtractor()

    test_url = "https://www.sec.gov/newsroom/speeches-statements/atkins-digital-finance-revolution-073125"
    print(f"\nExtracting: {test_url}")

    result = extractor.extract_full_speech(test_url)

    if result["success"]:
        files = extractor.save_full_speech(result, test_url)
        print(f"\nSaved {len(files)} files")
