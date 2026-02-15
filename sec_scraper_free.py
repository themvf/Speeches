#!/usr/bin/env python3
"""
SEC Speech Scraper - Free replacement for Firecrawl API
Uses curl_cffi + BeautifulSoup + markdownify to scrape SEC.gov public pages.
curl_cffi is needed because SEC.gov blocks standard requests (TLS fingerprinting).
"""

import time
import re
from curl_cffi import requests as cffi_requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin


class SECScraper:
    """Free SEC.gov scraper that maintains the same interface as FirecrawlHelper."""

    def __init__(self):
        self.session = cffi_requests.Session(impersonate="chrome")
        self._last_request_time = 0
        self._min_delay = 1.0  # seconds between requests

    def _rate_limit(self):
        """Enforce rate limiting to be respectful to SEC.gov."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_delay:
            time.sleep(self._min_delay - elapsed)
        self._last_request_time = time.time()

    def scrape_page(self, url: str, formats: Optional[List[str]] = None, **options) -> Dict[str, Any]:
        """
        Scrape a single page and return markdown content.

        Maintains the same return format as FirecrawlHelper.scrape_page() so that
        speech_analyzer.py and sec_speech_extractor.py work without changes.

        Returns:
            dict with structure: {"success": True, "data": {"data": {"content": "...", "markdown": "..."}}}
        """
        self._rate_limit()

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script, style, nav, footer elements
            for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # Try to find the main content area (SEC pages use article or main tags)
            main_content = (
                soup.find("article")
                or soup.find("main")
                or soup.find("div", class_="article-page")
                or soup.find("div", id="main-content")
                or soup.body
            )

            if main_content is None:
                main_content = soup

            # Convert to markdown
            markdown_content = md(
                str(main_content),
                heading_style="ATX",
                bullets="-",
                strip=["img"],
            )

            # Clean up excessive whitespace
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
            markdown_content = markdown_content.strip()

            # Also get raw HTML for the html format
            html_content = str(main_content)

            # Build metadata
            title_tag = soup.find("title")
            meta_desc = soup.find("meta", attrs={"name": "description"})

            metadata = {
                "title": title_tag.get_text(strip=True) if title_tag else "",
                "description": meta_desc["content"] if meta_desc and meta_desc.get("content") else "",
                "sourceURL": url,
            }

            return {
                "success": True,
                "data": {
                    "data": {
                        "content": markdown_content,
                        "markdown": markdown_content,
                        "html": html_content,
                        "metadata": metadata,
                    }
                },
                "credits_used": 0,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Request failed: {str(e)}",
                "credits_used": 0,
            }

    def discover_speech_urls(self, base_url: str = "https://www.sec.gov/newsroom/speeches-statements", max_pages: int = 5) -> List[Dict[str, str]]:
        """
        Discover speech URLs from the SEC speeches listing page.

        Returns:
            List of dicts with 'url', 'title', 'date', 'speaker', 'type' keys.
        """
        speeches = []
        seen_urls = set()

        for page in range(max_pages):
            self._rate_limit()
            page_url = f"{base_url}?page={page}" if page > 0 else base_url

            try:
                response = self.session.get(page_url, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # The SEC listing page uses a table with rows for each speech
                rows = soup.find_all("tr")
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) < 3:
                        continue

                    # Find the link in this row
                    link = row.find("a", href=True)
                    if not link:
                        continue
                    href = link.get("href", "")
                    if "/newsroom/speeches-statements/" not in href:
                        continue

                    full_url = urljoin("https://www.sec.gov", href)
                    if full_url in seen_urls:
                        continue
                    seen_urls.add(full_url)

                    # Extract date, title, speaker, type from table cells
                    date_text = cells[0].get_text(strip=True) if len(cells) > 0 else ""
                    title_text = link.get_text(strip=True)
                    speaker_text = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                    type_text = cells[3].get_text(strip=True) if len(cells) > 3 else ""

                    speeches.append({
                        "url": full_url,
                        "title": title_text,
                        "date": date_text,
                        "speaker": speaker_text,
                        "type": type_text,
                    })

                print(f"  Page {page + 1}: found {len(speeches)} speeches so far")

            except Exception as e:
                print(f"  Error on page {page + 1}: {e}")
                break

        return speeches

    def get_credit_status(self) -> Dict[str, Any]:
        """Compatibility stub - no credits needed with free scraping."""
        return {
            "remaining_credits": float("inf"),
            "monthly_limit": float("inf"),
            "current_usage": 0,
        }

    def print_credit_status(self):
        """Compatibility stub."""
        print("Using free scraper - no credit limits.")


# Convenience alias
FirecrawlHelper = SECScraper


if __name__ == "__main__":
    print("SEC Free Scraper - Test")
    print("=" * 50)

    scraper = SECScraper()

    # Test scraping a single speech
    test_url = "https://www.sec.gov/newsroom/speeches-statements/atkins-digital-finance-revolution-073125"
    print(f"\nScraping: {test_url}")

    result = scraper.scrape_page(test_url)

    if result["success"]:
        content = result["data"]["data"]["content"]
        print(f"Success! Content length: {len(content)} chars")
        print(f"First 200 chars:\n{content[:200]}...")
    else:
        print(f"Failed: {result['error']}")
