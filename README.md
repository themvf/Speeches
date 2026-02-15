# SEC Commissioner Speeches Dashboard

Streamlit dashboard for exploring and analyzing SEC Commissioner speeches. Includes sentiment analysis, topic modeling, and full-text search.

## Features

- **Overview** — Speech counts, word counts, and commissioner breakdown
- **Sentiment Analysis** — Keyword-based sentiment scoring per speech
- **Topic Analysis** — 6-category topic relevance (Digital Assets, Enforcement, Market Structure, Innovation, Investor Protection, Regulatory Framework)
- **Speech Explorer** — Full-text search and reading with filters

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Scraping New Speeches

The scraper uses `requests` + `BeautifulSoup` to fetch speeches directly from SEC.gov (no API keys needed):

```bash
python extract_all_speeches.py
```

## Project Structure

```
app.py                  # Streamlit dashboard
sec_scraper_free.py     # Free SEC.gov scraper
speech_analyzer.py      # Speech parsing and extraction
analysis_pipeline.py    # Sentiment, topic, commissioner analysis
extract_all_speeches.py # Batch extraction script
data/                   # Speech dataset (JSON)
```
