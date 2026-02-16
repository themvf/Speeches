# SEC Commissioner Speeches Dashboard

Streamlit dashboard for exploring and analyzing SEC Commissioner speeches. Includes sentiment analysis, topic modeling, full-text search, and in-app speech extraction.

## Features

- **Overview** — Speech counts, word counts, and commissioner breakdown
- **Sentiment Analysis** — Keyword-based sentiment scoring per speech
- **Topic Analysis** — 6-category topic relevance (Digital Assets, Enforcement, Market Structure, Innovation, Investor Protection, Regulatory Framework)
- **Speech Explorer** — Full-text search and reading with filters
- **Extract Speeches** — Discover and ingest new speeches by date range directly from the dashboard

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app works out of the box with the included local dataset. For persistent cloud storage (required for in-app extraction on Streamlit Cloud), set up Google Cloud Storage below.

## Google Cloud Storage Setup

Streamlit Cloud has an ephemeral filesystem, so extracted speeches need persistent storage. The app uses a GCS bucket.

### 1. Create a GCS Bucket

```bash
gsutil mb -l us-central1 gs://your-bucket-name
```

### 2. Create a Service Account

```bash
# Create service account
gcloud iam service-accounts create sec-speeches \
    --display-name="SEC Speeches App"

# Grant Storage Object Admin on the bucket
gsutil iam ch serviceAccount:sec-speeches@YOUR_PROJECT.iam.gserviceaccount.com:objectAdmin \
    gs://your-bucket-name

# Generate JSON key
gcloud iam service-accounts keys create key.json \
    --iam-account=sec-speeches@YOUR_PROJECT.iam.gserviceaccount.com
```

### 3. Upload Initial Dataset

```bash
gsutil cp data/all_speeches_final.json gs://your-bucket-name/all_speeches.json
```

### 4. Configure Secrets

**Locally:** Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in your service account key values.

**Streamlit Cloud:** Go to your app's Settings > Secrets and paste the contents of your `secrets.toml`.

## Scraping New Speeches

The scraper uses `curl_cffi` + `BeautifulSoup` to fetch speeches directly from SEC.gov (no API keys needed). `curl_cffi` is required because SEC.gov blocks standard HTTP clients via TLS fingerprinting.

**From the dashboard:** Use the "Extract Speeches" page to pick a date range, discover available speeches, and extract them.

**From the command line:**

```bash
python extract_all_speeches.py
```

## Project Structure

```
app.py                  # Streamlit dashboard (with extraction UI)
gcs_storage.py          # Google Cloud Storage read/write
sec_scraper_free.py     # Free SEC.gov scraper (curl_cffi)
speech_analyzer.py      # Speech parsing and extraction
analysis_pipeline.py    # Sentiment, topic, commissioner analysis
extract_all_speeches.py # Batch extraction script (CLI)
data/                   # Local speech dataset (JSON fallback)
.streamlit/             # Streamlit config and secrets template
```
