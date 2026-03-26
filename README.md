# SEC Commissioner Speeches Dashboard

Streamlit dashboard for exploring and analyzing SEC Commissioner speeches. Includes sentiment analysis, topic modeling, full-text search, and in-app speech extraction.

## Features

- **Overview** - Speech counts, word counts, and commissioner breakdown
- **Sentiment Analysis** - Keyword-based sentiment scoring per speech
- **Topic Analysis** - 6-category topic relevance (Digital Assets, Enforcement, Market Structure, Innovation, Investor Protection, Regulatory Framework)
- **Speech Explorer** - Full-text search and reading with filters
- **Extraction Workspace** - Discover and ingest SEC speeches plus connector documents from one admin section
- **Document Connectors** - Ingest SEC policy docs, DOJ USAO press releases, Federal Reserve speeches/testimony, and targeted NewsAPI articles into the shared knowledge base

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Vercel Migration Workspace

The Vercel migration scaffold now lives under `apps/web`.

```bash
cd apps/web
npm install
npm run dev
```

Migration planning documents:
- `docs/migration/vercel-migration-tracker.md`
- `docs/migration/api-contract-v1.md`

Initial Next.js API routes are available in `apps/web/app/api/*` for:
- metrics (`GET /api/metrics`)
- documents (`GET /api/documents`, `GET /api/documents/{id}`)
- jobs (`POST /api/jobs/ingest`, `POST /api/jobs/enrich`, `GET /api/jobs/{id}`)

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

Do not commit raw Google service-account JSON files to this repository. Keep the live credential only in local `.streamlit/secrets.toml` or in Streamlit Cloud Secrets.

Required for retrieval/index/chat:
- `[openai].api_key`

Optional connector key:
- `[newsapi].api_key` (enables the "News Connector: Financial Fraud & Policy" in Extraction)

## Scraping New Speeches

The scraper uses `curl_cffi` + `BeautifulSoup` to fetch speeches directly from SEC.gov (no API keys needed). `curl_cffi` is required because SEC.gov blocks standard HTTP clients via TLS fingerprinting.

**From the dashboard:** Use the "Extraction" page to run SEC speech extraction and document connectors.

**From the command line:**

```bash
python extract_all_speeches.py
```

## Project Structure

```text
app.py                  # Streamlit dashboard (with extraction UI)
gcs_storage.py          # Google Cloud Storage read/write
sec_scraper_free.py     # Free SEC.gov scraper (curl_cffi)
doj_usao_press_release_scraper.py  # DOJ USAO press-release scraper
federal_reserve_speech_testimony_scraper.py  # Federal Reserve speeches/testimony scraper
newsapi_financial_scraper.py  # NewsAPI discovery + article extractor
speech_analyzer.py      # Speech parsing and extraction
analysis_pipeline.py    # Sentiment, topic, commissioner analysis
extract_all_speeches.py # Batch extraction script (CLI)
data/                   # Local speech dataset (JSON fallback)
.streamlit/             # Streamlit config and secrets template
```
