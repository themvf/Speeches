# SEC Intelligence Console (Web)

Next.js frontend for the Streamlit-to-Vercel migration.

## Local Run

```bash
cd apps/web
npm install
cp .env.example .env.local
npm run dev
```

## Environment Variables

Primary config lives in `.env.local`.

- `DATA_SOURCE_MODE`: `auto`, `local`, or `gcs`
- `DATA_DIR_PATH`: local path fallback for JSON stores
- `GCS_BUCKET_NAME`, `GCS_CREDENTIALS_JSON`, `GCS_CREDENTIALS_PATH`: GCS-backed reads
- `GITHUB_ACTIONS_*`: dispatch and status for ingest/enrich/extract workflows
- `JOB_EXECUTION_MODE`: `github_actions` (default) or `local` for direct Python extraction
- `PYTHON_BIN`: python executable for local extraction mode (`python` by default)

Use `apps/web/.env.example` as the template.

If you want manual extraction without GitHub Actions, set `JOB_EXECUTION_MODE=local`. In that mode the API runs `run_connector_extraction_pipeline.py` directly and requires Python + dependencies in the runtime environment.

## Implemented API Routes (V1)

- `GET /api/metrics`
- `GET /api/documents`
- `GET /api/documents/{documentId}`
- `POST /api/jobs/ingest`
- `POST /api/jobs/enrich`
- `GET /api/jobs/{jobId}`

## Build Check

```bash
npm run build
```

## FINRA Comment-Letter Pilot (Notice 26-06)

```bash
python run_connector_extraction_pipeline.py ^
  --connector finra_comment_letter ^
  --base-url https://www.finra.org/rules-guidance/notices/26-06 ^
  --selection all ^
  --limit 50 ^
  --include-pdfs true ^
  --dry-run
```

Use the same command without `--dry-run` to persist results. If GCS credentials are configured and valid, records are written to `custom_documents.json` in your bucket.

## Deployment

Deploy as Vercel preview first, then promote after parity checks.
