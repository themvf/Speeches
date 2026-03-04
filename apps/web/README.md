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
- `GITHUB_ACTIONS_*`: dispatch and status for ingest/enrich workflows

Use `apps/web/.env.example` as the template.

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

## Deployment

Deploy as Vercel preview first, then promote after parity checks.