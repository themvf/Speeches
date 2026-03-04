# API Contract V1 (Migration)

This contract defines the first stable API between the new Next.js UI and backend services.

## Conventions
- JSON request and response bodies.
- All timestamps in ISO 8601 UTC (e.g., `2026-03-04T20:00:00Z`).
- Errors use the same envelope:

```json
{
  "ok": false,
  "error": "Human-readable message",
  "code": "MACHINE_CODE",
  "request_id": "req_123"
}
```

## 1) Dashboard Metrics

### `GET /api/metrics`
Returns high-level summary cards and trend slices.

Response:
```json
{
  "ok": true,
  "data": {
    "totals": {
      "documents": 1567,
      "organizations": 7,
      "enriched": 480,
      "pending_review": 38
    },
    "recent_ingest": {
      "last_run_at": "2026-03-04T19:55:28Z",
      "processed_count": 12,
      "failed_count": 1
    },
    "by_source_kind": [
      { "source_kind": "sec_speech", "count": 1021 },
      { "source_kind": "newsapi_article", "count": 412 }
    ]
  }
}
```

## 2) Documents List

### `GET /api/documents`
Query params:
- `q` string full-text query.
- `org` organization key or label.
- `source_kind` connector source kind.
- `status` enrichment status (`not_enriched`, `enriched`, `reviewed`, `fallback_enriched`).
- `date_from`, `date_to` ISO date.
- `page` integer, default `1`.
- `page_size` integer, default `25`, max `100`.
- `sort` one of `date_desc`, `date_asc`, `updated_desc`.

Response:
```json
{
  "ok": true,
  "data": {
    "items": [
      {
        "document_id": "doc_abc123",
        "title": "SEC Charges ...",
        "organization": "SEC",
        "source_kind": "newsapi_article",
        "url": "https://example.com/article",
        "published_at": "2026-03-03T14:20:00Z",
        "ingest_status": "existing",
        "enrichment_status": "enriched"
      }
    ],
    "page": 1,
    "page_size": 25,
    "total": 234
  }
}
```

## 3) Document Detail

### `GET /api/documents/{document_id}`
Response:
```json
{
  "ok": true,
  "data": {
    "metadata": {
      "document_id": "doc_abc123",
      "title": "...",
      "organization": "SEC",
      "source_kind": "newsapi_article",
      "url": "https://...",
      "published_at": "2026-03-03T14:20:00Z"
    },
    "content": {
      "full_text": "..."
    },
    "enrichment": {
      "status": "enriched",
      "summary": "...",
      "tags": ["fraud", "enforcement"],
      "keywords": ["SEC", "wire fraud"],
      "evidence_spans": []
    },
    "review": {
      "decision": "accepted",
      "notes": ""
    }
  }
}
```

## 4) Trigger Ingest Job

### `POST /api/jobs/ingest`
Body:
```json
{
  "limit": 10,
  "lookback_days": 7,
  "selection": "new_or_updated",
  "require_remote_persistence": true
}
```

Response:
```json
{
  "ok": true,
  "data": {
    "job_id": "gha_123456789",
    "provider": "github_actions",
    "status": "queued",
    "status_url": "/api/jobs/gha_123456789",
    "github_run_id": 123456789
  }
}
```

## 5) Trigger Enrichment Job

### `POST /api/jobs/enrich`
Body:
```json
{
  "mode": "only_missing_or_failed",
  "limit": 25,
  "heuristic_only": false,
  "source_kind": "newsapi_article"
}
```

Response shape is identical to ingest trigger.

## 6) Job Status

### `GET /api/jobs/{job_id}`
Response:
```json
{
  "ok": true,
  "data": {
    "job_id": "gha_123456789",
    "provider": "github_actions",
    "workflow": "Financial News Ingest (On Demand)",
    "status": "running",
    "github_run_id": 123456789,
    "html_url": "https://github.com/org/repo/actions/runs/123456789",
    "created_at": "2026-03-04T19:55:28Z",
    "started_at": "2026-03-04T19:55:30Z",
    "updated_at": "2026-03-04T19:56:01Z",
    "finished_at": "",
    "conclusion": "",
    "artifacts": []
  }
}
```

## 7) Connector Settings

### `GET /api/settings/connectors/news`
Returns normalized connector settings currently stored in GCS/local fallback.

### `PUT /api/settings/connectors/news`
Body fields:
- `query`, `lookback_days`, `max_pages`, `page_size`, `target_count`, `sort_by`
- `domains`, `exclude_domains`, `tags_csv`, `organization_label`

Response:
```json
{
  "ok": true,
  "data": {
    "saved": true,
    "settings": {
      "query": "...",
      "lookback_days": 7
    }
  }
}
```

## 8) Review Actions

### `POST /api/review/{document_id}`
Body:
```json
{
  "decision": "accepted",
  "notes": "Reviewed against source article."
}
```

Response:
```json
{
  "ok": true,
  "data": {
    "document_id": "doc_abc123",
    "decision": "accepted",
    "saved_at": "2026-03-04T20:15:00Z"
  }
}
```

## Security Notes
- Admin-only routes: all `POST`/`PUT`/`DELETE` routes above.
- Add request-level audit fields: `actor_id`, `actor_email`, `request_id`.
- Never return raw secret values in response or logs.
