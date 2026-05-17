# Bug Fixes & Improvements Log

Reference document covering all fixes, hardening, and automation work applied to this codebase. Organized by category.

---

## Silent Failure Fixes

### GCS Loader (`apps/web/lib/server/gcs-loader.ts`)
- Both `catch` blocks in `downloadGcsJson` and `uploadGcsJson` now log via `console.error` instead of swallowing errors silently.
- `cachedError` is updated on download failure so callers know the reason.

### Python Pipelines
- `run_connector_extraction_pipeline.py`: Added `import sys`; two bare `except: pass` blocks in `_load_existing_speech_url_keys` now log to stderr.
- `sync_knowledge_index.py`: `except Exception: pass` on vector store delete now logs a warning to stderr.
- `run_financial_news_pipeline.py`: Local file fallback `except Exception: pass` now logs to stderr via `_stderr()`.

### Admin API Routes
- `apps/web/app/admin/page.tsx`: All mutation handlers (`handleToggle`, `handleDelete`, `handleSave`) now wrap fetches in try/catch, check `res.ok`, and surface errors to the UI. Optimistic updates only apply after a confirmed success.
- `apps/web/app/api/metrics/route.ts`: Removed `custom.updated_at` from `processedCount` fallback chain (was inflating counts).

### RSS Refresh (`apps/web/app/api/intel/rss-refresh/route.ts`)
- Previously always returned `ok: true`. Now returns `ok: false` and HTTP 500 when every feed fails.

### Recap Generation (`apps/web/app/api/intel/recap/generate/route.ts`)
- Switched `Promise.all` → `Promise.allSettled` so a single topic failure doesn't abort the entire recap.
- Failed topics are logged to stderr.
- Corpus date filter fixed to check `published_date || date` (was missing `published_date`, causing corpus docs to never appear in recaps).

---

## Security Fixes

### Admin Cookie (`apps/web/app/api/admin/login/route.ts`, `apps/web/middleware.ts`)
- Cookie previously stored the raw `ADMIN_SECRET` value in plaintext.
- Login route now stores a SHA-256 hash of the secret.
- Middleware rewritten as `async` to use `crypto.subtle.digest` (Edge Runtime compatible) for hash comparison.
- Existing sessions were invalidated on deploy (users must re-login).

### Input Validation
- `apps/web/app/api/chats/ask/route.ts`: Prompt capped at 4000 chars; history sliced to last 6 messages; error message sanitized (no internal details leaked).
- `apps/web/app/api/search/route.ts`: Query capped at 2000 chars.
- `apps/web/lib/server/api-utils.ts`: `parseDate` now requires strict `YYYY-MM-DD` format — rejects freeform strings like `"Mon"` or `"3"`.
- `apps/web/app/api/admin/ticker/validate/route.ts`: JSON parse wrapped in try/catch; added empty string check.
- `apps/web/app/api/admin/ticker/route.ts`: POST validates each entry for non-empty symbol and name; normalizes to uppercase.
- `apps/web/app/api/documents/route.ts`: `docIds` param capped at 100 entries.
- `apps/web/app/api/intel/recap/route.ts`: Date param validated against `YYYY-MM-DD` regex before use.

### SSRF Prevention (`apps/web/app/api/admin/workflow/runs/route.ts`)
- Added `ALLOWED_WORKFLOWS` set (12 known workflow filenames).
- Returns 400 for any workflow not in the allowlist.
- Replaced hardcoded `REPO = "themvf/Speeches"` with `getGithubActionsConfig()` in `workflow/route.ts` and `workflow/cancel/route.ts`.

---

## Data Correctness Fixes

### Corpus Date Filtering
- `apps/web/app/api/intel/recap/generate/route.ts`: Date comparison now uses `published_date || date` field, matching how corpus documents store their dates.

### Article Date Filtering (`apps/web/lib/server/neon.ts`)
- `getRecentArticles`: Time-bounded queries now use `COALESCE(published_at, fetched_at)` in both `WHERE` and `ORDER BY` for correct date ordering when `published_at` is null.

### Trends Cutoff (`apps/web/app/api/trends/route.ts`)
- Filter now compares millisecond timestamps (`new Date(t.last_seen).getTime() >= cutoffMs`) instead of string comparison.

### Feed Latest Timestamp (`apps/web/app/api/intel/feed/route.ts`)
- `latestFetchedAt` computed via proper `reduce` over article timestamps rather than a potentially incorrect method.

### RSS GUID Deduplication (`apps/web/lib/server/rss-fetcher.ts`)
- `normalizeGuid` now generates a deterministic hash from `title + url` when neither a `guid` nor a `link` is present, preventing duplicate articles on re-ingestion.

### LLM Model (`trend_aggregation.py`)
- `CHAT_MODEL` was set to nonexistent `"gpt-5.1"`. Fixed to `"gpt-4o"`.
- `CHAT_MODEL_FALLBACKS` updated to `["gpt-4o-mini"]` (removed `gpt-4o` since it's now the primary).
- Placeholder trend description fallback changed from a generated string to `""` to avoid polluting output with fake descriptions.

---

## Performance & Reliability Fixes

### Fetch Timeouts (all server-side external calls)
Every outbound `fetch` call now uses `AbortController` to prevent Vercel function timeouts:

| File | Timeout | Upstream |
|---|---|---|
| `lib/server/rss-fetcher.ts` | 10s | RSS feeds |
| `lib/server/yahoo.ts` (quote + candles) | 8s | Yahoo Finance |
| `market/bonds/route.ts` | 8s | US Treasury |
| `market/chart/route.ts` | 8s | Yahoo Finance / CoinGecko |
| `market/commodities/route.ts` | 8s | Yahoo Finance (8 parallel) |
| `market/crypto/route.ts` | 8s | CoinGecko |
| `lib/server/openai-chat.ts` | 90s | OpenAI Responses API |
| `admin/workflow/route.ts` | 10s | GitHub API |
| `admin/workflow/cancel/route.ts` | 10s | GitHub API |

Note: `lib/server/gdelt-doc.ts` and `lib/server/gdelt-gkg.ts` already had `AbortController` timeouts.

### Frontend Polling Backoff (`apps/web/components/intelbeta-dashboard.tsx`)
- Replaced `setInterval` with recursive `setTimeout` + exponential backoff.
- On error: 15s → 30s → 60s → 120s cap. Resets to 15s on success.
- Prevents flood of requests when the feed endpoint is temporarily unavailable.

### Job Status Polling (`apps/web/components/job-status-badge.tsx`)
- Error branches now schedule a 30s retry instead of stopping polling permanently.

### RSS Feed Size Cap (`apps/web/lib/server/rss-fetcher.ts`)
- Feeds larger than 2 MB now throw immediately rather than parsing a potentially huge XML document.

### Document Cache Invalidation (`apps/web/lib/server/data-store.ts`)
- Added `invalidateDocumentCaches()` which clears `"sec_speeches"` and `"custom_documents"` cache keys.
- Called after successful document upload/delete in admin routes.

---

## Workflow & CI Fixes

### `financial-news-daily.yml`
- **DST double-fire**: Replaced NY-timezone hour check with explicit UTC hour conditions matching each cron trigger.
- **Enrichment step condition**: Now checks `processed_count != '' && processed_count != '0'` to skip enrichment when no articles were ingested.

### `knowledge-index-sync.yml`
- Was `pip install openai google-cloud-storage` — changed to `pip install -r requirements.txt` for full dependency parity with production runs.
- Added `set -euo pipefail` to the run step.
- Added `workflow_run` trigger watching "SEC Speech Sync (Scheduled)" and "Policy Extraction (On Demand)" so the knowledge index rebuilds automatically after new corpus documents land.

### `intelligence-evidence.yml`
- **Schedule**: Was `*/5 * * * *` (288 runs/day) — changed to `0 * * * *` (24 runs/day).
- **Hardcoded URL**: `AML_EVIDENCE_ENDPOINT_URL` had `?category=AML` baked in, causing the Capital Formation smoke test to always query the AML endpoint and fail its category assertion. Fixed by switching to `EVIDENCE_ENDPOINT_BASE` so each test appends the correct `?category=` param.
- Added `Lint` step (runs ESLint before Typecheck on push/PR).
- Added `Smoke Test Core App Endpoints` step (runs hourly against production).

### `python-tests.yml` (new)
- Runs all 60 Python unit tests on push/PR whenever `.py` files or `requirements.txt` change.
- `pytest` installed as a separate step (dev-only dependency, not in `requirements.txt`).

---

## Monitoring & Health Checks

### Daily Health Check (`run_daily_health_check.py`, `.github/workflows/daily-health-check.yml`)
- New script and workflow that runs daily at 09:00 UTC.
- Checks: GitHub Actions workflow failures (last 25h), enrichment failures, GCS connectivity, RSS feed refresh.
- Creates a GitHub Issue when failures are detected; closes cleanly when resolved.
- **Enrichment failure window**: Only counts failures with `updated_at` in the last 25h as "new" to avoid daily issue spam from historical failures.

### GitHub Secrets Added
| Secret | Purpose |
|---|---|
| `APP_URL` | Production URL for health check RSS test |
| `CRON_SECRET` | Bearer token for `/api/intel/rss-refresh` auth |

---

## ESLint Setup

### `apps/web/eslint.config.mjs` (new)
- Flat config using `@next/eslint-plugin-next` (`coreWebVitals`) and `@typescript-eslint` (`no-unused-vars`).
- Zero-warning tolerance enforced via `--max-warnings 0`.

### `apps/web/package.json`
- `lint` script changed from deprecated `next lint` to `eslint . --max-warnings 0`.

### Dead Code Removed
13 pre-existing unused-variable warnings fixed across 5 files:
- `enforcement/beta/route.ts`: `_agencyKey`
- `enforcement-beta-dashboard.tsx`: `_monthDateRange`, `_HeatmapPanel`, removed `visibleAgencies` useMemo
- `finra-dashboard.tsx`: removed unused `useRef` import
- `trends-dashboard.tsx`: `_MoverCard`, removed `showMovers`, `topRisers`, `topDecliners`
- `lib/server/gdelt-doc.ts`: `_buildOrQuery`, `_mapGdeltDocArticlesToKnownFocusAreaEvidence`

---

## Smoke Tests Added

### `apps/web/scripts/check-app-endpoints.ts` (new)
Runs hourly via `intelligence-evidence.yml` against production:

| Endpoint | Severity |
|---|---|
| `GET /api/intel/feed` | Critical |
| `GET /api/trends` | Critical |
| `GET /api/metrics` | Critical |
| `GET /api/market/crypto` | Critical |
| `GET /api/market/bonds` | Critical |
| `GET /api/intel/recap` | Critical |
| `GET /api/search` | Warn-only (OpenAI / vector store can be slow) |

Exit code 1 only on critical failures; search timeout surfaces as a warning.

---

## Settings Fix

### `apps/web/app/api/settings/connectors/news/route.ts`
- `doj_usao_exclude_terms` field was missing from the payload saved to GCS. Added to normalisation.
