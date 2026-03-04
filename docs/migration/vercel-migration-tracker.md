# Streamlit -> Vercel Migration Tracker

## Goal
Deliver a production-ready Next.js app on Vercel with better operator UX than Streamlit, while preserving existing ingestion and enrichment behavior.

## Constraints
- Keep Python ingestion/enrichment workers in place during phase 1.
- Do not run long scraping or enrichment synchronously from Vercel request handlers.
- Maintain GCS-backed persistence as source of truth for phase 1.

## Phase Plan

| Phase | Outcome | Target | Exit Criteria |
|---|---|---|---|
| P0 Foundation | New web app, design system, env model | Week 1 | Preview deploy stable and team can run locally |
| P1 Read UX | Dashboard, explorer, library, job center read views | Week 2 | Feature parity for read workflows |
| P2 Operator UX | Extraction controls, enrichment review, policy delta | Week 3-4 | Admin workflows complete in new UI |
| P3 Cutover | Parallel run and production switchover | Week 5-6 | No P1 regressions, Streamlit retired |

## Ticket Backlog

| ID | Ticket | Status | Depends On | Acceptance Criteria |
|---|---|---|---|---|
| MIG-001 | Scaffold Next.js app in `apps/web` | Complete | None | App boots locally and builds on Vercel |
| MIG-002 | Define design tokens and layout shell | Complete | MIG-001 | Colors, typography, spacing, and shell used by all pages |
| MIG-003 | Define environment/secrets contract | Complete | MIG-001 | Required env vars documented with startup validation |
| MIG-004 | Extract data-access logic from Streamlit into reusable services | Complete | MIG-003 | Core reads/writes callable without `st.*` |
| MIG-005 | Implement `GET /api/metrics` | Complete | MIG-004 | Returns overview KPIs used by dashboard cards |
| MIG-006 | Implement `GET /api/documents` with filtering/pagination | Complete | MIG-004 | Supports org/source-kind/text/date filters |
| MIG-007 | Implement `GET /api/documents/{id}` | Complete | MIG-006 | Returns metadata, content, and enrichment summary |
| MIG-008 | Implement job trigger API (`POST /api/jobs/ingest`, `POST /api/jobs/enrich`) | Complete | MIG-004 | Starts background run and returns job id |
| MIG-009 | Implement job status API (`GET /api/jobs/{id}`) | Complete | MIG-008 | Returns running/success/failure and latest logs |
| MIG-010 | Build Overview page | Todo | MIG-005 | KPI cards and trend summaries render from API |
| MIG-011 | Build Corpus Explorer page | Todo | MIG-006, MIG-007 | Table + detail panel + deep-linkable filters |
| MIG-012 | Build Document Library page | Todo | MIG-006, MIG-007 | Browse, inspect, and delete flow with guardrails |
| MIG-013 | Build Extraction Workspace page | Todo | MIG-008, MIG-009 | Configure and trigger connectors with status panel |
| MIG-014 | Build Enrichment Review page | Todo | MIG-007, MIG-009 | Queue, evidence view, and review actions |
| MIG-015 | Build Policy Delta Brief page | Todo | MIG-007 | Generate and display policy-delta outputs |
| MIG-016 | Add auth + RBAC for admin routes | Todo | MIG-013 | Mutating endpoints blocked for non-admin users |
| MIG-017 | Add audit logging for review/delete actions | Todo | MIG-016 | Audit entries persisted with actor and timestamp |
| MIG-018 | Add telemetry (Sentry + structured logs) | Todo | MIG-003 | Errors and job failures observable by request id |
| MIG-019 | Run one-week parallel validation | Todo | MIG-010..MIG-018 | No critical parity gaps, signoff recorded |
| MIG-020 | Production cutover to Vercel | Todo | MIG-019 | Traffic switched and Streamlit decommission date set |
| MIG-021 | Add dedicated manual GitHub workflows for ingest and enrich | Complete | MIG-008 | `financial-news-ingest.yml` and `financial-news-enrich.yml` dispatch successfully |

## Roles
- Product Owner: Prioritize migration tickets and sign off on parity.
- Frontend Engineer: Implement `apps/web` pages and UX states.
- Backend Engineer: Build API endpoints and job orchestration.
- Data Engineer: Maintain Python worker reliability and schedules.

## Definition of Done
- Feature has tests for primary path plus one error state.
- UX includes loading, empty, and failure states.
- Endpoint and payload documented in `docs/migration/api-contract-v1.md`.
- Change is running in Vercel preview environment.
