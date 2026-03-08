# Connector Readiness Checklist

Use this checklist before treating a connector as complete.

## Required Metadata

- Every extracted document must persist a stable `document_id`.
- Every extracted document must persist `source_kind`.
- Connector-specific fields required for downstream views must survive normalization and API serialization.
- Documents that should be enrichable must include non-empty `full_text`.

## Pipeline Coverage

- Extraction writes the connector output into `custom_documents.json`.
- Enrichment candidate building includes the connector's documents.
- Enrichment state persists `source_kind` and any connector-specific enrichment fields.
- Review tooling can filter and inspect the connector's documents directly.
- Search/retrieval indexes include the connector's documents when applicable.

## UI Coverage

- Streamlit shows the connector's documents in Extraction or Document Library views.
- Streamlit Enrichment Pipeline can isolate the connector by `source_kind`.
- Web APIs preserve the connector metadata needed by the page using it.
- Web UI exposes the connector through explicit filters or dedicated views when the workflow depends on it.

## Regression Checks

- Add or update a smoke test that proves the connector appears in enrichment candidates.
- Add or update a smoke test for any connector-specific grouping logic in the web layer.
- Verify counts by `source_kind` in the relevant UI after ingest.

## Release Gate

- Do not mark a connector done until extraction, enrichment, review, and UI visibility are all verified.
- If a connector is intentionally extract-only, document that explicitly in the connector coverage matrix.
- When adding a new `source_kind`, update the coverage matrix and at least one automated regression check in the same change.
