#!/usr/bin/env python3
"""Daily health check — reports failures across workflows, enrichment, GCS, and RSS."""

from __future__ import annotations

import base64
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from gcs_storage import GCSStorage
from source_health import (
    attach_latest_report,
    build_source_health_report,
    load_source_health,
    save_source_health,
    source_health_report_markdown,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _since_iso() -> str:
    return (_utc_now() - timedelta(hours=25)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_gcs_credentials(raw: str) -> dict | None:
    for attempt in (raw, raw.strip("'\"")):
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    try:
        return json.loads(base64.b64decode(raw).decode("utf-8"))
    except Exception:
        return None


# ─── Check: GitHub workflow failures ──────────────────────────────────────────

def check_workflow_failures(token: str, repo: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    since = _since_iso()
    failures: list[dict] = []

    try:
        url = f"https://api.github.com/repos/{repo}/actions/runs"
        params = {"created": f">{since}", "status": "failure", "per_page": 50}
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        for run in resp.json().get("workflow_runs", []):
            # Exclude this workflow itself from its own report
            if "health" in run["name"].lower():
                continue
            failures.append({
                "workflow": run["name"],
                "conclusion": run["conclusion"],
                "url": run["html_url"],
                "created_at": run["created_at"],
            })
    except Exception as e:
        return {"error": str(e), "count": 0, "failures": []}

    return {"count": len(failures), "failures": failures}


# ─── Check: Enrichment failures ───────────────────────────────────────────────

def check_enrichment_failures(bucket_name: str, credentials_info: dict) -> dict[str, Any]:
    try:
        from google.cloud import storage as gcs_lib
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_info(credentials_info)
        client = gcs_lib.Client(credentials=creds)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob("document_enrichment_state.json")

        if not blob.exists():
            return {"error": "document_enrichment_state.json not found in GCS", "count": 0, "failed_docs": []}

        data = json.loads(blob.download_as_text(encoding="utf-8"))
        entries = data.get("entries", {})

        cutoff = (_utc_now() - timedelta(hours=25)).isoformat()

        all_failed = [
            {
                "doc_id": k,
                "title": v.get("title", ""),
                "org_key": v.get("org_key", ""),
                "updated_at": v.get("updated_at", ""),
                "error": v.get("error", ""),
            }
            for k, v in entries.items()
            if v.get("status") == "failed"
        ]
        all_failed.sort(key=lambda x: x["updated_at"], reverse=True)

        # Only count failures updated in the last 25h as new — avoids daily issue spam
        # from documents that have been persistently failing for weeks.
        recent_failed = [d for d in all_failed if d["updated_at"] >= cutoff]

        return {
            "count": len(recent_failed),
            "total_historical": len(all_failed),
            "failed_docs": all_failed[:10],
        }
    except Exception as e:
        return {"error": str(e), "count": 0, "failed_docs": []}


# ─── Check: GCS connectivity ──────────────────────────────────────────────────

EXPECTED_BLOBS = [
    "all_speeches.json",
    "document_enrichment_state.json",
    "custom_documents.json",
]


def check_gcs_connectivity(bucket_name: str, credentials_info: dict) -> dict[str, Any]:
    try:
        from google.cloud import storage as gcs_lib
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_info(credentials_info)
        client = gcs_lib.Client(credentials=creds)
        bucket = client.bucket(bucket_name)

        blobs: dict[str, str] = {}
        for blob_name in EXPECTED_BLOBS:
            try:
                blob = bucket.blob(blob_name)
                if not blob.exists():
                    blobs[blob_name] = "missing"
                    continue
                text = blob.download_as_text(encoding="utf-8")
                json.loads(text)  # Validate parseable
                blobs[blob_name] = "ok"
            except Exception as e:
                blobs[blob_name] = f"error: {e}"

        return {"blobs": blobs}
    except Exception as e:
        return {"error": str(e), "blobs": {}}


# ─── Check: RSS feeds ─────────────────────────────────────────────────────────

def check_rss_feeds(app_url: str, cron_secret: str) -> dict[str, Any]:
    # Vercel owns the 10-minute FINRA member-firm rotation. The health check
    # should inspect the normal refresh result without paying for a duplicate
    # firm batch in the same time slot.
    url = f"{app_url.rstrip('/')}/api/intel/rss-refresh?finraFirmFeeds=0"
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {cron_secret}"},
            timeout=60,
        )
        data = resp.json()
        feeds = data.get("data", {}).get("feeds", [])
        failed = [f for f in feeds if f.get("error")]
        return {
            "ok": data.get("ok"),
            "total_feeds": len(feeds),
            "inserted": data.get("data", {}).get("inserted", 0),
            "failed_feeds": failed,
            "failed_count": len(failed),
        }
    except Exception as e:
        return {"error": str(e), "failed_count": -1}


def review_source_health_with_deepseek(report: dict[str, Any]) -> str:
    api_key = os.getenv("DEEPSEEK_API", "") or os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        return ""
    compact = {
        "generated_at": report.get("generated_at"),
        "recent_run_count": report.get("recent_run_count"),
        "recent_failed_run_count": report.get("recent_failed_run_count"),
        "error_categories": report.get("error_categories", {}),
        "failing_sources": report.get("failing_sources", [])[:12],
        "stale_sources": report.get("stale_sources", [])[:12],
        "quiet_sources": report.get("quiet_sources", [])[:12],
    }
    try:
        resp = requests.post(
            os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.getenv("SOURCE_HEALTH_REVIEW_MODEL", "deepseek-v4-flash"),
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You audit a news/source ingestion health log. Return a concise prioritized action list. "
                            "Mention only sources needing action and classify each as proxy/rotate, stale URL, parser fix, "
                            "credentials/API, healthy-no-new-items, or investigate."
                        ),
                    },
                    {"role": "user", "content": json.dumps(compact, ensure_ascii=False)},
                ],
                "temperature": 0,
            },
            timeout=45,
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
    except Exception as e:
        return f"DeepSeek review unavailable: {e}"


# ─── Report builder ───────────────────────────────────────────────────────────

def build_report(results: dict[str, Any]) -> tuple[str, str, int]:
    date_str = _utc_now().strftime("%Y-%m-%d")
    sections: list[str] = []
    failure_count = 0

    # Workflow failures
    wf = results.get("workflows", {})
    if wf.get("error"):
        sections.append(f"### Warning: Workflow Check Error\n{wf['error']}")
    elif wf.get("count", 0) > 0:
        failure_count += wf["count"]
        lines = [f"### FAIL: Workflow Failures ({wf['count']})\n"]
        for f in wf["failures"]:
            lines.append(f"- [{f['workflow']}]({f['url']}) — {f['conclusion']} at {f['created_at']}")
        sections.append("\n".join(lines))
    else:
        sections.append("### OK: Workflow Runs\nNo failures in the past 24h.")

    # Enrichment failures
    en = results.get("enrichment", {})
    if en.get("error"):
        sections.append(f"### Warning: Enrichment Check Error\n{en['error']}")
    elif en.get("count", 0) > 0:
        failure_count += en["count"]
        historical = en.get("total_historical", en["count"])
        header = f"### FAIL: Enrichment Failures ({en['count']} new in last 24h"
        if historical > en["count"]:
            header += f", {historical} total historical"
        header += ", showing up to 10)\n"
        lines = [header]
        for d in en.get("failed_docs", []):
            title_trunc = d["title"][:80] + ("…" if len(d["title"]) > 80 else "")
            lines.append(f"- `{d['doc_id']}` [{d['org_key']}] {title_trunc}")
            if d.get("error"):
                lines.append(f"  - Error: `{d['error'][:120]}`")
        sections.append("\n".join(lines))
    else:
        historical = en.get("total_historical", 0)
        note = f" ({historical} historical failures exist but are not new)" if historical > 0 else ""
        sections.append(f"### OK: Enrichment\nNo new enrichment failures in the last 24h.{note}")

    # GCS connectivity
    gcs = results.get("gcs", {})
    if gcs.get("error"):
        failure_count += 1
        sections.append(f"### FAIL: GCS Connectivity Error\n```\n{gcs['error']}\n```")
    else:
        blobs = gcs.get("blobs", {})
        blob_failures = {k: v for k, v in blobs.items() if v != "ok"}
        if blob_failures:
            failure_count += len(blob_failures)
            lines = ["### FAIL: GCS Blob Issues\n"]
            for name, status in blobs.items():
                icon = "OK" if status == "ok" else "FAIL"
                lines.append(f"- `{name}`: {icon} — {status}")
            sections.append("\n".join(lines))
        else:
            lines = ["### OK: GCS Connectivity\n"]
            for name in blobs:
                lines.append(f"- `{name}`: readable")
            sections.append("\n".join(lines))

    # RSS feeds
    rss = results.get("rss")
    if rss is None:
        sections.append("### Skipped: RSS Feeds\nAPP_URL or CRON_SECRET not configured.")
    elif rss.get("error"):
        sections.append(f"### Warning: RSS Check Error\n```\n{rss['error']}\n```")
    elif rss.get("failed_count", 0) > 0:
        failure_count += rss["failed_count"]
        lines = [f"### FAIL: RSS Feed Failures ({rss['failed_count']})\n"]
        for f in rss.get("failed_feeds", []):
            lines.append(f"- `{f.get('feedKey', 'unknown')}`: {f.get('error', '')}")
        sections.append("\n".join(lines))
    else:
        sections.append(
            f"### OK: RSS Feeds\n{rss.get('total_feeds', 0)} feeds healthy, "
            f"{rss.get('inserted', 0)} articles inserted."
        )

    source_health = results.get("source_health")
    if isinstance(source_health, dict):
        report = source_health.get("report", {}) if isinstance(source_health.get("report"), dict) else {}
        source_failures = len(report.get("failing_sources", []) or [])
        if source_failures:
            failure_count += source_failures
        section = source_health_report_markdown(report, str(source_health.get("ai_review", "") or ""))
        if source_health.get("storage_error"):
            section += f"\n\nSource health GCS warning: `{source_health['storage_error']}`"
        sections.append(section)
    else:
        sections.append("### Skipped: Source Health\nNo source health log was available yet.")

    repo = os.getenv("GITHUB_REPOSITORY", "")
    run_url = f"https://github.com/{repo}/actions/runs/{os.getenv('GITHUB_RUN_ID', '')}"
    body = (
        f"## Daily Health Report — {date_str}\n\n"
        + "\n\n".join(sections)
        + f"\n\n---\n_Generated at {_utc_now().strftime('%Y-%m-%dT%H:%M:%SZ')} — [view run]({run_url})_"
    )

    status = "FAILURES DETECTED" if failure_count > 0 else "All systems OK"
    title = f"Daily Health Report [{date_str}] — {failure_count} failure(s): {status}" if failure_count > 0 else f"Daily Health Report [{date_str}] — All systems OK"

    return title, body, failure_count


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    bucket_name = os.environ.get("GCS_BUCKET_NAME", "")
    gcs_creds_raw = os.environ.get("GCS_CREDENTIALS_JSON", "")
    app_url = os.environ.get("APP_URL", "")
    cron_secret = os.environ.get("CRON_SECRET", "")

    credentials_info = _parse_gcs_credentials(gcs_creds_raw) if gcs_creds_raw else None

    results: dict[str, Any] = {}
    storage: GCSStorage | None = None
    source_health_storage_error = ""

    print("Running workflow failure check...", flush=True)
    results["workflows"] = check_workflow_failures(token, repo) if (token and repo) else {"error": "GITHUB_TOKEN/GITHUB_REPOSITORY not set", "count": 0, "failures": []}

    if credentials_info and bucket_name:
        try:
            storage = GCSStorage(bucket_name, credentials_info)
        except Exception as e:
            source_health_storage_error = str(e)
        print("Running enrichment failure check...", flush=True)
        results["enrichment"] = check_enrichment_failures(bucket_name, credentials_info)
        print("Running GCS connectivity check...", flush=True)
        results["gcs"] = check_gcs_connectivity(bucket_name, credentials_info)
        print("Running source health check...", flush=True)
        source_payload = load_source_health(storage)
        source_report = build_source_health_report(source_payload)
        ai_review = review_source_health_with_deepseek(source_report)
        results["source_health"] = {
            "report": source_report,
            "ai_review": ai_review,
            "storage_error": source_health_storage_error,
        }
    else:
        results["enrichment"] = {"error": "GCS not configured", "count": 0, "failed_docs": []}
        results["gcs"] = {"error": "GCS not configured", "blobs": {}}
        print("WARNING: GCS credentials not available — skipping GCS checks.", file=sys.stderr)

    if app_url and cron_secret:
        print("Running RSS feed check...", flush=True)
        results["rss"] = check_rss_feeds(app_url, cron_secret)
    else:
        results["rss"] = None

    title, body, failure_count = build_report(results)
    if storage is not None and isinstance(results.get("source_health"), dict):
        try:
            source_payload = load_source_health(storage)
            source_report = results["source_health"].get("report", {}) if isinstance(results["source_health"].get("report"), dict) else {}
            ai_review = str(results["source_health"].get("ai_review", "") or "")
            save_source_health(
                attach_latest_report(source_payload, source_report, title, body, ai_review),
                storage,
            )
        except Exception as e:
            print(f"WARNING: Failed to save source health report: {e}", file=sys.stderr)

    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}\n")
    print(body)

    # Write outputs
    Path("health_report_title.txt").write_text(title, encoding="utf-8")
    Path("health_report_body.md").write_text(body, encoding="utf-8")
    Path("health_report.json").write_text(
        json.dumps({"title": title, "failure_count": failure_count, "results": results}, indent=2, default=str),
        encoding="utf-8",
    )

    github_output = os.environ.get("GITHUB_OUTPUT", "")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"has_failures={'true' if failure_count > 0 else 'false'}\n")

    fail_on_failure = os.getenv("HEALTH_CHECK_FAIL_ON_FAILURE", "true").strip().lower() not in {"0", "false", "no"}
    if failure_count > 0 and not fail_on_failure:
        print("HEALTH_CHECK_FAIL_ON_FAILURE=false; leaving workflow green after writing report.", flush=True)
    sys.exit(1 if failure_count > 0 and fail_on_failure else 0)


if __name__ == "__main__":
    main()
