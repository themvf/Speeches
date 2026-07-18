#!/usr/bin/env python3
"""Source health logging for scheduled extraction and enrichment jobs."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gcs_storage import GCSStorage


SOURCE_HEALTH_BLOB_NAME = "source_health_log.json"
SOURCE_HEALTH_LOCAL_PATH = Path(__file__).resolve().parent / "data" / SOURCE_HEALTH_BLOB_NAME
MAX_RUNS = 1500


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def parse_iso(value: Any) -> Optional[datetime]:
    text = normalize_space(value)
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except Exception:
        return None


def parse_gcs_credentials(raw: str) -> Optional[Dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return None
    candidates = [text]
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        candidates.append(text[1:-1].strip())
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    try:
        decoded = base64.b64decode(text, validate=True).decode("utf-8")
        parsed = json.loads(decoded)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        if isinstance(parsed, dict):
            return parsed
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    return None


def get_storage_from_env() -> Optional[GCSStorage]:
    bucket_name = normalize_space(os.getenv("GCS_BUCKET_NAME", ""))
    credentials = parse_gcs_credentials(os.getenv("GCS_CREDENTIALS_JSON", ""))
    if not bucket_name or not credentials:
        return None
    try:
        return GCSStorage(bucket_name, credentials)
    except Exception:
        return None


def empty_payload() -> Dict[str, Any]:
    return {
        "updated_at": "",
        "runs": [],
        "sources": {},
        "latest_report": None,
    }


def load_source_health(storage: Optional[GCSStorage] = None) -> Dict[str, Any]:
    if storage is not None:
        try:
            blob = storage.bucket.blob(SOURCE_HEALTH_BLOB_NAME)
            if blob.exists():
                payload = json.loads(blob.download_as_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return normalize_payload(payload)
        except Exception:
            pass
    if SOURCE_HEALTH_LOCAL_PATH.exists():
        try:
            return normalize_payload(json.loads(SOURCE_HEALTH_LOCAL_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return empty_payload()


def save_source_health(payload: Dict[str, Any], storage: Optional[GCSStorage] = None) -> None:
    normalized = normalize_payload(payload)
    normalized["updated_at"] = utc_now_iso()
    SOURCE_HEALTH_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_HEALTH_LOCAL_PATH.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
    if storage is not None:
        blob = storage.bucket.blob(SOURCE_HEALTH_BLOB_NAME)
        blob.upload_from_string(
            json.dumps(normalized, indent=2, ensure_ascii=False),
            content_type="application/json",
        )


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    sources = payload.get("sources", {})
    if not isinstance(sources, dict):
        sources = {}
    return {
        "updated_at": normalize_space(payload.get("updated_at", "")),
        "runs": [run for run in runs if isinstance(run, dict)][-MAX_RUNS:],
        "sources": {str(key): value for key, value in sources.items() if isinstance(value, dict)},
        "latest_report": payload.get("latest_report") if isinstance(payload.get("latest_report"), dict) else None,
    }


def first_error_from_summary(summary: Dict[str, Any]) -> str:
    direct = normalize_space(summary.get("error", ""))
    if direct:
        return direct
    for key in ("failed", "discovery_errors", "skipped_blocked"):
        value = summary.get(key)
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, dict):
                for field in ("error", "reason", "message", "title", "url"):
                    text = normalize_space(first.get(field, ""))
                    if text:
                        return text
                return normalize_space(json.dumps(first, ensure_ascii=False))[:500]
            text = normalize_space(first)
            if text:
                return text
    debug = summary.get("discovery_debug")
    if isinstance(debug, dict):
        errors = debug.get("errors")
        if isinstance(errors, list) and errors:
            return normalize_space(errors[0])
        feed_debug = debug.get("feed_discovery")
        if isinstance(feed_debug, dict) and isinstance(feed_debug.get("errors"), list) and feed_debug["errors"]:
            return normalize_space(feed_debug["errors"][0])
    return ""


def categorize_error(sample_error: str, summary: Dict[str, Any]) -> str:
    text = " ".join([
        sample_error,
        normalize_space(summary.get("error", "")),
        normalize_space(summary.get("connector", "")),
        normalize_space(summary.get("source_kind", "")),
    ]).lower()
    if not text and to_int(summary.get("discovered_count", 0)) == 0:
        return "no_discovery"
    if normalize_space(summary.get("command", "")) == "argparse_error":
        if "invalid choice" in text:
            return "invalid_choice"
        return "cli_usage_error"
    if "402" in text or "payment required" in text or "insufficient balance" in text or "insufficient credit" in text:
        return "billing"
    if "403" in text or "forbidden" in text:
        return "blocked_403"
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return "rate_limited_429"
    if "404" in text or "not found" in text:
        return "stale_404"
    if "connect tunnel failed" in text or "proxy error" in text or "proxyerror" in text:
        return "proxy_tunnel"
    if "tls connect error" in text or "openssl_internal" in text or "invalid library" in text:
        return "network_tls"
    if "xml" in text or "not well-formed" in text or "parse" in text:
        return "parser"
    if "api key" in text or "unauthorized" in text or "401" in text:
        return "auth"
    if "model_not_found" in text or "access to model" in text:
        return "model_access"
    if to_int(summary.get("failed_count", 0)) > 0:
        return "item_failures"
    if to_int(summary.get("discovered_count", 0)) > 0 and to_int(summary.get("processed_count", 0)) == 0:
        return "no_new_items"
    return "none" if not sample_error else "unknown"


def source_key_from_summary(summary: Dict[str, Any]) -> str:
    return (
        normalize_space(summary.get("connector", ""))
        or normalize_space(summary.get("source_kind", ""))
        or normalize_space(summary.get("command", ""))
        or normalize_space(summary.get("mode", ""))
        or "unknown"
    )


def status_from_summary(summary: Dict[str, Any], sample_error: str) -> str:
    if summary.get("ok") is False or sample_error and normalize_space(summary.get("command", "")) == "extract":
        return "failed"
    failed_count = to_int(summary.get("failed_count", 0))
    processed_count = to_int(summary.get("processed_count", 0))
    if failed_count > 0:
        return "partial" if processed_count > 0 else "failed"
    return "success"


def build_run_entry(summary: Dict[str, Any]) -> Dict[str, Any]:
    sample_error = first_error_from_summary(summary)
    source_key = source_key_from_summary(summary)
    status = status_from_summary(summary, sample_error)
    return {
        "id": normalize_space(os.getenv("GITHUB_RUN_ID", "")) or f"local-{utc_now_iso()}-{source_key}",
        "source_key": source_key,
        "command": normalize_space(summary.get("command", "")) or normalize_space(summary.get("mode", "")),
        "workflow": normalize_space(os.getenv("GITHUB_WORKFLOW", "")),
        "workflow_ref": normalize_space(os.getenv("GITHUB_REF_NAME", "")),
        "run_id": normalize_space(os.getenv("GITHUB_RUN_ID", "")),
        "run_attempt": normalize_space(os.getenv("GITHUB_RUN_ATTEMPT", "")),
        "status": status,
        "ran_at": normalize_space(summary.get("ran_at", "")) or utc_now_iso(),
        "discovered_count": to_int(summary.get("discovered_count", 0)),
        "filtered_count": to_int(summary.get("filtered_count", 0)),
        "candidate_count": to_int(summary.get("candidate_count", 0)),
        "selected_count": to_int(summary.get("selected_count", 0)),
        "processed_count": to_int(summary.get("processed_count", 0)),
        "saved_new": to_int(summary.get("saved_new", 0)),
        "saved_updates": to_int(summary.get("saved_updates", 0)),
        "failed_count": to_int(summary.get("failed_count", 0)),
        "enriched_count": to_int(summary.get("enriched_count", 0)),
        "fallback_enriched_count": to_int(summary.get("fallback_enriched_count", 0)),
        "used_models": summary.get("used_models", []) if isinstance(summary.get("used_models"), list) else [],
        "error_category": categorize_error(sample_error, summary),
        "sample_error": sample_error[:1000],
        "summary_path": normalize_space(os.getenv("SOURCE_HEALTH_SUMMARY_PATH", "")),
    }


def update_source_rollup(source: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
    previous_failures = to_int(source.get("consecutive_failures", 0))
    status = normalize_space(entry.get("status", "unknown"))
    ran_at = normalize_space(entry.get("ran_at", ""))
    successful = status in {"success", "partial"}
    return {
        "source_key": normalize_space(entry.get("source_key", "")),
        "last_run_at": ran_at,
        "last_status": status,
        "last_error_category": normalize_space(entry.get("error_category", "")),
        "last_error": normalize_space(entry.get("sample_error", "")),
        "last_workflow": normalize_space(entry.get("workflow", "")),
        "last_run_id": normalize_space(entry.get("run_id", "")),
        "last_counts": {
            "discovered": to_int(entry.get("discovered_count", 0)),
            "processed": to_int(entry.get("processed_count", 0)),
            "saved_new": to_int(entry.get("saved_new", 0)),
            "saved_updates": to_int(entry.get("saved_updates", 0)),
            "failed": to_int(entry.get("failed_count", 0)),
            "enriched": to_int(entry.get("enriched_count", 0)),
            "fallback_enriched": to_int(entry.get("fallback_enriched_count", 0)),
        },
        "last_success_at": ran_at if successful else normalize_space(source.get("last_success_at", "")),
        "consecutive_failures": 0 if successful else previous_failures + 1,
    }


def record_source_health(summary: Dict[str, Any], storage: Optional[GCSStorage] = None) -> None:
    try:
        storage = storage or get_storage_from_env()
        payload = load_source_health(storage)
        entry = build_run_entry(summary)
        payload["runs"].append(entry)
        payload["runs"] = payload["runs"][-MAX_RUNS:]
        sources = payload.setdefault("sources", {})
        key = normalize_space(entry.get("source_key", "unknown")) or "unknown"
        previous = sources.get(key, {}) if isinstance(sources.get(key), dict) else {}
        sources[key] = update_source_rollup(previous, entry)
        save_source_health(payload, storage)
    except Exception as exc:
        print(f"Source health logging failed: {exc}", flush=True)


def _extract_cli_flag_value(flag: str) -> str:
    """Best-effort extraction of a CLI flag's value directly from sys.argv.

    Used when argparse itself is about to fail (e.g. an unsupported
    --connector choice), before any parsed `args` object exists, so the
    attempted value can still be logged for diagnostics."""
    argv = sys.argv[1:]
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return ""


class RecordingArgumentParser(argparse.ArgumentParser):
    """ArgumentParser whose error() also logs a source-health failure entry
    before exiting.

    Plain argparse.ArgumentParser.error() prints a usage message and exits
    (SystemExit) before a script's own try/except around its real work ever
    runs - which is exactly where record_source_health() is normally called.
    That meant a CLI usage error (e.g. a --connector value dropped from
    SUPPORTED_CONNECTORS by an unrelated change elsewhere) was completely
    invisible to the failing/stale/quiet source-health dashboard and its
    daily GitHub issue: every scheduled run of a broken connector "failed"
    but left no trace anywhere health monitoring could see. This subclass
    logs the failure first, then defers to the normal argparse behavior
    (same usage message, same exit code) so CLI behavior is unchanged for
    callers - only its visibility to monitoring changes. Subparsers created
    via add_subparsers() inherit this class automatically.
    """

    def error(self, message: str) -> None:
        try:
            record_source_health(
                {
                    "ok": False,
                    "command": "argparse_error",
                    "connector": _extract_cli_flag_value("--connector"),
                    "source_kind": _extract_cli_flag_value("--source-kind"),
                    "error": message,
                    "ran_at": utc_now_iso(),
                }
            )
        except Exception as exc:
            print(f"Source health logging (argparse error) failed: {exc}", flush=True)
        super().error(message)


def build_source_health_report(payload: Dict[str, Any], lookback_hours: int = 25) -> Dict[str, Any]:
    normalized = normalize_payload(payload)
    now = datetime.now(UTC)
    cutoff = now - timedelta(hours=max(1, int(lookback_hours)))
    recent_runs: List[Dict[str, Any]] = []
    for run in normalized["runs"]:
        ran_at = parse_iso(run.get("ran_at"))
        if ran_at is not None and ran_at >= cutoff:
            recent_runs.append(run)

    failing_sources = []
    stale_sources = []
    quiet_sources = []
    for source in normalized["sources"].values():
        last_status = normalize_space(source.get("last_status", ""))
        failures = to_int(source.get("consecutive_failures", 0))
        last_run_at = parse_iso(source.get("last_run_at"))
        counts = source.get("last_counts", {}) if isinstance(source.get("last_counts"), dict) else {}
        if last_status == "failed" or failures > 0:
            failing_sources.append(source)
        elif last_run_at is not None and (now - last_run_at) > timedelta(hours=36):
            stale_sources.append(source)
        elif to_int(counts.get("discovered", 0)) == 0 and to_int(counts.get("processed", 0)) == 0:
            quiet_sources.append(source)

    categories: Dict[str, int] = {}
    for run in recent_runs:
        category = normalize_space(run.get("error_category", "none")) or "none"
        if category != "none":
            categories[category] = categories.get(category, 0) + 1

    return {
        "generated_at": utc_now_iso(),
        "lookback_hours": lookback_hours,
        "recent_run_count": len(recent_runs),
        "recent_failed_run_count": sum(1 for run in recent_runs if run.get("status") == "failed"),
        "recent_partial_run_count": sum(1 for run in recent_runs if run.get("status") == "partial"),
        "failing_sources": sorted(failing_sources, key=lambda item: (-to_int(item.get("consecutive_failures", 0)), normalize_space(item.get("source_key", ""))))[:25],
        "stale_sources": sorted(stale_sources, key=lambda item: normalize_space(item.get("last_run_at", "")))[:25],
        "quiet_sources": sorted(quiet_sources, key=lambda item: normalize_space(item.get("source_key", "")))[:25],
        "error_categories": categories,
        "recent_runs": sorted(recent_runs, key=lambda item: normalize_space(item.get("ran_at", "")), reverse=True)[:100],
    }


def source_health_report_markdown(report: Dict[str, Any], ai_review: str = "") -> str:
    lines = [
        "### Source Health",
        f"Recent source runs: {to_int(report.get('recent_run_count', 0))}; failed: {to_int(report.get('recent_failed_run_count', 0))}; partial: {to_int(report.get('recent_partial_run_count', 0))}.",
    ]
    categories = report.get("error_categories", {}) if isinstance(report.get("error_categories"), dict) else {}
    if categories:
        lines.append("Error categories: " + ", ".join(f"`{key}` {value}" for key, value in sorted(categories.items())))
    failing = report.get("failing_sources", []) if isinstance(report.get("failing_sources"), list) else []
    if failing:
        lines.append("\nFailing sources:")
        for source in failing[:10]:
            counts = source.get("last_counts", {}) if isinstance(source.get("last_counts"), dict) else {}
            lines.append(
                f"- `{source.get('source_key')}`: {source.get('last_error_category')} "
                f"({source.get('consecutive_failures')} consecutive); "
                f"discovered {counts.get('discovered', 0)}, processed {counts.get('processed', 0)}. "
                f"{normalize_space(source.get('last_error', ''))[:160]}"
            )
    stale = report.get("stale_sources", []) if isinstance(report.get("stale_sources"), list) else []
    if stale:
        lines.append("\nStale sources:")
        for source in stale[:10]:
            lines.append(f"- `{source.get('source_key')}`: last run {source.get('last_run_at') or 'unknown'}")
    quiet = report.get("quiet_sources", []) if isinstance(report.get("quiet_sources"), list) else []
    if quiet:
        lines.append("\nQuiet sources:")
        for source in quiet[:10]:
            lines.append(f"- `{source.get('source_key')}`: latest run found no items.")
    if ai_review:
        lines.append("\nDeepSeek review:\n" + ai_review.strip())
    return "\n".join(lines)


def attach_latest_report(payload: Dict[str, Any], report: Dict[str, Any], title: str, body: str, ai_review: str = "") -> Dict[str, Any]:
    normalized = normalize_payload(payload)
    normalized["latest_report"] = {
        "generated_at": normalize_space(report.get("generated_at", "")) or utc_now_iso(),
        "title": title,
        "body": body,
        "ai_review": ai_review,
        "report": report,
    }
    return normalized
