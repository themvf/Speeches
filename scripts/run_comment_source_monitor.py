#!/usr/bin/env python3
"""Run monitored SEC/FINRA rule-comment source ingestion and enrichment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MONITORS_BLOB_NAME = "comment_source_monitors.json"
MONITORS_LOCAL_PATH = DATA_DIR / MONITORS_BLOB_NAME

sys.path.insert(0, str(ROOT))

import run_financial_news_pipeline as core  # noqa: E402


SOURCE_TYPES = {
    "sec_rule_page",
    "sec_comment_url",
    "finra_rule_page",
    "finra_comment_url",
}


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def monitor_id(source_type: str, source_url: str) -> str:
    raw = f"{source_type}\n{source_url.strip()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def normalize_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    monitors = payload.get("monitors", [])
    if not isinstance(monitors, list):
        monitors = []
    return {
        "version": 1,
        "updated_at": str(payload.get("updated_at", "") or ""),
        "monitors": [item for item in monitors if isinstance(item, dict)],
    }


def load_storage():
    secrets = core._load_streamlit_secrets()
    storage, _status = core._get_gcs_storage(secrets)
    return storage


def load_monitors(storage=None) -> Dict[str, Any]:
    if storage is not None:
        try:
            blob = storage.bucket.blob(MONITORS_BLOB_NAME)
            if blob.exists():
                return normalize_payload(json.loads(blob.download_as_text(encoding="utf-8")))
        except Exception:
            pass
    if MONITORS_LOCAL_PATH.exists():
        try:
            return normalize_payload(json.loads(MONITORS_LOCAL_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return normalize_payload({})


def save_monitors(payload: Dict[str, Any], storage=None, require_remote: bool = False) -> None:
    normalized = normalize_payload(payload)
    normalized["updated_at"] = iso(utc_now())
    MONITORS_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MONITORS_LOCAL_PATH.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
    if storage is not None:
        storage.bucket.blob(MONITORS_BLOB_NAME).upload_from_string(
            json.dumps(normalized, indent=2, ensure_ascii=False),
            content_type="application/json",
        )
    elif require_remote:
        raise RuntimeError("GCS is required for durable comment-source monitor persistence.")


def parse_iso(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None


def upsert_monitor(payload: Dict[str, Any], source_type: str, source_url: str, monitor_days: int) -> Dict[str, Any]:
    now = utc_now()
    mid = monitor_id(source_type, source_url)
    expires_at = iso(now + timedelta(days=max(1, int(monitor_days))))
    monitors = payload.setdefault("monitors", [])
    for item in monitors:
        if str(item.get("id", "") or "") == mid:
            item.update(
                {
                    "source_type": source_type,
                    "source_url": source_url,
                    "active": True,
                    "expires_at": expires_at,
                    "updated_at": iso(now),
                }
            )
            return item
    item = {
        "id": mid,
        "source_type": source_type,
        "source_url": source_url,
        "active": True,
        "created_at": iso(now),
        "updated_at": iso(now),
        "expires_at": expires_at,
        "last_checked_at": "",
        "last_status": "",
        "last_error": "",
        "last_processed_count": 0,
    }
    monitors.append(item)
    return item


def active_monitors(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    now = utc_now()
    out = []
    for item in payload.get("monitors", []):
        expires_at = parse_iso(item.get("expires_at"))
        if item.get("active") is False:
            continue
        if expires_at is not None and expires_at < now:
            item["active"] = False
            item["last_status"] = "expired"
            continue
        out.append(item)
    return out


def task_plan(source_type: str, source_url: str) -> List[Tuple[str, List[str]]]:
    if source_type in {"sec_rule_page", "sec_comment_url"}:
        return [("sec_rule_comment", ["sec_rule_release", "sec_rule_comment"])]
    if source_type == "finra_rule_page":
        return [
            ("finra_regulatory_notice", ["finra_regulatory_notice"]),
            ("finra_comment_letter", ["finra_comment_letter"]),
        ]
    if source_type == "finra_comment_url":
        return [("finra_comment_letter", ["finra_comment_letter"])]
    raise RuntimeError(f"Unsupported source_type: {source_type}")


def run_command(args: List[str]) -> Tuple[int, str]:
    completed = subprocess.run(args, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return completed.returncode, completed.stdout[-4000:]


def load_summary(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def run_monitor(item: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    source_type = str(item.get("source_type", "") or "").strip()
    source_url = str(item.get("source_url", "") or "").strip()
    mid = str(item.get("id", "") or monitor_id(source_type, source_url))
    run_stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    total_processed = 0
    errors: List[str] = []
    summaries: List[str] = []

    for connector, enrich_source_kinds in task_plan(source_type, source_url):
        summary_path = ROOT / f"comment_monitor_{mid}_{connector}_{run_stamp}.json"
        cmd = [
            sys.executable,
            "run_connector_extraction_pipeline.py",
            "--connector",
            connector,
            "--base-url",
            source_url,
            "--selection",
            "new_or_updated",
            "--limit",
            str(max(1, int(args.extraction_limit))),
            "--max-pages",
            "1",
            "--include-pdfs",
            "true",
            "--include-rss",
            "false",
            "--summary-path",
            str(summary_path),
        ]
        if args.require_remote_persistence:
            cmd.append("--require-remote-persistence")
        code, output = run_command(cmd)
        summaries.append(str(summary_path.name))
        summary = load_summary(summary_path)
        processed = int(summary.get("processed_count", 0) or 0)
        total_processed += processed
        if code != 0:
            errors.append(f"{connector} extraction failed: {summary.get('error') or output}")
            continue

        for source_kind in enrich_source_kinds:
            enrich_summary_path = ROOT / f"comment_monitor_{mid}_{source_kind}_enrich_{run_stamp}.json"
            enrich_cmd = [
                sys.executable,
                "run_financial_news_pipeline.py",
                "enrich",
                "--source-kind",
                source_kind,
                "--mode",
                "only_missing_or_failed",
                "--doc-ids-from-summary",
                str(summary_path),
                "--limit",
                str(max(1, int(args.enrich_limit))),
                "--provider",
                args.provider,
                "--model",
                args.model,
                "--summary-path",
                str(enrich_summary_path),
            ]
            if args.require_remote_persistence:
                enrich_cmd.append("--require-remote-persistence")
            enrich_code, enrich_output = run_command(enrich_cmd)
            summaries.append(str(enrich_summary_path.name))
            if enrich_code != 0:
                enrich_summary = load_summary(enrich_summary_path)
                errors.append(f"{source_kind} enrichment failed: {enrich_summary.get('error') or enrich_output}")

    item["last_checked_at"] = iso(utc_now())
    item["last_processed_count"] = total_processed
    item["last_summary_files"] = summaries
    if errors:
        item["last_status"] = "failed"
        item["last_error"] = "; ".join(errors)[:2000]
    else:
        item["last_status"] = "success"
        item["last_error"] = ""
    return item


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SEC/FINRA comment-source monitors.")
    parser.add_argument("--source-type", choices=sorted(SOURCE_TYPES), default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--monitor-days", type=int, default=95)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--extraction-limit", type=int, default=50)
    parser.add_argument("--enrich-limit", type=int, default=50)
    parser.add_argument("--provider", choices=["openai", "deepseek"], default="deepseek")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--require-remote-persistence", action="store_true")
    parser.add_argument("--summary-path", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    storage = load_storage()
    payload = load_monitors(storage)
    target_ids = set()

    if args.source_url:
        if not args.source_type:
            raise RuntimeError("--source-type is required when --source-url is provided.")
        item = upsert_monitor(payload, args.source_type, args.source_url.strip(), args.monitor_days)
        target_ids.add(str(item.get("id", "")))
        save_monitors(payload, storage, require_remote=args.require_remote_persistence)

    monitors = active_monitors(payload)
    if target_ids:
        monitors = [item for item in monitors if str(item.get("id", "")) in target_ids]
    elif not args.run_all:
        monitors = []

    results = []
    for item in monitors:
        results.append(run_monitor(item, args))

    save_monitors(payload, storage, require_remote=args.require_remote_persistence)
    summary = {
        "ok": True,
        "ran_at": iso(utc_now()),
        "run_all": bool(args.run_all),
        "monitor_count": len(monitors),
        "results": [
            {
                "id": item.get("id"),
                "source_type": item.get("source_type"),
                "source_url": item.get("source_url"),
                "last_status": item.get("last_status"),
                "last_processed_count": item.get("last_processed_count"),
                "expires_at": item.get("expires_at"),
                "last_error": item.get("last_error"),
            }
            for item in results
        ],
    }
    if args.summary_path:
        Path(args.summary_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if any(item.get("last_status") == "failed" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
