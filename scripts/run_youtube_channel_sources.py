#!/usr/bin/env python3
"""Run saved YouTube channel transcript extraction and enrichment sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHANNELS_BLOB_NAME = "youtube_channel_sources.json"
LOCAL_CHANNELS_PATH = DATA_DIR / CHANNELS_BLOB_NAME

sys.path.insert(0, str(ROOT))

import run_financial_news_pipeline as core  # noqa: E402


DEFAULT_SEC_CHANNEL = {
    "id": "sec_views",
    "label": "SEC",
    "channel_ref": "https://www.youtube.com/user/SECViews",
    "active": True,
    "extraction_limit": 2,
    "enrich_limit": 2,
    "max_pages": 1,
    "connector": "sec_youtube_video",
}


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def iso(value: Optional[datetime] = None) -> str:
    return (value or utc_now()).isoformat().replace("+00:00", "Z")


def positive_int(value: Any, fallback: int, min_value: int = 1, max_value: int = 50) -> int:
    try:
        parsed = int(round(float(value)))
    except Exception:
        return fallback
    return max(min_value, min(max_value, parsed))


def channel_id(channel_ref: str) -> str:
    return hashlib.sha256(channel_ref.strip().lower().encode("utf-8")).hexdigest()[:16]


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return stem[:60] or "youtube_channel"


def normalize_channel(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    channel_ref = str(raw.get("channel_ref", "") or "").strip()
    if not channel_ref:
        return None
    connector = str(raw.get("connector", "") or "youtube_video").strip()
    if connector not in {"sec_youtube_video", "youtube_video"}:
        connector = "youtube_video"
    now = iso()
    return {
        "id": str(raw.get("id", "") or channel_id(channel_ref)).strip(),
        "label": str(raw.get("label", "") or channel_ref).strip(),
        "channel_ref": channel_ref,
        "active": raw.get("active") is not False,
        "extraction_limit": positive_int(raw.get("extraction_limit"), 2),
        "enrich_limit": positive_int(raw.get("enrich_limit"), 2),
        "max_pages": positive_int(raw.get("max_pages"), 1, 1, 10),
        "connector": connector,
        "added_at": str(raw.get("added_at", "") or now),
        "updated_at": str(raw.get("updated_at", "") or now),
        "last_run_at": str(raw.get("last_run_at", "") or ""),
        "last_status": str(raw.get("last_status", "") or ""),
        "last_error": str(raw.get("last_error", "") or ""),
    }


def normalize_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    raw_channels = payload.get("channels", [])
    if not isinstance(raw_channels, list):
        raw_channels = []
    channels = [item for item in (normalize_channel(raw) for raw in raw_channels) if item]
    if not channels:
        default_channel = dict(DEFAULT_SEC_CHANNEL)
        default_channel["added_at"] = iso()
        default_channel["updated_at"] = iso()
        channels.append(normalize_channel(default_channel))
        channels = [item for item in channels if item]
    return {
        "version": 1,
        "updated_at": str(payload.get("updated_at", "") or iso()),
        "channels": channels,
    }


def load_storage(require_remote: bool = False):
    secrets = core._load_streamlit_secrets()
    storage, _status = core._get_gcs_storage(secrets)
    if storage is None and require_remote:
        raise RuntimeError("GCS is required for durable YouTube channel source persistence.")
    return storage


def load_channels(storage=None) -> Dict[str, Any]:
    if storage is not None:
        try:
            blob = storage.bucket.blob(CHANNELS_BLOB_NAME)
            if blob.exists():
                return normalize_payload(json.loads(blob.download_as_text(encoding="utf-8")))
        except Exception as exc:
            print(f"[youtube-channels] Failed to load GCS config: {exc}", file=sys.stderr)
    if LOCAL_CHANNELS_PATH.exists():
        try:
            return normalize_payload(json.loads(LOCAL_CHANNELS_PATH.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"[youtube-channels] Failed to load local config: {exc}", file=sys.stderr)
    return normalize_payload({})


def save_channels(payload: Dict[str, Any], storage=None, require_remote: bool = False) -> None:
    normalized = normalize_payload(payload)
    normalized["updated_at"] = iso()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_CHANNELS_PATH.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
    if storage is not None:
        storage.bucket.blob(CHANNELS_BLOB_NAME).upload_from_string(
            json.dumps(normalized, indent=2, ensure_ascii=False),
            content_type="application/json",
        )
    elif require_remote:
        raise RuntimeError("GCS is required for durable YouTube channel source persistence.")


def run_command(cmd: List[str]) -> None:
    print("[youtube-channels] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def run_channel(channel: Dict[str, Any], require_remote: bool, provider: str, model: str) -> Dict[str, Any]:
    connector = str(channel.get("connector") or "youtube_video")
    source_kind = "sec_youtube_video" if connector == "sec_youtube_video" else "youtube_video"
    channel_ref = str(channel.get("channel_ref") or "").strip()
    label = str(channel.get("label") or channel_ref).strip()
    cid = str(channel.get("id") or channel_id(channel_ref)).strip()
    stem = safe_stem(f"{cid}_{label}")
    extraction_summary = f"youtube_channel_{stem}_extraction_summary.json"
    enrich_summary = f"youtube_channel_{stem}_enrich_summary.json"
    extraction_limit = positive_int(channel.get("extraction_limit"), 2)
    enrich_limit = positive_int(channel.get("enrich_limit"), 2)
    max_pages = positive_int(channel.get("max_pages"), 1, 1, 10)

    result: Dict[str, Any] = {
        "id": cid,
        "label": label,
        "channel_ref": channel_ref,
        "connector": connector,
        "source_kind": source_kind,
        "status": "running",
        "started_at": iso(),
        "extraction_summary": extraction_summary,
        "enrich_summary": enrich_summary,
    }
    try:
        extract_cmd = [
            sys.executable,
            "run_connector_extraction_pipeline.py",
            "--connector",
            connector,
            "--base-url",
            channel_ref,
            "--selection",
            "new_or_updated",
            "--limit",
            str(extraction_limit),
            "--max-pages",
            str(max_pages),
            "--summary-path",
            extraction_summary,
        ]
        if require_remote:
            extract_cmd.append("--require-remote-persistence")
        run_command(extract_cmd)

        enrich_cmd = [
            sys.executable,
            "run_financial_news_pipeline.py",
            "enrich",
            "--source-kind",
            source_kind,
            "--mode",
            "only_missing_or_failed",
            "--doc-ids-from-summary",
            extraction_summary,
            "--limit",
            str(enrich_limit),
            "--provider",
            provider,
            "--model",
            model,
            "--summary-path",
            enrich_summary,
        ]
        if require_remote:
            enrich_cmd.append("--require-remote-persistence")
        run_command(enrich_cmd)

        result["status"] = "success"
    except subprocess.CalledProcessError as exc:
        result["status"] = "failed"
        result["error"] = f"Command failed with exit code {exc.returncode}"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
    result["finished_at"] = iso()
    return result


def manual_channel_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    channel_ref = str(args.channel_ref or "").strip()
    connector = "sec_youtube_video" if "SECViews" in channel_ref or "secviews" in channel_ref.lower() else "youtube_video"
    return {
        "id": channel_id(channel_ref),
        "label": args.channel_label or "Manual YouTube Channel",
        "channel_ref": channel_ref,
        "active": True,
        "extraction_limit": positive_int(args.extraction_limit, 2),
        "enrich_limit": positive_int(args.enrich_limit, 2),
        "max_pages": positive_int(args.max_pages, 1, 1, 10),
        "connector": connector,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-ref", default="", help="Optional single channel override for this run.")
    parser.add_argument("--channel-label", default="", help="Optional label for a single channel override.")
    parser.add_argument("--extraction-limit", default="2")
    parser.add_argument("--enrich-limit", default="2")
    parser.add_argument("--max-pages", default="1")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--summary-path", default="youtube_channel_sources_run_summary.json")
    parser.add_argument("--require-remote-persistence", action="store_true")
    args = parser.parse_args()

    storage = load_storage(require_remote=args.require_remote_persistence)
    payload = load_channels(storage)
    manual_ref = str(args.channel_ref or "").strip()
    if manual_ref:
      channels = [manual_channel_from_args(args)]
      config_backed = False
    else:
      channels = [channel for channel in payload.get("channels", []) if channel.get("active") is not False]
      config_backed = True

    results: List[Dict[str, Any]] = []
    for channel in channels:
        if not channel or not str(channel.get("channel_ref", "") or "").strip():
            continue
        result = run_channel(channel, args.require_remote_persistence, args.provider, args.model)
        results.append(result)
        if config_backed:
            for saved in payload.get("channels", []):
                if saved.get("id") == channel.get("id"):
                    saved["last_run_at"] = result.get("finished_at", iso())
                    saved["last_status"] = result.get("status", "unknown")
                    saved["last_error"] = result.get("error", "")
                    saved["updated_at"] = iso()
                    break

    if config_backed:
        save_channels(payload, storage, require_remote=args.require_remote_persistence)

    summary = {
        "ok": all(item.get("status") == "success" for item in results),
        "mode": "single_override" if manual_ref else "saved_channels",
        "started_at": results[0].get("started_at") if results else "",
        "finished_at": iso(),
        "processed_count": len(results),
        "failed_count": sum(1 for item in results if item.get("status") != "success"),
        "results": results,
    }
    Path(args.summary_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
