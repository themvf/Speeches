#!/usr/bin/env python3
"""Run YouTube transcript extraction/enrichment for one ref or saved channels."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import run_financial_news_pipeline as core


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
YOUTUBE_SOURCES_BLOB_NAME = "youtube_channel_sources.json"
YOUTUBE_SOURCES_LOCAL_PATH = DATA_DIR / YOUTUBE_SOURCES_BLOB_NAME
DEFAULT_SEC_SOURCE = {
    "id": "secviews",
    "label": "SEC YouTube",
    "channel_ref": "https://www.youtube.com/user/SECViews",
    "active": True,
    "extraction_limit": 10,
    "max_pages": 1,
    "enrich_limit": 10,
}


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clamp_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        n = int(value)
    except Exception:
        n = int(default)
    return max(int(min_value), min(int(max_value), n))


def _safe_id(value: str, fallback: str = "youtube") -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return (cleaned or fallback)[:48]


def _normalize_source(source: Any, idx: int = 0) -> Optional[Dict[str, Any]]:
    if not isinstance(source, dict):
        return None
    channel_ref = _normalize_space(source.get("channel_ref") or source.get("channelRef") or source.get("url") or source.get("handle"))
    if not channel_ref:
        return None
    label = _normalize_space(source.get("label") or source.get("name")) or f"YouTube Source {idx + 1}"
    return {
        "id": _normalize_space(source.get("id")) or _safe_id(channel_ref, f"youtube-{idx + 1}"),
        "label": label,
        "channel_ref": channel_ref,
        "active": bool(source.get("active", True)),
        "extraction_limit": _clamp_int(source.get("extraction_limit") or source.get("extractionLimit"), 10, 1, 50),
        "max_pages": _clamp_int(source.get("max_pages") or source.get("maxPages"), 1, 1, 5),
        "enrich_limit": _clamp_int(source.get("enrich_limit") or source.get("enrichLimit"), 10, 1, 50),
    }


def _normalize_sources_payload(payload: Any) -> Dict[str, Any]:
    raw_sources = payload.get("sources", []) if isinstance(payload, dict) else []
    if not isinstance(raw_sources, list):
        raw_sources = []
    sources = [item for item in (_normalize_source(source, idx) for idx, source in enumerate(raw_sources)) if item]
    return {
        "version": 1,
        "updated_at": _normalize_space(payload.get("updated_at")) if isinstance(payload, dict) else "",
        "sources": sources,
    }


def _load_youtube_sources() -> Dict[str, Any]:
    secrets_payload = core._load_streamlit_secrets()
    storage, _status = core._get_gcs_storage(secrets_payload)
    if storage is not None:
        try:
            blob = storage.bucket.blob(YOUTUBE_SOURCES_BLOB_NAME)
            if blob.exists():
                payload = _normalize_sources_payload(json.loads(blob.download_as_text(encoding="utf-8")))
                YOUTUBE_SOURCES_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
                YOUTUBE_SOURCES_LOCAL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                return payload
        except Exception as exc:
            print(f"Remote load failed for {YOUTUBE_SOURCES_BLOB_NAME}: {exc}", file=sys.stderr)

    if YOUTUBE_SOURCES_LOCAL_PATH.exists():
        try:
            return _normalize_sources_payload(json.loads(YOUTUBE_SOURCES_LOCAL_PATH.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"Local load failed for {YOUTUBE_SOURCES_LOCAL_PATH}: {exc}", file=sys.stderr)

    return _normalize_sources_payload({"sources": [DEFAULT_SEC_SOURCE]})


def _run_command(cmd: List[str]) -> Dict[str, Any]:
    print("+ " + " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True)
    return {"returncode": completed.returncode, "ok": completed.returncode == 0}


def _run_source(source: Dict[str, Any], args: argparse.Namespace, idx: int) -> Dict[str, Any]:
    source_id = _safe_id(str(source.get("id") or source.get("label") or f"source-{idx + 1}"))
    extraction_summary = f"sec_youtube_video_extraction_summary_{source_id}.json"
    enrich_summary = f"sec_youtube_video_enrich_summary_{source_id}.json"

    extraction_cmd = [
        sys.executable,
        "run_connector_extraction_pipeline.py",
        "--connector",
        "sec_youtube_video",
        "--base-url",
        str(source["channel_ref"]),
        "--selection",
        "new_or_updated",
        "--limit",
        str(source["extraction_limit"]),
        "--max-pages",
        str(source["max_pages"]),
        "--summary-path",
        extraction_summary,
    ]
    if args.require_remote_persistence:
        extraction_cmd.append("--require-remote-persistence")

    extraction = _run_command(extraction_cmd)
    enrich: Dict[str, Any] = {"returncode": None, "ok": False, "skipped": True}
    if extraction["ok"]:
        enrich_cmd = [
            sys.executable,
            "run_financial_news_pipeline.py",
            "enrich",
            "--source-kind",
            "sec_youtube_video",
            "--mode",
            "only_missing_or_failed",
            "--doc-ids-from-summary",
            extraction_summary,
            "--limit",
            str(source["enrich_limit"]),
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--summary-path",
            enrich_summary,
        ]
        if args.require_remote_persistence:
            enrich_cmd.append("--require-remote-persistence")
        enrich = _run_command(enrich_cmd)

    return {
        "source": source,
        "extraction_summary": extraction_summary,
        "enrich_summary": enrich_summary,
        "extraction": extraction,
        "enrich": enrich,
        "ok": bool(extraction.get("ok") and enrich.get("ok")),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YouTube transcript extraction/enrichment runner")
    parser.add_argument("--channel-ref", default="", help="Manual YouTube video URL/id or channel ref. Blank loads saved channels.")
    parser.add_argument("--extraction-limit", type=int, default=10)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--enrich-limit", type=int, default=10)
    parser.add_argument("--provider", choices=["openai", "deepseek"], default="deepseek")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--require-remote-persistence", action="store_true")
    parser.add_argument("--summary-path", default="sec_youtube_video_pipeline_summary.json")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    manual_ref = _normalize_space(args.channel_ref)
    if manual_ref:
        sources = [
            {
                **DEFAULT_SEC_SOURCE,
                "id": "manual",
                "label": "Manual YouTube",
                "channel_ref": manual_ref,
                "extraction_limit": _clamp_int(args.extraction_limit, 1, 1, 50),
                "max_pages": _clamp_int(args.max_pages, 1, 1, 5),
                "enrich_limit": _clamp_int(args.enrich_limit, 1, 1, 50),
            }
        ]
        mode = "manual"
    else:
        payload = _load_youtube_sources()
        sources = [source for source in payload.get("sources", []) if source.get("active", True)]
        mode = "scheduled_sources"

    if not sources:
        summary = {
            "mode": mode,
            "generated_at": _utc_now_iso(),
            "source_count": 0,
            "success_count": 0,
            "failed_count": 0,
            "skipped": True,
            "message": "No active YouTube sources configured.",
            "results": [],
        }
        Path(args.summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    results = [_run_source(source, args, idx) for idx, source in enumerate(sources)]
    summary = {
        "mode": mode,
        "generated_at": _utc_now_iso(),
        "source_count": len(sources),
        "success_count": sum(1 for result in results if result.get("ok")),
        "failed_count": sum(1 for result in results if not result.get("ok")),
        "results": results,
    }
    Path(args.summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["success_count"] > 0 and summary["failed_count"] < summary["source_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
