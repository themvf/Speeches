#!/usr/bin/env python3
"""
One-time speaker metadata backfill for existing speech datasets.

Normalizes `metadata.speaker` and adds `metadata.speakers` (list of parsed
individual speakers) for both local JSON and GCS dataset storage.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tomllib
from typing import Any, Dict, Tuple

from gcs_storage import BLOB_NAME, GCSStorage
from sec_scraper_free import SECScraper
from speaker_utils import extract_speakers, format_speakers


@dataclass
class BackfillStats:
    total_speeches: int = 0
    changed_speaker_field: int = 0
    added_speakers_list: int = 0
    multi_speaker_entries: int = 0
    updated_from_listing_map: int = 0


def _normalize_dataset(
    data: Dict[str, Any],
    url_speaker_map: Dict[str, str] | None = None,
) -> Tuple[Dict[str, Any], BackfillStats]:
    stats = BackfillStats()
    speeches = data.get("speeches", [])

    for speech in speeches:
        metadata = speech.setdefault("metadata", {})
        raw_speaker = metadata.get("speaker", "")
        url = metadata.get("url", "")
        if url_speaker_map and url and url in url_speaker_map:
            raw_speaker = url_speaker_map[url]
            stats.updated_from_listing_map += 1
        old_speaker = str(raw_speaker).strip() if raw_speaker is not None else ""

        speaker_list = extract_speakers(old_speaker)
        normalized = format_speakers(old_speaker)

        if not normalized and old_speaker:
            normalized = old_speaker

        if normalized != old_speaker:
            metadata["speaker"] = normalized
            stats.changed_speaker_field += 1
        elif "speaker" not in metadata:
            metadata["speaker"] = normalized

        existing_list = metadata.get("speakers")
        if existing_list != speaker_list:
            metadata["speakers"] = speaker_list
            stats.added_speakers_list += 1

        if len(speaker_list) > 1:
            stats.multi_speaker_entries += 1

        stats.total_speeches += 1

    return data, stats


def _print_stats(label: str, stats: BackfillStats) -> None:
    print(f"\n[{label}]")
    print(f"Total speeches: {stats.total_speeches}")
    print(f"Updated from SEC listing map: {stats.updated_from_listing_map}")
    print(f"Changed metadata.speaker: {stats.changed_speaker_field}")
    print(f"Updated metadata.speakers list: {stats.added_speakers_list}")
    print(f"Entries with multiple speakers: {stats.multi_speaker_entries}")


def _load_gcs_from_secrets(secrets_path: Path) -> GCSStorage:
    if not secrets_path.exists():
        raise FileNotFoundError(f"Secrets file not found: {secrets_path}")

    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)

    if "gcs" not in secrets:
        raise KeyError("No [gcs] section found in secrets file")

    gcs_cfg = dict(secrets["gcs"])
    bucket_name = gcs_cfg.pop("bucket_name")
    return GCSStorage(bucket_name, gcs_cfg)


def build_listing_speaker_map(max_pages: int = 120) -> Dict[str, str]:
    """Build URL -> speaker text map from SEC listing pages."""
    scraper = SECScraper()
    entries = scraper.discover_speech_urls(max_pages=max_pages)
    speaker_map: Dict[str, str] = {}
    for entry in entries:
        url = entry.get("url", "")
        speaker = entry.get("speaker", "")
        if url:
            speaker_map[url] = speaker
    return speaker_map


def backfill_local(
    local_path: Path,
    create_backup: bool = True,
    url_speaker_map: Dict[str, str] | None = None,
) -> BackfillStats:
    if not local_path.exists():
        raise FileNotFoundError(f"Local dataset not found: {local_path}")

    with open(local_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data, stats = _normalize_dataset(data, url_speaker_map=url_speaker_map)

    if create_backup:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = local_path.with_suffix(local_path.suffix + f".bak.{ts}")
        shutil.copy2(local_path, backup)
        print(f"Local backup created: {backup}")

    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return stats


def backfill_gcs(
    secrets_path: Path,
    create_backup: bool = True,
    url_speaker_map: Dict[str, str] | None = None,
) -> BackfillStats:
    storage = _load_gcs_from_secrets(secrets_path)
    blob = storage.bucket.blob(BLOB_NAME)

    if not blob.exists():
        raise FileNotFoundError(f"GCS blob not found: {BLOB_NAME}")

    raw_text = blob.download_as_text(encoding="utf-8")
    data = json.loads(raw_text)
    data, stats = _normalize_dataset(data, url_speaker_map=url_speaker_map)

    if create_backup:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_name = f"{BLOB_NAME}.bak.{ts}"
        storage.bucket.blob(backup_name).upload_from_string(
            raw_text,
            content_type="application/json",
        )
        print(f"GCS backup created: {backup_name}")

    storage.save_speeches(data)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill speaker fields in existing datasets.")
    parser.add_argument(
        "--target",
        choices=["local", "gcs", "both"],
        default="both",
        help="Where to apply the backfill.",
    )
    parser.add_argument(
        "--local-path",
        default="data/all_speeches_final.json",
        help="Path to local dataset JSON.",
    )
    parser.add_argument(
        "--secrets-path",
        default=".streamlit/secrets.toml",
        help="Path to Streamlit secrets TOML for GCS credentials.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable backup creation before writing.",
    )
    parser.add_argument(
        "--skip-listing-map",
        action="store_true",
        help="Do not refresh speakers from SEC listing pages before normalization.",
    )
    parser.add_argument(
        "--listing-max-pages",
        type=int,
        default=120,
        help="Max SEC listing pages to scan when building URL->speaker map.",
    )
    args = parser.parse_args()

    do_backup = not args.no_backup
    local_path = Path(args.local_path)
    secrets_path = Path(args.secrets_path)
    url_speaker_map: Dict[str, str] | None = None

    if not args.skip_listing_map:
        print(
            f"Building SEC listing speaker map (up to {args.listing_max_pages} pages)..."
        )
        url_speaker_map = build_listing_speaker_map(max_pages=args.listing_max_pages)
        print(f"Listing map entries: {len(url_speaker_map)}")

    if args.target in ("local", "both"):
        stats = backfill_local(
            local_path,
            create_backup=do_backup,
            url_speaker_map=url_speaker_map,
        )
        _print_stats("LOCAL", stats)

    if args.target in ("gcs", "both"):
        stats = backfill_gcs(
            secrets_path,
            create_backup=do_backup,
            url_speaker_map=url_speaker_map,
        )
        _print_stats("GCS", stats)

    print("\nBackfill complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
