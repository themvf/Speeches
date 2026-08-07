#!/usr/bin/env python3
"""Report Neon compute/storage consumption so cost decisions use real numbers.

Read-only. Answers one question: is the database ever idle? Neon bills compute
by the hour and scales to zero when nothing queries it, so a scheduled job that
polls every few minutes can quietly convert a bursty workload into a 24/7 bill.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta

API_ROOT = "https://console.neon.tech/api/v2"


def _get(path: str, api_key: str) -> dict:
    request = urllib.request.Request(
        f"{API_ROOT}{path}",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--summary-path", default="")
    args = parser.parse_args()

    api_key = os.getenv("NEON_API_KEY", "").strip()
    project_id = os.getenv("NEON_PROJECT_ID", "").strip()
    if not api_key or not project_id:
        print(json.dumps({"ok": False, "error": "NEON_API_KEY and NEON_PROJECT_ID are required."}))
        return 1

    now = datetime.now(UTC).replace(microsecond=0)
    start = now - timedelta(days=max(1, args.days))
    summary: dict = {
        "ok": True,
        "ran_at": now.isoformat().replace("+00:00", "Z"),
        "window_days": args.days,
    }

    try:
        project = _get(f"/projects/{project_id}", api_key).get("project", {})
        summary["project"] = {
            "name": project.get("name", ""),
            # The autosuspend delay is the setting that lets compute reach zero.
            "suspend_timeout_seconds": project.get("default_endpoint_settings", {}).get("suspend_timeout_seconds"),
            "compute_last_active_at": project.get("compute_last_active_at", ""),
        }
        consumption = project.get("consumption_period_start", "")
        if consumption:
            summary["project"]["consumption_period_start"] = consumption
        for field in ("compute_time_seconds", "active_time_seconds", "data_storage_bytes_hour", "written_data_bytes"):
            if field in project:
                summary["project"][field] = project[field]
    except urllib.error.HTTPError as exc:
        summary["project_error"] = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}"
    except Exception as exc:  # noqa: BLE001
        summary["project_error"] = str(exc)

    try:
        granularity = "daily"
        path = (
            f"/consumption_history/projects?project_ids={project_id}"
            f"&from={start.isoformat().replace('+00:00', 'Z')}"
            f"&to={now.isoformat().replace('+00:00', 'Z')}"
            f"&granularity={granularity}"
        )
        history = _get(path, api_key)
        periods = (history.get("projects") or [{}])[0].get("periods") or []
        rows = []
        for period in periods:
            for entry in period.get("consumption") or []:
                rows.append(
                    {
                        "at": entry.get("timeframe_start", ""),
                        "active_time_seconds": entry.get("active_time_seconds", 0),
                        "compute_time_seconds": entry.get("compute_time_seconds", 0),
                        "written_data_bytes": entry.get("written_data_bytes", 0),
                        "synthetic_storage_size_bytes": entry.get("synthetic_storage_size_bytes", 0),
                    }
                )
        summary["daily"] = rows
        if rows:
            day_seconds = 86400
            active = [r["active_time_seconds"] for r in rows]
            summary["analysis"] = {
                "days_reported": len(rows),
                "mean_active_hours_per_day": round(sum(active) / len(active) / 3600, 2),
                "max_active_hours_per_day": round(max(active) / 3600, 2),
                # A ratio near 1.0 means the compute never suspends.
                "mean_active_fraction_of_day": round(sum(active) / len(active) / day_seconds, 3),
                "total_compute_hours": round(sum(r["compute_time_seconds"] for r in rows) / 3600, 2),
            }
    except urllib.error.HTTPError as exc:
        summary["history_error"] = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}"
    except Exception as exc:  # noqa: BLE001
        summary["history_error"] = str(exc)

    text = json.dumps(summary, indent=2)
    print(text)
    if args.summary_path:
        with open(args.summary_path, "w", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
