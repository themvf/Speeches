#!/usr/bin/env python3
"""Watchdog for the Polymarket macro wallet-intelligence pipeline.

Checks one thing: has a sync successfully recomputed wallet stats recently?
`polymarket_macro_wallet_stats.refreshed_at` advances only on a healthy run's
final step, so a stale maximum is a single signal that covers every way the
pipeline can stop producing:

  - polymarket-macro-sync.yml's cron never fired (GitHub scheduling degraded,
    or the workflow was auto-disabled after repo inactivity)
  - DATABASE_URL is unset/expired in Actions, so every run fails identically
  - the sync crashes, or Polymarket's API is down, for an extended stretch

Checking the outcome rather than each individual cause is deliberate: it also
catches failure modes nobody enumerated up front.

Runs from its OWN workflow on its OWN schedule - a check living inside
polymarket-macro-sync.yml could never detect that workflow not running at all.

Fails loud (non-zero exit) when unhealthy, including when the database is
simply unreachable: every other Neon read path in this project degrades
gracefully on purpose, but a watchdog that shrugs off a dead connection would
hide the exact failure it exists to catch. Non-zero also turns the Actions run
red, which is a second, independent signal alongside the GitHub issue.

Deliberately does not call record_source_health(): that writes to GCS, and
this check is otherwise 100% Neon-only. Adding a GCS dependency would make the
watchdog breakable by an outage unrelated to what it monitors (and would cut
against the SEC-20 GCS-egress cost work).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import neon_feeds

SOURCE_KEY = "macro_sync_watchdog"

# polymarket-macro-sync.yml runs "45 13 * * 1-5", "45 18 * * 1-5", "0 1 * * *".
# The weekday crons are Mon-Fri only, so the longest LEGITIMATE gap is the
# weekend's daily-01:00-to-daily-01:00 stretch: a full 24h. A tighter
# threshold would fire every single weekend and train everyone to ignore it.
DEFAULT_MAX_AGE_HOURS = 36

ISSUE_TITLE_PATH = "macro_watchdog_issue_title.txt"
ISSUE_BODY_PATH = "macro_watchdog_issue_body.md"


def evaluate(latest_refresh: Optional[datetime], row_count: int, max_age_hours: float,
             now: Optional[datetime] = None) -> Dict[str, Any]:
    """Pure decision step, so the thresholds are testable without a database."""
    now = now or datetime.now(UTC)
    if latest_refresh is None:
        return {"healthy": False, "reason": "no_rows",
                "detail": f"polymarket_macro_wallet_stats has no rows ({row_count} found) - "
                          "the pipeline has never successfully written stats, or they were wiped.",
                "age_hours": None, "row_count": row_count}
    if latest_refresh.tzinfo is None:
        latest_refresh = latest_refresh.replace(tzinfo=UTC)
    age_hours = (now - latest_refresh).total_seconds() / 3600
    if age_hours > max_age_hours:
        return {"healthy": False, "reason": "stale",
                "detail": f"Wallet stats last refreshed {age_hours:.1f}h ago "
                          f"(threshold {max_age_hours:.0f}h). The macro sync has stopped "
                          "completing successfully.",
                "age_hours": round(age_hours, 2), "row_count": row_count}
    return {"healthy": True, "reason": "fresh",
            "detail": f"Wallet stats refreshed {age_hours:.1f}h ago across {row_count} rows.",
            "age_hours": round(age_hours, 2), "row_count": row_count}


def build_issue_body(result: Dict[str, Any]) -> str:
    causes: List[str] = [
        "`polymarket-macro-sync.yml`'s schedule stopped firing - check "
        "`gh run list --workflow=polymarket-macro-sync.yml`. GitHub also "
        "auto-disables scheduled workflows after prolonged repo inactivity.",
        "`DATABASE_URL` is unset or expired **in GitHub Actions secrets** "
        "(Vercel keeps a separate copy - the web app working does not prove "
        "the Actions one is valid).",
        "The sync is running but failing partway - check the most recent run's "
        "summary JSON for a non-empty `errors` array.",
    ]
    lines = [
        f"**{result['detail']}**",
        "",
        "`polymarket_macro_wallet_stats.refreshed_at` only advances when a sync "
        "finishes recomputing wallet stats, so this means the macro "
        "wallet-intelligence pipeline has stopped producing. The `/market` "
        "Prediction Markets tab will keep serving the last-known data without "
        "any visible error, which is why this check exists.",
        "",
        "Likely causes, in the order worth checking:",
        "",
    ]
    lines += [f"{i}. {cause}" for i, cause in enumerate(causes, 1)]
    lines += [
        "",
        f"_Detected by `check_macro_sync_freshness.py` (reason: `{result['reason']}`, "
        f"rows: {result['row_count']})._",
    ]
    return "\n".join(lines)


def write_issue_files(result: Dict[str, Any]) -> bool:
    """Writes title/body only when unhealthy, so a healthy run stays silent -
    same 'clean week posts nothing' convention as ticker-prune-check.yml."""
    if result["healthy"]:
        return False
    age = result.get("age_hours")
    suffix = f"no data for {age:.0f}h" if age else "wallet stats missing"
    Path(ISSUE_TITLE_PATH).write_text(
        f"Macro sync watchdog: pipeline has stopped producing ({suffix})", encoding="utf-8")
    Path(ISSUE_BODY_PATH).write_text(build_issue_body(result), encoding="utf-8")
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-age-hours", type=float, default=DEFAULT_MAX_AGE_HOURS)
    args = parser.parse_args(argv)

    summary: Dict[str, Any] = {"source_key": SOURCE_KEY, "ran_at": datetime.now(UTC).isoformat(),
                               "max_age_hours": args.max_age_hours}
    try:
        freshness = neon_feeds.get_macro_wallet_stats_freshness()
        result = evaluate(freshness.get("latest_refresh"), int(freshness.get("row_count") or 0),
                          args.max_age_hours)
    except Exception as exc:
        # An unreachable database IS the failure this watchdog looks for.
        result = {"healthy": False, "reason": "database_unreachable",
                  "detail": f"Could not read polymarket_macro_wallet_stats: {exc}",
                  "age_hours": None, "row_count": 0}
    summary["result"] = result
    summary["issue_written"] = write_issue_files(result)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if result["healthy"] else 1


if __name__ == "__main__":
    sys.exit(main())
