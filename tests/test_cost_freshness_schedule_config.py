"""Regression guards for the cost-neutral freshness schedule consolidation."""

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_finra_firm_rotation_matches_the_dispatcher_cadence() -> None:
    """The two numbers must match each other, whatever the cadence is.

    The batch offset is floor(now / BATCH_SLOT_MS) * batchSize, so a slot
    shorter than the cron interval advances the offset by more than one batch
    per invocation and never fetches the firms in between - a silent coverage
    hole across 3,194 firms. Asserted as a relationship rather than a literal
    so changing the cadence is a two-file edit, not a three-file one.
    """
    finra_source = _text("apps/web/lib/server/finra-member-firm-rss.ts")
    vercel_config = json.loads(_text("apps/web/vercel.json"))

    slot_match = re.search(r"BATCH_SLOT_MS\s*=\s*(\d+)\s*\*\s*60_000", finra_source)
    assert slot_match, "BATCH_SLOT_MS must be declared as <minutes> * 60_000"
    slot_minutes = int(slot_match.group(1))

    schedules = [
        cron.get("schedule", "")
        for cron in vercel_config.get("crons", [])
        if cron.get("path") == "/api/intel/rss-refresh"
    ]
    assert schedules, "vercel.json must schedule /api/intel/rss-refresh"

    cron_match = re.match(r"^\*/(\d+)\s", schedules[0])
    assert cron_match, f"expected a */N minute field, got {schedules[0]!r}"
    cron_minutes = int(cron_match.group(1))

    assert slot_minutes == cron_minutes, (
        f"BATCH_SLOT_MS is {slot_minutes}m but the dispatcher runs every "
        f"{cron_minutes}m; firms between consecutive offsets would be skipped"
    )


def test_finra_firm_rotation_stays_inside_the_seven_day_news_window() -> None:
    """A slower cadence is fine; one that outruns `when:7d` loses news."""
    finra_source = _text("apps/web/lib/server/finra-member-firm-rss.ts")
    registry = json.loads(_text("apps/web/lib/generated/finra-member-firms.json"))

    slot_minutes = int(re.search(r"BATCH_SLOT_MS\s*=\s*(\d+)\s*\*\s*60_000", finra_source).group(1))
    batch_size = int(re.search(r"DEFAULT_BATCH_SIZE\s*=\s*(\d+)", finra_source).group(1))
    firms = sum(1 for firm in registry.get("firms", []) if firm.get("name") and firm.get("rssUrl"))

    cycle_minutes = math.ceil(firms / batch_size) * slot_minutes
    assert cycle_minutes < 7 * 24 * 60, (
        f"a full rotation takes {cycle_minutes / 60:.1f}h, which must stay under the "
        "168h `when:7d` query window or firm news is missed outright"
    )


def test_scheduled_health_check_skips_duplicate_finra_firm_batch() -> None:
    health_check = _text("run_daily_health_check.py")

    assert "/api/intel/rss-refresh?finraFirmFeeds=0" in health_check


def test_sec_speech_has_one_six_hour_scheduled_owner() -> None:
    dedicated_workflow = _text(".github/workflows/sec-speech-sync.yml")
    policy_workflow = _text(".github/workflows/policy-extraction-scheduled.yml")

    assert '- cron: "0 */6 * * *"' in dedicated_workflow
    assert "- connector: sec_speech" not in policy_workflow
