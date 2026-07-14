"""Regression guards for the cost-neutral freshness schedule consolidation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_finra_firm_rotation_matches_ten_minute_dispatcher() -> None:
    finra_source = _text("apps/web/lib/server/finra-member-firm-rss.ts")
    vercel_config = _text("apps/web/vercel.json")

    assert "const BATCH_SLOT_MS = 10 * 60_000;" in finra_source
    assert '"schedule": "*/10 * * * *"' in vercel_config


def test_scheduled_health_check_skips_duplicate_finra_firm_batch() -> None:
    health_check = _text("run_daily_health_check.py")

    assert "/api/intel/rss-refresh?finraFirmFeeds=0" in health_check


def test_sec_speech_has_one_six_hour_scheduled_owner() -> None:
    dedicated_workflow = _text(".github/workflows/sec-speech-sync.yml")
    policy_workflow = _text(".github/workflows/policy-extraction-scheduled.yml")

    assert '- cron: "0 */6 * * *"' in dedicated_workflow
    assert "- connector: sec_speech" not in policy_workflow
