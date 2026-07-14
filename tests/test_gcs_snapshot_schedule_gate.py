"""Regression guards for the SEC-20 emergency GCS egress brake.

The listed workflows read one or more monolithic Cloud Storage snapshots.  A
scheduled job must remain opt-in until its hot path has moved to bounded,
incremental storage.  Manual dispatch stays available for recovery/backfills.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
GATE = "vars.ENABLE_GCS_SNAPSHOT_SCHEDULES == 'true'"

SCHEDULED_SNAPSHOT_WORKFLOWS = {
    "agency-official-sites-3hour.yml",
    "bloomberg-public-hourly.yml",
    "connector-enrichment-6hour.yml",
    "connector-gap-6hour.yml",
    "crs-daily.yml",
    "cyber-sources-3hour.yml",
    "daily-health-check.yml",
    "financial-news-daily.yml",
    "intelligence-evidence.yml",
    "policy-extraction-scheduled.yml",
    "rss-full-ingestion-3hour.yml",
    "rule-comment-ingest.yml",
    "sec-speech-sync.yml",
    "sec-youtube-videos-daily.yml",
    "securities-market-sources-daily.yml",
    "senate-committee-sites-3hour.yml",
    "sentiment-scoring-daily.yml",
    "substack-public-2hour.yml",
    "trends-daily.yml",
}


def _workflow(filename: str) -> str:
    return (WORKFLOWS / filename).read_text(encoding="utf-8")


def test_scheduled_snapshot_readers_are_default_off_but_keep_manual_dispatch() -> None:
    for filename in sorted(SCHEDULED_SNAPSHOT_WORKFLOWS):
        workflow = _workflow(filename)
        assert "schedule:" in workflow, filename
        assert "workflow_dispatch:" in workflow, filename
        assert GATE in workflow, filename
        assert "github.event_name != 'schedule'" in workflow, filename


def test_knowledge_sync_does_not_follow_a_gated_scheduled_producer() -> None:
    workflow = _workflow("knowledge-index-sync.yml")

    assert "github.event.workflow_run.conclusion == 'success'" in workflow
    assert "github.event.workflow_run.event != 'schedule'" in workflow
    assert GATE in workflow


def test_neon_only_schedules_remain_enabled() -> None:
    for filename in ("reddit-attention-sweep-hourly.yml", "stock-attention-daily.yml"):
        assert GATE not in _workflow(filename), filename
