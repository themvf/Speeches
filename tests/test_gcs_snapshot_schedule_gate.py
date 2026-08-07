"""Regression guards for the SEC-20 emergency GCS egress brake.

The listed workflows read one or more monolithic Cloud Storage snapshots.  A
scheduled job must remain opt-in until its hot path has moved to bounded,
incremental storage.  Manual dispatch stays available for recovery/backfills.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
GATE = "vars.ENABLE_GCS_SNAPSHOT_SCHEDULES == 'true'"
NEON_PILOT_GATE = "vars.ENABLE_NEON_PILOT_SCHEDULES == 'true'"
NEON_SUBSTACK_GATE = "vars.ENABLE_NEON_SUBSTACK_SCHEDULES == 'true'"
RULE_COMMENT_GATE = "vars.ENABLE_RULE_COMMENT_INGEST != 'false'"

SCHEDULED_SNAPSHOT_WORKFLOWS = {
    "agency-official-sites-3hour.yml",
    "connector-enrichment-6hour.yml",
    "connector-gap-6hour.yml",
    "crs-daily.yml",
    "cyber-sources-3hour.yml",
    "daily-health-check.yml",
    "intelligence-evidence.yml",
    "policy-extraction-scheduled.yml",
    "rss-full-ingestion-3hour.yml",
    "sec-speech-sync.yml",
    "sec-youtube-videos-daily.yml",
    "securities-market-sources-daily.yml",
    "senate-committee-sites-3hour.yml",
    "sentiment-scoring-daily.yml",
    "trends-daily.yml",
}

NEON_PILOT_WORKFLOWS = {
    "bloomberg-public-hourly.yml": (
        "bloomberg_public_article",
        "Run Bloomberg Bounded Enrichment Catch-up",
        "always()",
        NEON_PILOT_GATE,
    ),
    "substack-public-2hour.yml": (
        "substack_public_article",
        "Run Substack Bounded Enrichment Catch-up",
        "always()",
        NEON_SUBSTACK_GATE,
    ),
    "financial-news-daily.yml": (
        "newsapi_article",
        "Run Financial News Bounded Enrichment Catch-up",
        "always() && steps.timegate.outputs.run_now == 'true'",
        NEON_PILOT_GATE,
    ),
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


def test_rule_comment_ingest_runs_on_its_own_gate_and_mirrors_into_neon() -> None:
    """/api/notices-comments reads Neon, so this ingest has to keep running.

    It is deliberately NOT on the shared brake: turning that back on would
    restart fifteen other snapshot readers.  Writes stay GCS-authoritative
    (these connectors are not neon-authoritative), so DATABASE_URL is what
    carries new notices and their enrichment into the table the page reads.
    """
    workflow = _workflow("rule-comment-ingest.yml")

    assert "schedule:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "github.event_name != 'schedule'" in workflow
    assert RULE_COMMENT_GATE in workflow
    assert GATE not in workflow
    assert "DATABASE_URL: ${{ secrets.DATABASE_URL }}" in workflow


def test_knowledge_sync_does_not_follow_a_gated_scheduled_producer() -> None:
    workflow = _workflow("knowledge-index-sync.yml")

    assert "github.event.workflow_run.conclusion == 'success'" in workflow
    assert "github.event.workflow_run.event != 'schedule'" in workflow
    assert GATE in workflow


def test_neon_only_schedules_remain_enabled() -> None:
    for filename in ("reddit-attention-sweep-hourly.yml", "stock-attention-daily.yml"):
        assert GATE not in _workflow(filename), filename


def test_neon_pilots_use_a_dedicated_gate_and_bounded_row_persistence() -> None:
    for filename, (source_kind, catchup_step, catchup_condition, schedule_gate) in NEON_PILOT_WORKFLOWS.items():
        workflow = _workflow(filename)
        assert "schedule:" in workflow, filename
        assert "workflow_dispatch:" in workflow, filename
        assert "github.event_name != 'schedule'" in workflow, filename
        assert schedule_gate in workflow, filename
        assert "ENABLE_GCS_SNAPSHOT_SCHEDULES" not in workflow, filename
        assert "group: sec20-neon-corpus-writers" in workflow, filename
        assert 'NEON_BACKFILL_VERIFIED: "true"' in workflow, filename
        assert workflow.count("--persistence-mode neon_authoritative") >= 3, filename
        assert f"--source-kind {source_kind}" in workflow, filename
        assert "--doc-ids-from-summary" in workflow, filename
        assert "--mode only_missing_or_failed" in workflow, filename
        assert "--limit 10" in workflow, filename
        catchup_header = f"- name: {catchup_step}\n        if: {catchup_condition}"
        assert catchup_header in workflow, filename
