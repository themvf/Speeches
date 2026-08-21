from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import check_macro_sync_freshness as watchdog


NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


def test_fresh_stats_are_healthy():
    result = watchdog.evaluate(NOW - timedelta(hours=5), 84565, 36, now=NOW)
    assert result["healthy"] is True
    assert result["reason"] == "fresh"
    assert result["age_hours"] == 5.0


def test_weekend_gap_does_not_false_positive():
    # The weekday crons are Mon-Fri, so a legitimate weekend gap runs a full
    # 24h between daily-01:00 runs. A threshold at or under that would fire
    # every weekend; this is the regression guard for that.
    result = watchdog.evaluate(NOW - timedelta(hours=24), 84565, watchdog.DEFAULT_MAX_AGE_HOURS, now=NOW)
    assert result["healthy"] is True


def test_stats_older_than_threshold_are_unhealthy():
    result = watchdog.evaluate(NOW - timedelta(hours=48), 84565, 36, now=NOW)
    assert result["healthy"] is False
    assert result["reason"] == "stale"
    assert result["age_hours"] == 48.0


def test_empty_table_is_unhealthy_rather_than_treated_as_fresh():
    result = watchdog.evaluate(None, 0, 36, now=NOW)
    assert result["healthy"] is False
    assert result["reason"] == "no_rows"


def test_naive_timestamp_is_treated_as_utc_not_crashed_on():
    naive = (NOW - timedelta(hours=2)).replace(tzinfo=None)
    result = watchdog.evaluate(naive, 10, 36, now=NOW)
    assert result["healthy"] is True
    assert result["age_hours"] == 2.0


def test_issue_files_are_written_only_when_unhealthy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert watchdog.write_issue_files({"healthy": True, "reason": "fresh", "detail": "ok",
                                       "age_hours": 3.0, "row_count": 10}) is False
    assert not (tmp_path / watchdog.ISSUE_TITLE_PATH).exists()

    assert watchdog.write_issue_files({"healthy": False, "reason": "stale",
                                       "detail": "Wallet stats last refreshed 48.0h ago",
                                       "age_hours": 48.0, "row_count": 84565}) is True
    body = (tmp_path / watchdog.ISSUE_BODY_PATH).read_text(encoding="utf-8")
    assert "48.0h ago" in body
    # The body must name the Actions-vs-Vercel secret split - the single most
    # likely cause, and the one most easily misdiagnosed as "the site works,
    # so the database is fine".
    assert "Vercel keeps a separate copy" in body


def test_unreachable_database_exits_nonzero_and_reports_rather_than_failing_soft(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch.object(watchdog.neon_feeds, "get_macro_wallet_stats_freshness",
                      side_effect=RuntimeError("could not connect to server")):
        exit_code = watchdog.main([])
    assert exit_code == 1
    body = (tmp_path / watchdog.ISSUE_BODY_PATH).read_text(encoding="utf-8")
    assert "could not connect to server" in body


def test_main_exits_zero_and_writes_nothing_when_healthy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch.object(watchdog.neon_feeds, "get_macro_wallet_stats_freshness",
                      return_value={"latest_refresh": datetime.now(UTC) - timedelta(hours=1),
                                    "row_count": 84565}):
        exit_code = watchdog.main([])
    assert exit_code == 0
    assert not (tmp_path / watchdog.ISSUE_TITLE_PATH).exists()
