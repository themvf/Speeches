from source_health import build_run_entry, build_source_health_report, update_source_rollup, utc_now_iso
from youtube_video_scraper import _is_proxy_billing_error


def test_build_run_entry_categorizes_blocked_403() -> None:
    entry = build_run_entry(
        {
            "connector": "fia_news_item",
            "command": "extract",
            "ok": False,
            "error": "403 Forbidden",
        }
    )

    assert entry["source_key"] == "fia_news_item"
    assert entry["status"] == "failed"
    assert entry["error_category"] == "blocked_403"

    rollup = update_source_rollup({}, entry)
    assert rollup["consecutive_failures"] == 1
    assert rollup["last_error_category"] == "blocked_403"


def test_build_run_entry_categorizes_provider_billing_before_proxy_failure() -> None:
    entry = build_run_entry(
        {
            "connector": "sec_youtube_video",
            "command": "extract",
            "ok": False,
            "error": "ProxyError: Tunnel connection failed: 402 Payment Required",
        }
    )

    assert entry["error_category"] == "billing"
    assert _is_proxy_billing_error(RuntimeError(entry["sample_error"])) is True


def test_build_source_health_report_groups_failing_and_quiet_sources() -> None:
    now = utc_now_iso()
    report = build_source_health_report(
        {
            "runs": [
                {
                    "source_key": "fia_news_item",
                    "status": "failed",
                    "ran_at": now,
                    "error_category": "blocked_403",
                }
            ],
            "sources": {
                "fia_news_item": {
                    "source_key": "fia_news_item",
                    "last_run_at": now,
                    "last_status": "failed",
                    "last_error_category": "blocked_403",
                    "consecutive_failures": 2,
                    "last_counts": {"discovered": 0, "processed": 0},
                },
                "substack_public_article": {
                    "source_key": "substack_public_article",
                    "last_run_at": now,
                    "last_status": "success",
                    "consecutive_failures": 0,
                    "last_counts": {"discovered": 0, "processed": 0},
                },
            },
        }
    )

    assert report["recent_failed_run_count"] == 1
    assert report["error_categories"] == {"blocked_403": 1}
    assert report["failing_sources"][0]["source_key"] == "fia_news_item"
    assert report["quiet_sources"][0]["source_key"] == "substack_public_article"
