"""Guards for comment-source monitor host/source_type reconciliation.

Three registry rows pointed the SEC scraper at finra.org and took a 403 on
every run, so the daily job was permanently red while still doing real work.
A job that always fails is a job nobody reads.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "run_comment_source_monitor", ROOT / "scripts" / "run_comment_source_monitor.py"
)
monitor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(monitor)


FINRA_NOTICE = "https://www.finra.org/rules-guidance/notices/25-06"
FINRA_COMMENTS = "https://www.finra.org/rules-guidance/notices/25-06#comments"
SEC_RULE = "https://www.sec.gov/rules-regulations/2026/03/s7-2026-09"


@pytest.mark.parametrize(
    "declared, url, expected",
    [
        # The exact production breakage: FINRA URLs filed under SEC types.
        ("sec_rule_page", FINRA_NOTICE, "finra_rule_page"),
        ("sec_rule_page", FINRA_COMMENTS, "finra_rule_page"),
        ("sec_comment_url", FINRA_COMMENTS, "finra_comment_url"),
        # Mirror image.
        ("finra_rule_page", SEC_RULE, "sec_rule_page"),
        ("finra_comment_url", SEC_RULE, "sec_comment_url"),
        # Already correct pairs are untouched.
        ("finra_rule_page", FINRA_NOTICE, "finra_rule_page"),
        ("finra_comment_url", FINRA_COMMENTS, "finra_comment_url"),
        ("sec_rule_page", SEC_RULE, "sec_rule_page"),
    ],
)
def test_resolve_source_type_fixes_host_family_and_keeps_role(declared, url, expected):
    assert monitor.resolve_source_type(declared, url) == expected


def test_resolve_source_type_leaves_unknown_hosts_alone():
    assert monitor.resolve_source_type("sec_rule_page", "https://example.gov/rules/1") == "sec_rule_page"
    assert monitor.resolve_source_type("sec_rule_page", "") == "sec_rule_page"


def test_resolve_source_type_matches_subdomains_not_lookalikes():
    assert monitor.resolve_source_type("sec_rule_page", "https://api.finra.org/x") == "finra_rule_page"
    # notfinra.org must not be treated as finra.org
    assert monitor.resolve_source_type("sec_rule_page", "https://notfinra.org/x") == "sec_rule_page"


def test_resolved_type_drives_a_reachable_task_plan():
    """The bad pair used to plan an SEC connector against a FINRA URL."""
    bad_plan = monitor.task_plan("sec_rule_page", FINRA_NOTICE)
    assert [connector for connector, _ in bad_plan] == ["sec_rule_comment"]

    fixed = monitor.resolve_source_type("sec_rule_page", FINRA_NOTICE)
    good_plan = monitor.task_plan(fixed, FINRA_NOTICE)
    assert [connector for connector, _ in good_plan] == [
        "finra_regulatory_notice",
        "finra_comment_letter",
    ]


def test_upsert_monitor_never_stores_a_mismatched_pair():
    payload = {"version": 1, "updated_at": "", "monitors": []}
    item = monitor.upsert_monitor(payload, "sec_rule_page", FINRA_NOTICE, 95)
    assert item["source_type"] == "finra_rule_page"
    assert payload["monitors"][0]["source_type"] == "finra_rule_page"


def test_run_monitor_heals_an_existing_bad_registry_row(monkeypatch):
    """Existing rows are corrected in place; the id is preserved."""
    planned = []

    def fake_task_plan(source_type, source_url):
        planned.append(source_type)
        return []

    monkeypatch.setattr(monitor, "task_plan", fake_task_plan)

    item = {
        "id": "81086ad0e0248aa36d7d6f24",
        "source_type": "sec_rule_page",
        "source_url": FINRA_NOTICE,
        "active": True,
    }

    class Args:
        extraction_limit = 50
        enrich_limit = 50
        provider = "deepseek"
        model = "deepseek-v4-pro"
        require_remote_persistence = False

    result = monitor.run_monitor(item, Args())

    assert planned == ["finra_rule_page"]
    assert result["source_type"] == "finra_rule_page"
    assert result["source_type_corrected_from"] == "sec_rule_page"
    assert result["id"] == "81086ad0e0248aa36d7d6f24"
    assert result["last_status"] == "success"
