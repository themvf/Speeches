"""Keep/delete rules for scripts/cleanup_neon_preview_branches.py. No network: both APIs are faked."""

import importlib.util
import io
import urllib.error
from argparse import Namespace
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "cleanup_neon_preview_branches", ROOT / "scripts" / "cleanup_neon_preview_branches.py"
)
cleanup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cleanup)

NOW = datetime(2026, 10, 4, 22, 0, tzinfo=UTC)
OLD = (NOW - timedelta(days=20)).isoformat()


def branch(name, id_=None, created=OLD, **extra):
    return {"id": id_ or f"br-{name.replace('/', '-')}", "name": name, "created_at": created, **extra}


def actions(decisions):
    return {d["name"]: (d["action"], d["reason"]) for d in decisions}


def http_error(code):
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(b"{}"))


def test_sweep_keeps_everything_that_is_not_a_finished_preview():
    branches = [
        branch("production", default=True),
        branch("vercel-dev"),
        branch("preview/claude/open-pr"),
        branch("preview/claude/protected", protected=True),
        branch("preview/claude/parent", id_="br-parent"),
        branch("child-of-preview", parent_id="br-parent"),
        branch("preview/claude/just-created", created=(NOW - timedelta(hours=2)).isoformat()),
        branch("preview/claude/recent-push"),
        branch("preview/claude/merged"),
        branch("preview/codex/git-branch-gone"),
    ]
    pushed = {"claude/recent-push": NOW - timedelta(hours=5), "claude/merged": NOW - timedelta(days=9)}
    result = actions(cleanup.plan(branches, {"claude/open-pr"}, pushed.get, NOW, 72))

    assert result["production"][0] == "keep"
    assert result["vercel-dev"] == ("keep", "not a preview branch")
    assert result["preview/claude/open-pr"] == ("keep", "open pull request")
    assert result["preview/claude/protected"] == ("keep", "protected branch")
    assert result["preview/claude/parent"] == ("keep", "has child branches")
    assert result["preview/claude/just-created"][0] == "keep"
    assert result["preview/claude/recent-push"] == ("keep", "git branch pushed within 72h")
    assert result["preview/claude/merged"][0] == "delete"
    assert result["preview/codex/git-branch-gone"][0] == "delete"


def test_unknown_git_activity_keeps_the_branch():
    def broken(_):
        raise RuntimeError("GitHub 502")

    result = actions(cleanup.plan([branch("preview/claude/x")], set(), broken, NOW, 72))
    assert result["preview/claude/x"][0] == "keep"
    assert "unknown" in result["preview/claude/x"][1]


def test_closed_pr_mode_deletes_only_that_branch_without_activity_checks():
    branches = [branch("preview/claude/closed", created=NOW.isoformat()), branch("preview/claude/other")]

    def never(_):
        raise AssertionError("closed-PR mode must not consult git activity")

    decisions = cleanup.plan(branches, set(), never, NOW, 72, only="claude/closed")
    assert actions(decisions) == {"preview/claude/closed": ("delete", "pull request closed")}


def test_closed_pr_mode_still_keeps_a_head_with_another_open_pr():
    decisions = cleanup.plan([branch("preview/claude/x")], {"claude/x"}, dict().get, NOW, 72, only="claude/x")
    assert actions(decisions)["preview/claude/x"] == ("keep", "open pull request")


def test_delete_waits_out_neon_operation_lock_and_tolerates_already_gone():
    calls, sleeps = [], []

    def request(method, url, headers):
        calls.append(method)
        if len(calls) < 3:
            raise http_error(423)
        return {}

    assert cleanup.delete_branch("p", "b", "k", request, sleeps.append) == "deleted"
    assert len(calls) == 3 and sleeps == [5, 10]

    def gone(method, url, headers):
        raise http_error(404)

    assert cleanup.delete_branch("p", "b", "k", gone, sleeps.append) == "already gone"


def fake_apis(branches, open_heads, deleted, fail_delete=()):
    def request(method, url, headers):
        if url.endswith("/branches") and "console.neon.tech" in url:
            return {"branches": branches}
        if "/pulls?" in url:
            return [{"head": {"ref": h}} for h in open_heads]
        if "api.github.com" in url and "/branches/" in url:
            raise http_error(404)
        if method == "DELETE":
            branch_id = url.rsplit("/", 1)[1]
            if branch_id in fail_delete:
                raise http_error(500)
            deleted.append(branch_id)
            return {}
        raise AssertionError(f"unexpected call {method} {url}")

    return request


ENV = {"NEON_API_KEY": "k", "NEON_PROJECT_ID": "p", "GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "o/r"}


def args(**overrides):
    base = {"execute": False, "branch": None, "min_age_hours": 72, "included_branches": 10}
    return Namespace(**{**base, **overrides})


def test_dry_run_deletes_nothing_and_estimates_the_fee():
    branches = [branch("production", default=True)] + [branch(f"preview/agent/b{i}") for i in range(14)]
    deleted = []
    summary, code = cleanup.run(args(), ENV, fake_apis(branches, {"agent/b0"}, deleted), now=NOW)

    assert code == 0 and deleted == []
    assert summary["mode"] == "dry_run"
    assert len(summary["would_delete"]) == 13
    assert summary["branches_before"] == 15 and summary["branches_after"] == 2
    assert summary["monthly_branch_fee_before_usd"] == 7.5
    assert summary["monthly_branch_fee_after_usd"] == 0


def test_execute_reports_failures_and_exits_nonzero():
    branches = [branch("preview/a/one", id_="b1"), branch("preview/a/two", id_="b2")]
    deleted = []
    summary, code = cleanup.run(args(execute=True), ENV, fake_apis(branches, set(), deleted, {"b2"}), now=NOW)

    assert deleted == ["b1"]
    assert summary["deleted"] == ["preview/a/one"]
    assert code == 1 and len(summary["errors"]) == 1


def test_missing_credentials_fail_before_any_call():
    def request(*_):
        raise AssertionError("no call expected")

    summary, code = cleanup.run(args(), {"NEON_API_KEY": "k"}, request, now=NOW)
    assert code == 1 and "NEON_PROJECT_ID" in summary["error"]
