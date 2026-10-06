#!/usr/bin/env python3
"""Delete Neon preview branches whose work is finished, so they stop billing.

The Vercel/Neon integration creates a `preview/<git-branch>` database branch for each preview
deployment, and nothing in this repo ever removed one. Neon's own "delete obsolete branches"
option fires only when the git branch is deleted, and GitHub keeps merged head branches here
(delete_branch_on_merge is off); the Vercel-managed integration instead waits for Vercel's
six-month deployment retention. Every branch past the plan's included allowance (10 on Launch,
25 on Scale) bills $1.50/month. On 2026-10-04 the project held 79 branches, two with an open PR.

Dry run by default. Only `preview/` branches are ever candidates. The default branch, protected
branches, branches with children, branches with an open PR and recently active branches are
always kept. Stdlib only, like report_neon_consumption.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta

NEON_API = "https://console.neon.tech/api/v2"
GITHUB_API = "https://api.github.com"
PREFIX = "preview/"
EXTRA_BRANCH_MONTHLY_USD = 1.50


def _request(method: str, url: str, headers: dict, timeout: int = 60):
    request = urllib.request.Request(url, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return json.loads(body) if body else {}


def _neon_headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}


def _github_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _parse_time(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def plan(branches, open_heads, last_commit_at, now, min_age_hours, only=None):
    """Decide each branch's fate. Pure apart from `last_commit_at`, so the rules are testable.

    `only` names one git branch (a just-closed PR's head): that preview branch is deleted without
    the activity checks, because the PR closing is the signal the work is done. Without it, every
    preview branch is swept, and recent activity keeps a branch whose PR has not been opened yet.
    `last_commit_at(git_branch)` returns the head commit time, None when the git branch is gone,
    or raises when GitHub cannot say; an unknown answer keeps the branch.
    """
    parents = {b.get("parent_id") for b in branches if b.get("parent_id")}
    cutoff = now - timedelta(hours=min_age_hours)
    decisions = []
    for branch in branches:
        name = branch.get("name", "")
        git_branch = name[len(PREFIX):] if name.startswith(PREFIX) else None
        if only is not None and git_branch != only:
            continue
        keep = None
        if git_branch is None:
            keep = "not a preview branch"
        elif branch.get("default"):
            keep = "default branch"
        elif branch.get("protected"):
            keep = "protected branch"
        elif branch.get("id") in parents:
            keep = "has child branches"
        elif git_branch in open_heads:
            keep = "open pull request"
        elif only is None:
            created = _parse_time(branch.get("created_at"))
            if created and created > cutoff:
                keep = f"created within {min_age_hours}h"
            else:
                try:
                    pushed = last_commit_at(git_branch)
                except Exception as exc:  # noqa: BLE001
                    keep = f"git activity unknown ({str(exc)[:80]})"
                else:
                    if pushed and pushed > cutoff:
                        keep = f"git branch pushed within {min_age_hours}h"
        decisions.append(
            {
                "id": branch.get("id", ""),
                "name": name,
                "action": "keep" if keep else "delete",
                "reason": keep or ("pull request closed" if only is not None else "no open PR and no recent activity"),
            }
        )
    return decisions


def open_pull_request_heads(repo: str, token: str, request=_request) -> set[str]:
    heads: set[str] = set()
    page = 1
    while True:
        pulls = request(
            "GET", f"{GITHUB_API}/repos/{repo}/pulls?state=open&per_page=100&page={page}", _github_headers(token)
        )
        heads.update(p["head"]["ref"] for p in pulls if (p.get("head") or {}).get("ref"))
        if len(pulls) < 100:
            return heads
        page += 1


def git_branch_last_commit(repo: str, token: str, git_branch: str, request=_request) -> datetime | None:
    url = f"{GITHUB_API}/repos/{repo}/branches/{urllib.parse.quote(git_branch, safe='')}"
    try:
        data = request("GET", url, _github_headers(token))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    commit = (data.get("commit") or {}).get("commit") or {}
    return _parse_time((commit.get("committer") or {}).get("date"))


def delete_branch(project_id: str, branch_id: str, api_key: str, request=_request, sleep=time.sleep, attempts=6) -> str:
    url = f"{NEON_API}/projects/{project_id}/branches/{branch_id}"
    for attempt in range(attempts):
        try:
            request("DELETE", url, _neon_headers(api_key))
            return "deleted"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return "already gone"
            # Neon runs one operation at a time per project and answers 423 while the previous
            # delete is still finishing, so wait it out rather than skipping the branch.
            if exc.code == 423 and attempt < attempts - 1:
                sleep(5 * (attempt + 1))
                continue
            raise
    return "deleted"


def monthly_branch_fee(count: int, included: int) -> float:
    return round(max(0, count - included) * EXTRA_BRANCH_MONTHLY_USD, 2)


def run(args, env, request=_request, sleep=time.sleep, now=None) -> tuple[dict, int]:
    api_key = env.get("NEON_API_KEY", "").strip()
    project_id = env.get("NEON_PROJECT_ID", "").strip()
    token = env.get("GITHUB_TOKEN", "").strip()
    repo = env.get("GITHUB_REPOSITORY", "").strip()
    missing = [k for k, v in (("NEON_API_KEY", api_key), ("NEON_PROJECT_ID", project_id),
                              ("GITHUB_TOKEN", token), ("GITHUB_REPOSITORY", repo)) if not v]
    if missing:
        return {"ok": False, "error": f"missing {', '.join(missing)}"}, 1

    now = now or datetime.now(UTC).replace(microsecond=0)
    branches = request("GET", f"{NEON_API}/projects/{project_id}/branches", _neon_headers(api_key)).get("branches", [])
    open_heads = open_pull_request_heads(repo, token, request)
    decisions = plan(
        branches,
        open_heads,
        lambda git_branch: git_branch_last_commit(repo, token, git_branch, request),
        now,
        args.min_age_hours,
        only=args.branch,
    )

    errors = []
    removed = 0
    for decision in decisions:
        if decision["action"] != "delete":
            continue
        if not args.execute:
            decision["result"] = "would delete"
            removed += 1
            continue
        try:
            decision["result"] = delete_branch(project_id, decision["id"], api_key, request, sleep)
            removed += 1
        except Exception as exc:  # noqa: BLE001
            detail = exc.read().decode("utf-8", "replace")[:200] if isinstance(exc, urllib.error.HTTPError) else ""
            decision["result"] = "failed"
            errors.append(f"{decision['name']}: {exc}{' ' + detail if detail else ''}")

    summary = {
        "ok": not errors,
        "ran_at": now.isoformat().replace("+00:00", "Z"),
        "mode": "execute" if args.execute else "dry_run",
        "scope": f"preview/{args.branch}" if args.branch is not None else "all preview branches",
        "branches_before": len(branches),
        "branches_after": len(branches) - removed,
        "open_pull_requests": len(open_heads),
        # The plan's allowance is not readable with a project-scoped key, so it is an input.
        "included_branches": args.included_branches,
        "monthly_branch_fee_before_usd": monthly_branch_fee(len(branches), args.included_branches),
        "monthly_branch_fee_after_usd": monthly_branch_fee(len(branches) - removed, args.included_branches),
        "deleted" if args.execute else "would_delete": [d["name"] for d in decisions if d.get("result") in ("deleted", "already gone", "would delete")],
        "kept": [{"name": d["name"], "reason": d["reason"]} for d in decisions if d["action"] == "keep"],
        "errors": errors,
    }
    return summary, 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="delete; without this the run only reports")
    parser.add_argument("--branch", help="only the preview branch for this git branch (a closed PR's head)")
    parser.add_argument("--min-age-hours", type=int, default=72,
                        help="sweep keeps branches created or pushed more recently than this")
    parser.add_argument("--included-branches", type=int, default=10, help="plan allowance: 10 Launch, 25 Scale")
    parser.add_argument("--summary-path", default="")
    args = parser.parse_args()

    summary, code = run(args, os.environ)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.summary_path:
        with open(args.summary_path, "w", encoding="utf-8") as handle:
            handle.write(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
