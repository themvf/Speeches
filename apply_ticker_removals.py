#!/usr/bin/env python3
"""SEC-53: automatically apply the high-confidence "deregistered" removals
from check_stale_tickers.py's weekly review file.

Deliberately narrow scope: only reason == "deregistered" and confidence ==
"high" (a Form 15 filing actually on record) gets auto-applied. Renames and
anything below high confidence stay flag-only, on purpose - a rename means
picking the right symbol out of possibly several candidates (a preferred-
share suffix, a bankruptcy-flagged ticker, a multi-symbol redomiciliation),
and that's exactly the kind of judgment call this script does not attempt.
See the 2026-08-20 cleanup commit for what that judgment actually looks like
in practice (28 removed, 10 renamed, 2 deliberately left alone after
individually checking each against EDGAR's formerNames/tickers/exchanges
fields) - this script only ever automates the least ambiguous slice of that.

A pure removal doesn't need an EDGAR refetch, so this skips
build_industry_config.py's 15-20 minute full rebuild entirely and edits the
already-committed artifacts directly:

  1. UNIVERSE in build_industry_config.py - the source-of-truth ticker list
  2. SUB_INDUSTRY_GROUPS in the same file, wherever the ticker is tagged
  3. apps/web/lib/server/industry-config.json - the served snapshot (drops
     the ticker from its industry's list; drops the whole industry if that
     empties it)
  4. industry_state.json's "latest" map - the filing-watch's per-ticker state
  5. The review file itself - removed tickers are dropped from
     "candidates" so next week's GitHub issue (and human eyes) only ever
     see what's still actually pending

Two guards, because the workflow pushes the result straight to main
unattended and build_industry_config.py is also what the HOURLY filing
watch rebuilds from - a bad edit here would break that too, not just
this weekly job:

  - A run proposing more than MAX_AUTO_REMOVALS removals applies nothing
    and leaves the batch flagged (anomaly circuit breaker).
  - Every edit is validated afterwards (builder source still parses as
    Python, UNIVERSE lost exactly the intended tickers and nothing else,
    config JSON still self-consistent). Any failure restores all four
    files byte-for-byte and exits non-zero, so the workflow commits
    nothing rather than committing damage.

Usage: python apply_ticker_removals.py [--review PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from typing import Any, Dict, List

REVIEW_PATH_DEFAULT = "data/ticker_prune_review.json"
BUILDER_PATH = "build_industry_config.py"
CONFIG_PATH = "apps/web/lib/server/industry-config.json"
STATE_PATH = "industry_state.json"

# Circuit breaker. Steady state should be 0-5 deregistrations a week for an
# ~890-company universe; the one genuinely large batch (26) was the initial
# backlog, cleared by hand. A run proposing more than this is anomalous
# (bad upstream data, a classification regression), so it auto-applies
# nothing and leaves the whole batch flagged for a human instead - the
# weekly issue still reports it, so refusing degrades to the old
# flag-only behavior rather than breaking the pipeline.
MAX_AUTO_REMOVALS = 25


def high_confidence_deregistrations(review: Dict[str, Any]) -> List[str]:
    """Pure filter - the one place the "what's safe to automate" policy
    lives, so it's unit-testable without touching any file."""
    return sorted(
        c["ticker"] for c in review.get("candidates", [])
        if c.get("reason") == "deregistered" and c.get("confidence") == "high"
    )


def _remove_tokens_from_block(text: str, block_start_marker: str, open_ch: str, close_ch: str, tickers: List[str]) -> tuple[str, List[str]]:
    """Removes `"TICKER",` tokens (exact quoted match) from the bracketed
    block that starts at block_start_marker, leaving everything else in the
    file untouched. Returns (new_text, tickers_actually_removed)."""
    start = text.index(block_start_marker)
    # block_start_marker is written to end with the block's own opening
    # bracket (e.g. "...List[str] = ["), so the marker's last character IS
    # open_at - searching for open_ch instead would find an EARLIER bracket
    # if the marker text itself contains one (e.g. "List[str]" has a "["
    # before the list literal actually starts).
    open_at = start + len(block_start_marker) - 1
    assert text[open_at] == open_ch, f"block_start_marker must end with {open_ch!r}"
    depth = 0
    end = open_at
    for i in range(open_at, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    block = text[open_at:end]
    removed: List[str] = []
    for tk in tickers:
        pattern = r'"' + re.escape(tk) + r'",\s*'
        new_block, n = re.subn(pattern, "", block)
        if n:
            block = new_block
            removed.append(tk)
    return text[:open_at] + block + text[end:], removed


def remove_from_builder_source(path: str, tickers: List[str]) -> Dict[str, List[str]]:
    """Removes tickers from both UNIVERSE and SUB_INDUSTRY_GROUPS in
    build_industry_config.py's source. Returns which tickers were actually
    found/removed in each block, for logging - a ticker missing from
    SUB_INDUSTRY_GROUPS is normal (most tickers aren't tagged), but one
    missing from UNIVERSE means it was already removed in an earlier run."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    text, removed_universe = _remove_tokens_from_block(text, "UNIVERSE: List[str] = [", "[", "]", tickers)
    text, removed_sub_industry = _remove_tokens_from_block(text, "SUB_INDUSTRY_GROUPS: Dict[str, List[str]] = {", "{", "}", tickers)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return {"universe": removed_universe, "sub_industry_groups": removed_sub_industry}


def remove_from_committed_config(path: str, tickers: List[str]) -> List[str]:
    """Drops the tickers' entries from the served snapshot directly (no
    EDGAR refetch needed for a removal). Drops an industry group entirely if
    removing its members empties it, so a fully-deregistered SIC bucket
    doesn't leave a zero-member row in the tab."""
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    ticker_set = set(tickers)
    removed: List[str] = []
    new_industries = []
    for industry in config.get("industries", []):
        kept = []
        for t in industry.get("tickers", []):
            if t.get("ticker") in ticker_set:
                removed.append(t["ticker"])
                continue
            kept.append(t)
        if kept:
            industry["tickers"] = kept
            new_industries.append(industry)
    config["industries"] = new_industries
    config["tickerCount"] = sum(len(i["tickers"]) for i in new_industries)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=1)
    return removed


def remove_from_state(path: str, tickers: List[str]) -> List[str]:
    try:
        with open(path, encoding="utf-8") as handle:
            state = json.load(handle)
    except FileNotFoundError:
        return []
    latest = state.get("latest", {})
    removed = [t for t in tickers if t in latest]
    for t in removed:
        del latest[t]
    state["latest"] = latest
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=1, sort_keys=True)
    return removed


def update_review_file(path: str, review: Dict[str, Any], applied: List[str]) -> None:
    """Drops the just-applied candidates from the review file so next week's
    (and any human's) view of it only shows what's still actually pending."""
    applied_set = set(applied)
    review["candidates"] = [c for c in review["candidates"] if c["ticker"] not in applied_set]
    review["flaggedCount"] = len(review["candidates"])
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(review, handle, indent=1)


def extract_universe_tickers(source_text: str) -> List[str]:
    """UNIVERSE's contents, parsed out of the builder source via AST.

    Uses ast.parse rather than exec/import on purpose: it validates that the
    file is still syntactically valid Python AND reads the list, in one step,
    without running any of the module's code. This is the check that catches
    a text-surgery bug corrupting the file - the exact failure mode that the
    List[str] bracket bug would have produced if it hadn't been caught in
    testing first."""
    tree = ast.parse(source_text)
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
        if target == "UNIVERSE":
            if not isinstance(node.value, ast.List):
                raise ValueError("UNIVERSE is no longer a list literal")
            return [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    raise ValueError("UNIVERSE assignment not found in builder source")


def validate_edits(builder_path: str, config_path: str, before_universe: List[str], expected_removed: List[str]) -> None:
    """Post-edit sanity check on the two artifacts a bad text edit could
    actually corrupt. Raises with a specific message on any mismatch; main()
    turns that into a full restore of every touched file, so a failed run
    leaves the repo exactly as it found it rather than committing damage."""
    with open(builder_path, encoding="utf-8") as handle:
        after_universe = extract_universe_tickers(handle.read())  # also re-parses the whole file

    expected = [t for t in before_universe if t not in set(expected_removed)]
    if after_universe != expected:
        unexpectedly_gone = sorted(set(expected) - set(after_universe))
        unexpectedly_kept = sorted(set(after_universe) - set(expected))
        raise ValueError(
            f"UNIVERSE mismatch after edit - unexpectedly removed: {unexpectedly_gone}, "
            f"unexpectedly still present: {unexpectedly_kept}"
        )

    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)  # re-parses; catches a corrupted JSON write
    actual = sum(len(i["tickers"]) for i in config.get("industries", []))
    if config.get("tickerCount") != actual:
        raise ValueError(f"config tickerCount ({config.get('tickerCount')}) disagrees with actual rows ({actual})")
    for industry in config.get("industries", []):
        if not industry.get("tickers"):
            raise ValueError(f"industry {industry.get('label')!r} left with zero members")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", default=REVIEW_PATH_DEFAULT)
    parser.add_argument("--builder", default=BUILDER_PATH, help="build_industry_config.py path (override for testing)")
    parser.add_argument("--config", default=CONFIG_PATH, help="Committed industry-config.json path (override for testing)")
    parser.add_argument("--state", default=STATE_PATH, help="industry_state.json path (override for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be removed without writing any file")
    args = parser.parse_args()

    with open(args.review, encoding="utf-8") as handle:
        review = json.load(handle)
    candidates = high_confidence_deregistrations(review)

    if not candidates:
        print("no high-confidence deregistrations to apply", file=sys.stderr)
        print("")
        return 0

    if len(candidates) > MAX_AUTO_REMOVALS:
        # Anomalous batch - apply nothing, leave everything flagged. The
        # weekly issue still lists them all, so this degrades to the old
        # flag-only behavior instead of either auto-applying a suspicious
        # mass removal or failing the workflow outright.
        print(
            f"REFUSING to auto-apply {len(candidates)} removals (cap is {MAX_AUTO_REMOVALS}) - "
            "leaving the whole batch flagged for manual review",
            file=sys.stderr,
        )
        print("")
        return 0

    print(f"applying {len(candidates)} high-confidence deregistration(s): {candidates}", file=sys.stderr)
    if args.dry_run:
        print("(dry run - no files written)", file=sys.stderr)
        print(" ".join(candidates))
        return 0

    # Snapshot every file we are about to touch. These edits span four files
    # and the workflow pushes the result straight to main, so a partial
    # application is the thing to avoid: any failure below restores all of
    # them and reports nothing applied.
    touched = [args.builder, args.config, args.state, args.review]
    originals: Dict[str, bytes] = {}
    for path in touched:
        try:
            with open(path, "rb") as handle:
                originals[path] = handle.read()
        except FileNotFoundError:
            pass  # remove_from_state tolerates a missing state file

    with open(args.builder, encoding="utf-8") as handle:
        before_universe = extract_universe_tickers(handle.read())

    try:
        source_result = remove_from_builder_source(args.builder, candidates)
        config_removed = remove_from_committed_config(args.config, candidates)
        state_removed = remove_from_state(args.state, candidates)
        update_review_file(args.review, review, candidates)
        validate_edits(args.builder, args.config, before_universe, source_result["universe"])
    except Exception as exc:
        for path, blob in originals.items():
            with open(path, "wb") as handle:
                handle.write(blob)
        print(f"VALIDATION FAILED, restored all files unchanged: {exc}", file=sys.stderr)
        print("")
        return 1

    missing_from_universe = [t for t in candidates if t not in source_result["universe"]]
    if missing_from_universe:
        print(f"  ! not found in UNIVERSE (already removed?): {missing_from_universe}", file=sys.stderr)
    missing_from_config = [t for t in candidates if t not in config_removed]
    if missing_from_config:
        print(f"  ! not found in committed config (already removed?): {missing_from_config}", file=sys.stderr)

    print(f"  UNIVERSE: removed {len(source_result['universe'])}", file=sys.stderr)
    print(f"  SUB_INDUSTRY_GROUPS: removed {len(source_result['sub_industry_groups'])}", file=sys.stderr)
    print(f"  committed config: removed {len(config_removed)}", file=sys.stderr)
    print(f"  filing-watch state: removed {len(state_removed)}", file=sys.stderr)
    print("  validated: builder still parses, UNIVERSE matches, config self-consistent", file=sys.stderr)

    print(" ".join(candidates))
    return 0


if __name__ == "__main__":
    sys.exit(main())
