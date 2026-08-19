#!/usr/bin/env python3
"""SEC-53 upkeep: detect new 10-Q/10-K filings for the Industries tab's
~917-ticker universe, same pattern as kpi_filing_watch.py (SEC-34).

Two detection modes:

  --detect  EDGAR market-wide "current filings" Atom feed
            (action=getcurrent), 2 requests total. Near-real-time (entries
            appear within minutes of EDGAR acceptance) but window-limited to
            the most recent ~100 filings per form ACROSS ALL EDGAR FILERS,
            not just ours - on a high-volume filing day a small filer in our
            917-ticker universe could scroll out of that window before an
            hourly poll catches it.
  --full    Per-CIK data.sec.gov submissions JSON (~917 requests, throttled -
            about 2 minutes, well under SEC's own suggested rate limit). The
            once-daily reconciliation that catches anything the hourly atom
            window missed. Unlike build_industry_config.py's full rebuild,
            this is submissions-only detection - no XBRL frames fetch, no
            edgar.Company() lookups - so it stays cheap even at 917 tickers.

Unlike kpi_filing_watch.py, there's no separate ciks map to keep in sync:
CIKs are read straight from the committed industry-config.json (already has
one per ticker), so this can never drift out of step with the universe the
way a second hand-maintained list could.

State lives in the committed industry_state.json: {"latest": {ticker:
{form, accession, filed}}}. Prints changed tickers (comma-separated, for
build_industry_config.py --tickers) on stdout; diagnostics go to stderr.
Never writes state unless --commit-state is passed - the workflow rebuilds
first and only then commits state + snapshot together, so a failed rebuild
can't strand a filing as "seen but not ingested" (same ordering guarantee as
SEC-34).

Amendments (10-Q/A, 10-K/A) are ignored by exact form match, matching
kpi_filing_watch.py - they rarely carry full XBRL and the original filing's
numbers are what the frames API and this tab track.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from typing import Any, Dict, List

from kpi_filing_watch import diff_against_state, parse_atom_entries

STATE_PATH = "industry_state.json"
CONFIG_PATH = "apps/web/lib/server/industry-config.json"
FORMS = ("10-Q", "10-K")
USER_AGENT = "PolicyResearchHub industry filing watch (joshbandes@gmail.com)"
ATOM_URL = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent"
            "&type={form}&company=&dateb=&owner=include&count=100&output=atom")
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
THROTTLE_S = 0.12


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def load_ciks_from_config(config_path: str) -> Dict[str, str]:
    """ticker -> CIK (no leading zeros, matching kpi_filing_watch's
    convention), read from the committed config rather than a second
    hand-maintained list - one less thing to keep in sync as the universe
    grows or a ticker gets pruned."""
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    ciks: Dict[str, str] = {}
    for industry in config.get("industries", []):
        for t in industry.get("tickers", []):
            if t.get("ticker") and t.get("cik"):
                ciks[t["ticker"]] = str(int(t["cik"]))
    return ciks


def detect_via_atom(ciks: Dict[str, str], latest: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    seen: List[Any] = []
    for form in FORMS:
        try:
            seen.extend(parse_atom_entries(_fetch(ATOM_URL.format(form=form))))
        except Exception as exc:
            print(f"[watch] atom fetch failed for {form}: {exc}", file=sys.stderr)
    return diff_against_state(seen, ciks, latest)


def detect_via_submissions(ciks: Dict[str, str], latest: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    changed: Dict[str, Dict[str, str]] = {}
    for i, (ticker, cik) in enumerate(ciks.items(), 1):
        time.sleep(THROTTLE_S)
        try:
            data = json.loads(_fetch(SUBMISSIONS_URL.format(cik=cik)))
        except Exception as exc:
            print(f"[watch] submissions fetch failed for {ticker}: {exc}", file=sys.stderr)
            continue
        recent = data.get("filings", {}).get("recent", {})
        for form, accession, filed in zip(recent.get("form", []), recent.get("accessionNumber", []), recent.get("filingDate", [])):
            if form not in FORMS:
                continue
            if latest.get(ticker, {}).get("accession") != accession:
                changed[ticker] = {"form": form, "accession": accession, "filed": filed}
            break  # newest 10-Q/10-K only
        if i % 200 == 0:
            print(f"[watch] {i}/{len(ciks)} checked, {len(changed)} changed so far", file=sys.stderr)
    return changed


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--detect", action="store_true", help="Atom feed (2 requests, near-real-time)")
    mode.add_argument("--full", action="store_true", help="Per-CIK submissions (complete reconciliation)")
    parser.add_argument("--commit-state", action="store_true",
                        help="Write detected accessions into industry_state.json (run AFTER a successful rebuild)")
    parser.add_argument("--state", default=STATE_PATH)
    parser.add_argument("--config", default=CONFIG_PATH)
    args = parser.parse_args(argv)

    ciks = load_ciks_from_config(args.config)
    try:
        with open(args.state, encoding="utf-8") as handle:
            state = json.load(handle)
    except FileNotFoundError:
        state = {}
    latest: Dict[str, Dict[str, str]] = state.get("latest", {})

    changed = detect_via_atom(ciks, latest) if args.detect else detect_via_submissions(ciks, latest)

    for ticker, info in sorted(changed.items()):
        print(f"[watch] {ticker}: new {info['form']} {info['accession']}", file=sys.stderr)
    if args.commit_state and changed:
        for ticker, info in changed.items():
            latest[ticker] = info
        state["latest"] = latest
        with open(args.state, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=1, sort_keys=True)
        print(f"[watch] state updated for {len(changed)} ticker(s)", file=sys.stderr)

    print(" ".join(sorted(changed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
