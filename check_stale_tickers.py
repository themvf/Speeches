#!/usr/bin/env python3
"""SEC-53 upkeep: weekly check for Industries-universe tickers that are no
longer public (acquired/merged/taken private, deregistered, or renamed).

Deliberately check-only, never auto-prunes UNIVERSE in build_industry_config.py
directly: flagging is cheap and reversible via a committed review file (same
git-as-store pattern as the KPI Tier C review queue and the attention
manipulation-defense review queue - annotate, don't silently suppress).
Removing/renaming an entry is a one-line hand edit to UNIVERSE once a flagged
row has been read, which keeps a human in the loop for the one part of this
that's a real judgment call (an ambiguous case might just be a data lag, not
an actual delisting).

Signals, cheapest/most authoritative first:
1. Presence in SEC's bulk company_tickers.json (~9k rows, one request) -
   the primary signal. SEC drops a ticker from this file once its EDGAR
   ticker association is no longer current (delisted, deregistered, or
   renamed). Reused fetch pattern from build_ticker_config.py (curl_cffi -
   sec.gov TLS-fingerprints and blocks plain `requests`).
2. For anything missing from that file, a per-CIK submissions.json fetch
   (same endpoint build_industry_config.py already uses) distinguishes:
   - "renamed": the CIK is still active and its own `tickers` field now
     lists a different symbol - not a delisting, a symbol change.
   - "deregistered": a Form 15 (15-12B/15-12G/15-15D, voluntary
     deregistration) appears in the recent filings.
   - "cik_not_found": the CIK itself 404s - strong delisting signal.
   - "no_active_ticker": the CIK is active but its own tickers field is
     empty - suggestive but not definitive.
   - "uncertain": still shows the same ticker in its own filing feed
     despite being absent from the bulk file - likely a data lag; flagged
     for a human to look at, not classified as a delisting.
3. A live Yahoo quote check (yahoo_market_data.fetch_daily_market_context)
   on flagged tickers only, as corroborating evidence attached to the
   review row - never the deciding signal by itself (Yahoo's endpoint is
   unofficial/no-SLA, per its own module docstring).

Usage:
    python check_stale_tickers.py [--config PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

CONFIG_PATH_DEFAULT = os.path.join("apps", "web", "lib", "server", "industry-config.json")
OUTPUT_PATH_DEFAULT = os.path.join("data", "ticker_prune_review.json")
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
USER_AGENT = "PolicyResearchHub ticker prune check (joshbandes@gmail.com)"
THROTTLE_S = 0.12
DEREGISTRATION_FORMS = ("15-12B", "15-12G", "15-15D")


def _fetch_bulk_tickers() -> Dict[str, str]:
    """ticker (upper) -> zero-padded CIK string, from SEC's bulk file.
    curl_cffi + browser impersonation, same as build_ticker_config.py -
    sec.gov TLS-fingerprints and blocks generic HTTP clients."""
    from curl_cffi import requests as cffi_requests

    resp = cffi_requests.get(SEC_TICKERS_URL, impersonate="chrome", timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    out: Dict[str, str] = {}
    for entry in payload.values():
        symbol = str(entry.get("ticker", "") or "").strip().upper()
        cik = entry.get("cik_str")
        if symbol and cik is not None:
            out[symbol] = str(int(cik)).zfill(10)
    if len(out) < 5000:
        raise RuntimeError(f"SEC returned only {len(out)} tickers - refusing to check against a suspiciously small universe")
    return out


def _fetch_submissions(cik: str) -> Optional[Dict[str, Any]]:
    req = urllib.request.Request(SUBMISSIONS_URL.format(cik=cik), headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def classify_candidate(ticker: str, cik: str, submissions: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Pure classification given a (possibly None, meaning 404) submissions
    payload - kept separate from the network call so it's unit-testable."""
    if submissions is None:
        return {"reason": "cik_not_found", "confidence": "high", "evidence": f"CIK {cik} returned 404 from EDGAR submissions"}

    current_tickers = [str(t).upper() for t in (submissions.get("tickers") or [])]
    recent_forms = (submissions.get("filings", {}) or {}).get("recent", {}).get("form", []) or []
    dereg_forms = sorted({f for f in recent_forms if any(f.upper().startswith(p) for p in DEREGISTRATION_FORMS)})

    if current_tickers and ticker not in current_tickers:
        return {
            "reason": "renamed",
            "confidence": "high",
            "evidence": f"CIK's own submissions now list {current_tickers} instead of {ticker}",
            "suggestedNewTicker": current_tickers[0],
        }
    if dereg_forms:
        return {
            "reason": "deregistered",
            "confidence": "high",
            "evidence": f"Recent deregistration filing(s): {dereg_forms}",
        }
    if not current_tickers:
        return {
            "reason": "no_active_ticker",
            "confidence": "medium",
            "evidence": "CIK is active in EDGAR but currently lists no ticker symbols",
        }
    return {
        "reason": "uncertain",
        "confidence": "low",
        "evidence": f"Missing from SEC's bulk ticker file, but the CIK's own feed still lists {ticker} with no deregistration filing - possibly a data lag",
    }


def _corroborate_with_yahoo(flagged: List[Dict[str, Any]]) -> None:
    """Attach a live Yahoo quote check to each flagged row, as corroborating
    evidence only - never the deciding signal (see module docstring). A
    separate function (rather than inlined) so tests can monkeypatch this to
    a no-op instead of making real network calls for every flagged ticker."""
    try:
        import yahoo_market_data
    except Exception as exc:
        print(f"  ! Yahoo corroboration skipped (import failed): {exc}", file=sys.stderr)
        return
    for row in flagged:
        try:
            ctx = yahoo_market_data.fetch_daily_market_context(row["ticker"])
            row["yahooQuoteAlive"] = ctx is not None
        except Exception as exc:
            print(f"  ! Yahoo check failed for {row['ticker']}: {exc}", file=sys.stderr)


def check_universe(config_path: str) -> Dict[str, Any]:
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    entries: List[Dict[str, str]] = []
    for industry in config.get("industries", []):
        for t in industry.get("tickers", []):
            if t.get("ticker") and t.get("cik"):
                entries.append({"ticker": t["ticker"], "name": t.get("name", ""), "cik": str(t["cik"]).zfill(10)})

    print(f"checking {len(entries)} tickers against SEC's bulk company_tickers.json...", file=sys.stderr)
    bulk = _fetch_bulk_tickers()
    print(f"  bulk file: {len(bulk)} active tickers", file=sys.stderr)

    flagged: List[Dict[str, Any]] = []
    for i, entry in enumerate(entries, 1):
        ticker, cik = entry["ticker"], entry["cik"]
        if bulk.get(ticker) == cik:
            continue  # still active under the same CIK, nothing to do
        time.sleep(THROTTLE_S)
        try:
            submissions = _fetch_submissions(cik)
        except Exception as exc:  # network hiccup on one ticker must not fail the whole run
            flagged.append({**entry, "reason": "check_failed", "confidence": "low", "evidence": str(exc)})
            continue
        classification = classify_candidate(ticker, cik, submissions)
        flagged.append({**entry, **classification})
        if i % 100 == 0:
            print(f"  {i}/{len(entries)} checked, {len(flagged)} flagged so far", file=sys.stderr)

    _corroborate_with_yahoo(flagged)

    return {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkedCount": len(entries),
        "flaggedCount": len(flagged),
        "candidates": flagged,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_PATH_DEFAULT)
    args = parser.parse_args()

    result = check_universe(args.config)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)

    print(f"\nwrote {args.output}")
    print(f"  checked: {result['checkedCount']} | flagged: {result['flaggedCount']}")
    by_reason: Dict[str, int] = {}
    for c in result["candidates"]:
        by_reason[c["reason"]] = by_reason.get(c["reason"], 0) + 1
    for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"    {reason:18} {n}")
    for c in result["candidates"]:
        note = f" -> suggest {c['suggestedNewTicker']}" if c.get("suggestedNewTicker") else ""
        print(f"  {c['ticker']:6} {c['reason']:16} ({c['confidence']}){note}: {c['evidence']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
