#!/usr/bin/env python3
"""SEC-50: 8-K + Form 4 catalyst events for the tracked industry universe.

Feeds the "why is this moving?" chips on the Movers and Reddit tabs. Two
EDGAR sources:

  --detect     intraday: the market-wide getcurrent Atom feed for 8-K and
               Form 4 (2 requests; note type=4 prefix-matches 424B*, so
               entries are re-filtered by EXACT form from the title, and
               Form 4 owner entries "(Reporting)" are skipped so the CIK is
               always the issuer's).
  --reconcile  daily: yesterday's form.idx daily index (1 request, the
               complete day) - catches anything the atom window missed.
               (Today's idx doesn't exist intraday - verified 2026-07-17.)

Both phases only register filings whose CIK is in the committed
industry-config.json universe (~522 tickers). Rows land immediately with
detail_status=pending; a capped per-run enrichment pass then fills:
  8-K   -> item codes from the issuer's submissions JSON (one request per
           distinct CIK, covers all its pending 8-Ks at once)
  Form 4-> a buy/sell summary parsed via edgartools (per-filing request);
           parse failures keep a generic label rather than blocking.

Requires DATABASE_URL. Research context only - not investment advice.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import neon_feeds
from source_health import record_source_health

SOURCE_KEY = "filing_catalyst_sync"
USER_AGENT = "PolicyResearchHub filing watch (joshbandes@gmail.com)"
ATOM_URL = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent"
            "&type={form}&company=&dateb=&owner=include&count=100&output=atom")
IDX_URL = "https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{qtr}/form.{ymd}.idx"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
INDUSTRY_CONFIG_PATH = "apps/web/lib/server/industry-config.json"
ENRICH_CAP_DETECT = 30
ENRICH_CAP_RECONCILE = 60
RETENTION_DAYS = 30
THROTTLE_S = 0.15

FORMS = ("8-K", "4")

# Atom titles look like: "8-K - COMPANY NAME (0001234567) (Filer)" or
# "4 - COMPANY NAME (0001234567) (Issuer)" / "(Reporting)" for the owner.
_ATOM_ENTRY_RE = re.compile(
    r"<title>(?P<form>[A-Z0-9/-]+)\s+-\s+.*?\((?P<cik>\d{10})\)\s+\((?P<role>Filer|Issuer|Reporting)\)</title>.*?"
    r"accession-number=(?P<accession>[\d-]+)",
    re.DOTALL,
)

# form.idx fixed-ish columns: FORM  COMPANY  CIK  DATE  path/accession.txt
_IDX_LINE_RE = re.compile(
    r"^(?P<form>\S[^ ]*)\s{2,}.*?\s(?P<cik>\d+)\s+(?P<date>\d{8})\s+edgar/data/\d+/(?P<accession>[\d-]+)\.txt",
)


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read().decode("utf-8", errors="replace")


def load_tracked_ciks(path: str = INDUSTRY_CONFIG_PATH) -> Dict[str, str]:
    """cik (no leading zeros) -> ticker, from the committed industry config."""
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    out: Dict[str, str] = {}
    for industry in config.get("industries", []):
        for entry in industry.get("tickers", []):
            cik = str(entry.get("cik") or "").lstrip("0")
            if cik:
                out[cik] = str(entry["ticker"])
    return out


def filing_index_url(cik: str, accession: str) -> str:
    return (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{accession.replace('-', '')}/{accession}-index.htm")


def parse_atom(xml: str, tracked: Dict[str, str]) -> List[Dict[str, Any]]:
    """Tracked-issuer filings from a getcurrent page. Exact form match only
    (the type= param prefix-matches, e.g. 4 -> 424B2); owner-side Form 4
    entries ('Reporting') are skipped so cik/ticker is the ISSUER."""
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for m in _ATOM_ENTRY_RE.finditer(xml):
        form = m.group("form")
        if form not in FORMS or m.group("role") == "Reporting":
            continue
        cik = m.group("cik").lstrip("0")
        ticker = tracked.get(cik)
        accession = m.group("accession")
        if not ticker or accession in seen:
            continue
        seen.add(accession)
        out.append({
            "accession": accession, "cik": cik, "ticker": ticker, "form": form,
            "filed_at": datetime.now(UTC), "url": filing_index_url(cik, accession),
        })
    return out


def parse_form_idx(text: str, tracked: Dict[str, str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for line in text.splitlines():
        m = _IDX_LINE_RE.match(line)
        if not m or m.group("form") not in FORMS:
            continue
        cik = m.group("cik").lstrip("0")
        ticker = tracked.get(cik)
        accession = m.group("accession")
        if not ticker or accession in seen:
            continue
        seen.add(accession)
        filed = datetime.strptime(m.group("date"), "%Y%m%d").replace(tzinfo=UTC)
        out.append({
            "accession": accession, "cik": cik, "ticker": ticker, "form": m.group("form"),
            "filed_at": filed, "url": filing_index_url(cik, accession),
        })
    return out


def eightk_items_from_submissions(cik: str) -> Dict[str, str]:
    """accession -> items string ('2.02,9.01') from the issuer's submissions
    JSON - one request covers every pending 8-K for that company."""
    data = json.loads(_fetch(SUBMISSIONS_URL.format(cik=cik)))
    recent = data.get("filings", {}).get("recent", {})
    out: Dict[str, str] = {}
    for accession, form, items in zip(
        recent.get("accessionNumber", []), recent.get("form", []), recent.get("items", [])
    ):
        if form == "8-K":
            out[str(accession)] = str(items or "")
    return out


def summarize_form4(transactions: List[Tuple[str, float, Optional[float]]]) -> str:
    """Pure: net a Form 4's non-derivative transactions into a chip label.
    transactions = [(code, shares, price_or_None)]. P/A add, S/D/F reduce."""
    bought = sold = 0.0
    value = 0.0
    for code, shares, price in transactions:
        code = (code or "").upper()
        try:
            shares = float(shares or 0)
        except (TypeError, ValueError):
            continue
        if code in ("P", "A"):
            bought += shares
            value += shares * float(price or 0)
        elif code in ("S", "D", "F"):
            sold += shares
            value -= shares * float(price or 0)
    net = bought - sold
    if net == 0:
        return "Insider transaction"
    direction = "bought" if net > 0 else "sold"
    dollars = abs(value)
    if dollars >= 1e6:
        amount = f"${dollars / 1e6:.1f}M"
    elif dollars >= 1e3:
        amount = f"${dollars / 1e3:.0f}K"
    else:
        amount = f"{abs(net):,.0f} shares"
    return f"Insider {direction} {amount}"


def _form4_summary_via_edgartools(accession: str) -> str:
    from edgar import find, set_identity
    set_identity("joshbandes@gmail.com")
    filing = find(accession)
    ownership = filing.obj()
    transactions: List[Tuple[str, float, Optional[float]]] = []
    table = getattr(ownership, "market_trades", None)
    frame = table.data if table is not None and hasattr(table, "data") else None
    if frame is not None and len(frame):
        for _, row in frame.iterrows():
            transactions.append((
                str(row.get("Code", row.get("code", ""))),
                row.get("Shares", row.get("shares", 0)),
                row.get("Price", row.get("price")),
            ))
    if not transactions:
        return "Insider transaction"
    return summarize_form4(transactions)


def enrich_pending(cap: int, summary: Dict[str, Any]) -> None:
    pending = neon_feeds.get_pending_filing_events(cap)
    summary["enrich_attempted"] = len(pending)
    items_cache: Dict[str, Dict[str, str]] = {}
    done = 0
    for event in pending:
        accession = str(event["accession"])
        try:
            if event["form"] == "8-K":
                cik = str(event["cik"])
                if cik not in items_cache:
                    time.sleep(THROTTLE_S)
                    items_cache[cik] = eightk_items_from_submissions(cik)
                items = items_cache[cik].get(accession, "")
                label = f"8-K items {items}" if items else "8-K filed"
                neon_feeds.update_filing_event_detail(accession, items, label)
            else:
                time.sleep(THROTTLE_S)
                text = _form4_summary_via_edgartools(accession)
                neon_feeds.update_filing_event_detail(accession, "", text)
            done += 1
        except Exception as exc:
            # Keep a generic-but-done label - a chip with a link is still
            # useful, and retrying a permanently-unparseable filing forever
            # would eat every run's cap.
            try:
                fallback = "8-K filed" if event["form"] == "8-K" else "Insider transaction"
                neon_feeds.update_filing_event_detail(accession, "", fallback)
            except Exception:
                pass
            summary.setdefault("enrich_errors", []).append(f"{event['ticker']} {accession}: {exc}")
    summary["enrich_done"] = done


def run(mode: str, summary: Dict[str, Any]) -> None:
    tracked = load_tracked_ciks()
    summary["tracked_ciks"] = len(tracked)

    filings: List[Dict[str, Any]] = []
    if mode == "detect":
        for form in FORMS:
            try:
                time.sleep(THROTTLE_S)
                filings.extend(parse_atom(_fetch(ATOM_URL.format(form=form)), tracked))
            except Exception as exc:
                summary.setdefault("errors", []).append(f"atom {form}: {exc}")
    else:
        day = datetime.now(UTC) - timedelta(days=1)
        # Walk back over weekends/holidays to the last day with an index.
        for _ in range(5):
            url = IDX_URL.format(year=day.year, qtr=(day.month - 1) // 3 + 1, ymd=day.strftime("%Y%m%d"))
            try:
                time.sleep(THROTTLE_S)
                filings = parse_form_idx(_fetch(url), tracked)
                summary["idx_day"] = day.strftime("%Y-%m-%d")
                break
            except Exception:
                day -= timedelta(days=1)
        else:
            summary.setdefault("errors", []).append("no daily form.idx found in the last 5 days")

    summary["matched_filings"] = len(filings)
    if filings:
        summary["inserted"] = neon_feeds.insert_filing_events(filings)

    enrich_pending(ENRICH_CAP_DETECT if mode == "detect" else ENRICH_CAP_RECONCILE, summary)
    summary["pruned"] = neon_feeds.prune_old_filing_events(RETENTION_DAYS)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--detect", action="store_true")
    group.add_argument("--reconcile", action="store_true")
    args = parser.parse_args(argv)
    mode = "detect" if args.detect else "reconcile"

    summary: Dict[str, Any] = {
        "source_key": SOURCE_KEY, "connector": SOURCE_KEY, "mode": mode,
        "errors": [], "ran_at": datetime.now(UTC).isoformat(),
    }
    try:
        run(mode, summary)
        summary["ok"] = not summary["errors"]
    except Exception as exc:
        summary["errors"].append(str(exc))
        summary["ok"] = False

    summary["failed_count"] = len(summary["errors"]) + len(summary.get("enrich_errors", []))
    summary["processed_count"] = summary.get("inserted", 0) + summary.get("enrich_done", 0)
    summary["discovered_count"] = summary.get("matched_filings", 0)
    record_source_health(summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
