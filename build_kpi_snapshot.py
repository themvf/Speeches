#!/usr/bin/env python3
"""SEC-8: bake the CBOE KPI tab's static snapshot for all public companies.

Extends the AAPL/GOOGL pilot to every public company in kpi_config.COMPANY_KPIS
using kpi_pilot's SEC-XBRL extraction, and writes the web shape
(apps/web/lib/server/kpi-pilot-data.json). Same static-snapshot-ahead-of-live
pattern as the CBOE pilot (SEC-19) and the Prediction Markets tab - SEC-9/10
build the live daily-refreshing pipeline behind the same API contract later.

Only KPIs that actually resolved (non-empty series) are written, so an
unmapped segment silently drops rather than shipping an empty card. Companies
that produce zero KPIs are omitted with a warning.

Usage: python build_kpi_snapshot.py [--quarters 8]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import kpi_config
import kpi_pilot

OUT_PATH = os.path.join("apps", "web", "lib", "server", "kpi-pilot-data.json")


def build(quarters: int) -> dict:
    raw = kpi_pilot.run(quarters, company_kpis=kpi_config.COMPANY_KPIS)
    companies: dict = {}
    dropped: list = []
    for ticker, cdata in raw["companies"].items():
        kpis_out = {}
        for kpi_key, kpi in cdata["kpis"].items():
            series = kpi.get("series") or []
            if not series:
                dropped.append(f"{ticker}.{kpi_key}")
                continue
            # Banks tag the income-statement provision line as a negative
            # (it reduces income); flip it so it renders as the positive
            # expense the companies report publicly ("provision of $2.5B").
            negate = kpi_key == "provision_credit_losses"
            kpis_out[kpi_key] = {
                "label": kpi["label"],
                "unit": kpi["unit"],
                "series": [
                    {"end": p["period_end"], "value": -p["value"] if negate else p["value"], "derived": p["derived"]}
                    for p in series
                ],
            }
        if not kpis_out:
            print(f"[{ticker}] no KPIs resolved - omitting", file=sys.stderr)
            continue
        companies[ticker] = {"name": kpi_config.NAMES.get(ticker, cdata["name"]), "kpis": kpis_out}

    return {
        "generatedAt": time.strftime("%Y-%m-%d", time.gmtime()),
        "source": "SEC XBRL via edgartools (SEC-8: all public companies; Tier A + segment revenues)",
        "companies": companies,
        "_dropped": dropped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarters", type=int, default=8)
    args = parser.parse_args()

    snapshot = build(args.quarters)
    dropped = snapshot.pop("_dropped")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=1)

    total_kpis = sum(len(c["kpis"]) for c in snapshot["companies"].values())
    print(f"\nwrote {OUT_PATH}")
    print(f"  companies: {len(snapshot['companies'])} | KPIs with data: {total_kpis}")
    if dropped:
        print(f"  dropped (no data): {len(dropped)} -> {', '.join(dropped)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
