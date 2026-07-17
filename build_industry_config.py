#!/usr/bin/env python3
"""SEC-53: bake the Industries tab config from EDGAR SIC classifications.

For a curated universe of liquid/tracked tickers (incl. the 22 CBOE KPI
companies and the Reddit-attention regulars), fetch each company's SIC code +
description from the data.sec.gov submissions API (free, no auth - the same
endpoint the KPI filing watch uses in production) and group tickers into
industries by SIC description. Writes the committed
apps/web/lib/server/industry-config.json the /api/market/industries route
reads - same static-config pattern as ticker_config.json / kpi_config.py:
industries only change when a company re-registers, so this doesn't earn a
live pipeline. Rerun + commit to refresh or grow the universe.

Usage: python build_industry_config.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from typing import Any, Dict, List

OUT_PATH = os.path.join("apps", "web", "lib", "server", "industry-config.json")
USER_AGENT = "PolicyResearchHub industry config (joshbandes@gmail.com)"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
THROTTLE_S = 0.15

# Curated universe: major names across industries + this app's tracked sets
# (CBOE KPI companies, Reddit-attention regulars). Grown by editing this list
# and rerunning.
UNIVERSE: List[str] = [
    # Semiconductors & equipment
    "NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU", "TXN", "AMAT", "TSM", "SMCI", "MRVL", "ARM",
    # Software / internet / megacap tech
    "MSFT", "GOOGL", "META", "AAPL", "AMZN", "NFLX", "CRM", "ORCL", "ADBE", "NOW",
    "PLTR", "SNOW", "UBER", "ABNB", "SHOP", "SPOT", "IBM", "CSCO", "DELL", "HPQ",
    # Banks & brokers
    "JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "PNC", "TFC", "SCHW",
    # Payments / fintech
    "V", "MA", "PYPL", "AXP", "SOFI", "HOOD", "COIN",
    # Crypto-adjacent
    "MARA", "RIOT", "CLSK", "MSTR",
    # Pharma
    "LLY", "PFE", "MRK", "JNJ", "ABBV", "BMY",
    # Biotech
    "MRNA", "AMGN", "GILD", "REGN", "VRTX",
    # Managed care / health services
    "UNH", "CVS", "CI", "HUM", "ELV",
    # Retail
    "WMT", "TGT", "COST", "HD", "LOW", "DG",
    # Apparel / consumer
    "NKE", "LULU", "KO", "PEP", "PG",
    # Autos & EV
    "TSLA", "F", "GM", "RIVN", "LCID", "NIO",
    # Energy
    "XOM", "CVX", "COP", "OXY", "SLB", "HAL",
    # Airlines
    "DAL", "UAL", "AAL", "LUV",
    # Defense / aerospace
    "LMT", "RTX", "NOC", "GD", "BA",
    # Media / telecom
    "DIS", "CMCSA", "T", "VZ", "TMUS", "WBD",
    # Restaurants
    "MCD", "SBUX", "CMG", "DPZ",
    # Industrials
    "CAT", "DE", "GE", "HON", "UNP",
    # Utilities / power
    "NEE", "DUK", "PCG",
    # Attention regulars
    "GME", "AMC", "BB",
]


def _fetch_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def group_by_industry(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pure: group {ticker,name,cik,sic,sic_description} records into
    industries keyed by SIC description, largest first (ties: label asc).
    Records missing a SIC land in an 'Unclassified' bucket."""
    buckets: Dict[str, Dict[str, Any]] = {}
    for record in records:
        label = str(record.get("sic_description") or "").strip() or "Unclassified"
        sic = str(record.get("sic") or "")
        bucket = buckets.setdefault(label, {"sic": sic, "label": label, "tickers": []})
        bucket["tickers"].append({
            "ticker": record["ticker"],
            "name": str(record.get("name") or record["ticker"]),
            "cik": str(record.get("cik") or ""),
        })
    industries = sorted(buckets.values(), key=lambda b: (-len(b["tickers"]), b["label"]))
    for industry in industries:
        industry["tickers"].sort(key=lambda t: t["ticker"])
    return industries


def main() -> int:
    from edgar import Company, set_identity
    set_identity("joshbandes@gmail.com")

    records: List[Dict[str, Any]] = []
    for i, ticker in enumerate(UNIVERSE, 1):
        try:
            company = Company(ticker)
            cik = int(company.cik)
        except Exception as exc:
            print(f"[{ticker}] CIK lookup failed, skipping: {exc}", file=sys.stderr)
            continue
        try:
            time.sleep(THROTTLE_S)
            data = _fetch_json(SUBMISSIONS_URL.format(cik=cik))
        except Exception as exc:
            print(f"[{ticker}] submissions fetch failed, skipping: {exc}", file=sys.stderr)
            continue
        records.append({
            "ticker": ticker,
            "name": str(data.get("name") or company.name or ticker).title(),
            "cik": str(cik),
            "sic": str(data.get("sic") or ""),
            "sic_description": str(data.get("sicDescription") or ""),
        })
        if i % 20 == 0:
            print(f"  {i}/{len(UNIVERSE)} fetched", file=sys.stderr)

    industries = group_by_industry(records)
    payload = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "SEC EDGAR submissions API (SIC classification)",
        "tickerCount": len(records),
        "industries": industries,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)

    print(f"\nwrote {OUT_PATH}")
    print(f"  tickers: {len(records)} | industries: {len(industries)}")
    for industry in industries[:12]:
        members = ", ".join(t["ticker"] for t in industry["tickers"][:8])
        print(f"  {industry['label'][:44]:44} ({len(industry['tickers'])}): {members}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
