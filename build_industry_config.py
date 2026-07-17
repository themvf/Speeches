#!/usr/bin/env python3
"""SEC-53/SEC-56: bake the Industries tab config from EDGAR.

Two EDGAR sources, both free/no-auth:

1. SIC classification - one submissions API call per ticker (the same
   endpoint the KPI filing watch uses). Gives each company's registered
   industry, which is how peers get grouped.
2. Financials - the XBRL *frames* API, which returns a concept for EVERY
   filer in one request (e.g. NetIncomeLoss/CY2026Q1 -> ~4.9k entities). So
   the whole universe costs ~20 requests instead of one-per-company.

Writes the committed apps/web/lib/server/industry-config.json that
/api/market/industries reads - same static-config pattern as
ticker_config.json / kpi_config.py. SIC changes only on re-registration and
financials only on filings, so this doesn't earn a live pipeline; rerun and
commit to refresh or grow the universe.

Market cap is deliberately NOT baked: shares outstanding is, and the route
multiplies it by the live price it already fetches for the expanded industry
- so market cap stays current for free.

Usage: python build_industry_config.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

OUT_PATH = os.path.join("apps", "web", "lib", "server", "industry-config.json")
USER_AGENT = "PolicyResearchHub industry config (joshbandes@gmail.com)"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
FRAMES_URL = "https://data.sec.gov/api/xbrl/frames/{taxonomy}/{concept}/{unit}/{period}.json"
THROTTLE_S = 0.12

# Calendar quarters to try, newest first. Frames normalize odd fiscal
# calendars into the nearest calendar quarter, so checking a few back-quarters
# catches filers that haven't reported the newest one yet.
QUARTERS = ["CY2026Q1", "CY2025Q4", "CY2025Q3"]

# Revenue's top-line tag varies by filer (tech vs banks vs NVDA) - same
# fallback chain kpi_config proved.
REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "RevenuesNetOfInterestExpense",
]
PROFIT_CONCEPTS = ["NetIncomeLoss", "ProfitLoss"]
# Expenses are DERIVED (revenue - profit), not taken from a filed tag, on
# purpose: filers tag CostsAndExpenses/OperatingExpenses inconsistently (for
# NVDA CostsAndExpenses is opex-only at $7.6B against $81.6B revenue, so it
# doesn't reconcile), and a column that means different things per company is
# worse than useless in a PEER table. Deriving gives one definition - total
# cost incl. tax - that's identical across peers and ties out exactly with
# revenue and profit. The UI footnotes it.

UNIVERSE: List[str] = [
    # ── Semiconductors & equipment ──
    "NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU", "TXN", "AMAT", "LRCX", "KLAC",
    "ADI", "NXPI", "MCHP", "ON", "MPWR", "SWKS", "QRVO", "TER", "ENTG", "TSM",
    "ASML", "ARM", "MRVL", "SMCI", "WOLF", "AMKR", "GFS",
    # ── Software & cloud ──
    "MSFT", "ORCL", "CRM", "ADBE", "NOW", "INTU", "SNOW", "DDOG", "MDB", "TEAM",
    "WDAY", "PANW", "CRWD", "ZS", "OKTA", "NET", "FTNT", "SPLK", "HUBS", "ZM",
    "DOCU", "TWLO", "PLTR", "AI", "PATH", "U", "RBLX", "EA", "TTWO", "ADSK",
    "ANSS", "CDNS", "SNPS", "VEEV", "TYL", "PTC", "SSNC", "BB",
    # ── Internet & media ──
    "GOOGL", "META", "AMZN", "NFLX", "DIS", "CMCSA", "WBD", "PARA", "SPOT",
    "PINS", "SNAP", "MTCH", "BMBL", "YELP", "TRIP", "EXPE", "BKNG", "ABNB",
    "UBER", "LYFT", "DASH", "ETSY", "EBAY", "SHOP", "SE", "BABA", "JD", "PDD",
    # ── Hardware & devices ──
    "AAPL", "DELL", "HPQ", "HPE", "IBM", "CSCO", "ANET", "JNPR", "NTAP", "STX",
    "WDC", "PSTG", "ZBRA", "GRMN", "SONO",
    # ── Banks ──
    "JPM", "BAC", "C", "WFC", "USB", "PNC", "TFC", "COF", "MTB", "FITB",
    "HBAN", "RF", "KEY", "CFG", "ZION", "CMA", "ALLY", "NYCB", "FHN", "SNV",
    # ── Brokers, asset managers & exchanges ──
    "GS", "MS", "SCHW", "BLK", "BX", "KKR", "APO", "CG", "TROW", "BEN",
    "IVZ", "AMP", "RJF", "LPLA", "HOOD", "IBKR", "VIRT", "CME", "ICE", "NDAQ", "CBOE",
    # ── Payments & fintech ──
    "V", "MA", "PYPL", "AXP", "FIS", "FISV", "GPN", "TOST", "AFRM", "SOFI", "UPST", "SQ",
    # ── Crypto-linked ──
    "COIN", "MARA", "RIOT", "CLSK", "HUT", "MSTR", "BITF", "CIFR", "WULF",
    # ── Insurance ──
    "BRK-B", "PGR", "ALL", "TRV", "CB", "AIG", "MET", "PRU", "AFL", "HIG",
    "LNC", "GL", "CINF", "WRB", "MKL",
    # ── Pharma ──
    "LLY", "PFE", "MRK", "JNJ", "ABBV", "BMY", "ZTS", "VTRS", "OGN", "JAZZ", "NVO", "AZN",
    # ── Biotech ──
    "AMGN", "GILD", "REGN", "VRTX", "MRNA", "BIIB", "ALNY", "BMRN", "INCY",
    "SRPT", "NBIX", "EXAS", "IONS", "UTHR", "RARE",
    # ── Medical devices & life-science tools ──
    "ABT", "TMO", "DHR", "MDT", "SYK", "BSX", "BDX", "ISRG", "EW", "ZBH",
    "BAX", "A", "IQV", "RMD", "DXCM", "PODD", "ALGN", "HOLX", "WAT", "MTD", "ILMN",
    # ── Managed care & health services ──
    "UNH", "CVS", "CI", "HUM", "ELV", "CNC", "MOH", "MCK", "COR", "CAH", "HCA", "UHS", "DVA",
    # ── Retail & e-commerce ──
    "WMT", "TGT", "COST", "HD", "LOW", "DG", "DLTR", "BJ", "KR", "ACI",
    "ROST", "TJX", "BURL", "GPS", "M", "JWN", "KSS", "BBY", "ORLY", "AZO",
    "AAP", "TSCO", "ULTA", "FIVE", "GME", "BBWI", "W", "CHWY", "CVNA", "KMX",
    # ── Apparel & luxury ──
    "NKE", "LULU", "DECK", "SKX", "CROX", "VFC", "RL", "PVH", "HBI", "UAA", "ONON", "BIRK",
    # ── Food, beverage & household ──
    "KO", "PEP", "PG", "MDLZ", "GIS", "K", "HSY", "CAG", "CPB", "SJM",
    "KHC", "HRL", "TSN", "STZ", "TAP", "MNST", "CELH", "KDP", "CL", "KMB", "CHD", "EL",
    # ── Restaurants ──
    "MCD", "SBUX", "CMG", "YUM", "QSR", "DPZ", "DRI", "TXRH", "WEN", "SHAK", "CAVA", "WING",
    # ── Autos & EV ──
    "TSLA", "F", "GM", "RIVN", "LCID", "NIO", "LI", "XPEV", "STLA", "HMC", "TM", "APTV", "LEA", "BWA",
    # ── Energy: majors, E&P, services ──
    "XOM", "CVX", "COP", "OXY", "EOG", "PXD", "DVN", "FANG", "HES", "MRO",
    "APA", "CTRA", "SLB", "HAL", "BKR", "NOV", "FTI", "PSX", "VLO", "MPC", "OKE", "KMI", "WMB", "LNG",
    # ── Utilities & power ──
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "ED", "WEC", "ES",
    "PEG", "SRE", "PCG", "EIX", "FE", "AEE", "CMS", "DTE", "PPL", "VST", "CEG", "NRG",
    # ── Airlines & travel ──
    "DAL", "UAL", "AAL", "LUV", "ALK", "JBLU", "SAVE", "HA", "RCL", "CCL", "NCLH", "MAR", "HLT", "H", "WH",
    # ── Aerospace & defense ──
    "BA", "LMT", "RTX", "NOC", "GD", "LHX", "HII", "TDG", "HWM", "TXT", "SPR", "AXON", "LDOS", "BAH",
    # ── Industrials & machinery ──
    "CAT", "DE", "GE", "HON", "MMM", "EMR", "ETN", "ITW", "PH", "CMI",
    "PCAR", "ROK", "DOV", "IR", "SWK", "FAST", "GWW", "URI", "WM", "RSG",
    # ── Transport & logistics ──
    "UPS", "FDX", "UNP", "CSX", "NSC", "ODFL", "JBHT", "CHRW", "XPO", "SAIA",
    # ── Telecom ──
    "T", "VZ", "TMUS", "LUMN", "USM", "TDS",
    # ── REITs & real estate ──
    "AMT", "PLD", "CCI", "EQIX", "DLR", "SPG", "O", "PSA", "WELL", "VTR",
    "AVB", "EQR", "MAA", "ESS", "INVH", "ARE", "BXP", "VNO", "IRM", "WY", "CBRE", "Z", "OPEN",
    # ── Materials & chemicals ──
    "LIN", "APD", "SHW", "ECL", "DD", "DOW", "LYB", "PPG", "NUE", "STLD",
    "CLF", "X", "AA", "FCX", "NEM", "MOS", "CF", "ALB", "VMC", "MLM",
    # ── Homebuilders ──
    "DHI", "LEN", "PHM", "NVR", "TOL", "KBH", "TMHC", "MTH",
    # ── Gaming & leisure ──
    "LVS", "MGM", "WYNN", "CZR", "PENN", "DKNG", "BYD", "CHDN", "PLNT", "PTON",
    # ── Attention regulars ──
    "AMC", "BBAI", "SOUN", "IONQ", "RGTI", "QBTS", "LUNR", "RKLB", "ASTS", "JOBY", "ACHR", "NNE", "OKLO", "SMR",
]


def _fetch_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def fetch_frame_metric(concepts: List[str], unit: str, taxonomy: str = "us-gaap", instant: bool = False) -> Dict[str, Dict[str, Any]]:
    """cik -> {val, end} for the first hit per company, scanning newest
    quarter first and, within a quarter, the concept priority order. Period
    recency dominates so peers compare like-for-like periods; the concept
    chain only decides which tag to trust inside that quarter."""
    out: Dict[str, Dict[str, Any]] = {}
    for period in QUARTERS:
        frame = f"{period}I" if instant else period
        for concept in concepts:
            url = FRAMES_URL.format(taxonomy=taxonomy, concept=concept, unit=unit, period=frame)
            try:
                time.sleep(THROTTLE_S)
                data = _fetch_json(url).get("data", [])
            except Exception as exc:
                print(f"  ! frame {concept} {frame}: {exc}", file=sys.stderr)
                continue
            for row in data:
                cik = str(row.get("cik") or "")
                if not cik or cik in out:
                    continue
                try:
                    val = float(row.get("val"))
                except (TypeError, ValueError):
                    continue
                out[cik] = {"val": val, "end": str(row.get("end") or "")}
            print(f"  frame {concept:52} {frame:9} -> {len(data):5} rows (cum {len(out)})", file=sys.stderr)
    return out


def build_financials() -> Dict[str, Dict[str, Any]]:
    """cik -> financial record assembled from the bulk frames."""
    print("fetching XBRL frames (bulk, all filers per request)...", file=sys.stderr)
    revenue = fetch_frame_metric(REVENUE_CONCEPTS, "USD")
    profit = fetch_frame_metric(PROFIT_CONCEPTS, "USD")
    shares = fetch_frame_metric(["EntityCommonStockSharesOutstanding"], "shares", taxonomy="dei", instant=True)

    ciks = set(revenue) | set(profit) | set(shares)
    out: Dict[str, Dict[str, Any]] = {}
    for cik in ciks:
        rev = revenue.get(cik)
        pro = profit.get(cik)
        sha = shares.get(cik)
        # Only derive when both sides come from the SAME reported period, so
        # the three columns always reconcile.
        exp_val: Optional[float] = None
        if rev and pro and rev.get("end") == pro.get("end"):
            exp_val = rev["val"] - pro["val"]
        out[cik] = {
            "revenue": rev["val"] if rev else None,
            "profit": pro["val"] if pro else None,
            "expenses": exp_val,
            "sharesOutstanding": sha["val"] if sha else None,
            "periodEnd": (rev or pro or {}).get("end") or None,
        }
    return out


def group_by_industry(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pure: group ticker records into industries keyed by SIC description,
    largest first (ties: label asc). Missing SIC -> 'Unclassified'."""
    buckets: Dict[str, Dict[str, Any]] = {}
    for record in records:
        label = str(record.get("sic_description") or "").strip() or "Unclassified"
        bucket = buckets.setdefault(label, {"sic": str(record.get("sic") or ""), "label": label, "tickers": []})
        entry = {k: record[k] for k in ("ticker", "name", "cik") if k in record}
        for key in ("revenue", "profit", "expenses", "sharesOutstanding", "periodEnd"):
            if record.get(key) is not None:
                entry[key] = record[key]
        bucket["tickers"].append(entry)
    industries = sorted(buckets.values(), key=lambda b: (-len(b["tickers"]), b["label"]))
    for industry in industries:
        industry["tickers"].sort(key=lambda t: t["ticker"])
    return industries


def main() -> int:
    from edgar import Company, set_identity
    set_identity("joshbandes@gmail.com")

    financials = build_financials()
    print(f"financial records from frames: {len(financials)}\n", file=sys.stderr)

    records: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for i, ticker in enumerate(UNIVERSE, 1):
        try:
            company = Company(ticker)
            cik = int(company.cik)
        except Exception:
            skipped.append(ticker)
            continue
        try:
            time.sleep(THROTTLE_S)
            data = _fetch_json(SUBMISSIONS_URL.format(cik=cik))
        except Exception:
            skipped.append(ticker)
            continue
        record: Dict[str, Any] = {
            "ticker": ticker,
            "name": str(data.get("name") or company.name or ticker).title(),
            "cik": str(cik),
            "sic": str(data.get("sic") or ""),
            "sic_description": str(data.get("sicDescription") or ""),
        }
        record.update(financials.get(str(cik), {}))
        # Foreign private issuers (20-F/40-F filers, e.g. TSM) report ORDINARY
        # shares while the US listing trades as ADRs at a different ratio, so
        # shares x US price overstates market cap badly (TSM came out at
        # $10.6T). Only keep the share count for domestic 10-K/10-Q filers;
        # everyone else shows "-" rather than a wrong number.
        forms = set(data.get("filings", {}).get("recent", {}).get("form", []))
        if not ({"10-K", "10-Q"} & forms):
            record.pop("sharesOutstanding", None)
        records.append(record)
        if i % 50 == 0:
            print(f"  {i}/{len(UNIVERSE)} classified", file=sys.stderr)

    industries = group_by_industry(records)
    with_fin = sum(1 for r in records if r.get("revenue") is not None or r.get("profit") is not None)
    payload = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "SEC EDGAR: SIC from submissions API, financials from XBRL frames API",
        "tickerCount": len(records),
        "industries": industries,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)

    print(f"\nwrote {OUT_PATH}")
    print(f"  tickers: {len(records)} | industries: {len(industries)} | with financials: {with_fin}")
    if skipped:
        print(f"  skipped (no CIK/submissions): {len(skipped)} -> {', '.join(skipped)}", file=sys.stderr)
    multi = [i for i in industries if len(i["tickers"]) >= 3]
    print(f"  industries with 3+ members: {len(multi)}")
    for industry in industries[:14]:
        members = ", ".join(t["ticker"] for t in industry["tickers"][:9])
        print(f"  {industry['label'][:42]:42} ({len(industry['tickers']):2}): {members}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
