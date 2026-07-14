#!/usr/bin/env python3
"""SEC-17 pilot: quarterly history for Apple's and Alphabet's CBOE-listed
KPIs, extracted from SEC XBRL via edgartools (capability verified on
SEC-12). Two-company spike ahead of the full SEC-9/SEC-10 build - the KPI
mappings below are the prototype for kpi_config.json.

Requires: pip install edgartools (>=4.18). Not yet in requirements.txt -
that lands with SEC-10. Network: data.sec.gov only.

Usage:
    python kpi_pilot.py [--quarters 8] [--out kpi_pilot_output.json]

Output: JSON of per-KPI quarterly series (period_end-sorted), with Q4
values derived by FY-minus-9M subtraction where a quarter is only
reported inside a 10-K (flagged "derived": true - for EPS this is an
approximation, since share counts differ across quarters).

Research context only - not investment advice.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

# KPI map: (company, kpi_key) -> concept + exact dimension requirements.
# dims=None means the undimensioned (consolidated) fact.
REVENUE_CONCEPT = "RevenueFromContractWithCustomerExcludingAssessedTax"
PILOT_KPIS: Dict[str, List[Dict[str, Any]]] = {
    "AAPL": [
        {"kpi_key": "eps_diluted", "label": "Diluted EPS", "unit": "usd_per_share",
         "concept": "EarningsPerShareDiluted", "dims": None},
        {"kpi_key": "total_net_sales", "label": "Total net sales", "unit": "usd",
         "concept": REVENUE_CONCEPT, "dims": None},
        {"kpi_key": "iphone_net_sales", "label": "iPhone net sales", "unit": "usd",
         "concept": REVENUE_CONCEPT, "dims": {"dim_srt_ProductOrServiceAxis": "aapl:IPhoneMember"}},
        {"kpi_key": "services_net_sales", "label": "Services net sales", "unit": "usd",
         "concept": REVENUE_CONCEPT, "dims": {"dim_srt_ProductOrServiceAxis": "us-gaap:ServiceMember"}},
        {"kpi_key": "americas_net_sales", "label": "Americas net sales", "unit": "usd",
         "concept": REVENUE_CONCEPT, "dims": {"dim_us-gaap_StatementBusinessSegmentsAxis": "aapl:AmericasSegmentMember"}},
        {"kpi_key": "greater_china_net_sales", "label": "Greater China net sales", "unit": "usd",
         "concept": REVENUE_CONCEPT, "dims": {"dim_us-gaap_StatementBusinessSegmentsAxis": "aapl:GreaterChinaSegmentMember"}},
    ],
    "GOOGL": [
        {"kpi_key": "eps_diluted", "label": "Diluted EPS", "unit": "usd_per_share",
         "concept": "EarningsPerShareDiluted", "dims": None},
        {"kpi_key": "revenues", "label": "Revenues", "unit": "usd",
         "concept": "Revenues", "dims": None,
         "fallback_concept": REVENUE_CONCEPT},
        # YouTube ads facts carry a co-dimension (they're disclosed inside
        # the Google Services segment) - both axes must be specified.
        {"kpi_key": "youtube_ads_revenues", "label": "YouTube ads revenues", "unit": "usd",
         "concept": REVENUE_CONCEPT, "dims": {
             "dim_srt_ProductOrServiceAxis": "goog:YouTubeAdvertisingRevenueMember",
             "dim_us-gaap_StatementBusinessSegmentsAxis": "goog:GoogleServicesMember"}},
        {"kpi_key": "google_cloud_revenues", "label": "Google Cloud revenues", "unit": "usd",
         "concept": REVENUE_CONCEPT, "dims": {"dim_us-gaap_StatementBusinessSegmentsAxis": "goog:GoogleCloudMember"}},
    ],
}

QUARTER_DAYS = (80, 100)   # a fiscal quarter's duration window
YEAR_DAYS = (350, 380)     # a fiscal year's duration window

# Axes a fact may carry WITHOUT being requested, with the members that are
# semantically neutral for our purposes. Discovered in this pilot: Apple's
# newer filings tag segment revenue with ConsolidationItemsAxis =
# OperatingSegmentsMember (older filings omit it), which is just the
# "operating segments view" marker, not a finer slice.
NEUTRAL_DIMS: Dict[str, set] = {
    "dim_srt_ConsolidationItemsAxis": {"us-gaap:OperatingSegmentsMember"},
}


def _duration_days(row: Dict[str, Any]) -> Optional[int]:
    from datetime import date
    try:
        start = date.fromisoformat(str(row["period_start"]))
        end = date.fromisoformat(str(row["period_end"]))
        return (end - start).days
    except Exception:
        return None


def _matches_dims(row: Dict[str, Any], dims: Optional[Dict[str, str]], dim_cols: List[str]) -> bool:
    required = dims or {}
    for col, member in required.items():
        if str(row.get(col)) != member:
            return False
    # every OTHER dim column must be empty (a fact tagged with extra axes
    # is a different, finer-grained slice than the KPI wants) - except the
    # known-neutral axis/member combinations.
    for col in dim_cols:
        if col in required:
            continue
        value = row.get(col)
        if _is_na(value):
            continue
        if str(value) in NEUTRAL_DIMS.get(col, set()):
            continue
        return False
    return True


def _is_na(value: Any) -> bool:
    return value is None or value != value or value == ""  # NaN-safe


def _extract_series(filings_facts: List[Dict[str, Any]], kpi: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Builds the quarterly series for one KPI across all collected facts:
    direct quarterly durations first, then Q4 derivation from FY - 9M for
    periods only covered inside a 10-K."""
    quarterly: Dict[str, Dict[str, Any]] = {}
    yearly: Dict[str, Dict[str, Any]] = {}
    for row in filings_facts:
        days = _duration_days(row)
        if days is None:
            continue
        point = {"period_start": str(row["period_start"]), "period_end": str(row["period_end"]),
                 "value": float(row["numeric_value"]), "derived": False}
        if QUARTER_DAYS[0] <= days <= QUARTER_DAYS[1]:
            quarterly[point["period_end"]] = point
        elif YEAR_DAYS[0] <= days <= YEAR_DAYS[1]:
            yearly[point["period_end"]] = point

    # Q4 derivation: for each FY whose end-date has no direct quarterly
    # fact, subtract the three quarters that fall inside that FY window.
    from datetime import date
    for fy_end, fy_point in yearly.items():
        if fy_end in quarterly:
            continue
        fy_start = date.fromisoformat(fy_point["period_start"])
        fy_end_d = date.fromisoformat(fy_end)
        inside = [q for q in quarterly.values()
                  if fy_start <= date.fromisoformat(q["period_start"])
                  and date.fromisoformat(q["period_end"]) < fy_end_d]
        if len(inside) == 3:
            q4_value = fy_point["value"] - sum(q["value"] for q in inside)
            q4_start = max(date.fromisoformat(q["period_end"]) for q in inside)
            quarterly[fy_end] = {"period_start": str(q4_start), "period_end": fy_end,
                                 "value": round(q4_value, 4), "derived": True}

    return sorted(quarterly.values(), key=lambda p: p["period_end"])


def run(quarters: int) -> Dict[str, Any]:
    from edgar import Company, set_identity
    set_identity("joshbandes@gmail.com")

    # enough filings to cover `quarters` quarters: ~3 10-Qs + 1 10-K per FY
    n_10q = max(6, quarters)
    n_10k = max(2, quarters // 4 + 1)

    output: Dict[str, Any] = {"companies": {}}
    for ticker, kpis in PILOT_KPIS.items():
        company = Company(ticker)
        print(f"[{ticker}] {company.name} (CIK {company.cik})", file=sys.stderr)
        filings = list(company.get_filings(form="10-Q").latest(n_10q)) \
            + list(company.get_filings(form="10-K").latest(n_10k))

        # concept -> accumulated fact rows across all filings
        concepts = {k["concept"] for k in kpis} | {k.get("fallback_concept") for k in kpis if k.get("fallback_concept")}
        facts_by_concept: Dict[str, List[Dict[str, Any]]] = {c: [] for c in concepts}
        dim_cols_by_concept: Dict[str, set] = {c: set() for c in concepts}
        for filing in filings:
            print(f"  parsing {filing.form} {filing.filing_date}", file=sys.stderr)
            try:
                xbrl = filing.xbrl()
            except Exception as exc:
                print(f"  ! xbrl parse failed: {exc}", file=sys.stderr)
                continue
            for concept in concepts:
                try:
                    df = xbrl.facts.query().by_concept(concept).to_dataframe()
                except Exception:
                    continue
                if df.empty:
                    continue
                dim_cols = [c for c in df.columns if c.startswith("dim_")]
                dim_cols_by_concept[concept].update(dim_cols)
                facts_by_concept[concept].extend(df.to_dict("records"))

        company_out = {"name": company.name, "cik": company.cik, "kpis": {}}
        for kpi in kpis:
            for concept in [kpi["concept"]] + ([kpi["fallback_concept"]] if kpi.get("fallback_concept") else []):
                dim_cols = sorted(dim_cols_by_concept.get(concept, set()))
                rows = [r for r in facts_by_concept.get(concept, [])
                        if _matches_dims(r, kpi["dims"], dim_cols) and r.get("numeric_value") == r.get("numeric_value")]
                # Pilot finding: a concept like Alphabet's `Revenues` also
                # matches footnote facts (hedging-style rows) whose values
                # collide with the income-statement fact under period
                # dedupe, corrupting the series. For undimensioned KPIs,
                # prefer primary-statement rows when any exist (segment
                # KPIs live in disclosure sections, so only apply this
                # where dims is None).
                if kpi["dims"] is None:
                    stmt_rows = [r for r in rows if str(r.get("statement_type")) == "IncomeStatement"]
                    if stmt_rows:
                        rows = stmt_rows
                series = _extract_series(rows, kpi)
                if series:
                    break
            company_out["kpis"][kpi["kpi_key"]] = {
                "label": kpi["label"], "unit": kpi["unit"], "concept": concept,
                "dims": kpi["dims"], "series": series[-quarters:],
            }
            latest = series[-1] if series else None
            print(f"  {kpi['kpi_key']:26} points={len(series)} latest={latest}", file=sys.stderr)
        output["companies"][ticker] = company_out
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarters", type=int, default=8)
    parser.add_argument("--out", default="kpi_pilot_output.json")
    args = parser.parse_args()
    output = run(args.quarters)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
