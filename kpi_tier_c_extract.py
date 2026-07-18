#!/usr/bin/env python3
"""SEC-13 Tier C: extract operational KPIs (absent from XBRL) from 8-K
earnings releases via DeepSeek, with evidence-span verification.

Per company in kpi_config.TIER_C_KPIS:
1. Find the latest 8-K carrying item 2.02 (results of operations) from the
   company's EDGAR submissions JSON.
2. Fetch that filing's EX-99* earnings-release exhibit (HTML; PDF exhibits
   are skipped with a per-company note - a v1 caveat, mainly TSLA decks).
3. Ask DeepSeek (deepseek-v4-pro - quality tier; ~8 docs/quarter is trivial
   volume) for each configured KPI: value in base units, period label, and a
   VERBATIM evidence quote.
4. Verify each evidence quote actually appears in the exhibit text
   (whitespace/quote-normalized substring - same convention as the
   enrichment pipeline's _evidence_snippet_verified). Unverifiable evidence
   marks the value evidenceVerified=false; it still lands pending_review.
5. Write apps/web/lib/server/kpi-tier-c-data.json. Every new value has
   status "pending_review"; the Market tab renders ONLY "approved" values.
   Review = flip status in this committed JSON (git-as-store, same
   architecture decision as SEC-34 - deliberately no Neon/admin-panel
   dependency; the JSON diff IS the review artifact). Re-runs preserve any
   approved/rejected decision for the same (kpi, period) and only replace an
   entry when a NEWER period shows up, which resets status to
   pending_review.

Requires: DEEPSEEK_API. Network: data.sec.gov + www.sec.gov + api.deepseek.com.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from kpi_config import NAMES, TIER_C_KPIS

# source_health drags in google-cloud-storage; this workflow deliberately
# runs on a minimal pip install (requests + bs4, no GCS creds), so health
# recording is best-effort - same stdlib-fast-path spirit as SEC-34.
try:
    from source_health import record_source_health
except ImportError:  # pragma: no cover - exercised only in the minimal env
    def record_source_health(summary):  # type: ignore[misc]
        return None

SOURCE_KEY = "kpi_tier_c_extract"
OUTPUT_PATH = Path("apps/web/lib/server/kpi-tier-c-data.json")
STATE_PATH = Path("kpi_state.json")
UA = {"User-Agent": "sec-speeches research joshbandes@gmail.com"}
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-pro"
MAX_TEXT_CHARS = 80_000


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_ciks() -> Dict[str, str]:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {k: str(v) for k, v in state.get("ciks", {}).items()}


def find_latest_earnings_8k(cik: str) -> Optional[Dict[str, str]]:
    """Latest 8-K whose items include 2.02, from the submissions JSON."""
    url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
    data = requests.get(url, headers=UA, timeout=30).json()
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        items = str(recent.get("items", [""] * len(forms))[i] or "")
        if "2.02" not in items:
            continue
        return {
            "accession": str(recent.get("accessionNumber", [""] * len(forms))[i] or ""),
            "filed_at": str(recent.get("filingDate", [""] * len(forms))[i] or ""),
            "primary": str(recent.get("primaryDocument", [""] * len(forms))[i] or ""),
        }
    return None


def pick_exhibit(cik: str, accession: str, primary: str) -> Tuple[Optional[str], str]:
    """(exhibit_url, note). Picks the largest EX-99* HTML file in the filing
    directory; falls back to the largest non-primary .htm."""
    acc = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}"
    listing = requests.get(f"{base}/index.json", headers=UA, timeout=30).json()
    files = listing.get("directory", {}).get("item", [])
    def size_of(f: Dict[str, Any]) -> int:
        try:
            return int(f.get("size") or 0)
        except (TypeError, ValueError):
            return 0
    ex99_html = [f for f in files if re.search(r"ex[-_]?99", str(f.get("name", "")), re.I)
                 and str(f.get("name", "")).lower().endswith((".htm", ".html"))]
    if ex99_html:
        best = max(ex99_html, key=size_of)
        return f"{base}/{best['name']}", "ex99"
    ex99_pdf = [f for f in files if re.search(r"ex[-_]?99", str(f.get("name", "")), re.I)
                and str(f.get("name", "")).lower().endswith(".pdf")]
    if ex99_pdf:
        return None, "ex99 exhibit is a PDF - skipped (v1 caveat)"
    other_html = [f for f in files if str(f.get("name", "")).lower().endswith((".htm", ".html"))
                  and str(f.get("name", "")) != primary and "index" not in str(f.get("name", "")).lower()]
    if other_html:
        best = max(other_html, key=size_of)
        return f"{base}/{best['name']}", "fallback_html"
    return None, "no HTML exhibit found"


def exhibit_text(url: str) -> str:
    html = requests.get(url, headers=UA, timeout=60).text
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    if len(text) > MAX_TEXT_CHARS:
        # Cut the middle (boilerplate-heavy) rather than the tail, where
        # press releases often put the metric tables.
        half = MAX_TEXT_CHARS // 2
        text = text[:half] + " ... " + text[-half:]
    return text


def _normalize_for_match(value: str) -> str:
    value = value.replace("’", "'").replace("‘", "'")
    value = value.replace("“", '"').replace("”", '"')
    value = value.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", value).strip().lower()


def evidence_verified(evidence: str, source_text: str) -> bool:
    ev = _normalize_for_match(evidence)
    return bool(ev) and ev in _normalize_for_match(source_text)


def build_prompt(ticker: str, kpis: List[Dict[str, str]], text: str) -> List[Dict[str, str]]:
    kpi_lines = "\n".join(
        f'- "{k["kpi_key"]}": {k["label"]}. {k["hint"]}' for k in kpis
    )
    system = (
        "You extract operational metrics from a company's earnings press release. "
        "Respond with ONLY a JSON object. For each requested kpi_key, the value is either null "
        "(metric not stated in the text) or an object {\"value\": <plain number in base units>, "
        "\"period\": \"<the fiscal period the number is for, e.g. Q2 2026>\", "
        "\"evidence\": \"<short VERBATIM quote from the text containing the number>\"}. "
        "Never estimate, never derive from other numbers, never use a prior period's value."
    )
    user = f"Company: {NAMES.get(ticker, ticker)} ({ticker})\nRequested metrics:\n{kpi_lines}\n\nPress release text:\n{text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_deepseek(messages: List[Dict[str, str]], api_key: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    resp = requests.post(
        DEEPSEEK_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": MODEL, "messages": messages, "temperature": 0,
              "response_format": {"type": "json_object"}},
        timeout=180,
    )
    resp.raise_for_status()
    body = resp.json()
    usage = body.get("usage", {}) or {}
    tokens = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
    }
    content = body["choices"][0]["message"]["content"]
    return json.loads(content), tokens


def parse_extraction(raw: Dict[str, Any], kpis: List[Dict[str, str]], source_text: str) -> Dict[str, Dict[str, Any]]:
    """Validate the model payload into storable entries. Non-numeric values
    and missing evidence are dropped; unverifiable evidence is kept but
    flagged evidenceVerified=false."""
    out: Dict[str, Dict[str, Any]] = {}
    for kpi in kpis:
        key = kpi["kpi_key"]
        entry = raw.get(key)
        if not isinstance(entry, dict):
            continue
        try:
            value = float(entry.get("value"))
        except (TypeError, ValueError):
            continue
        evidence = str(entry.get("evidence", "") or "").strip()
        if not evidence:
            continue
        out[key] = {
            "label": kpi["label"],
            "unit": kpi["unit"],
            "value": value,
            "period": str(entry.get("period", "") or ""),
            "evidence": evidence,
            "evidenceVerified": evidence_verified(evidence, source_text),
            "status": "pending_review",
            "extractedAt": _utc_now_iso(),
        }
    return out


def merge_company(existing: Dict[str, Any], fresh_kpis: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Review decisions survive re-runs: an existing approved/rejected entry
    for the same period is kept verbatim; a new period replaces the entry
    and resets it to pending_review."""
    merged = dict(existing.get("kpis", {}))
    for key, fresh in fresh_kpis.items():
        prior = merged.get(key)
        if (
            isinstance(prior, dict)
            and prior.get("period") == fresh["period"]
            and prior.get("status") in ("approved", "rejected")
        ):
            continue
        merged[key] = fresh
    return merged


def run(tickers: Optional[List[str]] = None) -> Dict[str, Any]:
    api_key = os.environ.get("DEEPSEEK_API", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API is not set.")
    ciks = load_ciks()
    targets = {t: TIER_C_KPIS[t] for t in (tickers or sorted(TIER_C_KPIS)) if t in TIER_C_KPIS}

    existing: Dict[str, Any] = {"companies": {}}
    if OUTPUT_PATH.exists():
        existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))

    summary: Dict[str, Any] = {
        "source_key": SOURCE_KEY, "connector": SOURCE_KEY, "ran_at": _utc_now_iso(),
        "model": MODEL, "companies": {}, "errors": [],
        "tokens": {"prompt_tokens": 0, "completion_tokens": 0},
    }
    companies = dict(existing.get("companies", {}))
    for ticker, kpis in targets.items():
        cik = ciks.get(ticker)
        if not cik:
            summary["errors"].append(f"{ticker}: no CIK in kpi_state.json")
            continue
        try:
            filing = find_latest_earnings_8k(cik)
            if not filing:
                summary["companies"][ticker] = {"note": "no 8-K item 2.02 in recent filings"}
                continue
            exhibit_url, note = pick_exhibit(cik, filing["accession"], filing["primary"])
            if not exhibit_url:
                summary["companies"][ticker] = {"note": note, "filed_at": filing["filed_at"]}
                continue
            text = exhibit_text(exhibit_url)
            raw, tokens = call_deepseek(build_prompt(ticker, kpis, text), api_key)
            summary["tokens"]["prompt_tokens"] += tokens["prompt_tokens"]
            summary["tokens"]["completion_tokens"] += tokens["completion_tokens"]
            fresh = parse_extraction(raw, kpis, text)
            prior_company = companies.get(ticker, {})
            companies[ticker] = {
                "name": NAMES.get(ticker, ticker),
                "sourceUrl": exhibit_url,
                "filedAt": filing["filed_at"],
                "kpis": merge_company(prior_company, fresh),
            }
            summary["companies"][ticker] = {
                "extracted": len(fresh),
                "verified": sum(1 for v in fresh.values() if v["evidenceVerified"]),
                "exhibit": note, "filed_at": filing["filed_at"],
            }
        except Exception as exc:
            summary["errors"].append(f"{ticker}: {exc}")

    payload = {
        "generatedAt": _utc_now_iso(),
        "source": f"LLM extraction ({MODEL}) from 8-K earnings releases (EX-99)",
        "companies": companies,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    extracted_total = sum(int(c.get("extracted", 0) or 0) for c in summary["companies"].values())
    summary["processed_count"] = extracted_total
    summary["discovered_count"] = len(targets)
    summary["failed_count"] = len(summary["errors"])
    summary["ok"] = extracted_total > 0 and not summary["errors"]
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default="", help="Comma-separated subset (default: all Tier C companies)")
    parser.add_argument("--summary-path", default="kpi_tier_c_summary.json")
    args = parser.parse_args(argv)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or None
    try:
        summary = run(tickers)
    except Exception as exc:
        summary = {"source_key": SOURCE_KEY, "connector": SOURCE_KEY,
                   "ran_at": _utc_now_iso(), "ok": False, "errors": [str(exc)],
                   "processed_count": 0, "failed_count": 1, "discovered_count": 0}
    record_source_health(summary)
    Path(args.summary_path).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
