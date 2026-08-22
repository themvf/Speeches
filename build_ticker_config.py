#!/usr/bin/env python3
"""Generates ticker_config.json — the stock-ticker universe used by
ticker_resolver.py (see docs/stock-attention-spec.md §5).

Run manually (locally or via workflow_dispatch) when the ticker universe
should be refreshed — quarterly is plenty; this is NOT fetched per sweep.
The output is a committed artifact so the sweep has zero network
dependencies beyond Reddit itself.

Sources fetched at generation time only:
- SEC's public company_tickers.json (via curl_cffi — sec.gov blocks
  generic HTTP clients via TLS fingerprinting; same reason
  sec_scraper_free.py uses curl_cffi).
- A common-English-words list (google-10000-english) used to compute the
  ambiguous-symbol set: any valid ticker that is also a common word (or
  common finance/Reddit slang, or <= 2 chars) requires an explicit
  $-cashtag to count as a mention. This gating is the spec's #1 quality
  defense — see the StonkWhisper/ApeWisdom false-positive evidence in
  spec §5.

Usage:
    python build_ticker_config.py [--output ticker_config.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from typing import Dict, Set

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMMON_WORDS_URL = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"

# Finance/Reddit vocabulary that collides with real ticker symbols and is
# used constantly in ALL-CAPS on the swept subreddits. Seeded from the
# spec's review of live ApeWisdom/StonkWhisper false positives; grow this
# during Phase 5 validation. Entries that aren't actual tickers are
# harmless (the intersection with the universe is what ships).
CURATED_AMBIGUOUS_SEED: Set[str] = {
    "DD", "CEO", "CFO", "COO", "IPO", "ATH", "ATL", "OTM", "ITM", "FD", "PT",
    "PM", "AH", "ER", "EOD", "EOW", "EPS", "PE", "EV", "AI", "IMO", "IMHO",
    "YOLO", "FOMO", "USA", "USD", "GDP", "FBI", "SEC", "IRS", "ETF", "API",
    "CPI", "FED", "WSB", "TLDR", "TL", "DR", "OP", "RIP", "LFG", "HODL",
    "OR", "IT", "ALL", "ARE", "FOR", "ON", "SO", "BE", "GO", "NOW", "OPEN",
    "YOU", "UP", "AM", "CD", "EU", "VT", "LOT", "RR", "OS", "SAM", "PL",
    "BDC", "CC", "ES", "WTI", "CAN", "HAS", "BIG", "GAIN", "LOSS", "MOON",
    "PUMP", "DUMP", "HOLD", "BUY", "SELL", "CALL", "PUTS", "PLAY", "REAL",
    "NEXT", "EDIT", "POST", "LINK", "TRX",
    # Domain acronyms that collide with real tickers, found by reading the
    # document-corpus indexer's dry runs (2026-08-21). Each is far more often
    # the term than the company in financial and regulatory prose:
    #   DAO  decentralized autonomous organization
    #   RSI  relative strength index      (ticker: Rush Street Interactive)
    #   TDS  traffic distribution system  (ticker: Telephone and Data Systems)
    #   ASIC application-specific integrated circuit
    #   WSE  wafer-scale engine
    #   ASA  Norwegian corporate suffix, e.g. "Transocean ASA"
    #   ADV  Form ADV
    # Gating only affects BARE uppercase matches - cashtags ($TDS) still
    # resolve, so a genuine mention of the company is not lost.
    "DAO", "RSI", "TDS", "ASIC", "WSE", "ASA", "ADV",
}

# Symbols the common-words list catches (e.g. "amd" appears in web-corpus
# word lists via hardware pages) but whose exact-uppercase form in finance
# subreddits overwhelmingly means the ticker. Overrides the wordlist gate;
# never overrides the <=2-char rule. Deliberately conservative - genuinely
# common caps-rant words (SHOP, COST, CAT, KEY, FAST, WELL, LOW, SPOT) stay
# gated, with curated-name recall below instead.
CURATED_UNAMBIGUOUS_OVERRIDES: Set[str] = {
    "AMD", "IBM", "CRM", "SNAP", "DELL", "SONY",
}

# Curated company-name -> ticker map (spec §5 tier 3, confidence 0.7).
# Deliberately hand-picked: never auto-derive name matching from all ~9k
# SEC official titles ("TARGET" in a sentence must not count as TGT).
# Multi-word phrases are matched with word boundaries, case-insensitive.
# Validated against the fetched universe at generation time; entries whose
# symbol is missing from the universe are dropped with a warning.
CURATED_NAMES: Dict[str, str] = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta platforms": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "gamestop": "GME",
    "palantir": "PLTR",
    "coinbase": "COIN",
    "robinhood": "HOOD",
    "micron": "MU",
    "intel": "INTC",
    "boeing": "BA",
    "disney": "DIS",
    "paypal": "PYPL",
    "broadcom": "AVGO",
    "oracle": "ORCL",
    "salesforce": "CRM",
    "qualcomm": "QCOM",
    "starbucks": "SBUX",
    "moderna": "MRNA",
    "pfizer": "PFE",
    "exxon": "XOM",
    "chevron": "CVX",
    "jpmorgan": "JPM",
    "goldman sachs": "GS",
    "bank of america": "BAC",
    "wells fargo": "WFC",
    "citigroup": "C",
    "blackrock": "BLK",
    "berkshire": "BRK-B",
    "rocket lab": "RKLB",
    "super micro": "SMCI",
    "strategy inc": "MSTR",
    "microstrategy": "MSTR",
    "rivian": "RIVN",
    "lucid motors": "LCID",
    "opendoor": "OPEN",
    "sofi": "SOFI",
    "shopify": "SHOP",
    "spotify": "SPOT",
    "costco": "COST",
    "caterpillar": "CAT",
}


def _fetch_sec_tickers() -> Dict[str, str]:
    from curl_cffi import requests as cffi_requests

    resp = cffi_requests.get(SEC_TICKERS_URL, impersonate="chrome", timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    tickers: Dict[str, str] = {}
    for entry in payload.values():
        symbol = str(entry.get("ticker", "") or "").strip().upper()
        title = str(entry.get("title", "") or "").strip()
        if symbol:
            tickers[symbol] = title
    if len(tickers) < 5000:
        raise RuntimeError(
            f"SEC returned only {len(tickers)} tickers - refusing to write a suspiciously small universe"
        )
    return tickers


def _fetch_common_words() -> Set[str]:
    import requests

    resp = requests.get(COMMON_WORDS_URL, timeout=60)
    resp.raise_for_status()
    words = {line.strip().lower() for line in resp.text.splitlines() if line.strip()}
    if len(words) < 5000:
        raise RuntimeError(
            f"common-words list returned only {len(words)} entries - refusing to compute ambiguity from it"
        )
    return words


def build_config() -> Dict[str, object]:
    tickers = _fetch_sec_tickers()
    common_words = _fetch_common_words()

    ambiguous = sorted(
        symbol
        for symbol in tickers
        if len(symbol) <= 2
        or (
            symbol not in CURATED_UNAMBIGUOUS_OVERRIDES
            and (symbol.lower() in common_words or symbol in CURATED_AMBIGUOUS_SEED)
        )
    )

    names: Dict[str, str] = {}
    for phrase, symbol in sorted(CURATED_NAMES.items()):
        if symbol in tickers:
            names[phrase] = symbol
        else:
            print(f"[build_ticker_config] WARNING: curated name {phrase!r} -> {symbol} not in SEC universe; dropped", file=sys.stderr)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_url": SEC_TICKERS_URL,
        "ticker_count": len(tickers),
        "ambiguous_count": len(ambiguous),
        "tickers": dict(sorted(tickers.items())),
        "ambiguous": ambiguous,
        "names": names,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="ticker_config.json")
    args = parser.parse_args()

    config = build_config()
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=1)
        handle.write("\n")
    print(
        f"[build_ticker_config] wrote {args.output}: {config['ticker_count']} tickers, "
        f"{config['ambiguous_count']} ambiguous, {len(config['names'])} curated names"  # type: ignore[arg-type]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
