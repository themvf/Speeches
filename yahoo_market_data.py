"""Python-side Yahoo chart-endpoint fetcher for the stock attention daily
rollup (docs/stock-attention-enhancements-spec.md item 2).

Ports apps/web/lib/server/yahoo.ts's response-parsing logic (same
unofficial endpoint, same meta-field fallback chain) rather than adding a
Node dependency to a Python workflow. Deliberately narrower than the TS
version - only what the rollup needs: latest close/volume plus a 20-day
volume baseline for the "unusual volume" divergence flag.

Yahoo's chart endpoint is unofficial (no auth, no documented SLA) - callers
must treat every field as possibly missing and every request as possibly
failing. Never raises; always returns None on any problem so a bad symbol
or a Yahoo outage degrades one row's market-context columns to NULL
instead of failing the whole rollup.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; market-data/1.0)"}
REQUEST_TIMEOUT_SECONDS = 8
# Spec's "throttle ~5/s" - one request every 0.2s keeps a 300-ticker rollup
# under a minute of fetch time without leaning on Yahoo's undocumented limits.
MIN_REQUEST_INTERVAL_SECONDS = 0.2

_last_request_at = 0.0


def _throttle() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
    _last_request_at = time.monotonic()


def fetch_daily_market_context(symbol: str) -> Optional[Dict[str, Any]]:
    """Latest close/pct-change/volume plus a 20-trading-day volume baseline
    for `symbol`. Returns None on any failure (bad symbol, network error,
    malformed response, insufficient history) rather than raising."""
    _throttle()
    try:
        resp = requests.get(
            YAHOO_CHART_URL.format(symbol=symbol),
            params={"range": "1mo", "interval": "1d"},
            headers=YAHOO_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
        result = (payload.get("chart", {}) or {}).get("result") or []
        if not result:
            return None
        series = result[0]
        timestamps: List[int] = series.get("timestamp") or []
        quote = ((series.get("indicators") or {}).get("quote") or [{}])[0]
        closes: List[Optional[float]] = quote.get("close") or []
        volumes: List[Optional[float]] = quote.get("volume") or []
        if not timestamps or len(closes) != len(timestamps) or len(volumes) != len(timestamps):
            return None

        # Walk from the end to find the latest day with both a close and a
        # volume - the most recent bar can be null intraday before Yahoo
        # finalizes it.
        latest_idx = None
        for i in range(len(timestamps) - 1, -1, -1):
            if closes[i] is not None and volumes[i] is not None:
                latest_idx = i
                break
        if latest_idx is None:
            return None

        latest_close = float(closes[latest_idx])
        latest_volume = int(volumes[latest_idx])

        prior_closes = [c for c in closes[:latest_idx] if c is not None]
        prior_pct = None
        if prior_closes:
            prev_close = prior_closes[-1]
            if prev_close:
                prior_pct = ((latest_close - prev_close) / prev_close) * 100

        prior_volumes = [v for v in volumes[:latest_idx] if v is not None]
        volume_vs_20d = None
        # Require a handful of prior days before trusting a baseline -
        # a 1-2 day average is not a meaningful "normal volume" reference.
        if len(prior_volumes) >= 5:
            baseline = sum(prior_volumes[-20:]) / len(prior_volumes[-20:])
            if baseline > 0:
                volume_vs_20d = latest_volume / baseline

        return {
            "price_close": latest_close,
            "price_pct": prior_pct,
            "volume": latest_volume,
            "volume_vs_20d": volume_vs_20d,
        }
    except Exception:
        return None


def fetch_market_context_batch(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Sequential, throttled fetch for a bounded ticker list (the day's
    rolled-up tickers, ~100-300 typically) - not for arbitrary large batches."""
    out: Dict[str, Dict[str, Any]] = {}
    for symbol in symbols:
        context = fetch_daily_market_context(symbol)
        if context is not None:
            out[symbol] = context
    return out
