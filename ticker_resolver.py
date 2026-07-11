"""Three-tier stock-ticker resolution against ticker_config.json
(see docs/stock-attention-spec.md §5). Python-only by design — resolution
happens at ingestion time in the Reddit sweep; the web tier only ever sees
already-resolved tickers, so there is deliberately no TS port (and the
~9k-entry config must NOT be added to the webpack-inlined
entity-aliases.json — spec §5.3).

Tiers, strictest first:
1. $SYMBOL cashtag            -> confidence 1.0 (unambiguous author intent)
2. Bare UPPERCASE symbol      -> confidence 1.0, only if NOT in the
                                  ambiguous set (common words / finance
                                  slang / <=2 chars require a cashtag)
3. Curated company-name match -> confidence 0.7

A missing/unreadable config raises immediately rather than degrading:
unlike entity aliasing (where fail-soft keeps enrichment alive), a sweep
that resolves nothing has no purpose, and a silent empty universe would
look like "quiet day on Reddit" instead of the packaging bug it is.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Pattern, Tuple

TICKER_CONFIG_PATH = Path(__file__).resolve().parent / "ticker_config.json"

CASHTAG_CONFIDENCE = 1.0
BARE_SYMBOL_CONFIDENCE = 1.0
NAME_MATCH_CONFIDENCE = 0.7

_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5})(?![A-Za-z0-9])")
_BARE_SYMBOL_RE = re.compile(r"(?<![A-Za-z0-9$])([A-Z]{1,5})(?![A-Za-z0-9])")

_config_cache: dict | None = None
_name_patterns_cache: List[Tuple[Pattern[str], str]] | None = None


def _load_config() -> dict:
    global _config_cache
    if _config_cache is None:
        payload = json.loads(TICKER_CONFIG_PATH.read_text(encoding="utf-8"))
        tickers = payload.get("tickers")
        if not isinstance(tickers, dict) or not tickers:
            raise RuntimeError(f"{TICKER_CONFIG_PATH} has no tickers - regenerate with build_ticker_config.py")
        _config_cache = {
            "universe": set(tickers.keys()),
            "titles": tickers,
            "ambiguous": set(payload.get("ambiguous", [])),
            "names": dict(payload.get("names", {})),
        }
    return _config_cache


def _name_patterns() -> List[Tuple[Pattern[str], str]]:
    global _name_patterns_cache
    if _name_patterns_cache is None:
        cfg = _load_config()
        patterns: List[Tuple[Pattern[str], str]] = []
        for phrase, symbol in cfg["names"].items():
            escaped = re.escape(phrase).replace(r"\ ", r"\s+")
            patterns.append((re.compile(rf"\b{escaped}\b", re.IGNORECASE), symbol))
        _name_patterns_cache = patterns
    return _name_patterns_cache


def ticker_title(symbol: str) -> str:
    """Company title for a symbol, or '' if unknown."""
    return str(_load_config()["titles"].get(symbol, "") or "")


def resolve_tickers(text: str) -> Dict[str, float]:
    """All tickers mentioned in `text`, as {symbol: confidence}. When the
    same symbol matches through multiple tiers, the highest confidence
    wins."""
    cfg = _load_config()
    universe: set = cfg["universe"]
    ambiguous: set = cfg["ambiguous"]
    raw = str(text or "")
    if not raw.strip():
        return {}

    found: Dict[str, float] = {}

    def _record(symbol: str, confidence: float) -> None:
        if confidence > found.get(symbol, 0.0):
            found[symbol] = confidence

    # Tier 1: cashtags (case-insensitive - "$gme" shows the same intent).
    for match in _CASHTAG_RE.finditer(raw):
        symbol = match.group(1).upper()
        if symbol in universe:
            _record(symbol, CASHTAG_CONFIDENCE)

    # Tier 2: bare symbols - exact uppercase only, gated by the ambiguous
    # set. "NVDA" counts; "ALL" (in "I LOST ALL MY MONEY") does not.
    for match in _BARE_SYMBOL_RE.finditer(raw):
        symbol = match.group(1)
        if symbol in universe and symbol not in ambiguous:
            _record(symbol, BARE_SYMBOL_CONFIDENCE)

    # Tier 3: curated company names ("Robinhood is down" -> HOOD).
    for pattern, symbol in _name_patterns():
        if pattern.search(raw):
            _record(symbol, NAME_MATCH_CONFIDENCE)

    return found
