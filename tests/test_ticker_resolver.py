"""Tests for the ticker universe config and the three-tier resolver
(docs/stock-attention-spec.md §5)."""

import json

import ticker_resolver
from ticker_resolver import TICKER_CONFIG_PATH, resolve_tickers, ticker_title


# ─── committed config integrity ─────────────────────────────────────────────

def test_config_file_loads_and_is_plausible():
    payload = json.loads(TICKER_CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["ticker_count"] > 5000
    assert payload["ticker_count"] == len(payload["tickers"])
    assert payload["ambiguous_count"] == len(payload["ambiguous"])
    # every ambiguous symbol and curated-name target is in the universe
    universe = set(payload["tickers"])
    assert set(payload["ambiguous"]) <= universe
    assert set(payload["names"].values()) <= universe


def test_known_false_positive_symbols_are_ambiguous():
    """The exact symbols observed polluting StonkWhisper/ApeWisdom's live
    leaderboards (spec §5) must require a cashtag."""
    payload = json.loads(TICKER_CONFIG_PATH.read_text(encoding="utf-8"))
    ambiguous = set(payload["ambiguous"])
    for symbol in ["ALL", "IT", "NOW", "OPEN", "YOU", "DD", "AI", "TRX"]:
        if symbol in payload["tickers"]:
            assert symbol in ambiguous, symbol


def test_liquid_meme_tickers_are_not_ambiguous():
    payload = json.loads(TICKER_CONFIG_PATH.read_text(encoding="utf-8"))
    ambiguous = set(payload["ambiguous"])
    # AMD/IBM are in common-word lists (web-corpus artifacts) but are
    # explicitly overridden as unambiguous in build_ticker_config.py.
    for symbol in ["NVDA", "GME", "TSLA", "MSFT", "AMD", "IBM", "PLTR"]:
        assert symbol in payload["tickers"], symbol
        assert symbol not in ambiguous, symbol


# ─── tier 1: cashtags ────────────────────────────────────────────────────────

def test_cashtag_counts_at_full_confidence():
    assert resolve_tickers("$GME to the moon") == {"GME": 1.0}


def test_cashtag_is_case_insensitive():
    assert resolve_tickers("loading up on $gme calls") == {"GME": 1.0}


def test_cashtag_overrides_ambiguity_gate():
    # ALL (Allstate) is ambiguous bare, but an explicit cashtag counts.
    assert resolve_tickers("$ALL is undervalued") == {"ALL": 1.0}


def test_unknown_cashtag_is_ignored():
    fake = "ZZZZQ"
    assert fake not in json.loads(TICKER_CONFIG_PATH.read_text(encoding="utf-8"))["tickers"]
    assert resolve_tickers(f"${fake} lol") == {}


def test_cashtag_does_not_match_inside_longer_token():
    # "$GMEXTRA" must not count as GME.
    assert "GME" not in resolve_tickers("$GMEXTRA")


# ─── tier 2: bare symbols ───────────────────────────────────────────────────

def test_bare_unambiguous_symbol_counts():
    assert resolve_tickers("NVDA earnings tomorrow") == {"NVDA": 1.0}


def test_bare_ambiguous_symbol_is_rejected():
    # The caps-rant false positive: ALL is a real ticker (Allstate).
    assert resolve_tickers("I LOST ALL MY MONEY") == {}
    assert resolve_tickers("This DD is solid, trust me") == {}


def test_lowercase_bare_symbol_is_rejected():
    assert resolve_tickers("thinking about gme again") == {}


def test_bare_symbol_respects_token_boundaries():
    assert resolve_tickers("BANVDAX") == {}


# ─── tier 3: curated names ──────────────────────────────────────────────────

def test_name_match_counts_at_lower_confidence():
    assert resolve_tickers("Robinhood is down again") == {"HOOD": 0.7}
    assert resolve_tickers("gamestop earnings leak") == {"GME": 0.7}


def test_name_plus_cashtag_keeps_highest_confidence():
    assert resolve_tickers("GameStop $GME yolo") == {"GME": 1.0}


def test_multiword_name_matches_across_whitespace():
    assert resolve_tickers("bank    of america downgrade") == {"BAC": 0.7}


# ─── misc ───────────────────────────────────────────────────────────────────

def test_empty_text_returns_empty():
    assert resolve_tickers("") == {}
    assert resolve_tickers(None) == {}


def test_multiple_tickers_in_one_text():
    result = resolve_tickers("rotating out of NVDA into $MU and Palantir")
    assert result == {"NVDA": 1.0, "MU": 1.0, "PLTR": 0.7}


def test_ticker_title_lookup():
    assert "NVIDIA" in ticker_title("NVDA").upper()
    assert ticker_title("ZZZZQ") == ""
