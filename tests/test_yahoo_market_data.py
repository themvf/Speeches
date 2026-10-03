"""Tests for the Python-side Yahoo chart fetcher (enhancement item 2).
No real network access - requests.get is mocked throughout."""

from unittest.mock import MagicMock, patch

import yahoo_market_data as ymd


def _chart_response(timestamps, closes, volumes, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {
        "chart": {
            "result": [{
                "timestamp": timestamps,
                "indicators": {"quote": [{"close": closes, "volume": volumes}]},
            }]
        }
    }
    return resp


def test_fetch_daily_market_context_happy_path():
    timestamps = list(range(20))
    closes = [100.0 + i for i in range(19)] + [120.0]
    volumes = [1_000_000] * 19 + [3_000_000]
    with patch.object(ymd, "_throttle"):
        with patch.object(ymd.requests, "get", return_value=_chart_response(timestamps, closes, volumes)):
            result = ymd.fetch_daily_market_context("NVDA")

    assert result["price_close"] == 120.0
    assert result["volume"] == 3_000_000
    # prior close was closes[18] == 118.0 -> pct = (120-118)/118*100
    assert round(result["price_pct"], 4) == round((120.0 - 118.0) / 118.0 * 100, 4)
    # volume_vs_20d compares latest volume to the mean of prior volumes
    assert result["volume_vs_20d"] == 3_000_000 / 1_000_000


def test_fetch_daily_market_context_handles_null_latest_bar():
    # Yahoo can return a null close/volume for the most recent, not-yet-
    # finalized bar - the resolver should walk back to the last real one.
    timestamps = [0, 1, 2]
    closes = [10.0, 11.0, None]
    volumes = [100, 200, None]
    with patch.object(ymd, "_throttle"):
        with patch.object(ymd.requests, "get", return_value=_chart_response(timestamps, closes, volumes)):
            result = ymd.fetch_daily_market_context("XYZ")
    assert result["price_close"] == 11.0
    assert result["volume"] == 200


def test_fetch_daily_market_context_insufficient_history_skips_baseline():
    timestamps = [0, 1]
    closes = [10.0, 11.0]
    volumes = [100, 200]
    with patch.object(ymd, "_throttle"):
        with patch.object(ymd.requests, "get", return_value=_chart_response(timestamps, closes, volumes)):
            result = ymd.fetch_daily_market_context("XYZ")
    assert result["volume_vs_20d"] is None  # fewer than 5 prior days


def test_fetch_daily_market_context_non_200_returns_none():
    with patch.object(ymd, "_throttle"):
        with patch.object(ymd.requests, "get", return_value=_chart_response([], [], [], status_code=404)):
            assert ymd.fetch_daily_market_context("FAKE") is None


def test_fetch_daily_market_context_empty_result_returns_none():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"chart": {"result": []}}
    with patch.object(ymd, "_throttle"):
        with patch.object(ymd.requests, "get", return_value=resp):
            assert ymd.fetch_daily_market_context("FAKE") is None


def test_fetch_daily_market_context_network_error_returns_none_not_raise():
    with patch.object(ymd, "_throttle"):
        with patch.object(ymd.requests, "get", side_effect=ConnectionError("boom")):
            assert ymd.fetch_daily_market_context("NVDA") is None


def test_fetch_daily_market_context_malformed_json_returns_none():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.side_effect = ValueError("bad json")
    with patch.object(ymd, "_throttle"):
        with patch.object(ymd.requests, "get", return_value=resp):
            assert ymd.fetch_daily_market_context("NVDA") is None


def test_fetch_market_context_batch_skips_failures():
    timestamps = list(range(6))
    closes = [10.0] * 6
    volumes = [100] * 6
    good_resp = _chart_response(timestamps, closes, volumes)

    def _get(url, params=None, headers=None, timeout=None):
        if "BAD" in url:
            raise ConnectionError("boom")
        return good_resp

    with patch.object(ymd, "_throttle"):
        with patch.object(ymd.requests, "get", side_effect=_get):
            result = ymd.fetch_market_context_batch(["GOOD", "BAD"])

    assert "GOOD" in result
    assert "BAD" not in result


def _chart_response_with_meta(timestamps, closes, volumes, meta, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {
        "chart": {
            "result": [{
                "timestamp": timestamps, "meta": meta,
                "indicators": {"quote": [{"close": closes, "volume": volumes}]},
            }]
        }
    }
    return resp


# Field shapes taken from a live AAPL response on 2026-09-21: a daily bar's timestamp is its
# session open (13:30Z), the session runs to 20:00Z, and regularMarketTime sits at 20:00:02Z
# once that session has closed.
OPEN_AT, CLOSE_AT = 1758461400, 1758484800   # 2026-09-21 13:30Z -> 20:00Z
PRIOR_OPEN = OPEN_AT - 86400


def test_a_session_still_in_progress_is_not_used_as_a_whole_day():
    """Yahoo emits today's bar mid-session with only the volume traded so far.

    Dividing that by an average of whole days understates the ratio -- the same mistake as
    comparing a rolling window against per-interval values.
    """
    timestamps = [PRIOR_OPEN - 86400 * i for i in range(8, 0, -1)] + [PRIOR_OPEN, OPEN_AT]
    closes = [100.0] * 8 + [110.0, 111.0]
    volumes = [1_000_000] * 8 + [2_000_000, 120_000]   # the last bar is one hour into the day
    meta = {"regularMarketTime": OPEN_AT + 3600, "currentTradingPeriod": {"regular": {"start": OPEN_AT, "end": CLOSE_AT}}}
    with patch.object(ymd, "_throttle"):
        with patch.object(ymd.requests, "get", return_value=_chart_response_with_meta(timestamps, closes, volumes, meta)):
            result = ymd.fetch_daily_market_context("AAPL")
    assert result["volume"] == 2_000_000          # yesterday's complete bar, not today's 120k
    assert result["price_close"] == 110.0
    assert result["volume_vs_20d"] == 2.0         # 2,000,000 / 1,000,000, not 0.12


def test_the_same_session_is_used_once_it_has_closed():
    timestamps = [PRIOR_OPEN - 86400 * i for i in range(8, 0, -1)] + [PRIOR_OPEN, OPEN_AT]
    closes = [100.0] * 8 + [110.0, 111.0]
    volumes = [1_000_000] * 8 + [2_000_000, 3_000_000]
    meta = {"regularMarketTime": CLOSE_AT + 2, "currentTradingPeriod": {"regular": {"start": OPEN_AT, "end": CLOSE_AT}}}
    with patch.object(ymd, "_throttle"):
        with patch.object(ymd.requests, "get", return_value=_chart_response_with_meta(timestamps, closes, volumes, meta)):
            result = ymd.fetch_daily_market_context("AAPL")
    assert result["volume"] == 3_000_000 and result["price_close"] == 111.0


def test_a_bar_from_before_the_current_session_is_never_treated_as_partial():
    # Pre-open, or over a weekend, the newest bar is the previous session and must be used.
    timestamps = [PRIOR_OPEN - 86400 * i for i in range(8, 0, -1)] + [PRIOR_OPEN]
    meta = {"regularMarketTime": PRIOR_OPEN + 23400, "currentTradingPeriod": {"regular": {"start": OPEN_AT, "end": CLOSE_AT}}}
    with patch.object(ymd, "_throttle"):
        with patch.object(ymd.requests, "get", return_value=_chart_response_with_meta(
                timestamps, [100.0] * 8 + [110.0], [1_000_000] * 8 + [2_000_000], meta)):
            result = ymd.fetch_daily_market_context("AAPL")
    assert result["volume"] == 2_000_000


def test_a_response_without_meta_keeps_the_previous_behaviour():
    # The guard must never drop a bar just because the provider omitted the session fields.
    for meta in ({}, {"currentTradingPeriod": {}}, {"regularMarketTime": None}):
        assert ymd._bar_is_the_open_session(meta, OPEN_AT) is False
    assert ymd._bar_is_the_open_session(None, OPEN_AT) is False
    assert ymd._bar_is_the_open_session(
        {"regularMarketTime": True, "currentTradingPeriod": {"regular": {"start": OPEN_AT, "end": CLOSE_AT}}},
        OPEN_AT) is False   # booleans are ints in Python; they are not timestamps
