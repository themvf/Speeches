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
