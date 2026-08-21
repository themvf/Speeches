import report_polymarket_distributions as dist


def _w(events, wins, entry=0.5, wallet="0xa"):
    return {"wallet": wallet, "markets": events, "wins": wins, "entry_avg": entry}


def test_flags_a_large_sustained_edge():
    # 159 markets won at an average price of 0.35 is +65 points of edge, which
    # nobody sustains in a liquid market. The clamp bug produced this shape.
    rows = [_w(159, 159, 0.35, "0xperfect"), _w(10, 10, 0.2, "0xsmall")]
    report = dist.analyze(rows, "markets")
    assert report["implausible_edges"]["count"] == 1, "only the large-sample one"
    assert report["implausible_edges"]["sample"][0]["wallet"] == "0xperfect"


def test_a_perfect_win_rate_at_near_certainty_prices_is_not_flagged():
    """Buying at 0.999 and winning every time is a real strategy earning a
    tenth of a cent, not an anomaly. Flagging win rate alone buried the one
    genuine case among a dozen of these."""
    rows = [_w(136, 136, 0.999, "0xchalk")]
    report = dist.analyze(rows, "markets")
    assert report["implausible_edges"]["count"] == 0


def test_population_edge_near_zero_is_not_flagged():
    # What a fairly-measured population looks like: no systematic edge.
    rows = [_w(50, 25, 0.50), _w(50, 26, 0.52), _w(50, 24, 0.48)]
    report = dist.analyze(rows, "markets")
    assert report["population_edge_suspicious"] is False


def test_population_edge_far_above_zero_is_flagged_as_measurement_bias():
    # If the average wallet appears to beat the price it paid by 30 points,
    # the population is not unusually skilled - the measurement is biased.
    rows = [_w(50, 40, 0.50), _w(50, 42, 0.50), _w(50, 41, 0.50)]
    report = dist.analyze(rows, "markets")
    assert report["population_edge_suspicious"] is True


def test_edge_is_weighted_by_sample_so_one_lucky_wallet_cannot_move_it():
    # 500 events at zero edge, one event at +90 points.
    rows = [_w(500, 250, 0.50), _w(1, 1, 0.10, "0xlucky")]
    report = dist.analyze(rows, "markets")
    assert abs(report["edge"]["sample_weighted_mean"]) < 0.01
    # The unweighted mean is badly skewed by that single wallet, which is
    # exactly why the weighted figure is the one that gets judged.
    assert report["edge"]["mean"] > 0.4


def test_rows_without_an_entry_price_are_excluded_from_edge_not_counted_as_zero():
    rows = [_w(50, 25, None), _w(50, 25, 0.50)]
    report = dist.analyze(rows, "markets")
    assert report["edge"]["n"] == 1
    assert report["wallets"] == 2
