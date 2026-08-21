import analyze_macro_archetype_bands as diag


def test_band_for_separates_every_reason_a_wallet_fails_to_qualify():
    # Not enough events for the cohort's bar.
    assert diag.band_for(4, 4, 100, 100, 90, 100, 0.5, "headline_cpi", 0.4) == diag.BAND_INSUFFICIENT_EVENTS
    # Enough events, but most of the cost landed after release.
    assert diag.band_for(10, 5, 50, 100, 10, 30, 0.5, "headline_cpi", 0.4) == diag.BAND_LOW_TIMING_COVERAGE
    # Never re-settled, so there is no price to judge the win rate against.
    assert diag.band_for(10, 6, 50, 100, 90, 100, 0.5, "headline_cpi", None) == diag.BAND_NO_ENTRY_PRICE
    # Wins 60% but paid 0.70 for it: correctly excluded, and distinguishable
    # from "not enough data" - this wallet has been measured and has no edge.
    assert diag.band_for(10, 6, 50, 100, 90, 100, 0.5, "headline_cpi", 0.70) == diag.BAND_NO_EDGE


def test_band_for_matches_classify_wallet_for_named_archetypes():
    import polymarket_macro_sync as macro
    cases = [
        (10, 7, 100, 100, 70, 100, 0.5, "fed_decision", 0.40),
        (10, 8, 100, 100, 10, 100, 0.8, "fed_decision", 0.60),
        (10, 4, 150, 100, 80, 100, 0.2, "fed_decision", 0.20),
        (5, 4, 100, 100, 70, 100, 0.5, "us_gdp", 0.40),
        (10, 6, 50, 100, 90, 100, 0.5, "fed_decision", 0.70),
    ]
    for events, wins, pnl, cost, pc, tc, entry, cohort, entry_avg in cases:
        archetype = macro.classify_wallet(events, wins, pnl, cost, pc, tc, entry, cohort, entry_avg)
        band = diag.band_for(events, wins, pnl, cost, pc, tc, entry, cohort, entry_avg)
        # The diagnostic may name the REASON more precisely, but must never
        # disagree about whether the wallet qualified.
        qualified = band in (diag.BAND_EARLY_SHARP, diag.BAND_RELEASE_SCALPER, diag.BAND_LONGSHOT)
        assert qualified == (archetype != "unclassified"), (band, archetype)
        if qualified:
            assert band == archetype


def _row(wallet, cohort, events, wins, pnl, cost, predictive_cost, timing_cost, entry=None, name="", entry_avg=0.4):
    return {"wallet": wallet, "cohort": cohort, "name": name, "events": events, "wins": wins,
            "pnl": pnl, "cost": cost, "predictive_cost": predictive_cost, "timing_cost": timing_cost,
            "win_entry_avg": entry, "entry_avg": entry_avg}


def test_analyze_excludes_generalist_rows_and_profiles_the_dead_zone():
    rows = [
        _row("a", "headline_cpi", 10, 6, 50, 100, 40, 100, entry=0.5),  # dead zone
        _row("a", "macro_generalist", 22, 12, 200, 300, 150, 300, entry=0.5),  # must be excluded
        _row("b", "headline_cpi", 3, 2, 20, 20, 15, 20, entry=0.4),  # insufficient events
        _row("c", "fed_decision", 10, 7, 100, 100, 70, 100, entry=0.5),  # early_sharp
    ]
    report = diag.analyze(rows)
    assert report["total_wallet_cohort_pairs"] == 3  # generalist row excluded
    assert sum(report["overall_bands"].values()) == 3
    assert report["overall_bands"][diag.BAND_INSUFFICIENT_EVENTS] == 1
    assert diag.BAND_DEAD_ZONE not in report["overall_bands"], "edge-gating should leave the dead zone empty"


def test_analyze_reports_empty_dead_zone_profile_when_nothing_lands_there():
    rows = [_row("a", "fed_decision", 3, 2, 20, 20, 15, 20, entry=0.4)]
    report = diag.analyze(rows)
    assert report["overall_bands"] == {diag.BAND_INSUFFICIENT_EVENTS: 1}
    assert report["dead_zone_profile"]["win_rate"] == {"n": 0, "mean": None, "median": None, "p25": None, "p75": None}
