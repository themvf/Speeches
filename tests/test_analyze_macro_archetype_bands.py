import analyze_macro_archetype_bands as diag


def test_band_for_splits_the_two_implicit_unclassified_reasons():
    # Not enough events for the cohort's bar.
    assert diag.band_for(4, 4, 100, 100, 90, 100, 0.5, "headline_cpi") == diag.BAND_INSUFFICIENT_EVENTS
    # Enough events, but most of the cost landed after release (low timing coverage).
    assert diag.band_for(10, 5, 50, 100, 10, 30, 0.5, "headline_cpi") == diag.BAND_LOW_TIMING_COVERAGE
    # Enough events, full timing coverage, but predictive_share sits in the
    # 0.25-0.60 gap that no named archetype covers and win rate/entry don't
    # satisfy longshot either - this is the actual dead zone.
    assert diag.band_for(10, 6, 50, 100, 40, 100, 0.5, "headline_cpi") == diag.BAND_DEAD_ZONE


def test_band_for_matches_classify_wallet_for_named_archetypes():
    import polymarket_macro_sync as macro
    cases = [
        (10, 7, 100, 100, 70, 100, 0.5, "fed_decision"),
        (10, 8, 100, 100, 10, 100, 0.8, "fed_decision"),
        (10, 3, 150, 100, 80, 100, 0.2, "fed_decision"),
        (5, 4, 100, 100, 70, 100, 0.5, "us_gdp"),
    ]
    for events, wins, pnl, cost, predictive_cost, timing_cost, entry, cohort in cases:
        archetype = macro.classify_wallet(events, wins, pnl, cost, predictive_cost, timing_cost, entry, cohort)
        band = diag.band_for(events, wins, pnl, cost, predictive_cost, timing_cost, entry, cohort)
        assert band == archetype, (events, wins, pnl, cost, predictive_cost, timing_cost, entry, cohort)


def _row(wallet, cohort, events, wins, pnl, cost, predictive_cost, timing_cost, entry=None, name=""):
    return {"wallet": wallet, "cohort": cohort, "name": name, "events": events, "wins": wins,
            "pnl": pnl, "cost": cost, "predictive_cost": predictive_cost, "timing_cost": timing_cost,
            "win_entry_avg": entry}


def test_analyze_excludes_generalist_rows_and_profiles_the_dead_zone():
    rows = [
        _row("a", "headline_cpi", 10, 6, 50, 100, 40, 100, entry=0.5),  # dead zone
        _row("a", "macro_generalist", 22, 12, 200, 300, 150, 300, entry=0.5),  # must be excluded
        _row("b", "headline_cpi", 3, 2, 20, 20, 15, 20, entry=0.4),  # insufficient events
        _row("c", "fed_decision", 10, 7, 100, 100, 70, 100, entry=0.5),  # early_sharp
    ]
    report = diag.analyze(rows)
    assert report["total_wallet_cohort_pairs"] == 3  # generalist row excluded
    assert report["overall_bands"] == {
        diag.BAND_DEAD_ZONE: 1,
        diag.BAND_INSUFFICIENT_EVENTS: 1,
        diag.BAND_EARLY_SHARP: 1,
    }
    assert report["per_cohort_bands"]["headline_cpi"] == {diag.BAND_DEAD_ZONE: 1, diag.BAND_INSUFFICIENT_EVENTS: 1}
    assert report["dead_zone_profile"]["win_rate"]["n"] == 1
    assert report["dead_zone_profile"]["win_rate"]["mean"] == 0.6
    assert report["dead_zone_profile"]["pnl"]["mean"] == 50


def test_analyze_reports_empty_dead_zone_profile_when_nothing_lands_there():
    rows = [_row("a", "fed_decision", 3, 2, 20, 20, 15, 20, entry=0.4)]
    report = diag.analyze(rows)
    assert report["overall_bands"] == {diag.BAND_INSUFFICIENT_EVENTS: 1}
    assert report["dead_zone_profile"]["win_rate"] == {"n": 0, "mean": None, "median": None, "p25": None, "p75": None}
