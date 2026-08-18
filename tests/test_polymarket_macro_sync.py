from datetime import UTC, datetime, timedelta

import polymarket_macro_sync as macro


def test_classifies_ten_recurring_release_families_and_normalizes_keys():
    release = datetime(2026, 8, 7, tzinfo=UTC)
    cases = {
        "Fed Decision in July?": ("fed_decision", "fed_decision:2026-07"),
        "How many jobs added in July?": ("nonfarm_payrolls", "nonfarm_payrolls:2026-07"),
        "July Unemployment Rate": ("unemployment", "unemployment:2026-07"),
        "July Inflation US - Annual": ("headline_cpi", "headline_cpi:2026-07"),
        "Core CPI MoM - July 2026": ("core_cpi", "core_cpi:2026-07"),
        "US GDP growth in Q2 2026?": ("us_gdp", "us_gdp:2026-Q2"),
        "Core PCE YoY - July 2026": ("core_pce", "core_pce:2026-07"),
        "Core PCE MoM - July 2026": ("core_pce", "core_pce:2026-07"),
        "ISM Manufacturing PMI - July 2026": ("ism_manufacturing", "ism_manufacturing:2026-07"),
        "ISM Services PMI - July 2026": ("ism_services", "ism_services:2026-07"),
        "PPI YoY - July 2026": ("ppi", "ppi:2026-07"),
        "Producer Price Index (PPI) YoY - July 2026": ("ppi", "ppi:2026-07"),
    }
    for title, expected in cases.items():
        assert macro.classify_macro_event(title, release) == expected
    assert macro.classify_macro_event("US recession by end of 2026?", release) is None
    # Discontinued Feb-Mar 2026 series (verified live 2026-08-18); must never
    # silently start matching a different, unrelated title shape.
    assert macro.classify_macro_event("How many jobless claims during the week ending Feb 7?", release) is None


def test_release_time_uses_official_eastern_publication_time():
    date_only = datetime(2026, 7, 29, tzinfo=UTC)
    assert macro.scheduled_release_at("fed_decision", date_only) == datetime(2026, 7, 29, 18, 0, tzinfo=UTC)
    assert macro.scheduled_release_at("headline_cpi", date_only) == datetime(2026, 7, 29, 12, 30, tzinfo=UTC)
    # ISM's Report on Business goes out at 10:00am ET, not the 8:30am ET
    # default shared by the BLS/BEA releases (payrolls/CPI/PCE/PPI/GDP).
    assert macro.scheduled_release_at("ism_manufacturing", date_only) == datetime(2026, 7, 29, 14, 0, tzinfo=UTC)
    assert macro.scheduled_release_at("ism_services", date_only) == datetime(2026, 7, 29, 14, 0, tzinfo=UTC)
    assert macro.scheduled_release_at("core_pce", date_only) == datetime(2026, 7, 29, 12, 30, tzinfo=UTC)
    assert macro.scheduled_release_at("ppi", date_only) == datetime(2026, 7, 29, 12, 30, tzinfo=UTC)


def _fill(wallet, release, hours_before, outcome="Yes", price=0.5, size=10):
    return {"proxyWallet": wallet, "name": wallet, "outcome": outcome, "side": "BUY",
            "size": size, "price": price, "filled_at": release - timedelta(hours=hours_before)}


def test_aggregate_release_counts_brackets_once_and_buckets_entry_timing():
    release = datetime(2026, 8, 12, 12, 30, tzinfo=UTC)
    markets = [
        {"condition_id": "a", "winner": "Yes", "release_at": release},
        {"condition_id": "b", "winner": "No", "release_at": release},
    ]
    fills = {
        "a": [_fill("early", release, 48), _fill("late", release, 0.5)],
        "b": [_fill("early", release, 12, outcome="No"), _fill("scalper", release, -1, outcome="No")],
    }
    result = macro.aggregate_release(markets, fills)
    assert set(result) == {"early", "late", "scalper"}
    assert result["early"]["early_cost"] == 5
    assert result["early"]["pre_release_cost"] == 5
    assert result["late"]["late_cost"] == 5
    assert result["scalper"]["post_release_cost"] == 5


def test_wallet_classifier_requires_sample_and_timing_coverage():
    assert macro.classify_wallet(9, 9, 100, 100, 90, 100, 0.5) == "unclassified"
    assert macro.classify_wallet(10, 7, 100, 100, 70, 100, 0.5) == "early_sharp"
    assert macro.classify_wallet(10, 8, 100, 100, 10, 100, 0.8) == "release_scalper"
    assert macro.classify_wallet(10, 3, 150, 100, 80, 100, 0.2) == "longshot"
    assert macro.classify_wallet(10, 8, 100, 100, 10, 40, 0.8) == "unclassified"
