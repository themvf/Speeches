import math
from datetime import datetime,timedelta,timezone
import pytest
from crypto_hour_profile import (hourly_returns,coin_days,profile,_rotation_p,_shift,_tstat,_RNG,
                                 untestable_hours)

NOW=datetime(2026,9,21,0,tzinfo=timezone.utc)
T0=NOW-timedelta(days=7)

def candles(hours=168,start=T0,price=lambda i:100.0,volume=lambda i:10.0,skip=()):
    return [dict(hour=start+timedelta(hours=i),close=price(i),volume=volume(i))
            for i in range(hours) if i not in skip]

def test_returns_are_attributed_to_the_later_hour_and_never_bridge_a_gap():
    pts=candles(5,price=lambda i:100.0*(1.1**i))
    rows=hourly_returns(pts,T0,NOW)
    assert [r[0] for r in rows]==[T0+timedelta(hours=i) for i in (1,2,3,4)]
    assert all(r[1]==pytest.approx(math.log(1.1)) for r in rows)
    # Dropping hour 2 removes the returns at hour 2 (no candle) and hour 3 (no predecessor).
    gapped=hourly_returns(candles(5,price=lambda i:100.0*(1.1**i),skip=(2,)),T0,NOW)
    assert [r[0] for r in gapped]==[T0+timedelta(hours=1),T0+timedelta(hours=4)] or \
           [r[0] for r in gapped]==[T0+timedelta(hours=1)]
    assert all(r[1]==pytest.approx(math.log(1.1)) for r in gapped[:1])

def test_window_excludes_candles_outside_it_but_still_uses_the_prior_hour_for_the_first_return():
    pts=candles(10,start=T0-timedelta(hours=5))
    rows=hourly_returns(pts,T0,NOW)
    assert all(T0<=h<NOW for h,_ in rows)

def test_absent_hours_count_as_zero_volume_and_partial_days_are_dropped():
    pts=candles(48,volume=lambda i:0.0 if i%24 in (3,4) else 10.0)
    days=coin_days(pts,T0,T0+timedelta(days=2))
    assert len(days)==2 and all(len(d)==24 for d in days)
    assert all(sum(v for _,v in d)==pytest.approx(1.0) for d in days)
    # A day whose candles are missing entirely still yields 24 entries once zero-filled.
    sparse=coin_days(candles(48,skip=tuple(range(24,46))),T0,T0+timedelta(days=2))
    assert len(sparse)==2
    # Without zero filling, a day short of 20 present hours is dropped.
    assert len(coin_days(candles(48,skip=tuple(range(24,46))),T0,T0+timedelta(days=2),zero_fill=False))==1

def test_a_day_with_no_volume_at_all_is_skipped_rather_than_dividing_by_zero():
    assert coin_days(candles(24,volume=lambda i:0.0),T0,T0+timedelta(days=1))==[]

def _wobble(i,h,scale=0.05):
    """Deterministic pseudo-noise, so a synthetic hour has variance the way a real one does."""
    return scale*(((i*37+h*101)%17)/17.0-0.5)

def test_rotation_test_finds_an_injected_hour_effect_and_clears_a_profile_with_no_alignment():
    noisy=[[(h,1.0+_wobble(i,h)) for h in range(24)] for i in range(14)]
    _,p_flat=_rotation_p({'A':noisy},1.0,7,iterations=300)
    assert p_flat>0.2  # no hour is special, so rotation reproduces the observed statistic
    spiked=[[(h,(5.0 if h==13 else 1.0)+_wobble(i,h)) for h in range(24)] for i in range(14)]
    _,p_spiked=_rotation_p({'A':spiked},1.0,7,iterations=300)
    assert p_spiked<0.01  # the same hour every day is exactly what rotation destroys

def test_an_hour_with_no_variance_is_reported_rather_than_silently_untested():
    # A permanently dead hour has an identical value every day, so its t-statistic is undefined.
    dead=[[(h,0.0 if h==9 else 1.0+_wobble(i,h)) for h in range(24)] for i in range(14)]
    flagged=untestable_hours(dead,1.0)
    assert [f['hour'] for f in flagged]==[9]
    assert flagged[0]['reason']=='every observation identical' and flagged[0]['n']==14
    assert untestable_hours([[(h,1.0+_wobble(i,h)) for h in range(24)] for i in range(14)],1.0)==[]

def test_shift_relabels_hours_without_changing_values():
    groups=[[(0,1.0),(1,2.0),(23,3.0)]]
    assert _shift(groups,-4)==[[(20,1.0),(21,2.0),(19,3.0)]]

def test_tstat_needs_enough_points_and_rejects_a_degenerate_sample():
    from crypto_hour_profile import MIN_HOUR_OBS
    assert _tstat([1.0]*(MIN_HOUR_OBS-1))is None          # too few observations
    assert _tstat([2.0]*MIN_HOUR_OBS)is None              # no spread at all
    assert _tstat([1.0,2.0,3.0,4.0,5.0],null=3.0)==pytest.approx(0.0)

def test_three_nearly_identical_values_cannot_manufacture_a_huge_t():
    # Real case: ZEC's hourly series had [-0.62%, -0.59%, -0.59%] at one hour, which as an n=3
    # sample gives t = -56 off a -0.6% mean. Such an hour must be reported, never tested.
    assert _tstat([-0.0062,-0.0059,-0.0059])is None
    groups=[[(18,v)] for v in (-0.0062,-0.0059,-0.0059)]
    flagged=untestable_hours(groups,0.0)
    assert {f['hour'] for f in flagged}>={18}
    assert next(f for f in flagged if f['hour']==18)['reason'].startswith('fewer than')

def test_rng_is_deterministic_so_p_values_reproduce():
    assert [_RNG(3).randint(24) for _ in range(5)]==[_RNG(3).randint(24) for _ in range(5)]

def test_profile_flags_a_repeating_busy_hour_and_still_clears_a_constant_return():
    # Price rises by the same factor every hour, so no hour has a distinctive return,
    # while volume repeats a spike at 13:00 UTC every day.
    series={'A':candles(168,price=lambda i:100.0*(1.001**i),
                        volume=lambda i:(50.0 if i%24==13 else 1.0)+_wobble(i//24,i%24,0.2))}
    report=profile(series,7,now=NOW,iterations=300)
    assert report['coins']['A']['returns']==167
    assert report['coins']['A']['hourly_sd']==pytest.approx(0.0,abs=1e-12)
    assert 'return_p_global' not in report['coins']['A']  # dust-level variance is not a testable series
    assert report['coins']['A']['return_test']=='price did not move enough to measure'
    assert report['pooled']['volume_share']['p_global']['day']<0.01
    busiest=max(report['pooled']['volume_share']['table'],key=lambda r:r['mean'])
    assert busiest['utc_hour']==13
    assert report['pooled']['volume_share']['untestable_hours']==[]

def test_et_labels_track_the_offset_actually_in_effect():
    report=profile({'A':candles(168,volume=lambda i:10.0+_wobble(i//24,i%24,2.0))},7,now=NOW,iterations=50)
    assert report['et_offset_hours']==-4  # September in New York is EDT
    for row in report['pooled']['volume_share']['table']:
        assert row['et_hour']==(row['utc_hour']-4)%24

def test_a_window_too_short_to_test_says_so_instead_of_reporting_p_equals_one():
    from crypto_hour_profile import MIN_HOUR_OBS
    # Three days gives at most four observations per hour, one short of what a t-statistic needs.
    report=profile({'A':candles(72,start=NOW-timedelta(days=3),
                                price=lambda i:100.0+_wobble(i//24,i%24,4.0))},3,now=NOW,iterations=50)
    assert 'return_p_global' not in report['coins']['A']
    assert 'too short to test' in report['coins']['A']['return_test']
    assert 'p_global' not in report['pooled']['returns']
    assert 'too short to test' in report['pooled']['returns']['test']
    assert len(report['pooled']['returns']['untestable_hours'])==24
    assert len(report['pooled']['returns']['table'])==24  # the profile still shows, only the test is withheld

def test_profile_skips_a_coin_with_too_few_returns_without_failing_the_run():
    report=profile({'A':candles(168,price=lambda i:100.0+_wobble(i//24,i%24,4.0)),'B':candles(6)},
                   7,now=NOW,iterations=100)
    assert 'return_p_global' not in report['coins']['B']
    assert report['coins']['B']['return_test']=='fewer than 24 returns in the window'
    assert report['coins']['B']['returns']==5
    assert 'return_p_global' in report['coins']['A']


def test_a_rolling_volume_series_is_kept_out_of_the_volume_profile():
    """CoinGecko reports a rolling 24h total, so ZEC's volume is not hourly volume.

    Averaging a smoothed window against real hourly series would flatten the profile toward
    uniform. The series is excluded and named, never silently mixed in.
    """
    real={'kind':'ohlcv','points':candles(168,price=lambda i:100.0+_wobble(i//24,i%24,4.0),
                                          volume=lambda i:(50.0 if i%24==13 else 1.0)+_wobble(i//24,i%24,0.2))}
    rolling={'kind':'price_observation','points':candles(168,price=lambda i:100.0+_wobble(i//24,i%24,3.0),
                                                        volume=lambda i:1000.0+i)}
    report=profile({'A':real,'B':rolling},7,now=NOW,iterations=200)
    assert [e['coin'] for e in report['volume_excluded']]==['B']
    assert 'rolling 24h' in report['volume_excluded'][0]['reason']
    assert report['pooled']['volume_share']['coins']==1        # only the real series contributed
    assert report['coins']['B']['returns']==167                 # its RETURNS are still analysed
    assert report['pooled']['volatility']['coins']==2

def test_three_nulls_are_reported_and_the_coin_null_is_never_the_most_permissive():
    """The coin null gives each coin one vote, so it cannot be looser than the day null by much."""
    per={c:[[(h,(3.0 if h==13 else 1.0)+_wobble(i+ord(c),h)) for h in range(24)] for i in range(10)]
         for c in 'ABCDE'}
    ps={m:_rotation_p(per,1.0,11,m,iterations=400)[1] for m in ('day','coin','market')}
    assert ps['day']<0.01                       # a fixed hour every day, every coin
    assert ps['coin']<=0.2 and ps['market']<=0.2
    assert ps['coin']>=ps['day']                # stricter null can only cost significance

def test_profile_exposes_all_three_nulls_for_each_pooled_metric():
    series={c:{'kind':'ohlcv','points':candles(168,price=lambda i:100.0+_wobble(i//24,i%24,4.0),
                                               volume=lambda i:(9.0 if i%24==13 else 1.0)+_wobble(i//24,i%24,0.2))}
            for c in ('A','B','C')}
    report=profile(series,7,now=NOW,iterations=200)
    assert set(report['pooled']['volume_share']['p_global'])=={'day','coin','market'}
    assert report['pooled']['volume_share']['coins']==3


def test_split_half_separates_a_persistent_shape_from_a_reshuffled_one():
    from crypto_hour_profile import split_half
    # Same busy hour in both halves: the profile should replicate.
    steady={c:[[(h,(4.0 if h==13 else 1.0)+_wobble(i+ord(c),h)) for h in range(24)] for i in range(12)]
            for c in 'ABC'}
    got=split_half(steady,1.0,5,iterations=400)
    assert got['r']>0.8 and got['p']<0.05 and got['coins']==3
    # Busy hour MOVES between halves: in-sample the profile is peaky, out of sample it does not hold.
    drifting={c:[[(h,(4.0 if h==(13 if i<6 else 2) else 1.0)+_wobble(i+ord(c),h)) for h in range(24)]
                 for i in range(12)] for c in 'ABC'}
    moved=split_half(drifting,1.0,5,iterations=400)
    assert moved['r']<got['r'] and moved['p']>0.05

def test_split_half_declines_to_answer_when_there_are_too_few_coins_or_days():
    from crypto_hour_profile import split_half
    assert split_half({'A':[[(h,1.0) for h in range(24)] for _ in range(8)]},1.0,5,iterations=50) is None
    assert split_half({'A':[[(1,1.0)]],'B':[[(1,2.0)]]},1.0,5,iterations=50) is None
