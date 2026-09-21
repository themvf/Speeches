"""Time-of-day profile for every tracked coin: does the hour of day predict return, activity or volatility?

Reads the hourly archive (`crypto_market_hourly_latest`) when DATABASE_URL is set, and otherwise
re-fetches the same pinned-pool candles from the public providers, so the answer is reproducible
without database access.

Every reported p-value is a day-rotation permutation test whose statistic is max|t| across the 24
hours, so it already accounts for having looked at all 24 hours. Rotating each calendar day's values
by a random offset destroys hour-of-day alignment while preserving the value distribution and the
within-day serial structure exactly. A per-hour t-statistic on its own is NOT evidence of a
time-of-day effect; only the global p-value is.

Absent candles mean no trades that hour, not missing data, so volume profiles count them as zero.
Returns are never bridged across an absent hour.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
import statistics as st
from zoneinfo import ZoneInfo

from crypto_coins import markets, archive_start

ET = ZoneInfo('America/New_York')
ITERATIONS = 3000
SEED = 20260921
MIN_SD = 1e-9  # Below this the price never really moved; a permutation p-value would be noise.
# A t-statistic on three observations is not a measurement. Three coincidentally-similar values
# give a near-zero standard error and an enormous t off a trivial mean -- ZEC's CoinGecko series
# produced t = -56 from a -0.6% mean this way. Hours below this count are reported, never tested.
MIN_HOUR_OBS = 5


class _RNG:
    """Deterministic LCG so a rerun of the same window reproduces the same p-values."""

    def __init__(self, seed): self.s = seed & 0xFFFFFFFF

    def next(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s / 0x100000000

    def randint(self, n): return int(self.next() * n) % n if n else 0


def _tstat(xs, null=0.0, min_obs=MIN_HOUR_OBS):
    n = len(xs)
    if n < min_obs: return None
    sd = st.stdev(xs)
    return None if sd == 0 else (st.fmean(xs) - null) / (sd / math.sqrt(n))


def load_archive(conn, coins, since):
    """Hourly candles per coin from the pinned default source."""
    out = {}
    with conn, conn.cursor() as cur:
        for coin in coins:
            cur.execute('''SELECT h.hour,h.close,h.volume FROM crypto_market_hourly_latest h
                JOIN crypto_market_sources s ON s.id=h.source_id
                WHERE s.coin=%s AND s.is_default AND h.hour>=%s ORDER BY h.hour''', (coin, since))
            rows = cur.fetchall()
            if rows: out[coin] = [{'hour': r[0], 'close': float(r[1]), 'volume': float(r[2])} for r in rows]
    return out


def load_live(coins, since, fetch=None, wait=None):
    """Same candles straight from the providers, using the archive's own pool-pin rule."""
    import time
    import requests
    import crypto_market_history as mh
    fetch = fetch or requests.get
    wait = wait or time.sleep
    now = datetime.now(timezone.utc)

    def get(url):
        for attempt in range(4):
            wait(2.5)  # Public market APIs are shared and rate-limited; never burst requests.
            response = fetch(url, timeout=30, allow_redirects=False, headers={'Accept': 'application/json'})
            if response.status_code == 429 and attempt < 3:
                try: delay = float(getattr(response, 'headers', {}).get('Retry-After', '15'))
                except (ValueError, TypeError): delay = 15
                if not math.isfinite(delay) or delay > 30: raise ValueError('Market cooldown exceeds retry bound')
                wait(max(3, delay))
                continue
            if response.status_code != 200: raise ValueError('Market HTTP ' + str(response.status_code))
            return response.json()
        raise ValueError('Market rate limit')

    out, errors = {}, []
    for coin, (network, address, start) in markets().items():
        if coin not in coins: continue
        try:
            selected = mh.pools(get(mh.GECKO + network + '/tokens/' + address + '/pools'), address, network)
            if not selected: errors.append(coin + ': no indexed pool'); continue
            # Same pin rule as the archive: the oldest pool among those archived.
            pool = min(selected, key=lambda p: (p['created'], p['id']))
            raw = get(mh.GECKO + network + '/pools/' + pool['id'] + '/ohlcv/hour?aggregate=1&limit=1000'
                      '&currency=usd&include_empty_intervals=false&token=' + pool['side'])
            pts = mh.normalize_hourly(raw, 'ohlcv', now, start=start)
            out[coin] = {'kind': 'ohlcv', 'points': [{'hour': p['hour'], 'close': p['close'], 'volume': p['volume']}
                                                     for p in pts if p['hour'] >= since]}
        except Exception as exc:  # noqa: BLE001 - one coin's provider failure is a coverage gap, not a failed run.
            errors.append('%s: %s %s' % (coin, type(exc).__name__, str(exc)[:160]))
    if 'ZEC' in coins:
        try:
            raw = get(mh.ZEC_HOURLY_URL)
            pts = mh.normalize_hourly(raw, 'price_observation', now, start=archive_start('ZEC'))
            out['ZEC'] = {'kind': 'price_observation',
                          'points': [{'hour': p['hour'], 'close': p['close'], 'volume': p['volume']}
                                     for p in pts if p['hour'] >= since]}
        except Exception as exc:  # noqa: BLE001
            errors.append('ZEC: %s %s' % (type(exc).__name__, str(exc)[:160]))
    return out, errors


def hourly_returns(points, since, until):
    """(hour, log return) for candles in the window whose immediately preceding hour is present."""
    by = {p['hour']: p for p in points}
    rows = []
    for p in sorted(points, key=lambda p: p['hour']):
        prev = by.get(p['hour'] - timedelta(hours=1))
        if not prev or not since <= p['hour'] < until: continue
        if p['close'] <= 0 or prev['close'] <= 0: continue
        rows.append((p['hour'], math.log(p['close'] / prev['close'])))
    return rows


def coin_days(points, since, until, zero_fill=True):
    """Per calendar day, [(hour, share of that day's volume)]; absent hours count as zero volume."""
    by = {p['hour']: p for p in points}
    first = min(by) if by else None
    days = {}
    t = since
    while t < until:
        if first is not None and t >= first:
            if t in by: days.setdefault(t.date(), []).append((t.hour, by[t]['volume']))
            elif zero_fill: days.setdefault(t.date(), []).append((t.hour, 0.0))
        t += timedelta(hours=1)
    out = []
    for values in days.values():
        if len(values) < (24 if zero_fill else 20): continue
        total = sum(v for _, v in values)
        if total > 0: out.append([(h, v / total) for h, v in values])
    return out


def _flatten(per_coin):
    return [g for groups in per_coin.values() for g in groups]


def _bucket(groups):
    acc = {h: [] for h in range(24)}
    for g in groups:
        for h, v in g: acc[h].append(v)
    return acc


def _rotate(group, k):
    values = [v for _, v in group]
    values = values[k:] + values[:k]
    return [(group[i][0], values[i]) for i in range(len(group))]


def untestable_hours(per_coin_or_groups, null):
    """Hours with too few observations to test, or with every observation identical.

    Such an hour contributes nothing to the global statistic. A constant hour is exactly the shape a
    permanently dead trading hour would take, so it is reported rather than dropped in silence: an
    hour missing from the test must never read as an hour that was tested and cleared.
    """
    groups = (_flatten(per_coin_or_groups) if isinstance(per_coin_or_groups, dict)
              else per_coin_or_groups)
    acc = _bucket(groups)
    return [{'hour': h, 'n': len(acc[h]),
             'reason': 'fewer than %d observations' % MIN_HOUR_OBS if len(acc[h]) < MIN_HOUR_OBS
                       else 'every observation identical'}
            for h in range(24) if _tstat(acc[h], null) is None]


def _rotation_p(per_coin, null, seed, mode='day', iterations=ITERATIONS):
    """Permutation p-value under one of three nulls, in increasing strictness.

    day    each coin-day is rotated independently, so a coin's days count as separate evidence.
    coin   one offset per coin rotates all of its days together. This is the honest unit for a
           cross-coin claim: a coin contributes one vote however many days it has, and any
           day-to-day correlation within a coin survives the shuffle instead of being averaged away.
    market one offset per calendar day is applied to every coin at once, preserving contemporaneous
           cross-coin co-movement, so a single market-wide move cannot be counted once per coin.

    'day' is anti-conservative when a coin's daily profiles are correlated for reasons unrelated to
    the clock, which is exactly the case this data cannot rule out. Report all three.
    """
    def stat(per):
        acc = _bucket(_flatten(per))
        return max((abs(t) for t in (_tstat(v, null) for v in acc.values()) if t is not None), default=0.0)
    rng = _RNG(seed)
    observed = stat(per_coin)
    most_days = max((len(g) for g in per_coin.values()), default=0)
    ge = 0
    for _ in range(iterations):
        shuffled = {}
        if mode == 'coin':
            for coin, groups in per_coin.items():
                k = rng.randint(24)
                shuffled[coin] = [_rotate(g, k % len(g)) for g in groups if g]
        elif mode == 'market':
            offsets = [rng.randint(24) for _ in range(most_days)]
            for coin, groups in per_coin.items():
                shuffled[coin] = [_rotate(g, offsets[i] % len(g)) for i, g in enumerate(groups) if g]
        else:
            for coin, groups in per_coin.items():
                shuffled[coin] = [_rotate(g, rng.randint(len(g))) for g in groups if g]
        if stat(shuffled) >= observed: ge += 1
    return observed, (ge + 1) / (iterations + 1)


def _table(groups, null, tz_offset):
    acc = {h: [] for h in range(24)}
    for g in groups:
        for h, v in g: acc[h].append(v)
    return [{'utc_hour': h, 'et_hour': (h + tz_offset) % 24, 'n': len(acc[h]),
             'mean': st.fmean(acc[h]) if acc[h] else None,
             'median': st.median(acc[h]) if acc[h] else None,
             't': _tstat(acc[h], null)} for h in range(24)]


def _shift(groups, offset):
    return [[((h + offset) % 24, v) for h, v in g] for g in groups]


def profile(series, days, now=None, iterations=ITERATIONS):
    """Return, volume-share and volatility profiles by hour of day, with global permutation p-values."""
    now = (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    since = now - timedelta(days=days)
    # utcoffset() is already the UTC->Eastern shift (-4 during EDT, -5 during EST); do not negate it.
    et_offset = int(now.astimezone(ET).utcoffset().total_seconds() // 3600)
    report = {'window_days': days, 'since': since.isoformat(), 'until': now.isoformat(),
              'et_offset_hours': et_offset, 'iterations': iterations, 'coins': {}, 'pooled': {}}

    ret_groups, vol_groups, absret_groups = {}, {}, {}
    report['volume_excluded'] = []
    for coin, entry_in in sorted(series.items()):
        points = entry_in['points'] if isinstance(entry_in, dict) else entry_in
        kind = entry_in.get('kind', 'ohlcv') if isinstance(entry_in, dict) else 'ohlcv'
        rows = hourly_returns(points, since, now)
        # CoinGecko's market_chart reports a ROLLING 24-HOUR total, not the volume traded in that
        # hour: the series drifts ~2% an hour where a real hourly series swings 50%. Feeding it into
        # an hour-of-day volume profile would be averaging a smoothed window against real hours.
        cd = coin_days(points, since, now) if kind == 'ohlcv' else []
        if kind != 'ohlcv':
            report['volume_excluded'].append({'coin': coin, 'kind': kind,
                                              'reason': 'provider reports a rolling 24h total, not hourly volume'})
        entry = {'returns': len(rows), 'coin_days': len(cd),
                 'distinct_days': len({h.date() for h, _ in rows}),
                 'hourly_sd': st.stdev([r for _, r in rows]) if len(rows) > 2 else None}
        if len(rows) >= 24 and (entry['hourly_sd'] or 0) > MIN_SD:
            byday = {}
            for h, r in rows: byday.setdefault(h.date(), []).append((h.hour, r))
            groups = [g for g in byday.values() if len(g) >= 2]
            testable = 24 - len(untestable_hours(groups, 0.0))
            if testable:
                # A single coin has no cross-coin structure, so the day null is the only one available.
                obs, p = _rotation_p({coin: groups}, 0.0, SEED + abs(hash(coin)) % 9999, 'day', iterations)
                entry['return_max_abs_t'] = obs
                entry['return_p_global'] = p
                entry['testable_hours'] = testable
            else:
                # Every hour has fewer than MIN_HOUR_OBS observations: the window is too short
                # to test this coin at all. Saying so is not the same as finding no effect.
                entry['return_test'] = ('no hour has %d observations in this window; '
                                        'too short to test' % MIN_HOUR_OBS)
            # Minimum detectable recurring same-hour move, 80% power, two-sided 5%, 24-hour penalty.
            nd = entry['distinct_days']
            entry['mde_pct'] = 100 * (math.exp(2.8 * entry['hourly_sd'] / math.sqrt(nd)) - 1) if nd else None
            entry['mde_pct_corrected'] = 100 * (math.exp(3.6 * entry['hourly_sd'] / math.sqrt(nd)) - 1) if nd else None
            mean = st.fmean([r for _, r in rows]); sd = entry['hourly_sd']
            ret_groups[coin] = [[(h, (r - mean) / sd) for h, r in g] for g in groups]
            base = st.fmean([abs(r) for _, r in rows]) or 1
            absret_groups[coin] = [[(h, abs(r) / base) for h, r in g] for g in groups]
        else:
            entry['return_test'] = ('fewer than 24 returns in the window' if len(rows) < 24
                                    else 'price did not move enough to measure')
        if cd: vol_groups[coin] = cd
        report['coins'][coin] = entry

    for name, per_coin, null in [('returns', ret_groups, 0.0), ('volume_share', vol_groups, 1 / 24),
                                 ('volatility', absret_groups, 1.0)]:
        if not per_coin: continue
        flat = _flatten(per_coin)
        untestable = untestable_hours(per_coin, null)
        block = {'units': len(flat), 'coins': len(per_coin), 'untestable_hours': untestable,
                 'table': _table(flat, null, et_offset),
                 'note': 'One test per metric. The ET column relabels the same hours, so it is not a '
                         'second, independent confirmation. Days are bucketed on their UTC date. '
                         'Read p_global.coin as the headline: it is the strictest null and treats '
                         'each coin as one unit.'}
        if len(untestable) == 24:
            # No hour can be tested, but the profile itself is still worth showing; what must not
            # happen is a p-value that reads as "tested and found nothing".
            block['test'] = 'no hour has %d observations; too short to test' % MIN_HOUR_OBS
        else:
            block['p_global'] = {}
            for mode in ('day', 'coin', 'market'):
                obs, p = _rotation_p(per_coin, null, SEED + len(name) + len(mode), mode, iterations)
                block['max_abs_t'] = obs
                block['p_global'][mode] = p
            block['split_half'] = split_half(per_coin, null, SEED + len(name) * 3, iterations)
        report['pooled'][name] = block
    return report


def _profile_vector(per_coin, null):
    acc = _bucket(_flatten(per_coin))
    return [st.fmean(acc[h]) - null if acc[h] else None for h in range(24)]


def _correlation(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3: return None
    xs = [x for x, _ in pairs]; ys = [y for _, y in pairs]
    mx, my = st.fmean(xs), st.fmean(ys)
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return None if den == 0 else sum((x - mx) * (y - my) for x, y in pairs) / den


def split_half(per_coin, null, seed, iterations=ITERATIONS):
    """Does the hour-of-day shape measured in the first half still hold in the second?

    This is the question a global p-value cannot answer. A profile can be significantly
    non-flat in-sample and still be a different shape next week. Each coin's days are split in
    half chronologically; the correlation between the two halves' 24-hour profiles is compared
    against a market-wide rotation null, which re-dates the second half while preserving
    cross-coin co-movement exactly.
    """
    first, second = {}, {}
    for coin, groups in per_coin.items():
        if len(groups) < 4: continue
        mid = len(groups) // 2
        first[coin], second[coin] = groups[:mid], groups[mid:]
    if len(first) < 2: return None
    a = _profile_vector(first, null)
    observed = _correlation(a, _profile_vector(second, null))
    if observed is None: return None
    rng = _RNG(seed)
    most = max(len(g) for g in second.values())
    ge = 0
    for _ in range(iterations):
        offsets = [rng.randint(24) for _ in range(most)]
        shuffled = {c: [_rotate(g, offsets[i] % len(g)) for i, g in enumerate(groups) if g]
                    for c, groups in second.items()}
        if (_correlation(a, _profile_vector(shuffled, null)) or -1) >= observed: ge += 1
    return {'r': observed, 'p': (ge + 1) / (iterations + 1), 'coins': len(first)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', type=int, default=14, help='window length in days')
    parser.add_argument('--live', action='store_true', help='re-fetch from providers instead of the archive')
    parser.add_argument('--iterations', type=int, default=ITERATIONS)
    parser.add_argument('--coins', help='comma-separated symbols; default is every tracked coin')
    args = parser.parse_args()

    coins = [c.strip().upper() for c in args.coins.split(',')] if args.coins else list(markets()) + ['ZEC']
    since = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(days=args.days)
    errors = []
    if args.live or not os.environ.get('DATABASE_URL'):
        series, errors = load_live(coins, since)
        source = 'providers'
    else:
        import psycopg2
        conn = psycopg2.connect(os.environ['DATABASE_URL'], connect_timeout=15)
        try: series = load_archive(conn, coins, since)
        finally: conn.close()
        source = 'archive'
    if not series:
        print(json.dumps({'ok': False, 'source': source, 'errors': errors or ['no hourly candles in window']}))
        raise SystemExit(1)
    report = profile(series, args.days, iterations=args.iterations)
    report['source'] = source
    report['errors'] = errors
    report['ok'] = True
    print(json.dumps(report, indent=1, default=str))


if __name__ == '__main__':
    main()
