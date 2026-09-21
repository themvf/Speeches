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
            out[coin] = [{'hour': p['hour'], 'close': p['close'], 'volume': p['volume']} for p in pts if p['hour'] >= since]
        except Exception as exc:  # noqa: BLE001 - one coin's provider failure is a coverage gap, not a failed run.
            errors.append('%s: %s %s' % (coin, type(exc).__name__, str(exc)[:160]))
    if 'ZEC' in coins:
        try:
            raw = get(mh.ZEC_HOURLY_URL)
            pts = mh.normalize_hourly(raw, 'price_observation', now, start=archive_start('ZEC'))
            out['ZEC'] = [{'hour': p['hour'], 'close': p['close'], 'volume': p['volume']} for p in pts if p['hour'] >= since]
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


def _bucket(groups):
    acc = {h: [] for h in range(24)}
    for g in groups:
        for h, v in g: acc[h].append(v)
    return acc


def untestable_hours(groups, null):
    """Hours with too few observations to test, or with every observation identical.

    Such an hour contributes nothing to the global statistic. A constant hour is exactly the shape a
    permanently dead trading hour would take, so it is reported rather than dropped in silence: an
    hour missing from the test must never read as an hour that was tested and cleared.
    """
    acc = _bucket(groups)
    return [{'hour': h, 'n': len(acc[h]),
             'reason': 'fewer than %d observations' % MIN_HOUR_OBS if len(acc[h]) < MIN_HOUR_OBS
                       else 'every observation identical'}
            for h in range(24) if _tstat(acc[h], null) is None]


def _rotation_p(groups, null, seed, iterations=ITERATIONS):
    """groups: [[(hour, value), ...]] one list per rotatable unit (a coin-day). Returns (max|t|, p)."""
    def stat(gs):
        return max((abs(t) for t in (_tstat(v, null) for v in _bucket(gs).values()) if t is not None), default=0.0)
    rng = _RNG(seed)
    observed = stat(groups)
    ge = 0
    for _ in range(iterations):
        rotated = []
        for g in groups:
            k = rng.randint(len(g))
            values = [v for _, v in g]
            values = values[k:] + values[:k]
            rotated.append([(g[i][0], values[i]) for i in range(len(g))])
        if stat(rotated) >= observed: ge += 1
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

    ret_groups, vol_groups, absret_groups = [], [], []
    for coin, points in sorted(series.items()):
        rows = hourly_returns(points, since, now)
        cd = coin_days(points, since, now)
        entry = {'returns': len(rows), 'coin_days': len(cd),
                 'distinct_days': len({h.date() for h, _ in rows}),
                 'hourly_sd': st.stdev([r for _, r in rows]) if len(rows) > 2 else None}
        if len(rows) >= 24 and (entry['hourly_sd'] or 0) > MIN_SD:
            byday = {}
            for h, r in rows: byday.setdefault(h.date(), []).append((h.hour, r))
            groups = [g for g in byday.values() if len(g) >= 2]
            testable = 24 - len(untestable_hours(groups, 0.0))
            if testable:
                obs, p = _rotation_p(groups, 0.0, SEED + abs(hash(coin)) % 9999, iterations)
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
            ret_groups.extend([[(h, (r - mean) / sd) for h, r in g] for g in groups])
            base = st.fmean([abs(r) for _, r in rows]) or 1
            absret_groups.extend([[(h, abs(r) / base) for h, r in g] for g in groups])
        else:
            entry['return_test'] = ('fewer than 24 returns in the window' if len(rows) < 24
                                    else 'price did not move enough to measure')
        vol_groups.extend(cd)
        report['coins'][coin] = entry

    for name, groups, null in [('returns', ret_groups, 0.0), ('volume_share', vol_groups, 1 / 24),
                               ('volatility', absret_groups, 1.0)]:
        if not groups: continue
        untestable = untestable_hours(groups, null)
        if len(untestable) == 24:
            # No hour can be tested, but the profile itself is still worth showing; what must not
            # happen is a p-value that reads as "tested and found nothing".
            report['pooled'][name] = {'units': len(groups), 'untestable_hours': untestable,
                                      'table': _table(groups, null, et_offset),
                                      'test': 'no hour has %d observations; too short to test' % MIN_HOUR_OBS}
            continue
        obs, p = _rotation_p(groups, null, SEED + len(name), iterations)
        report['pooled'][name] = {
            'max_abs_t': obs, 'p_global': p, 'units': len(groups),
            'untestable_hours': untestable,
            'table': _table(groups, null, et_offset),
            'note': 'One test per metric. The ET column relabels the same hours, so it is not a '
                    'second, independent confirmation. Days are bucketed on their UTC date.'}
    return report


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
