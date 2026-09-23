"""Price-independent adoption evidence. No wallet identities or prices in permanent summaries."""
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal
from hashlib import sha256
import json
from statistics import median
from .metrics import number

METHOD = 'adoption-v1'


def summarize(day, snapshots, holders, expected):
    """Only complete, supply-reconciled stored enumerations qualify. BP is filtered by caller."""
    if not snapshots or len(snapshots) != expected or any(not s['holders_complete'] for s in snapshots):
        return None
    by_asset = defaultdict(list)
    for h in holders:
        by_asset[h['asset_id']].append(h)
    for s in snapshots:
        rows = by_asset[s['asset_id']]
        if len(rows) != s['unique_holders'] or number(s['token_supply']) is None:
            return None  # Raw history may have expired; never invent an old ownership distribution.
    excluded = {h['wallet_address'] for h in holders if h['excluded']}
    wallets, whole = defaultdict(set), set()
    assets = {}
    for s in snapshots:
        eligible = [h for h in by_asset[s['asset_id']] if h['wallet_address'] not in excluded and h['balance_tokens'] > 0]
        assets[str(s['asset_id'])] = dict(supply=str(s['token_supply']), holders=len(eligible))
        for h in eligible:
            wallets[h['wallet_address']].add(s['asset_id'])
            if h['balance_tokens'] >= 1:
                whole.add(h['wallet_address'])
    return dict(date=str(day), methodology=METHOD, assets=assets, holders=len(wallets),
                whole_token_holders=len(whole), multi_asset_holders=sum(len(a) >= 2 for a in wallets.values()),
                exclusion_fingerprint=sha256('\n'.join(sorted(excluded)).encode()).hexdigest(),
                status='Estimated', source='Stored finalized Solana supply and reconciled Helius ownership',
                limitation='Nonzero non-system wallets, not individuals or $100 meaningful holders. Dust and unlabelled custody can distort counts. One-token holders are a sensitivity check, not an economic threshold.')


def assess(history, day, period, env=None):
    env = env or {}
    settings = {name: Decimal(env.get(key, default)) for name, key, default in (
        ('holder_threshold', 'BACKPACK_ADOPTION_HOLDER_PCT_30D', '1'),
        ('supply_threshold', 'BACKPACK_ADOPTION_SUPPLY_PCT_30D', '0.1'),
        ('rapid_threshold', 'BACKPACK_ADOPTION_RAPID_PCT_30D', '10'),
        ('breadth_threshold', 'BACKPACK_ADOPTION_BREADTH_PCT', '60'),
        ('momentum_threshold', 'BACKPACK_ADOPTION_MOMENTUM_PP', '0.25'))}
    if period not in (7, 30, 90) or any(not v.is_finite() or v < 0 for v in settings.values()) or settings['rapid_threshold'] <= settings['holder_threshold'] or not 50 < settings['breadth_threshold'] <= 100:
        raise ValueError('Invalid adoption settings')
    result = dict(date=str(day), period_days=period, methodology=METHOD, settings={k: str(v) for k,v in settings.items()},
                  state='Insufficient evidence', momentum='Insufficient evidence',
                  reason='Needs consecutive complete ownership observations for the same securities and system exclusions.',
                  observed_days=0, required_days=period+1)
    indexed = {str(r['date']): r for r in history}
    def window(end):
        rows = [indexed.get(str(end-timedelta(days=i))) for i in range(period+1)]
        if any(r is None for r in rows): return None
        latest, first = rows[0], rows[-1]
        cohort = set(latest['assets'])
        if not cohort or any(set(r['assets']) != cohort or r['exclusion_fingerprint'] != latest['exclusion_fingerprint'] or r['methodology'] != METHOD for r in rows): return None
        if first['holders'] <= 0 or any(number(a['supply']) is None or number(a['supply']) < 0 for r in rows for a in r['assets'].values()): return None
        # Registered but never-issued securities are observed zeros, not missing evidence.
        issued = {a for a in cohort if any(number(r['assets'][a]['supply']) > 0 for r in rows)}
        if not issued: return None
        if any(number(first['assets'][a]['supply']) == 0 for a in issued):
            return None  # A new issuance needs a positive comparable baseline; never divide by zero.
        rates = [(number(latest['assets'][a]['supply'])/number(first['assets'][a]['supply'])-1)*100 for a in issued]
        holder_rate = (Decimal(latest['holders'])/first['holders']-1)*100
        return dict(cohort=sorted(cohort), issued_cohort=sorted(issued), unissued_securities=len(cohort-issued), fingerprint=latest['exclusion_fingerprint'],
                    holders=latest['holders'], previous_holders=first['holders'],
                    holder_growth_pct=holder_rate, holder_rate_30d=holder_rate*30/period,
                    median_supply_growth_pct=median(rates), supply_rate_30d=median(rates)*30/period,
                    expanding_supply_pct=Decimal(sum(r*30/period > settings['supply_threshold'] for r in rates))*100/len(rates),
                    contracting_supply_pct=Decimal(sum(r*30/period < -settings['supply_threshold'] for r in rates))*100/len(rates),
                    growing_holder_breadth_pct=Decimal(sum(latest['assets'][a]['holders'] > first['assets'][a]['holders'] for a in issued))*100/len(issued),
                    declining_holder_breadth_pct=Decimal(sum(latest['assets'][a]['holders'] < first['assets'][a]['holders'] for a in issued))*100/len(issued),
                    whole_token_growth_pct=(Decimal(latest['whole_token_holders'])/first['whole_token_holders']-1)*100 if first['whole_token_holders'] else None,
                    multi_asset_holders=latest['multi_asset_holders'], previous_multi_asset_holders=first['multi_asset_holders'])
    for i in range(period+1):
        if str(day-timedelta(days=i)) not in indexed: break
        result['observed_days'] += 1
    current = window(day)
    if current is None: return result
    result.update(current)
    h,s = current['holder_rate_30d'],current['supply_rate_30d']
    broad = current['expanding_supply_pct'] >= settings['breadth_threshold'] and current['growing_holder_breadth_pct'] >= settings['breadth_threshold']
    shrinking = current['contracting_supply_pct'] >= settings['breadth_threshold'] and current['declining_holder_breadth_pct'] >= settings['breadth_threshold']
    if h > settings['holder_threshold'] and s > settings['supply_threshold'] and broad:
        result.update(state='Growing rapidly' if h >= settings['rapid_threshold'] else 'Growing slowly', reason='Ownership and median token supply are expanding across a broad share of tracked securities.')
        if current['whole_token_growth_pct'] is not None and current['whole_token_growth_pct'] <= 0:
            result.update(state='Mixed', reason='Nonzero wallet counts grew, but wallets holding at least one token did not. Check dust and ownership distribution.')
    elif h < -settings['holder_threshold'] and s < -settings['supply_threshold'] and shrinking:
        result.update(state='Declining', reason='Ownership and token supply are contracting broadly across the tracked securities.')
    elif abs(h) <= settings['holder_threshold'] and abs(s) <= settings['supply_threshold']:
        result.update(state='Status quo', reason='Aggregate ownership and median supply changes remain within the disclosed noise bands; inspect breadth for offsetting changes.')
    else:
        result.update(state='Mixed', reason='Ownership, issuance or breadth disagree; no single direction is forced.')
    previous = window(day-timedelta(days=period))
    if previous and previous['cohort'] == current['cohort'] and previous['issued_cohort'] == current['issued_cohort'] and previous['fingerprint'] == current['fingerprint']:
        dh,ds = h-previous['holder_rate_30d'],s-previous['supply_rate_30d']
        threshold = settings['momentum_threshold']
        momentum = 'Accelerating' if dh > threshold and ds > threshold else 'Slowing' if dh < -threshold and ds < -threshold else 'Mixed / steady'
        if result['state']=='Declining': momentum={'Accelerating':'Contraction easing','Slowing':'Contraction deepening'}.get(momentum,momentum)
        result.update(momentum=momentum, previous_holder_rate_30d=previous['holder_rate_30d'], previous_supply_rate_30d=previous['supply_rate_30d'])
    return result


def capture_adoption(conn, day, env):
    """Derive missing summaries from retained historical evidence, never rewrite snapshots."""
    from .collector import fetch_all
    from psycopg2.extras import Json
    start = day-timedelta(days=180)
    existing = fetch_all(conn,'SELECT date,data FROM backpack_adoption_daily WHERE date BETWEEN %s AND %s',(start,day))
    saved = {r['date'] for r in existing}
    days = fetch_all(conn,'SELECT date,assets_expected FROM backpack_ecosystem_daily_snapshots WHERE date BETWEEN %s AND %s ORDER BY date',(start,day))
    for d in days:
        if d['date'] in saved: continue
        snapshots=fetch_all(conn,"SELECT s.* FROM backpack_asset_daily_snapshots s JOIN backpack_assets a ON a.id=s.asset_id WHERE s.date=%s AND a.asset_type<>'bp'",(d['date'],))
        ids=[r['asset_id'] for r in snapshots]
        holders=fetch_all(conn,'SELECT asset_id,wallet_address,balance_tokens,excluded FROM backpack_asset_holder_daily_snapshots WHERE date=%s AND asset_id=ANY(%s)',(d['date'],ids))
        summary=summarize(d['date'],snapshots,holders,d['assets_expected'])
        if summary is None: continue
        with conn,conn.cursor() as cur:
            cur.execute('INSERT INTO backpack_adoption_daily(date,data) VALUES(%s,%s) ON CONFLICT DO NOTHING',(d['date'],Json(summary)))
    history=[r['data'] for r in fetch_all(conn,'SELECT data FROM backpack_adoption_daily WHERE date BETWEEN %s AND %s ORDER BY date',(start,day))]
    # Do not freeze a failed/partial day: retries can still complete its holder capture.
    if not any(r['date']==str(day) for r in history): return
    with conn,conn.cursor() as cur:
        for period in (7,30,90):
            result=assess(history,day,period,env)
            cur.execute('INSERT INTO backpack_adoption_assessments(date,period_days,data) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING',
                        (day,period,Json(result,dumps=lambda v:json.dumps(v,default=str))))
