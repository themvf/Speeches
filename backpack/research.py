"""Conservative, reproducible environment classification. No investment score."""
from datetime import timedelta
from decimal import Decimal
from .metrics import number


def classify(history, day, period=7, issuance_threshold=Decimal('1'), trading_threshold=Decimal('20'), growth_threshold=Decimal('1')):
    if period not in (7,30) or any(not x.is_finite() or x<0 for x in (issuance_threshold,trading_threshold,growth_threshold)):
        raise ValueError('Invalid classification settings')
    result=dict(date=day,period_days=period,state='Unavailable',reason='Requires consecutive complete observations and comparable asset coverage.',
        issuance_aum_pct=None,aum_growth_pct=None,holder_growth_pct=None,trading_growth_pct=None,
        issuance_threshold_pct=issuance_threshold,trading_threshold_pct=trading_threshold,growth_threshold_pct=growth_threshold,
        methodology='v1: net issuance / starting AUM; AUM and meaningful-holder growth; two adjacent complete trading windows. Not an investment recommendation.')
    bydate={r['date']:r for r in history}
    rows=[bydate.get(day-timedelta(days=i)) for i in range(2*period)]
    if any(r is None for r in rows):return result
    if any(r.get('valuation_current') is not True for r in rows):
        result['reason']='Stale or missing equity references prevent classification.'
        return result
    cohort=rows[0].get('cohort')
    if not cohort or any(r.get('cohort')!=cohort for r in rows):return result
    # AUM and holder paths must be observed throughout, not just at endpoints.
    if any(number(r.get(k)) is None for r in rows for k in ('reference_aum_usd','meaningful_holders','daily_swap_volume_usd')):
        result['reason']='Incomplete adoption or market-wide DEX coverage. Observed swap samples cannot classify trading growth.'
        return result
    current,previous=rows[0],rows[period]
    aum,holders=number(previous['reference_aum_usd']),number(previous['meaningful_holders'])
    old_volume=sum(number(r['daily_swap_volume_usd']) for r in rows[period:])
    deltas=[number(r.get('net_supply_change_usd')) for r in rows[:period]]
    if aum<=0 or holders<=0 or old_volume<=0 or any(x is None for x in deltas):
        result['reason']='A positive AUM, holder and trading baseline and every issuance delta are required.'
        return result
    issuance=sum(deltas)/aum*100
    aum_growth=(number(current['reference_aum_usd'])/aum-1)*100
    holder_growth=(number(current['meaningful_holders'])/holders-1)*100
    volume_growth=(sum(number(r['daily_swap_volume_usd']) for r in rows[:period])/old_volume-1)*100
    result.update(issuance_aum_pct=issuance,aum_growth_pct=aum_growth,holder_growth_pct=holder_growth,trading_growth_pct=volume_growth)
    high_issuance=issuance>issuance_threshold
    high_trading=volume_growth>trading_threshold
    broad_growth=aum_growth>growth_threshold and holder_growth>growth_threshold
    if high_issuance and broad_growth:
        result.update(state='Expansion' if high_trading else 'Accumulation',reason='Net issuance, AUM and meaningful holders exceed configured growth thresholds; trading determines the quadrant.')
    elif not high_issuance and aum_growth<=growth_threshold and holder_growth<=growth_threshold:
        result.update(state='Churn' if high_trading else 'Stagnant',reason='Issuance, AUM and holders do not exceed adoption thresholds; trading determines the quadrant. Stagnant can include contraction.')
    else:
        result.update(state='Mixed',reason='Adoption measures disagree. No quadrant is forced when issuance, AUM and holder breadth diverge.')
    return result


def capture_research(conn,day,env):
    from .collector import fetch_all,insert_many
    # Mirror only explicitly approved securities; BP is excluded by asset type.
    with conn,conn.cursor() as cur:
        cur.execute('''INSERT INTO tokenized_security_assets(issuer_id,network,mint,token_symbol,underlying_symbol,
            underlying_exchange,underlying_name,asset_type,official_source,verified_at,verification_status,backpack_asset_id,active)
            SELECT 'backpack','solana',solana_mint,token_symbol,underlying_symbol,underlying_exchange,underlying_name,
                asset_type,official_source,source_verified_at,verification_status,id,active
            FROM backpack_assets WHERE asset_type<>'bp' AND issuer='Backpack' AND verification_status<>'pending'
            ON CONFLICT(network,mint) DO UPDATE SET active=excluded.active
            WHERE tokenized_security_assets.backpack_asset_id=excluded.backpack_asset_id''')
        cur.execute('''INSERT INTO tokenized_security_daily_snapshots(asset_id,date,reference_aum_usd,net_issuance_usd,
            meaningful_holders,daily_swap_volume_usd,source,observed_at,methodology)
            SELECT a.id,s.date,s.reference_aum_usd,s.net_supply_change_usd,s.holders_over_100,s.daily_swap_volume_usd,
                s.source,s.captured_at,'backpack-v1: reference supply x underlying price; asset holder counts must not be summed across assets'
            FROM backpack_asset_daily_snapshots s JOIN tokenized_security_assets a ON a.backpack_asset_id=s.asset_id
            WHERE s.date=%s ON CONFLICT DO NOTHING''',(day,))
    history=fetch_all(conn,'SELECT * FROM backpack_ecosystem_daily_snapshots WHERE date BETWEEN %s AND %s',(day-timedelta(days=59),day))
    captured=fetch_all(conn,"SELECT s.date,s.asset_id,s.underlying_price_timestamp,s.captured_at FROM backpack_asset_daily_snapshots s JOIN backpack_assets a ON a.id=s.asset_id WHERE s.date BETWEEN %s AND %s AND a.asset_type<>'bp'",(day-timedelta(days=59),day))
    cohorts={}
    for r in captured:cohorts.setdefault(r['date'],set()).add(r['asset_id'])
    for r in history:
        r['cohort']=cohorts.get(r['date']) if len(cohorts.get(r['date'],()))==r['assets_expected'] else None
        components=[s for s in captured if s['date']==r['date']]
        r['valuation_current']=bool(components) and all(s['underlying_price_timestamp'] is not None and timedelta(0)<=s['captured_at']-s['underlying_price_timestamp']<=timedelta(days=4) for s in components)
    if not any(r['date']==day for r in history):return
    settings=[Decimal(env.get(key,default)) for key,default in [('BACKPACK_ISSUANCE_GROWTH_PCT','1'),('BACKPACK_TRADING_GROWTH_PCT','20'),('BACKPACK_ADOPTION_GROWTH_PCT','1')]]
    records=[classify(history,day,p,*settings) for p in (7,30)]
    with conn,conn.cursor() as cur:insert_many(cur,'backpack_environment_daily',records)
