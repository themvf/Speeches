"""Permanent daily analytical records. Scope 0 means tracked security ecosystem, excluding BP."""
from datetime import timedelta
from decimal import Decimal


def precompute(conn,day):
    from .collector import fetch_all,insert_many
    snapshots=fetch_all(conn,'SELECT * FROM backpack_asset_daily_snapshots WHERE date BETWEEN %s AND %s',(day-timedelta(days=90),day))
    ecosystem=fetch_all(conn,'SELECT * FROM backpack_ecosystem_daily_snapshots WHERE date BETWEEN %s AND %s',(day-timedelta(days=90),day))
    security_ids={r['id'] for r in fetch_all(conn,"SELECT id FROM backpack_assets WHERE asset_type<>'bp'")}
    cohorts={}
    for r in snapshots:
        if r['asset_id'] in security_ids:cohorts.setdefault(r['date'],set()).add(r['asset_id'])
    scopes={0:ecosystem}
    for s in snapshots:scopes.setdefault(s['asset_id'],[]).append(s)
    output=[]
    def record(scope,metric,period,value,method):
        output.append(dict(date=day,scope_asset_id=scope,metric=metric,period_days=period,value=value,
            status='Estimated' if value is not None else 'Unavailable',methodology=method))
    for scope,history in scopes.items():
        bydate={r['date']:r for r in history}
        current=bydate.get(day)
        if not current:continue
        for period in (7,30,90):
            days=[bydate.get(day-timedelta(days=i)) for i in range(period)]
            stable=all(r and (scope!=0 or r['assets_expected']==current['assets_expected'] and cohorts.get(r['date'])==cohorts.get(day)) for r in days)
            issuance=sum(r['net_supply_change_usd'] for r in days) if stable and all(r['net_supply_change_usd'] is not None for r in days) else None
            record(scope,'net_issuance_usd',period,issuance,'v1: sum every consecutive daily supply delta valued at its reference price; tracked-universe size must remain unchanged')
            before=bydate.get(day-timedelta(days=period))
            for field in ('reference_aum_usd','meaningful_holders' if scope==0 else 'holders_over_100'):
                valid=before and current.get(field) is not None and before.get(field) and (scope!=0 or before['assets_expected']==current['assets_expected'] and cohorts.get(before['date'])==cohorts.get(day))
                value=(Decimal(current[field])/Decimal(before[field])-1)*100 if valid else None
                record(scope,field+'_growth_pct',period,value,'v1: exact-date percentage change; missing or zero baseline unavailable; registered universe, not full market')
    latest=next((r for r in ecosystem if r['date']==day),None)
    if latest and latest['reference_aum_usd'] is not None:
        securities=fetch_all(conn,"SELECT s.reference_aum_usd FROM backpack_asset_daily_snapshots s JOIN backpack_assets a ON a.id=s.asset_id WHERE s.date=%s AND a.asset_type<>'bp' AND a.active AND a.verification_status<>'pending'",(day,))
        values=sorted([r['reference_aum_usd'] for r in securities if r['reference_aum_usd'] is not None],reverse=True)
        total=latest['reference_aum_usd']
        if len(values)==latest['assets_expected'] and sum(values)==total:
            for n in (1,3,5,10):
                record(0,f'top_{n}_aum_pct',0,sum(values[:n])/total*100 if total else None,'v1: ranked reference AUM / reconciled tracked security AUM; BP excluded')
            record(0,'long_tail_aum_usd',0,sum(values[10:]),'v1: AUM outside top ten registered securities')
    with conn,conn.cursor() as cur:insert_many(cur,'backpack_analytical_daily_metrics',output)
