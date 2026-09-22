"""Provider-free, explicit billing imports and reproducible 7/30-day cost reviews."""
import csv
from datetime import date,timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

UNITS={'cost_usd':'USD','compute_cu_hours':'CU-hours','egress_bytes':'bytes',
       'api_requests':'requests','function_invocations':'invocations','active_cpu_seconds':'seconds',
       'cache_hit_pct':'percent','average_payload_bytes':'bytes','credits':'credits'}
PROVIDERS={'Neon','Vercel','Helius','Jupiter','Alpaca'}


def parse_billing(path):
    path=Path(path)
    if path.stat().st_size>5_000_000:raise ValueError('Billing file exceeds 5 MB')
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if set(reader.fieldnames or [])!={'date','provider','scope','metric','value','unit','source'}:raise ValueError('Invalid billing columns')
        records=[];keys=set()
        for row in reader:
            day=date.fromisoformat(row['date']);value=Decimal(row['value']);url=urlparse(row['source'])
            if day>date.today() or row['provider'] not in PROVIDERS or row['scope'] not in ('shared','backpack'):
                raise ValueError('Invalid billing date, provider or scope')
            if row['metric'] not in UNITS or row['unit']!=UNITS[row['metric']] or not value.is_finite() or value<0:
                raise ValueError('Invalid billing metric, value or unit')
            if row['metric']=='cache_hit_pct' and value>100:raise ValueError('Invalid percentage')
            if url.scheme!='https' or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError('Source must be a credential-free HTTPS evidence reference')
            key=(day,row['provider'],row['scope'],row['metric'])
            if key in keys:raise ValueError('Duplicate billing observation')
            keys.add(key);records.append(dict(row,date=day,value=value))
    if not records:raise ValueError('Empty billing file')
    return records


def import_billing(conn,path):
    records=parse_billing(path)
    # All-or-nothing import; repeat identical exports are harmless, conflicting evidence is rejected.
    with conn,conn.cursor() as cur:
        for r in records:
            key=(r['date'],r['provider'],r['scope'],r['metric'])
            cur.execute('''INSERT INTO backpack_billing_observations(date,provider,scope,metric,value,unit,source)
                VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING''',(*key,r['value'],r['unit'],r['source']))
            cur.execute('SELECT value,unit,source FROM backpack_billing_observations WHERE date=%s AND provider=%s AND scope=%s AND metric=%s',key)
            if cur.fetchone()!=(r['value'],r['unit'],r['source']):raise ValueError('Conflicting immutable billing observation')
    return {'billing_observations_imported_or_confirmed':len(records)}


def review_window(storage,billing,days):
    if days not in (7,30):raise ValueError('Use a 7 or 30 day review')
    totals={}
    for r in storage:totals[r['date']]=totals.get(r['date'],0)+int(r['total_bytes'])
    end=max(totals) if totals else None
    result=dict(period_days=days,as_of=end,status='Unavailable',storage_bytes=None,storage_growth_bytes=None,
        projected_storage_bytes_30d=None,infrastructure_cost_usd=None,projected_30d_infrastructure_cost_usd=None,
        methodology='Storage projection extrapolates allocated-byte growth, not billed storage. Dollar projection requires daily Backpack-attributed Neon AND Vercel cost exports; shared-account charges excluded.')
    if end is None:return result
    result['storage_bytes']=totals[end]
    if all(end-timedelta(days=i) in totals for i in range(days+1)):
        growth=totals[end]-totals[end-timedelta(days=days)]
        result.update(status='Storage observed; billing incomplete',storage_growth_bytes=growth,
            projected_storage_bytes_30d=max(0,Decimal(totals[end])+Decimal(growth)*30/days))
    costs={(r['date'],r['provider']):Decimal(r['value']) for r in billing if r['scope']=='backpack' and r['metric']=='cost_usd' and r['provider'] in ('Neon','Vercel')}
    keys=[(end-timedelta(days=i),provider) for i in range(days) for provider in ('Neon','Vercel')]
    if all(key in costs for key in keys):
        cost=sum(costs[k] for k in keys)
        result.update(infrastructure_cost_usd=cost,projected_30d_infrastructure_cost_usd=cost*30/days,
            status='Observed billing; projection is a constant-usage scenario')
    return result


def cost_review(conn):
    from .collector import fetch_all
    storage=fetch_all(conn,'SELECT date,total_bytes FROM backpack_storage_observations WHERE date>=(SELECT max(date)-30 FROM backpack_storage_observations)')
    billing=fetch_all(conn,'SELECT * FROM backpack_billing_observations ORDER BY date DESC LIMIT 10000')
    return dict(windows=[review_window(storage,billing,n) for n in (7,30)],billing=billing,
        note='Imported observations retain source and attribution. Provider API charges are separate from Neon/Vercel infrastructure cost. Zero is valid only if explicitly present in the export.')
