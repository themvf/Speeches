"""Disposable Postgres only. Set BACKPACK_TEST_DATABASE_URL explicitly to run."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
import os
import uuid
import pytest
from backpack.collector import setup, run, fetch_all
from backpack.metrics import BP_MINT
from backpack.providers import SourceError

URL=os.environ.get('BACKPACK_TEST_DATABASE_URL')
pytestmark=pytest.mark.skipif(not URL,reason='Requires disposable BACKPACK_TEST_DATABASE_URL')

class FakeProviders:
    def __init__(self):
        self.env={}
        self.usage=defaultdict(int)
        self.failed=set()
    def supply(self,mint,independent=False):
        if mint in self.failed: raise SourceError('Simulated isolated asset failure')
        return D(1000),6,100
    def rpc(self,method,params): return int(datetime.now(timezone.utc).timestamp())
    def equity(self,symbol):return D(20),datetime.now(timezone.utc),D(19)
    def price(self,mint):return D(21),datetime.now(timezone.utc)
    def holders(self,mint):return [{'address':mint+'account','owner':'shared-wallet','amount':1000000000}],100,101
    def validation_market(self,mint):return {'priceUsd':'21','volume':{'h24':0}}

@pytest.fixture(scope="session")
def database():
    import psycopg2
    connection=psycopg2.connect(URL)
    yield connection
    connection.close()

@pytest.fixture
def conn(database):
    connection=database
    schema='backpack_test_'+uuid.uuid4().hex
    with connection,connection.cursor() as cur:
        cur.execute('CREATE SCHEMA '+schema)
        cur.execute('SET search_path TO '+schema)
    setup(connection)
    try:yield connection
    finally:
        connection.rollback()
        with connection,connection.cursor() as cur:
            cur.execute('DROP SCHEMA '+schema+' CASCADE')


def asset(conn,mint,symbol):
    with conn,conn.cursor() as cur:
        cur.execute('''INSERT INTO backpack_assets(token_symbol,token_name,solana_mint,underlying_symbol,underlying_exchange,
            underlying_name,asset_type,issuer,official_source,source_verified_at,verification_status)
            VALUES(%s,%s,%s,%s,'XNYS',%s,'common_stock','Backpack','https://example.test/official',now(),'manual_approved') RETURNING id''',
            (symbol,symbol,mint,symbol,symbol))
        return cur.fetchone()[0]

def test_capture_retry_immutability_reconciliation_and_partial_failure(conn):
    a=asset(conn,'mint-a','A');b=asset(conn,'mint-b','B')
    p=FakeProviders();p.failed.add('mint-b')
    result=run(conn,p)
    assert result['failed']==1 and result['succeeded']==2
    assert len(fetch_all(conn,'SELECT * FROM backpack_ecosystem_daily_snapshots'))==0
    snap=fetch_all(conn,'SELECT * FROM backpack_asset_daily_snapshots WHERE asset_id=%s',(a,))[0]
    assert snap['reference_aum_usd']==20000 and snap['holders_over_100']==1
    assert snap['net_supply_change_tokens'] is None
    captured=snap['captured_at']
    p.failed.clear()
    result=run(conn,p)
    assert result['skipped']==2 and result['succeeded']==1
    total=fetch_all(conn,'SELECT * FROM backpack_ecosystem_daily_snapshots')[0]
    assert total['reference_aum_usd']==40000
    assert total['meaningful_holders']==1 and total['multi_asset_2']==1
    assert total['daily_swap_volume_usd'] is None
    assert fetch_all(conn,'SELECT captured_at FROM backpack_asset_daily_snapshots WHERE asset_id=%s',(a,))[0]['captured_at']==captured
    result=run(conn,p)
    assert result['skipped']==3
    assert len(fetch_all(conn,'SELECT * FROM backpack_asset_daily_snapshots'))==3
    assert len(fetch_all(conn,'SELECT * FROM backpack_ecosystem_daily_snapshots'))==1
    assert len(fetch_all(conn,'SELECT * FROM backpack_data_quality_events'))>20
    assert fetch_all(conn,'SELECT staked_bp,estimated_circulating_supply FROM backpack_bp_daily_snapshots')[0]=={'staked_bp':None,'estimated_circulating_supply':None}

def test_yesterday_supply_generates_today_issuance(conn):
    a=asset(conn,'mint-a','A')
    yesterday=datetime.now(timezone.utc).date()-timedelta(days=1)
    with conn,conn.cursor() as cur:
        run_id=str(uuid.uuid4())
        cur.execute('INSERT INTO backpack_ingestion_runs(run_id,snapshot_date) VALUES(%s,%s)',(run_id,yesterday))
        cur.execute('''INSERT INTO backpack_asset_daily_snapshots(asset_id,date,run_id,captured_at,slot,source,token_supply,decimals,data_quality_score,quality_status)
            VALUES(%s,%s,%s,now()-interval '1 day',50,'test',900,6,20,'Partial')''',(a,yesterday,run_id))
    run(conn,FakeProviders())
    row=fetch_all(conn,'SELECT * FROM backpack_asset_daily_snapshots WHERE asset_id=%s ORDER BY date DESC',(a,))[0]
    assert row['net_supply_change_tokens']==100 and row['net_supply_change_usd']==2000

def test_pending_registry_not_collected(conn):
    a=asset(conn,'mint-pending','P')
    with conn,conn.cursor() as cur:cur.execute("UPDATE backpack_assets SET verification_status='pending' WHERE id=%s",(a,))
    result=run(conn,FakeProviders())
    assert result['attempted']==1 # BP only
    assert not fetch_all(conn,'SELECT * FROM backpack_asset_daily_snapshots WHERE asset_id=%s',(a,))

def test_live_rpc_cannot_manufacture_history(conn):
    with pytest.raises(ValueError,match='backfill'):run(conn,FakeProviders(),datetime.now(timezone.utc).date()-timedelta(days=2))

def test_active_lease_blocks_second_worker(conn):
    with conn,conn.cursor() as cur:
        cur.execute("INSERT INTO backpack_job_leases VALUES('daily',%s,now()+interval '10 minutes')",(str(uuid.uuid4()),))
    assert run(conn,FakeProviders())['status']=='already_running'


def test_label_evidence_is_immutable_and_whale_cohorts_are_persisted(conn):
    with conn,conn.cursor() as cur:
        cur.execute("""INSERT INTO backpack_wallet_labels VALUES
            ('shared-wallet','Treasury','Verified issuer','high','https://example.test/treasury',now(),'Exact address evidence')""")
    p=FakeProviders();p.env['BACKPACK_WHALE_THRESHOLDS_USD']='500000,1000000,2000000'
    run(conn,p)
    labels=fetch_all(conn,'SELECT label,label_entity,label_confidence,label_source,excluded FROM backpack_asset_holder_daily_snapshots')
    assert labels==[dict(label='Treasury',label_entity='Verified issuer',label_confidence='high',label_source='https://example.test/treasury',excluded=True)]
    cohorts=fetch_all(conn,'SELECT * FROM backpack_bp_whale_daily_snapshots ORDER BY threshold_usd')
    assert [r['threshold_usd'] for r in cohorts]==[100000,500000,1000000,2000000]
    assert all(r['whale_count']==0 and r['new_whales'] is None for r in cohorts)
    with conn,conn.cursor() as cur:
        cur.execute("UPDATE backpack_wallet_labels SET label='Unknown',confidence='low',source='https://example.test/revoked'")
    run(conn,p)
    assert fetch_all(conn,'SELECT label,label_entity,label_confidence,label_source,excluded FROM backpack_asset_holder_daily_snapshots')==labels
    assert len(fetch_all(conn,'SELECT * FROM backpack_bp_whale_daily_snapshots'))==4
    setup(conn) # additive migration rerun preserves historical evidence
    assert fetch_all(conn,'SELECT label,label_entity,label_confidence,label_source,excluded FROM backpack_asset_holder_daily_snapshots')==labels


def test_readiness_missing_credentials_persists_without_fake_capture(conn):
    from backpack.readiness import preflight, audit
    p=FakeProviders()
    result=preflight(conn,p)
    checks={r['check_name']:r for r in result['checks']}
    assert not result['ready_for_security_capture']
    assert checks['helius_das']['status']=='Unavailable'
    assert checks['starter_universe']['status']=='Unavailable'
    assert len(fetch_all(conn,'SELECT * FROM backpack_readiness_checks'))==8
    assert not fetch_all(conn,'SELECT * FROM backpack_asset_daily_snapshots')
    assert audit(conn)['assets']==[]
    assert fetch_all(conn,"SELECT verification_status FROM backpack_assets WHERE asset_type='bp'")[0]['verification_status']=='official'


def test_audit_reproduces_aum_but_does_not_claim_signoff(conn):
    from backpack.readiness import audit
    asset(conn,'mint-a','A')
    run(conn,FakeProviders())
    result=audit(conn)
    row=next(r for r in result['assets'] if r['symbol']=='A')
    assert row['stored_reference_aum']==row['reproduced_reference_aum']==20000
    assert row['holder_supply_difference']==0
    assert row['total_swap_volume'] is None
    assert result['manual_signoff'].startswith('Required')


def test_starter_registry_revalidates_exact_mints_and_isolates_failures(conn):
    import json
    from pathlib import Path
    from backpack.registry import seed_starter
    manifest=json.loads(Path('backpack/starter_universe.json').read_text())['assets']
    primary=[dict(symbol=r['token_symbol'],tokens=[dict(blockchain='Solana',contractAddress=r['solana_mint'])]) for r in manifest]
    primary[0]['tokens'][0]['contractAddress']='WRONG_MINT'
    securities=[dict(asset=r['token_symbol'],name=r['underlying_name'],cusip=r['cusip']) for r in manifest]
    p=FakeProviders()
    p.request=lambda provider,method,url: primary if url.endswith('/assets') else securities
    result=seed_starter(conn,p)
    assert result['failed']==1
    registered=fetch_all(conn,"SELECT * FROM backpack_assets WHERE asset_type<>'bp'")
    assert len(registered)==13
    assert all(r['verification_status']=='official' and r['launch_date'] is None for r in registered)
    assert not fetch_all(conn,'SELECT * FROM backpack_asset_daily_snapshots')
    assert seed_starter(conn,p)['failed']==1
    assert len(fetch_all(conn,"SELECT * FROM backpack_assets WHERE asset_type<>'bp'"))==13


def test_dashboard_sql_templates_execute_on_empty_and_populated_schema(conn):
    """Exercise the actual web reader SQL; duplicate date output aliases must not break sorting."""
    import re
    from pathlib import Path
    source=Path('apps/web/lib/server/backpack-store.ts').read_text()
    templates=re.findall(r'sql`([^`]+)`',source)
    assert len(templates)>=10
    def execute(day,asset_id):
        values={'day':day,'assetId??null':asset_id,'assetId':asset_id}
        for template in templates:
            params=[]
            def bind(match):
                params.append(values[match.group(1)])
                return '%s'
            query=re.sub(r'\$\{([^}]+)\}',bind,template)
            fetch_all(conn,query,tuple(params))
    execute(None,None)
    a=asset(conn,'mint-a','A')
    run(conn,FakeProviders())
    execute(datetime.now(timezone.utc).date(),a)


def test_current_holders_baseline_events_and_safe_retention(conn):
    from backpack.storage import persist_holders, maintain, cost_report
    a=asset(conn,'mint-state','STATE')
    run(conn,FakeProviders())
    today=datetime.now(timezone.utc).date()
    assert not fetch_all(conn,'SELECT * FROM backpack_holder_events')  # baseline is not arrival history
    assert fetch_all(conn,'SELECT balance_tokens FROM backpack_current_holders WHERE asset_id=%s',(a,))[0]['balance_tokens']==1000
    assert fetch_all(conn,'SELECT aggregates_validated FROM backpack_holder_checkpoints WHERE asset_id=%s',(a,))[0]['aggregates_validated']
    # A later complete enumeration replaces current state atomically and preserves changes.
    tomorrow=today+timedelta(days=1)
    from backpack.collector import insert
    baseline=fetch_all(conn,'SELECT * FROM backpack_asset_daily_snapshots WHERE asset_id=%s',(a,))[0]
    with conn,conn.cursor() as cur:
        insert(cur,'backpack_asset_daily_snapshots',dict(baseline,date=tomorrow))
        persist_holders(cur,a,tomorrow,[dict(wallet_address='new-wallet',balance_tokens=D(1000),value_usd=D(20000),excluded=False,label='Unknown')],{})
    assert [r['wallet_address'] for r in fetch_all(conn,'SELECT * FROM backpack_current_holders WHERE asset_id=%s',(a,))]==['new-wallet']
    events=fetch_all(conn,'SELECT * FROM backpack_holder_events WHERE asset_id=%s',(a,))
    assert {'NEW_HOLDER','EXITED_HOLDER'} <= {e['event_type'] for e in events}
    with conn,conn.cursor() as cur:
        # Unknown legacy/raw day has no attestation and must survive cleanup.
        old=today-timedelta(days=60)
        insert(cur,'backpack_asset_daily_snapshots',dict(baseline,date=old,holders_complete=False))
        cur.execute("INSERT INTO backpack_asset_holder_daily_snapshots(asset_id,date,wallet_address,balance_tokens,excluded,source,label,slot) VALUES(%s,%s,'unvalidated',1,false,'fixture','Unknown',1)",(a,old))
    maintain(conn,today+timedelta(days=40),{})
    raw=fetch_all(conn,'SELECT * FROM backpack_asset_holder_daily_snapshots WHERE asset_id=%s',(a,))
    assert len(raw)==1 and raw[0]['wallet_address']=='unvalidated'
    assert len(fetch_all(conn,'SELECT * FROM backpack_asset_daily_snapshots WHERE asset_id=%s',(a,)))==3
    assert len(fetch_all(conn,'SELECT * FROM backpack_holder_events WHERE asset_id=%s',(a,)))==len(events)
    assert cost_report(conn)['projected_monthly_cost_usd'] is None


def test_raw_swap_retention_requires_attestation_and_preserves_pinned_evidence(conn):
    from backpack.storage import maintain
    a=asset(conn,'mint-retention','RET')
    today=datetime.now(timezone.utc).date()
    old=today-timedelta(days=60)
    run(conn,FakeProviders())
    from backpack.collector import insert
    baseline=fetch_all(conn,'SELECT * FROM backpack_asset_daily_snapshots WHERE asset_id=%s',(a,))[0]
    with conn,conn.cursor() as cur:
        insert(cur,'backpack_asset_daily_snapshots',dict(baseline,date=old+timedelta(days=1),daily_swap_volume_usd=100,unique_traders=2))
        insert(cur,'backpack_asset_dex_daily_snapshots',dict(asset_id=a,date=old+timedelta(days=1),venue='All',trades=2,median_trade_size=50,average_trade_size=50,p95_trade_size=50,max_trade_size=50,coverage_status='Verified',source='fixture'))
        for signature in ('ordinary','anomaly','unvalidated'):
            at=old if signature!='unvalidated' else old-timedelta(days=1)
            cur.execute("INSERT INTO backpack_transactions(asset_id,signature,event_kind,timestamp,source,slot) VALUES(%s,%s,'swap',%s,'fixture',1)",(a,signature,datetime.combine(at,datetime.min.time(),timezone.utc)))
        cur.execute("INSERT INTO backpack_transaction_retention_checks VALUES(%s,%s,true,'fixture validates permanent aggregate',now())",(a,old))
        cur.execute("INSERT INTO backpack_transaction_evidence(asset_id,signature,event_kind,reason,source) VALUES(%s,'anomaly','swap','material event','review')",(a,))
    maintain(conn,today,{})
    assert {r['signature'] for r in fetch_all(conn,'SELECT signature FROM backpack_transactions')}=={'anomaly','unvalidated'}
    maintain(conn,today,{})
    assert len(fetch_all(conn,'SELECT * FROM backpack_transaction_evidence'))==1


def test_precomputed_metrics_preserve_unknowns_and_reconcile_composition(conn):
    a=asset(conn,'mint-analytics','ANALYTICS')
    result=run(conn,FakeProviders())
    assert result['status']=='completed'
    rows=fetch_all(conn,'SELECT * FROM backpack_analytical_daily_metrics')
    assert next(r['value'] for r in rows if r['scope_asset_id']==0 and r['metric']=='top_1_aum_pct')==100
    assert all(r['value'] is None for r in rows if r['metric']=='net_issuance_usd')
    assert next(r['value'] for r in rows if r['scope_asset_id']==0 and r['metric']=='long_tail_aum_usd')==0
    count=len(rows)
    run(conn,FakeProviders())
    assert len(fetch_all(conn,'SELECT * FROM backpack_analytical_daily_metrics'))==count


def test_interrupted_asset_transaction_rolls_back_and_retry_preserves_successes(conn,monkeypatch):
    import backpack.collector as collector
    a=asset(conn,'mint-interrupt','INTERRUPT')
    original=collector.persist_holders
    def interrupted(cur,asset_id,*args):
        original(cur,asset_id,*args)
        if asset_id==a:raise RuntimeError('Simulated interrupted commit')
    monkeypatch.setattr(collector,'persist_holders',interrupted)
    result=run(conn,FakeProviders())
    assert result['failed']==1 and result['succeeded']==1
    for table in ('backpack_asset_daily_snapshots','backpack_current_holders','backpack_holder_checkpoints','backpack_asset_holder_daily_snapshots'):
        assert not fetch_all(conn,'SELECT * FROM '+table+' WHERE asset_id=%s',(a,))
    monkeypatch.setattr(collector,'persist_holders',original)
    retried=run(conn,FakeProviders())
    assert retried['succeeded']==1 and retried['skipped']==1
    assert len(fetch_all(conn,'SELECT * FROM backpack_current_holders WHERE asset_id=%s',(a,)))==1


def test_incomplete_enumeration_preserves_prior_current_state_and_withholds_analytics(conn):
    from backpack.collector import collect_asset,calendar_for
    a=asset(conn,'mint-incomplete','INCOMPLETE')
    today=datetime.now(timezone.utc).date();yesterday=today-timedelta(days=1)
    old_run=str(uuid.uuid4())
    with conn,conn.cursor() as cur:cur.execute('INSERT INTO backpack_ingestion_runs(run_id,snapshot_date) VALUES(%s,%s)',(old_run,yesterday))
    registry=fetch_all(conn,'SELECT * FROM backpack_assets WHERE id=%s',(a,))[0]
    collect_asset(conn,FakeProviders(),registry,old_run,yesterday,{},calendar_for(yesterday))
    p=FakeProviders();original=p.holders
    p.holders=lambda mint:([{'address':'incomplete','owner':'wrong-owner','amount':1}],100,101) if mint=='mint-incomplete' else original(mint)
    run(conn,p)
    state=fetch_all(conn,'SELECT * FROM backpack_current_holders WHERE asset_id=%s',(a,))
    assert len(state)==1 and state[0]['wallet_address']=='shared-wallet' and state[0]['last_seen_at']==yesterday
    snap=fetch_all(conn,'SELECT * FROM backpack_asset_daily_snapshots WHERE asset_id=%s AND date=%s',(a,today))[0]
    assert snap['holders_complete'] is False and snap['holders_over_100'] is None and snap['new_holders'] is None
    assert not fetch_all(conn,'SELECT * FROM backpack_holder_events WHERE asset_id=%s',(a,))


def test_competitor_mirror_excludes_bp_pending_and_preserves_daily_records(conn):
    a=asset(conn,'mint-neutral','NEUTRAL');b=asset(conn,'mint-pending','PENDING')
    with conn,conn.cursor() as cur:cur.execute("UPDATE backpack_assets SET verification_status='pending' WHERE id=%s",(b,))
    run(conn,FakeProviders())
    assert len(fetch_all(conn,'SELECT * FROM tokenized_security_assets'))==1
    rows=fetch_all(conn,'SELECT * FROM tokenized_security_daily_snapshots')
    assert len(rows)==1 and rows[0]['reference_aum_usd']==20000 and rows[0]['daily_swap_volume_usd'] is None
    assert not fetch_all(conn,'SELECT * FROM tokenized_security_market_snapshots')
    assert {r['state'] for r in fetch_all(conn,'SELECT * FROM backpack_environment_daily')}=={'Unavailable'}
    run(conn,FakeProviders())
    assert fetch_all(conn,'SELECT * FROM tokenized_security_daily_snapshots')==rows


def test_billing_import_is_idempotent_and_conflict_rolls_back_whole_export(conn,tmp_path):
    from backpack.cost_review import import_billing
    p=tmp_path/'billing.csv';header='date,provider,scope,metric,value,unit,source\n'
    first='2026-01-01,Neon,backpack,cost_usd,0.01,USD,https://example.test/invoice\n'
    p.write_text(header+first);import_billing(conn,p);import_billing(conn,p)
    p.write_text(header+first.replace('Neon','Vercel')+first.replace('0.01','0.02'))
    with pytest.raises(ValueError):import_billing(conn,p)
    rows=fetch_all(conn,'SELECT * FROM backpack_billing_observations')
    assert len(rows)==1 and rows[0]['value']==D('0.01')


def test_operations_sql_executes_without_provider_calls(conn):
    from pathlib import Path
    import re
    source=Path('apps/web/lib/server/backpack-operations.ts').read_text()
    for stage in range(2):
        for query in re.findall(r'sql`([^`]+)`',source):fetch_all(conn,query)
        if stage==0:run(conn,FakeProviders())
