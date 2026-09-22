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
