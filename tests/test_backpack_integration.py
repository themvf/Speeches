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
