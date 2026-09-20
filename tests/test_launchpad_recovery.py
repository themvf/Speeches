from datetime import timedelta
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import launchpad_archive as archive
from launchpad_chains import SOLANA, ROBINHOOD
from test_launchpad_archive import (db, NOW, TOKEN, SOL_TOKEN, SOL_DEEP, SOL_OUTSIDE,
                                   pool, sol_pool, responder, sol_responder, Response)


def seed_sweep(db,network,at,newest,complete=True):
    with db,db.cursor() as cur:
        cur.execute('''INSERT INTO launchpad_sweeps
                       (network,started_at,newest_pool_at,complete,gap_seconds)
                       VALUES (%s,%s,%s,%s,0)''',(network,at,newest,complete))


def seed_graduate(db,token=SOL_TOKEN,sampled=True,age=60,measure=None):
    with db,db.cursor() as cur:
        cur.execute('''INSERT INTO launchpad_tokens
                       (network,token_address,dex,first_seen_at,last_seen_at,graduated,graduated_at,
                        cohort_sampled,measure_pool,measure_pool_reason)
                       VALUES ('solana',%s,'pumpswap',%s,%s,true,%s,%s,%s,%s)''',
                    (token,NOW,NOW,NOW-timedelta(minutes=age),sampled,measure,
                     'deepest graduate pool' if measure else None))


def test_watermark_uses_latest_same_chain_even_if_incomplete(db):
    seed_sweep(db,'robinhood',NOW-timedelta(hours=20),NOW-timedelta(hours=20))
    seed_sweep(db,'robinhood',NOW-timedelta(minutes=5),NOW-timedelta(minutes=5),False)
    seed_sweep(db,'solana',NOW-timedelta(minutes=1),NOW)
    seed_sweep(db,None,NOW,NOW)
    calls=[]
    def fetch(url,**kwargs):
        calls.append(url)
        return responder({1:[pool(created=(NOW-timedelta(minutes=6)).isoformat())]})(url,**kwargs)
    out=archive.sweep(db,fetch=fetch,now=NOW,wait=lambda _:None)
    assert out['gap_seconds']==0 and out['pages']==1
    assert not any('page=2' in url for url in calls)
    with db,db.cursor() as cur:
        cur.execute('SELECT network FROM launchpad_sweeps ORDER BY id DESC LIMIT 1')
        assert cur.fetchone()[0]=='robinhood'


def test_other_chain_cannot_hide_gap_or_supply_first_watermark(db):
    seed_sweep(db,'solana',NOW,NOW)
    out=archive.sweep(db,fetch=responder({1:[pool()]}),now=NOW,wait=lambda _:None)
    assert out['gap_seconds'] is None
    later=NOW+timedelta(minutes=30)
    out=archive.sweep(db,fetch=responder({1:[pool(created=later.isoformat())]}),
                      now=later,wait=lambda _:None)
    assert out['gap_seconds']>0
    # A real gap is retained, but it doesn't freeze the next watermark.
    out=archive.sweep(db,fetch=responder({1:[pool(created=later.isoformat())]}),
                      now=later+timedelta(minutes=5),wait=lambda _:None)
    assert out['gap_seconds']==0 and out['pages']==1


def test_report_and_daily_do_not_blend_chains_or_guess_legacy(db):
    for minute in range(0,60,2):
        seed_sweep(db,'solana',NOW-timedelta(minutes=minute),NOW)
    seed_sweep(db,None,NOW,NOW)
    seed_graduate(db)
    robin=archive.report(db,now=NOW,hours=1)
    sol=archive.report(db,now=NOW,hours=1,chain=SOLANA)
    assert robin['sweeps']['recorded']==0 and not robin['continuous']
    assert robin['tokens']['discovered']==0 and robin['detection_lag_seconds']['measured']==0
    assert robin['daily']==[]
    assert sol['sweeps']['recorded']==30
    assert sol['unattributed_legacy_sweeps']['count']==1 and not sol['continuous']
    assert sol['daily'][0]['sweeps']==30


def test_report_detects_trailing_outage_even_with_enough_sweeps(db):
    for minute in range(30,60):
        seed_sweep(db,'solana',NOW-timedelta(minutes=minute),NOW)
    out=archive.report(db,now=NOW,hours=1,chain=SOLANA)
    assert out['sweeps']['recorded']==out['sweeps']['expected']
    assert out['sweeps']['trailing_silence_seconds']==1800 and not out['continuous']


def test_schema_pending_is_read_only_and_search_path_aware(db):
    with db,db.cursor() as cur:
        cur.execute('ALTER TABLE launchpad_sweeps DROP COLUMN network CASCADE')
    db.set_session(readonly=True)
    assert not archive.schema_ready(db)
    db.set_session(readonly=False)
    archive.setup(db)
    db.set_session(readonly=True)
    assert archive.schema_ready(db)
    archive.report(db,now=NOW,hours=1)


def test_retry_cannot_spend_capture_reserve(monkeypatch):
    clock=[0.0];calls=[]
    monkeypatch.setattr(archive.time,'monotonic',lambda:clock[0])
    def wait(seconds):clock[0]+=seconds
    def fetch(url,**kwargs):
        calls.append(kwargs['timeout'])
        response=Response({},429);response.headers={'Retry-After':'30'}
        return response
    get=archive.bounded_get(fetch,wait,2,lambda:12)
    with pytest.raises(ValueError,match='retry exceeds phase budget'):get('test')
    assert clock[0]==2 and calls==[10]
    clock[0]=12
    with pytest.raises(ValueError,match='budget exhausted'):get('test')
    assert len(calls)==1


def test_worker_does_not_fetch_pools_outside_cohort(db):
    seed_graduate(db,SOL_OUTSIDE,sampled=False)
    calls=[]
    def fetch(url,**_):
        calls.append(url)
        assert url.endswith('/info')
        return Response({'data':{'attributes':{}}})
    out=archive.enrich(db,fetch=fetch,now=NOW,wait=lambda _:None)
    assert out['processed_this_run']==1 and len(calls)==1
    with db,db.cursor() as cur:
        cur.execute('SELECT measure_pool_reason FROM launchpad_tokens')
        assert cur.fetchone()[0]=='outside ladder cohort'


def test_failed_pool_lookup_is_not_recorded_as_no_pools(db):
    seed_graduate(db)
    def failing(url,**_):
        return Response({},500) if url.endswith('/pools') else Response({'data':{'attributes':{}}})
    archive.enrich(db,fetch=failing,now=NOW,wait=lambda _:None)
    with db,db.cursor() as cur:
        cur.execute('SELECT measure_pool_reason FROM launchpad_tokens')
        assert cur.fetchone()[0] is None
    good={'data':[sol_pool(dex='pumpswap',address=SOL_DEEP)]}
    archive.enrich(db,fetch=sol_responder({},pool_list=good,
                   pool_payload=sol_pool(dex='pumpswap',address=SOL_DEEP)),now=NOW,wait=lambda _:None)
    with db,db.cursor() as cur:
        cur.execute('SELECT measure_pool,measure_pool_timing FROM launchpad_tokens')
        assert cur.fetchone()==(SOL_DEEP,'late')


def test_worker_reserves_time_and_batches_ladder(db,monkeypatch):
    seed_graduate(db,'first',measure=SOL_DEEP)
    seed_graduate(db,'second',measure='OtherPool')
    # Metadata consumes 30 seconds each. Its 33-second allocation cannot eat the
    # ladder's remaining 18 seconds, even while there is more metadata queued.
    clock=[0.0];urls=[]
    monkeypatch.setattr(archive.time,'monotonic',lambda:clock[0])
    def fetch(url,**kwargs):
        urls.append(url)
        if url.endswith('/info'):
            clock[0]+=min(30,kwargs['timeout'])
            if kwargs['timeout']<30:
                import requests
                raise requests.Timeout('simulated slow response')
            return Response({'data':{'attributes':{}}})
        assert '/pools/multi/' in url
        return Response({'data':[sol_pool(dex='pumpswap',address=SOL_DEEP),
                                 sol_pool(dex='pumpswap',address='OtherPool')]})
    out=archive.enrich(db,replace(SOLANA,enrich_minutes=1),fetch=fetch,now=NOW,wait=lambda _:None)
    assert out['rungs_filled']==2
    assert sum('/pools/multi/' in u for u in urls)==1
    coverage=archive.ladder_health(db,SOLANA,NOW+timedelta(minutes=2),NOW-timedelta(hours=2))
    assert coverage[0]['eligible']==2 and coverage[0]['filled']==2
    assert coverage[0]['median_lateness_seconds']>=3300


def test_ladder_denominators_exclude_future_rungs_and_other_chains(db):
    seed_graduate(db,age=7)
    rows=archive.ladder_health(db,SOLANA,NOW,NOW-timedelta(days=1))
    assert len(rows)==1 and rows[0]['rung_minutes']==5
    assert rows[0]['missing']==1 and rows[0]['without_pool']==1
    assert archive.ladder_health(db,ROBINHOOD,NOW,NOW-timedelta(days=1))==[]


def test_report_workflow_script_runs_all_six_read_only_sections(db,tmp_path):
    # The fixture schema is isolated; propagate it to child connections, force
    # PostgreSQL read-only mode, and execute the actual production script.
    env=dict(os.environ,DATABASE_URL=os.environ['CRYPTO_SOCIAL_TEST_DATABASE_URL'],
             PGOPTIONS='-c search_path=launchpad_test -c default_transaction_read_only=on')
    result=subprocess.run([sys.executable,'scripts/graduation_archive_report.py','--output',str(tmp_path)],
                           env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert len(list(tmp_path.glob('*.json')))==6
    payload=json.loads((tmp_path/'solana-report.json').read_text())
    assert payload['network']=='solana' and payload['continuous'] is False
