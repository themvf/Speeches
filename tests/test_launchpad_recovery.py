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
        cur.execute('DELETE FROM launchpad_schema_revision')
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


def test_api_slots_are_shared_across_connections_and_deadline_safe(db):
    import psycopg2
    other=psycopg2.connect(os.environ['CRYPTO_SOCIAL_TEST_DATABASE_URL'],
                          options='-c search_path=launchpad_test')
    try:
        first=archive.reserve_api_slot(db,30)
        second=archive.reserve_api_slot(other,30)
        assert first<1 and second>1.5
        with db,db.cursor() as cur:
            cur.execute("SELECT next_request_at FROM launchpad_api_budget WHERE name='gecko'")
            before=cur.fetchone()[0]
        with pytest.raises(ValueError,match='phase deadline'):
            archive.reserve_api_slot(other,1)
        with db,db.cursor() as cur:
            cur.execute("SELECT next_request_at FROM launchpad_api_budget WHERE name='gecko'")
            assert cur.fetchone()[0]==before
        archive.defer_api(other,20)
        with pytest.raises(ValueError,match='phase deadline'):
            archive.reserve_api_slot(db,10)
        assert archive.reserve_api_slot(db,40)>18
    finally:other.close()


def test_worker_leases_survive_connection_switches_and_expired_owner_cannot_release(db):
    import psycopg2
    other=psycopg2.connect(os.environ['CRYPTO_SOCIAL_TEST_DATABASE_URL'],
                          options='-c search_path=launchpad_test')
    try:
        owner=archive.acquire_worker(db,SOLANA,'sweep')
        assert owner and archive.acquire_worker(other,SOLANA,'sweep') is None
        # Chain and job isolation: enrichment and Robinhood remain independent.
        assert archive.acquire_worker(other,SOLANA,'enrich')
        assert archive.acquire_worker(other,ROBINHOOD,'sweep')
        archive.release_worker(other,SOLANA,'sweep',owner)
        owner=archive.acquire_worker(other,SOLANA,'sweep')
        assert owner
        with db,db.cursor() as cur:
            cur.execute("UPDATE launchpad_worker_leases SET expires_at=clock_timestamp()-interval '1 second'")
        replacement=archive.acquire_worker(db,SOLANA,'sweep')
        assert replacement and replacement!=owner
        archive.release_worker(other,SOLANA,'sweep',owner)
        assert archive.acquire_worker(other,SOLANA,'sweep') is None
        archive.release_worker(other,SOLANA,'sweep',replacement)
        assert archive.acquire_worker(db,SOLANA,'sweep')
    finally:other.close()


def test_robinhood_ladder_can_reserve_shared_slots_without_nested_transaction(db,monkeypatch):
    import requests
    from test_launchpad_archive import GRAD_POOL
    with db,db.cursor() as cur:
        cur.execute('''INSERT INTO launchpad_tokens
                       (network,token_address,dex,first_seen_at,last_seen_at,graduated,graduated_at,
                        cohort_sampled,measure_pool,measure_pool_reason)
                       VALUES ('robinhood',%s,'pons-v2-dex',%s,%s,true,%s,true,%s,'launchpad destination')''',
                    (TOKEN,NOW,NOW,NOW-timedelta(hours=1),GRAD_POOL))
    monkeypatch.setattr(requests,'get',responder({1:[]}))
    # fetch=None uses the real database coordination path; network is stubbed.
    result=archive.sweep(db,now=NOW,wait=lambda _:None)
    assert result['observations']==1 and not result['errors']
    with db,db.cursor() as cur:
        cur.execute('SELECT count(*) FROM launchpad_worker_leases')
        assert cur.fetchone()[0]==0


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
    daily=archive.daily(db,chain=SOLANA,now=NOW)
    assert daily[0]['pools_measured']==0 and daily[0]['pools_at_graduation']==0
    assert daily[0]['at_graduation_share'] is None


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
    result=subprocess.run([sys.executable,'scripts/graduation_archive_report.py','--output',str(tmp_path),
                           '--cohort-since',NOW.isoformat()],
                           env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert len(list(tmp_path.glob('*.json')))==6
    payload=json.loads((tmp_path/'solana-report.json').read_text())
    assert payload['network']=='solana' and payload['continuous'] is False
    assert payload['post_repair_cohort']['cohort_graduates']==0


def test_current_schema_setup_needs_no_write_or_ddl_locks(db):
    # A ready collector's setup must even work in a read-only transaction.
    db.set_session(readonly=True)
    archive.setup(db,SOLANA)
    archive.setup(db,ROBINHOOD)
    db.set_session(readonly=False)


def test_schema_revision_is_stable_across_process_hash_seeds(db):
    env=dict(os.environ,DATABASE_URL=os.environ['CRYPTO_SOCIAL_TEST_DATABASE_URL'],
             PGOPTIONS='-c search_path=launchpad_test -c default_transaction_read_only=on',
             PYTHONHASHSEED='733')
    code="import os,psycopg2,launchpad_archive as a; c=psycopg2.connect(os.environ['DATABASE_URL']); a.setup(c); c.close()"
    result=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_cohort_report_excludes_history_and_deduplicates_captures(db):
    from launchpad_cohort_report import cohort_report
    seed_graduate(db,'historical',age=120)
    seed_graduate(db,'fresh',age=30,measure=SOL_DEEP)
    seed_graduate(db,'miss',age=10)
    seed_graduate(db,'outside',age=15,sampled=False)
    seed_graduate(db,'future',age=-5)
    with db,db.cursor() as cur:
        cur.execute("UPDATE launchpad_tokens SET measure_pool_timing='late' WHERE token_address='fresh'")
        for trades in (0,2):
            cur.execute('''INSERT INTO launchpad_trade_captures
                           (network,token_address,capture_started_at,trades)
                           VALUES ('solana','fresh',%s,%s)''',(NOW-timedelta(minutes=29),trades))
    db.set_session(readonly=True)
    out=cohort_report(db,SOLANA,NOW-timedelta(hours=1),NOW)
    assert out['graduates']==3 and out['cohort_graduates']==2
    assert out['captures']['captured']==1 and out['captures']['nonempty']==1
    assert out['captures']['coverage']==0.5 and out['sample_state']=='too_small'
    assert out['pool_selection']['at_graduation_coverage']==0 and out['pool_selection']['late']==1
    assert out['global_backlog']['oldest_age_seconds']==7200
    assert out['cohort_backlog']['oldest_age_seconds']==1800
    assert out['missed_capture_examples'][0]['token_address']=='miss'
    assert out['sweeps']['longest_silence_seconds']==3600
    assert cohort_report(db,ROBINHOOD,NOW-timedelta(hours=1),NOW)['graduates']==0
    db.set_session(readonly=False)


@pytest.mark.parametrize('value',['2026-09-20','2026-09-20T15:58:00','nonsense'])
def test_cohort_cutoff_requires_timezone(value):
    from launchpad_cohort_report import cutoff
    with pytest.raises(ValueError,match='timezone'):cutoff(value)


def test_arrival_capture_precedes_state_rate_limit(db):
    urls=[]
    graduation=sol_pool(dex='pumpswap',address=SOL_DEEP)
    delegate=sol_responder({1:[graduation]},pool_list={'data':[graduation]})
    def fetch(url,**kwargs):
        urls.append(url)
        if '/tokens/multi/' in url:
            response=Response({},429);response.headers={'Retry-After':'300'}
            return response
        return delegate(url,**kwargs)
    out=archive.sweep(db,SOLANA,fetch=fetch,now=NOW,wait=lambda _:None)
    assert out['trade_captures']==1
    assert next(i for i,u in enumerate(urls) if '/trades?' in u)<next(i for i,u in enumerate(urls) if '/tokens/multi/' in u)
    assert any('rate limit' in e for e in out['errors'])


def test_state_only_graduate_still_gets_capture(db):
    curve=sol_pool()
    multi={'data':[{'attributes':{'address':SOL_TOKEN,'launchpad_details':{
        'completed':True,'completed_at':NOW.isoformat(),'migrated_destination_pool_address':SOL_DEEP}}}]}
    fetch=sol_responder({1:[curve]},multi=multi,
                       pool_list={'data':[sol_pool(dex='pumpswap',address=SOL_DEEP)]})
    out=archive.sweep(db,SOLANA,fetch=fetch,now=NOW,wait=lambda _:None)
    assert out['trade_captures']==1 and out['graduations']==1


def test_capture_error_retains_token_and_provider_cause(db):
    graduation=sol_pool(dex='pumpswap',address=SOL_DEEP)
    delegate=sol_responder({1:[graduation]},pool_list={'data':[graduation]})
    def fetch(url,**kwargs):
        if '/trades?' in url:return Response({},503)
        return delegate(url,**kwargs)
    out=archive.sweep(db,SOLANA,fetch=fetch,now=NOW,wait=lambda _:None)
    assert out['trade_captures']==0
    assert any(SOL_TOKEN in e and 'HTTP 503' in e for e in out['errors'])
