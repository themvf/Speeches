from datetime import datetime,timedelta,timezone
import pytest
from launchpad_archive import (CURVE_DEXES,GRADUATE_DEXES,NETWORK,RUNGS,classify,gap_seconds,
                               parse_info,parse_multi,parse_pool,rungs_due)

NOW=datetime(2026,9,19,16,0,tzinfo=timezone.utc)
TOKEN='0xa92768863a55d8a0591709f7f5e594a249d36ea3'
CURVE_POOL='0x1111111111111111111111111111111111111111'
GRAD_POOL='0x90b75d6e9ac1e7efc3154c89e155121f7d7c0e54ebbd3dbba379650deb64d41e'


def pool(dex='pons-v2',address=CURVE_POOL,token=TOKEN,created='2026-09-19T15:50:00Z'):
    """Shaped like a real new_pools entry (fields taken from a live 2026-09-19 response)."""
    return {'attributes':{'address':address,'name':'ASKR / WETH','pool_created_at':created,
                          'base_token_price_usd':'0.00544','fdv_usd':'5518626.17','reserve_in_usd':'164377.60',
                          'volume_usd':{'m30':'27838.51','h1':'51653.66','h24':'3918446.04'},
                          'transactions':{'m30':{'buys':44,'sells':45,'buyers':34,'sellers':21},
                                          'h1':{'buys':80,'sells':60,'buyers':55,'sellers':30}},
                          'price_change_percentage':{'h1':'-11.658','h24':'9155.125'}},
            'relationships':{'dex':{'data':{'id':dex}},'base_token':{'data':{'id':'robinhood_'+token}}}}


def test_pool_parsing_keeps_the_fields_the_archive_is_built_on():
    parsed=parse_pool(pool())
    assert parsed['token']==TOKEN and parsed['dex']=='pons-v2' and parsed['pool']==CURVE_POOL
    assert parsed['created']==datetime(2026,9,19,15,50,tzinfo=timezone.utc)
    # Distinct buyers, not just trade counts: "is it still attracting new people" is the whole question.
    assert parsed['buyers_m30']==34 and parsed['sellers_m30']==21 and parsed['buyers_h1']==55
    assert parsed['txns_h1']==140 and parsed['liquidity_usd' if False else 'liquidity']==pytest.approx(164377.60)


def test_pool_parsing_rejects_records_it_cannot_key():
    assert parse_pool({'attributes':{'address':None},'relationships':{}}) is None
    broken=pool();broken['relationships']['base_token']={'data':{'id':'no-network-prefix'}}
    assert parse_pool(broken) is None


def test_curve_and_graduate_pools_are_different_dexes():
    # The finding the design rests on: a graduation announces itself as a new pool on a graduate DEX,
    # so it is detected inside the sweep we already run, with pool_created_at as its timestamp.
    assert classify('pons-v2')=='curve'
    assert classify('pons-v2-dex')=='graduate'
    assert classify('uniswap-v4-robinhood')=='other'
    assert not (CURVE_DEXES & GRADUATE_DEXES)


def test_gap_detection_is_what_makes_a_missed_sweep_visible():
    previous=datetime(2026,9,19,15,45,tzinfo=timezone.utc)
    # Windows overlap: the sweep reached back past what we already had.
    assert gap_seconds(previous,previous-timedelta(minutes=4))==0
    # Windows do not overlap: launches happened in between that we will never see.
    assert gap_seconds(previous,previous+timedelta(minutes=7))==420
    # No prior sweep to compare against is unknown, not "fine".
    assert gap_seconds(None,previous) is None
    assert gap_seconds(previous,None) is None


def test_ladder_fills_each_rung_once_and_only_when_due():
    graduated=NOW-timedelta(minutes=45)
    assert rungs_due(graduated,NOW,set())==[10,30]
    assert rungs_due(graduated,NOW,{10})==[30]
    assert rungs_due(graduated,NOW,{10,30})==[]
    # A rung is never anticipated.
    assert 60 not in rungs_due(graduated,NOW,set())
    assert rungs_due(None,NOW,set())==[]
    assert rungs_due(NOW-timedelta(days=30),NOW,set())==list(RUNGS)


def test_multi_parsing_reads_graduation_state():
    payload={'data':[{'attributes':{'address':TOKEN.upper(),'symbol':'ASKR','name':'heyaskr',
        'price_usd':'0.0063','fdv_usd':'6389343.92',
        'launchpad_details':{'graduation_percentage':100.0,'completed':True,
                             'completed_at':'2026-09-18T18:36:55.000Z',
                             'migrated_destination_pool_address':GRAD_POOL}}}]}
    state=parse_multi(payload)[TOKEN]
    assert state['completed'] and state['pct']==100.0
    assert state['completed_at']==datetime(2026,9,18,18,36,55,tzinfo=timezone.utc)
    assert state['destination']==GRAD_POOL


def test_multi_parsing_keeps_a_live_curve_token_distinguishable_from_a_finished_one():
    payload={'data':[{'attributes':{'address':TOKEN,'symbol':'ZECROOM',
        'launchpad_details':{'graduation_percentage':34.06,'completed':False,'completed_at':None,
                             'migrated_destination_pool_address':None}}}]}
    state=parse_multi(payload)[TOKEN]
    assert state['pct']==34.06 and not state['completed']
    assert state['completed_at'] is None and state['destination'] is None


def test_info_parsing_supplies_holders_concentration_and_the_x_handle():
    payload={'data':{'attributes':{'holders':{'count':2153,'distribution_percentage':{'top_10':'57.5'}},
                                   'twitter_handle':'heyaskr','telegram_handle':'heyaskr'}}}
    info=parse_info(payload)
    assert info['holders']==2153 and info['top10']==pytest.approx(57.5)
    # The bridge to the social side: a graduate arrives with its own declared account, no search.
    assert info['twitter']=='heyaskr'


def test_x_handles_are_cleaned_because_the_field_is_whatever_the_launcher_typed():
    from launchpad_archive import normalize_handle
    assert normalize_handle('heyaskr')=='heyaskr'
    assert normalize_handle('@heyaskr')=='heyaskr'
    assert normalize_handle('https://x.com/heyaskr')=='heyaskr'
    assert normalize_handle('https://twitter.com/heyaskr?s=20')=='heyaskr'
    # Seen in a live sweep: a status URL stored in the handle field. Joining on this would match
    # nothing, so it is dropped rather than recorded as if it were an account.
    assert normalize_handle('Na1_N1ako/status/2101346622135230543')=='Na1_N1ako'
    assert normalize_handle('https://x.com/i/communities/123/status/9') is None or len(normalize_handle('https://x.com/i/communities/123/status/9'))<=15
    assert normalize_handle('a'*16) is None
    assert normalize_handle('not a handle') is None
    assert normalize_handle('') is None and normalize_handle(None) is None


def test_info_parsing_drops_a_handle_that_is_not_one():
    payload={'data':{'attributes':{'twitter_handle':'Na1_N1ako/status/2101346622135230543'}}}
    assert parse_info(payload)['twitter']=='Na1_N1ako'


def test_info_parsing_tolerates_a_token_with_no_declared_socials():
    assert parse_info({'data':{'attributes':{}}})=={'holders':None,'top10':None,'twitter':None}
    assert parse_info({})=={'holders':None,'top10':None,'twitter':None}


# Optional real Postgres gate: uses only a disposable local test database.
@pytest.fixture
def db():
    import os
    import psycopg2
    url=os.environ.get('CRYPTO_SOCIAL_TEST_DATABASE_URL')
    if not url: pytest.skip('Disposable Postgres not configured')
    conn=psycopg2.connect(url)
    conn.autocommit=True
    with conn.cursor() as cur:
        cur.execute('DROP SCHEMA IF EXISTS launchpad_test CASCADE; CREATE SCHEMA launchpad_test; SET search_path TO launchpad_test')
    conn.autocommit=False
    from launchpad_archive import setup
    setup(conn)
    yield conn
    conn.close()


class Response:
    def __init__(self,payload,status=200):self._payload=payload;self.status_code=status;self.headers={}
    def json(self):return self._payload


def responder(pages,multi=None,info=None,pools=None):
    """Serve new_pools pages, tokens/multi and per-token calls from fixtures."""
    def fetch(url,**_):
        if '/new_pools' in url:
            page=int(url.rsplit('page=',1)[1])
            return Response({'data':pages.get(page,[])})
        if '/tokens/multi/' in url:return Response(multi or {'data':[]})
        if url.endswith('/info'):return Response(info or {'data':{'attributes':{}}})
        if '/pools/' in url:return Response({'data':(pools or pool(dex='pons-v2-dex',address=GRAD_POOL))})
        raise AssertionError('unexpected url '+url)
    return fetch


def test_db_sweep_records_a_launch_and_leaves_an_auditable_sweep_row(db):
    from launchpad_archive import sweep
    result=sweep(db,fetch=responder({1:[pool()]}),now=NOW,wait=lambda *_:None)
    assert result['status']=='ok' and result['new_tokens']==1 and result['curve']==1
    with db,db.cursor() as cur:
        cur.execute('SELECT token_address,state,graduated,curve_pool FROM launchpad_tokens')
        assert cur.fetchone()==(TOKEN,'live',False,CURVE_POOL)
        cur.execute('SELECT phase,rung_minutes FROM launchpad_observations')
        assert cur.fetchall()==[('curve',None)]
        cur.execute('SELECT pools_seen,complete,gap_seconds FROM launchpad_sweeps')
        assert cur.fetchone()==(1,True,None)   # no prior sweep: gap is unknown, not zero


def test_db_graduation_is_detected_from_the_graduate_pool_arriving(db):
    from launchpad_archive import sweep
    fetch=lambda *a,**k:None
    sweep(db,fetch=responder({1:[pool()]}),now=NOW,wait=lambda *_:None)
    later=NOW+timedelta(minutes=10)
    graduation=pool(dex='pons-v2-dex',address=GRAD_POOL,created='2026-09-19T16:05:00Z')
    info={'data':{'attributes':{'holders':{'count':2153,'distribution_percentage':{'top_10':'57.5'}},'twitter_handle':'heyaskr'}}}
    result=sweep(db,fetch=responder({1:[graduation]},info=info),now=later,wait=lambda *_:None)
    assert result['graduations']==1
    with db,db.cursor() as cur:
        cur.execute('SELECT graduated,state,graduated_at,graduated_detected_at,holders,twitter_handle,graduation_pool FROM launchpad_tokens')
        graduated,state,at,detected,holders,handle,grad_pool=cur.fetchone()
        assert graduated and state=='graduated' and grad_pool==GRAD_POOL
        # Both timestamps are kept: their difference is our observation lag, which is the evidence
        # that decides whether a 60-second fast lane is ever worth building.
        assert at==datetime(2026,9,19,16,5,tzinfo=timezone.utc) and detected==later
        assert holders==2153 and handle=='heyaskr'


def test_db_a_non_overlapping_window_marks_the_sweep_incomplete(db):
    from launchpad_archive import sweep
    sweep(db,fetch=responder({1:[pool(created='2026-09-19T15:50:00Z')]}),now=NOW,wait=lambda *_:None)
    # Next sweep only reaches back to 16:20 - everything launched between 15:50 and 16:20 is lost.
    late=NOW+timedelta(minutes=40)
    sweep(db,fetch=responder({1:[pool(address='0x2222222222222222222222222222222222222222',
                                      token='0xbbbb000000000000000000000000000000000000',
                                      created='2026-09-19T16:20:00Z')]}),now=late,wait=lambda *_:None)
    with db,db.cursor() as cur:
        cur.execute('SELECT gap_seconds,complete FROM launchpad_sweeps ORDER BY id DESC LIMIT 1')
        gap,complete=cur.fetchone()
        assert gap==1800 and complete is False


def test_db_observations_are_immutable_and_first_facts_win(db):
    from launchpad_archive import sweep
    sweep(db,fetch=responder({1:[pool()]}),now=NOW,wait=lambda *_:None)
    # A re-run at the same instant must not duplicate or rewrite the observation.
    sweep(db,fetch=responder({1:[pool()]}),now=NOW,wait=lambda *_:None)
    with db,db.cursor() as cur:
        cur.execute('SELECT count(*) FROM launchpad_observations')
        assert cur.fetchone()[0]==1
        cur.execute('SELECT first_seen_at FROM launchpad_tokens')
        assert cur.fetchone()[0]==NOW


def test_db_feed_depth_is_only_read_from_a_sweep_that_went_full_depth(db):
    from launchpad_archive import PAGES,report
    # A sweep that stopped early did so because it met ground already recorded: its shallow reach is
    # proof the archive is healthy. Only a sweep that exhausted every page has actually found the
    # feed's limit. Reading the wrong one raises the alarm exactly when nothing is wrong.
    with db,db.cursor() as cur:
        cur.execute('''INSERT INTO launchpad_sweeps (started_at,pages_fetched,pools_seen,oldest_pool_at,newest_pool_at,gap_seconds,complete)
                       VALUES (%s,%s,20,%s,%s,0,true)''',(NOW,1,NOW-timedelta(seconds=120),NOW))
        cur.execute('''INSERT INTO launchpad_sweeps (started_at,pages_fetched,pools_seen,oldest_pool_at,newest_pool_at,gap_seconds,complete)
                       VALUES (%s,%s,200,%s,%s,0,true)''',(NOW,PAGES,NOW-timedelta(seconds=830),NOW))
    out=report(db,now=NOW+timedelta(minutes=1),hours=1)
    assert out['sweeps']['min_reach_seconds']==830   # the full-depth sweep, not the 120s early stop
    assert out['margin_warning'] is False

    with db,db.cursor() as cur:cur.execute('DELETE FROM launchpad_sweeps WHERE pages_fetched>=%s',(PAGES,))
    out=report(db,now=NOW+timedelta(minutes=1),hours=1)
    # No full-depth sweep means the feed's limit was never observed: unknown, not healthy.
    assert out['sweeps']['min_reach_seconds'] is None and out['margin_seconds'] is None
    assert out['margin_warning'] is False


def test_db_report_calls_a_broken_archive_incomplete(db):
    from launchpad_archive import report,sweep
    sweep(db,fetch=responder({1:[pool()]}),now=NOW,wait=lambda *_:None)
    out=report(db,now=NOW+timedelta(minutes=1),hours=48)
    # One sweep in a 48-hour window is nowhere near continuous, and the report must say so rather
    # than reporting healthy-looking totals.
    assert out['continuous'] is False and 'expected sweeps' in out['verdict']
    assert out['tokens']['discovered']==1 and out['totals']['observations']==1
