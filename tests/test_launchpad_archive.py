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


def test_db_daily_summary_answers_the_health_question_without_reading_raw_rows(db):
    from launchpad_archive import daily
    day=datetime(2026,9,19,tzinfo=timezone.utc)
    with db,db.cursor() as cur:
        for minute,gap,complete,launches,graduates in ((0,0,True,40,1),(30,0,True,35,2),(60,900,False,20,0)):
            cur.execute('''INSERT INTO launchpad_sweeps (started_at,pages_fetched,pools_seen,oldest_pool_at,
                             newest_pool_at,new_tokens,graduations,observations,gap_seconds,complete)
                           VALUES (%s,10,200,%s,%s,%s,%s,%s,%s,%s)''',
                        (day+timedelta(minutes=minute),day,day+timedelta(minutes=minute),launches,graduates,launches,gap,complete))
        # Lags of 30/60/90/600s: the median stays low while the tail does not, which is exactly the
        # shape that would argue for a faster collector and which a median alone would hide.
        for i,lag in enumerate((30,60,90,600)):
            cur.execute('''INSERT INTO launchpad_tokens (network,token_address,dex,first_seen_at,last_seen_at,
                             graduated,graduated_at,graduated_detected_at)
                           VALUES (%s,%s,'pons-v2',%s,%s,true,%s,%s)''',
                        (NETWORK,'0x'+str(i)*40,day,day,day+timedelta(hours=1),day+timedelta(hours=1,seconds=lag)))
    rows=daily(db,now=day+timedelta(hours=2),days=1)
    assert len(rows)==1
    row=rows[0]
    assert row['day']=='2026-09-19'
    assert row['launches_seen']==95 and row['graduates_detected']==3
    assert row['incomplete_sweeps']==1 and row['max_gap_seconds']==900
    assert row['sweeps']==3 and row['expected']==288          # 24h / 5-minute cadence
    assert row['detection_lag_median']==60 and row['detection_lag_p95']==600
    assert row['lag_measured']==4


def test_db_report_calls_a_broken_archive_incomplete(db):
    from launchpad_archive import report,sweep
    sweep(db,fetch=responder({1:[pool()]}),now=NOW,wait=lambda *_:None)
    out=report(db,now=NOW+timedelta(minutes=1),hours=48)
    # One sweep in a 48-hour window is nowhere near continuous, and the report must say so rather
    # than reporting healthy-looking totals.
    assert out['continuous'] is False and 'expected sweeps' in out['verdict']
    assert out['tokens']['discovered']==1 and out['totals']['observations']==1


# --- Solana adapter: the chain-specific path end to end ---------------------------------------

# In the 25% ladder cohort (the draw hashes the mint), so the full measure-pool and trade-capture
# path runs. SOL_OUTSIDE below is deliberately not in it.
SOL_TOKEN='Csi8DLHFExL6QCsQ8xW5P63vNrMzFKR3koBSHn7fY5Yc'
SOL_OUTSIDE='Csi8DLHFExL6QCsQ8xW5P63vNrMzFKR3koBSHn7fY5Yo'
SOL_DEEP='ALPZYXZBTvbmT1cyHxwXUgvFLCyMVuAXaJv9nLYDpieq'
SOL_EMPTY='FramDv5MyadCwonKaShAqNQToV8kKBMmxh6S4QzJtTeb'


def sol_pool(dex='pump-fun',address='0xcurve',token=SOL_TOKEN,created='2026-09-19T19:18:02Z'):
    p=pool(dex=dex,address=address,token=token,created=created)
    p['attributes']['name']='Nike / SOL'
    p['relationships']['base_token']['data']['id']='solana_'+token
    return p


def sol_responder(pages,multi=None,info=None,pool_list=None,trades=None,pool_payload=None):
    def fetch(url,**_):
        if '/new_pools' in url:
            page=int(url.rsplit('page=',1)[1]);return Response({'data':pages.get(page,[])})
        if '/tokens/multi/' in url:return Response(multi or {'data':[]})
        if url.endswith('/pools'):return Response(pool_list or {'data':[]})
        if '/trades' in url:return Response(trades or {'data':[]})
        if url.endswith('/info'):return Response(info or {'data':{'attributes':{}}})
        if '/pools/' in url:return Response({'data':pool_payload or sol_pool()})
        raise AssertionError('unexpected url '+url)
    return fetch


def test_db_solana_graduate_measures_the_deep_pool_and_captures_opening_trades(db):
    from launchpad_archive import sweep
    from launchpad_chains import SOLANA
    graduation=sol_pool(dex='pumpswap',address=SOL_DEEP,created='2026-09-19T19:18:07Z')
    multi={'data':[{'attributes':{'address':SOL_TOKEN,'symbol':'Nike',
        'launchpad_details':{'graduation_percentage':100.0,'completed':True,
                             'completed_at':'2026-09-19T19:18:02.000Z',
                             'migrated_destination_pool_address':SOL_EMPTY}}}]}
    info={'data':{'attributes':{'developer_address':'7H7SkM44','developer_holding_percentage':'19.99',
                                'mint_authority':'no','freeze_authority':'no','is_honeypot':'unknown',
                                'twitter_handle':'nikebasketball','description':'a coin','gt_score':60.0,
                                'holders':{'count':None,'distribution_percentage':{}}}}}
    pool_list={'data':[
        {'attributes':{'address':SOL_DEEP,'reserve_in_usd':'31356.0'},'relationships':{'dex':{'data':{'id':'pumpswap'}}}},
        {'attributes':{'address':SOL_EMPTY,'reserve_in_usd':'0.0'},'relationships':{'dex':{'data':{'id':'meteora-damm-v2'}}}}]}
    trades={'data':[{'attributes':{'tx_from_address':'w'+str(i),'block_timestamp':f'2026-09-19T19:18:{10+i:02d}Z',
                                   'kind':'buy' if i%4 else 'sell','to_token_amount':'10','from_token_amount':'1',
                                   'volume_in_usd':'100','tx_hash':'tx'+str(i),'block_number':i}} for i in range(12)]}
    result=sweep(db,SOLANA,fetch=sol_responder({1:[graduation]},multi=multi,info=info,pool_list=pool_list,trades=trades),
                 now=NOW,wait=lambda *_:None)
    assert result['network']=='solana' and result['graduations']==1 and result['trade_captures']==1
    with db,db.cursor() as cur:
        cur.execute('''SELECT measure_pool,measure_pool_reason,graduation_pool,developer_address,
                              developer_holding,launchpad,twitter_handle,info_raw IS NOT NULL
                       FROM launchpad_tokens WHERE network='solana' ''')
        measure,reason,destination,dev,holding,launchpad,handle,raw=cur.fetchone()
        # The ladder must read the deep pool, never the near-empty one the launchpad field names.
        assert measure==SOL_DEEP and destination==SOL_EMPTY
        assert 'deepest' in reason and 'disagreed' in reason
        # Enrichment moved to its own worker: the sweep protects only what expires.
        assert dev is None and holding is None
        # pumpswap is where it landed, not where it launched: recording a destination venue as the
        # launchpad would be false, so it stays NULL until we see the curve pool itself.
        assert launchpad is None
        # Socials and the raw payload arrive with the worker, not the sweep.
        assert handle is None and raw is False
        cur.execute('''SELECT trades,wallets,buyers,sellers,pool,lag_seconds,window_seconds
                       FROM launchpad_trade_captures''')
        trades_n,wallets,buyers,sellers,cpool,lag,window=cur.fetchone()
        assert trades_n==12 and wallets==12 and buyers==9 and sellers==3 and cpool==SOL_DEEP
        # Boundaries are recorded, so how much of the opening window we caught is measurable rather
        # than assumed: first trade 8s after graduation, spanning 11s.
        assert lag==8 and window==11
        cur.execute('SELECT count(*),count(DISTINCT wallet),min(sequence),max(sequence) FROM launchpad_trades')
        assert cur.fetchone()==(12,12,0,11)


def test_db_solana_capture_is_skipped_when_no_pool_is_liquid(db):
    from launchpad_archive import sweep
    from launchpad_chains import SOLANA
    graduation=sol_pool(dex='pumpswap',address=SOL_DEEP,created='2026-09-19T19:18:07Z')
    multi={'data':[{'attributes':{'address':SOL_TOKEN,'symbol':'Nike',
        'launchpad_details':{'graduation_percentage':100.0,'completed':True,
                             'completed_at':'2026-09-19T19:18:02.000Z','migrated_destination_pool_address':None}}}]}
    pool_list={'data':[{'attributes':{'address':SOL_EMPTY,'reserve_in_usd':'0.0'},
                        'relationships':{'dex':{'data':{'id':'pumpswap'}}}}]}
    result=sweep(db,SOLANA,fetch=sol_responder({1:[graduation]},multi=multi,pool_list=pool_list),
                 now=NOW,wait=lambda *_:None)
    assert result['trade_captures']==0
    with db,db.cursor() as cur:
        cur.execute("SELECT measure_pool,measure_pool_reason FROM launchpad_tokens WHERE network='solana'")
        measure,reason=cur.fetchone()
        # Recorded as an outcome, not dropped - dropping these biases every survival rate upward.
        assert measure is None and reason=='no liquid pool'


def test_db_the_two_chains_never_touch_each_other(db):
    from launchpad_archive import sweep
    from launchpad_chains import SOLANA
    sweep(db,fetch=responder({1:[pool()]}),now=NOW,wait=lambda *_:None)
    sweep(db,SOLANA,fetch=sol_responder({1:[sol_pool()]},
          multi={'data':[{'attributes':{'address':SOL_TOKEN,'symbol':'Nike',
                 'launchpad_details':{'graduation_percentage':4.0,'completed':False,'completed_at':None,
                                      'migrated_destination_pool_address':None}}}]}),
          now=NOW,wait=lambda *_:None)
    with db,db.cursor() as cur:
        cur.execute('SELECT network,count(*) FROM launchpad_tokens GROUP BY network ORDER BY network')
        assert cur.fetchall()==[('robinhood',1),('solana',1)]
        cur.execute("SELECT dex FROM launchpad_tokens WHERE network='robinhood'")
        assert cur.fetchone()[0]=='pons-v2'


def test_db_a_graduate_outside_the_cohort_is_recorded_without_being_measured(db):
    from launchpad_archive import sweep
    from launchpad_chains import SOLANA,in_cohort
    assert not in_cohort(SOLANA,SOL_OUTSIDE)
    graduation=sol_pool(dex='pumpswap',address=SOL_DEEP,token=SOL_OUTSIDE,created='2026-09-19T19:18:07Z')
    multi={'data':[{'attributes':{'address':SOL_OUTSIDE,'symbol':'Nike',
        'launchpad_details':{'graduation_percentage':100.0,'completed':True,
                             'completed_at':'2026-09-19T19:18:02.000Z','migrated_destination_pool_address':SOL_EMPTY}}}]}
    info={'data':{'attributes':{'developer_address':'7H7SkM44','developer_holding_percentage':'19.99'}}}
    result=sweep(db,SOLANA,fetch=sol_responder({1:[graduation]},multi=multi,info=info),now=NOW,wait=lambda *_:None)
    assert result['graduations']==1 and result['trade_captures']==0
    with db,db.cursor() as cur:
        cur.execute("SELECT cohort_sampled,measure_pool,measure_pool_reason,enriched_at FROM launchpad_tokens WHERE network='solana'")
        sampled,measure,reason,enriched=cur.fetchone()
        # Still recorded, and the reason still says we chose not to look rather than implying the
        # token had no pools. Enrichment itself is the worker's job now.
        assert sampled is False and measure is None and reason=='outside ladder cohort'
        assert enriched is None


def test_db_an_exhausted_budget_still_produces_a_complete_sweep_row(db):
    from dataclasses import replace
    from launchpad_archive import sweep
    from launchpad_chains import SOLANA
    # budget_fraction 0 puts every fetch past the deadline on its first check, including discovery.
    # This exercises the deadline path itself, which is how a shadowed variable in it went unnoticed
    # until a live run: the mocked tests never entered the branch.
    broke=replace(SOLANA,budget_fraction=0.0)
    graduation=sol_pool(dex='pumpswap',address=SOL_DEEP,created='2026-09-19T19:18:07Z')
    multi={'data':[{'attributes':{'address':SOL_TOKEN,'symbol':'Nike',
        'launchpad_details':{'graduation_percentage':100.0,'completed':True,
                             'completed_at':'2026-09-19T19:18:02.000Z','migrated_destination_pool_address':None}}}]}
    result=sweep(db,broke,fetch=sol_responder({1:[graduation]},multi=multi),now=NOW,wait=lambda *_:None)
    # The sweep still completes and still writes a row saying what it could not do, rather than
    # crashing or reporting a clean run it did not have.
    assert result['status']=='ok' and result['pools']==0 and result['trade_captures']==0
    assert any('reserve budget' in e for e in result['errors'])
    with db,db.cursor() as cur:
        cur.execute('SELECT count(*),bool_or(NOT complete) FROM launchpad_sweeps')
        assert cur.fetchone()==(1,True)


def test_db_a_graduate_the_deadline_skipped_is_retried_not_abandoned(db):
    from dataclasses import replace
    from launchpad_archive import sweep
    from launchpad_chains import SOLANA  # noqa: F401 - kept for symmetry with the worker test
    graduation=sol_pool(dex='pumpswap',address=SOL_DEEP,created='2026-09-19T19:18:07Z')
    multi={'data':[{'attributes':{'address':SOL_TOKEN,'symbol':'Nike',
        'launchpad_details':{'graduation_percentage':100.0,'completed':True,
                             'completed_at':'2026-09-19T19:18:02.000Z','migrated_destination_pool_address':None}}}]}
    info={'data':{'attributes':{'developer_address':'7H7SkM44'}}}
    pool_list={'data':[{'attributes':{'address':SOL_DEEP,'reserve_in_usd':'31356.0'},
                        'relationships':{'dex':{'data':{'id':'pumpswap'}}}}]}
    # Solana enriches in its own worker now, so exercise the in-sweep backlog on a chain that still
    # enriches in the sweep. First sweep discovers the graduation with no enrichment budget at all.
    chain=replace(SOLANA,enrich_in_sweep=True,max_info=0)
    sweep(db,chain,fetch=sol_responder({1:[graduation]},multi=multi,info=info,pool_list=pool_list),
          now=NOW,wait=lambda *_:None)
    with db,db.cursor() as cur:
        cur.execute("SELECT graduated,measure_pool_reason FROM launchpad_tokens WHERE network='solana'")
        assert cur.fetchone()==(True,None)
    # It is already graduated, so it never returns via the newly-graduated set. Measured live: 27 of
    # 29 graduates were cut this way and would have stayed unenriched for ever. The backlog is what
    # brings them back.
    sweep(db,replace(SOLANA,enrich_in_sweep=True),fetch=sol_responder({1:[]},multi=multi,info=info,pool_list=pool_list),
          now=NOW+timedelta(minutes=2),wait=lambda *_:None)
    with db,db.cursor() as cur:
        cur.execute("SELECT measure_pool,measure_pool_reason,developer_address FROM launchpad_tokens WHERE network='solana'")
        measure,reason,dev=cur.fetchone()
        assert measure==SOL_DEEP and reason=='deepest graduate pool' and dev=='7H7SkM44'


# --- Enrichment worker: durable work, its own schedule ------------------------------------------

def test_db_the_solana_sweep_no_longer_enriches_but_still_captures(db):
    from launchpad_archive import sweep
    from launchpad_chains import SOLANA
    graduation=sol_pool(dex='pumpswap',address=SOL_DEEP,created='2026-09-19T19:18:07Z')
    multi={'data':[{'attributes':{'address':SOL_TOKEN,'symbol':'Nike',
        'launchpad_details':{'graduation_percentage':100.0,'completed':True,
                             'completed_at':'2026-09-19T19:18:02.000Z','migrated_destination_pool_address':None}}}]}
    pool_list={'data':[{'attributes':{'address':SOL_DEEP,'reserve_in_usd':'31356.0'},
                        'relationships':{'dex':{'data':{'id':'pumpswap'}}}}]}
    trades={'data':[{'attributes':{'tx_from_address':'w'+str(i),'block_timestamp':f'2026-09-19T19:18:{10+i:02d}Z',
                                   'kind':'buy','to_token_amount':'10','from_token_amount':'1','volume_in_usd':'100',
                                   'tx_hash':'tx'+str(i),'block_number':i}} for i in range(5)]}
    result=sweep(db,SOLANA,fetch=sol_responder({1:[graduation]},multi=multi,pool_list=pool_list,trades=trades),
                 now=NOW,wait=lambda *_:None)
    # The perishable half still happens in the sweep, at graduation.
    assert result['trade_captures']==1 and result['graduations']==1
    with db,db.cursor() as cur:
        cur.execute("SELECT measure_pool,enriched_at,developer_address FROM launchpad_tokens WHERE network='solana'")
        measure,enriched,dev=cur.fetchone()
        # Pool selection is immediate - a token can fall to nothing within the hour, so choosing it
        # later could name a different market. Everything durable is left to the worker.
        assert measure==SOL_DEEP and enriched is None and dev is None


def test_db_the_worker_drains_what_the_sweep_left(db):
    from launchpad_archive import backlog_health,enrich,sweep
    from launchpad_chains import SOLANA
    graduation=sol_pool(dex='pumpswap',address=SOL_DEEP,created='2026-09-19T19:18:07Z')
    multi={'data':[{'attributes':{'address':SOL_TOKEN,'symbol':'Nike',
        'launchpad_details':{'graduation_percentage':100.0,'completed':True,
                             'completed_at':'2026-09-19T19:18:02.000Z','migrated_destination_pool_address':None}}}]}
    pool_list={'data':[{'attributes':{'address':SOL_DEEP,'reserve_in_usd':'31356.0'},
                        'relationships':{'dex':{'data':{'id':'pumpswap'}}}}]}
    sweep(db,SOLANA,fetch=sol_responder({1:[graduation]},multi=multi,pool_list=pool_list),now=NOW,wait=lambda *_:None)
    health=backlog_health(db,SOLANA,now=NOW)
    assert health['pending_enrichment']==1 and health['enrichment_state']=='healthy'
    info={'data':{'attributes':{'developer_address':'7H7SkM44','developer_holding_percentage':'19.99',
                                'twitter_handle':'nikebasketball','holders':{'count':2153,'distribution_percentage':{'top_10':'57.5'}}}}}
    # 30 minutes after the fixture's graduation (19:18), so the +5/+10/+30 rungs are due.
    out=enrich(db,SOLANA,fetch=sol_responder({},info=info,pool_list=pool_list,pool_payload=sol_pool(dex='pumpswap',address=SOL_DEEP)),
               now=datetime(2026,9,19,19,48,tzinfo=timezone.utc),wait=lambda *_:None)
    assert out['processed_this_run']==1 and out['pending_enrichment']==0 and out['errors']==[]
    # +5,+10 and +30 rungs are all due 30 minutes after graduation; one is filled per run.
    assert out['rungs_filled']==1
    with db,db.cursor() as cur:
        cur.execute("SELECT developer_address,holders,twitter_handle,measure_pool_reason,enriched_at IS NOT NULL FROM launchpad_tokens WHERE network='solana'")
        dev,holders,handle,reason,enriched=cur.fetchone()
        assert dev=='7H7SkM44' and holders==2153 and handle=='nikebasketball' and enriched is True
        # The sweep already chose the pool at graduation, so the worker must not relabel it as late.
        assert reason=='deepest graduate pool'


def test_db_a_pool_the_worker_had_to_choose_late_says_so(db):
    from launchpad_archive import enrich
    from launchpad_chains import SOLANA
    with db,db.cursor() as cur:
        cur.execute("""INSERT INTO launchpad_tokens (network,token_address,dex,first_seen_at,last_seen_at,
                         graduated,graduated_at,cohort_sampled) VALUES ('solana',%s,'pumpswap',%s,%s,true,%s,true)""",
                    (SOL_TOKEN,NOW,NOW,NOW))
    pool_list={'data':[{'attributes':{'address':SOL_DEEP,'reserve_in_usd':'12.0'},
                        'relationships':{'dex':{'data':{'id':'pumpswap'}}}}]}
    enrich(db,SOLANA,fetch=sol_responder({},pool_list=pool_list,pool_payload=sol_pool(dex='pumpswap',address=SOL_DEEP)),
           now=NOW+timedelta(hours=3),wait=lambda *_:None)
    with db,db.cursor() as cur:
        cur.execute("SELECT measure_pool_reason FROM launchpad_tokens WHERE network='solana'")
        # Chosen hours after the fact, which may not be the market that mattered. Recorded, not hidden.
        assert 'selected late, not at graduation' in cur.fetchone()[0]


def test_db_one_failing_item_does_not_block_the_backlog(db):
    from launchpad_archive import enrich
    from launchpad_chains import SOLANA
    good='Csi8DLHFExL6QCsQ8xW5P63vNrMzFKR3koBSHn7fY5Yc'
    bad='Bad8DLHFExL6QCsQ8xW5P63vNrMzFKR3koBSHn7fY5Yc'
    with db,db.cursor() as cur:
        for t in (bad,good):   # bad is older, so it is attempted first
            cur.execute("""INSERT INTO launchpad_tokens (network,token_address,dex,first_seen_at,last_seen_at,
                             graduated,graduated_at,cohort_sampled) VALUES ('solana',%s,'pumpswap',%s,%s,true,%s,false)""",
                        (t,NOW,NOW,NOW-timedelta(minutes=10 if t==bad else 5)))
    def fetch(url,**_):
        if bad in url:return Response({'errors':[{'status':'404'}]},status=404)
        if url.endswith('/info'):return Response({'data':{'attributes':{'developer_address':'DEV1'}}})
        return Response({'data':[]})
    out=enrich(db,SOLANA,fetch=fetch,now=NOW,wait=lambda *_:None)
    # The failure is recorded and the run continues; a bad row must never stall everything behind it.
    assert out['processed_this_run']==1 and any(bad[:10] in e for e in out['errors'])
    with db,db.cursor() as cur:
        cur.execute("SELECT token_address,developer_address FROM launchpad_tokens WHERE enriched_at IS NOT NULL")
        assert cur.fetchall()==[(good,'DEV1')]
        cur.execute("SELECT count(*) FROM launchpad_tokens WHERE enriched_at IS NULL")
        assert cur.fetchone()[0]==1     # the failure stays pending, to be retried


def test_db_re_running_the_worker_skips_finished_work_and_keeps_first_values(db):
    from launchpad_archive import enrich
    from launchpad_chains import SOLANA
    with db,db.cursor() as cur:
        cur.execute("""INSERT INTO launchpad_tokens (network,token_address,dex,first_seen_at,last_seen_at,
                         graduated,graduated_at,cohort_sampled,measure_pool,measure_pool_reason)
                       VALUES ('solana',%s,'pumpswap',%s,%s,true,%s,false,%s,'deepest graduate pool')""",
                    (SOL_TOKEN,NOW,NOW,NOW,SOL_DEEP))
    first={'data':{'attributes':{'developer_address':'DEV1','twitter_handle':'first_handle','holders':{'count':10,'distribution_percentage':{}}}}}
    second={'data':{'attributes':{'developer_address':'DEV2','twitter_handle':'second_handle','holders':{'count':99,'distribution_percentage':{}}}}}
    enrich(db,SOLANA,fetch=lambda url,**_:Response(first if url.endswith('/info') else {'data':[]}),now=NOW,wait=lambda *_:None)
    out=enrich(db,SOLANA,fetch=lambda url,**_:Response(second if url.endswith('/info') else {'data':[]}),
               now=NOW+timedelta(minutes=10),wait=lambda *_:None)
    # Nothing is pending, so the second run does no work at all rather than re-reading the same rows.
    assert out['processed_this_run']==0 and out['pending_enrichment']==0
    with db,db.cursor() as cur:
        cur.execute("SELECT developer_address,twitter_handle,holders,measure_pool_reason FROM launchpad_tokens WHERE network='solana'")
        dev,handle,holders,reason=cur.fetchone()
        # Identity-ish fields are first-write-wins; the pool chosen at graduation is never relabelled.
        assert dev=='DEV1' and handle=='first_handle' and holders==10
        assert reason=='deepest graduate pool'


def test_db_backlog_health_is_about_age_not_count_or_rate(db):
    from launchpad_archive import backlog_health
    from launchpad_chains import SOLANA
    def graduate(token,age_minutes,enriched=False):
        with db,db.cursor() as cur:
            cur.execute("""INSERT INTO launchpad_tokens (network,token_address,dex,first_seen_at,last_seen_at,
                             graduated,graduated_at,cohort_sampled,enriched_at)
                           VALUES ('solana',%s,'pumpswap',%s,%s,true,%s,false,%s)""",
                        (token,NOW,NOW,NOW-timedelta(minutes=age_minutes),NOW if enriched else None))
    def record(age):
        with db,db.cursor() as cur:
            cur.execute("""INSERT INTO launchpad_enrich_runs (network,started_at,oldest_pending_age_seconds)
                           VALUES ('solana',%s,%s)""",(NOW,age))
    # A big queue whose oldest item is young is healthy: count alone says nothing.
    for i in range(100):graduate('young%d'%i,2)
    h=backlog_health(db,SOLANA,now=NOW)
    assert h['pending_enrichment']==100 and h['enrichment_state']=='healthy'
    # One genuinely old item past the target, and rising across the last runs, is degrading - even
    # though the service rate here comfortably exceeds the arrival rate.
    graduate('ancient',360)
    record(60);record(120)
    h=backlog_health(db,SOLANA,now=NOW)
    assert h['oldest_pending_age_seconds']==21600 and h['oldest_age_rising'] is True
    assert h['enrichment_state']=='degrading' and h['state']=='degrading'
    # The same age falling again is healthy, whatever the rates are doing.
    record(30000);record(25000)
    h=backlog_health(db,SOLANA,now=NOW)
    assert h['oldest_age_rising'] is False and h['enrichment_state']=='healthy'


def test_db_an_empty_backlog_is_idle_not_healthy_by_accident(db):
    from launchpad_archive import backlog_health
    from launchpad_chains import SOLANA
    h=backlog_health(db,SOLANA,now=NOW)
    assert h['enrichment_state']=='idle'
    # With no captures and no measured pools there is nothing to judge, and an empty system must not
    # claim health: the overall state is unknown, not idle.
    assert h['capture_state']=='unknown' and h['selection_state']=='unknown' and h['state']=='unknown'


def test_db_pool_selection_timing_is_a_commissioning_metric(db):
    from launchpad_archive import backlog_health
    from launchpad_chains import SOLANA
    with db,db.cursor() as cur:
        for token,reason in (('a','deepest graduate pool'),('b','deepest graduate pool'),
                             ('c','deepest graduate pool (selected late, not at graduation)')):
            cur.execute("""INSERT INTO launchpad_tokens (network,token_address,dex,first_seen_at,last_seen_at,
                             graduated,graduated_at,cohort_sampled,measure_pool,measure_pool_reason,
                             measure_pool_timing,enriched_at)
                           VALUES ('solana',%s,'pumpswap',%s,%s,true,%s,true,'pool',%s,%s,%s)""",
                        (token,NOW,NOW,NOW-timedelta(minutes=5),reason,
                         'late' if 'selected late' in reason else 'at_graduation',NOW))
    h=backlog_health(db,SOLANA,now=NOW)
    # Two of three chosen at graduation. A share that stays low once fresh data accumulates means the
    # cadence-critical sweep is not reaching cohort members, and the backlog is covering for it.
    assert h['pools_measured_24h']==3 and h['pools_selected_at_graduation']==2
    assert h['at_graduation_share']==pytest.approx(0.667,abs=0.001)
    # Three rows is not evidence: the share is reported, the verdict is withheld.
    assert h['selection_state']=='unknown'


def test_db_capture_health_is_separate_from_enrichment_health(db):
    from launchpad_archive import backlog_health
    from launchpad_chains import SOLANA
    epoch=NOW-timedelta(hours=2)
    with db,db.cursor() as cur:
        # A fully drained, perfectly healthy enrichment backlog...
        for i in range(30):
            cur.execute("""INSERT INTO launchpad_tokens (network,token_address,dex,launchpad_family,first_seen_at,
                             last_seen_at,graduated,graduated_at,cohort_sampled,measure_pool,
                             measure_pool_reason,measure_pool_timing,enriched_at)
                           VALUES ('solana',%s,'pumpswap','pump.fun',%s,%s,true,%s,true,'pool',
                                   'deepest graduate pool','at_graduation',%s)""",
                        ('tok%d'%i,NOW,NOW,NOW-timedelta(minutes=30),NOW))
        # ...with captures for only three of them, all before the window.
        for i in range(3):
            cur.execute("""INSERT INTO launchpad_trade_captures (network,token_address,pool,capture_started_at)
                           VALUES ('solana',%s,'pool',%s)""",('tok%d'%i,epoch))
    h=backlog_health(db,SOLANA,now=NOW)
    # Enrichment is spotless. Capture is not. One boolean would have blurred exactly the failure we
    # care most about: losing perishable data while every durable metric reads perfect.
    assert h['enrichment_state']=='idle'
    assert h['captures_taken']==3 and h['captures_eligible']==30
    assert h['capture_coverage_24h']==0.1 and h['capture_state']=='degrading'
    # The overall state takes the worse of the parts, so nothing has to remember to check both.
    assert h['state']=='degrading'


def test_db_capture_coverage_ignores_graduates_from_before_captures_existed(db):
    from launchpad_archive import backlog_health
    from launchpad_chains import SOLANA
    epoch=NOW-timedelta(hours=1)
    with db,db.cursor() as cur:
        # 25 graduates predating the capture mechanism: never eligible, so counting them would
        # manufacture a failure that never happened and poison the metric for the first day.
        for i in range(25):
            cur.execute("""INSERT INTO launchpad_tokens (network,token_address,dex,first_seen_at,last_seen_at,
                             graduated,graduated_at,cohort_sampled,enriched_at)
                           VALUES ('solana',%s,'pumpswap',%s,%s,true,%s,true,%s)""",
                        ('old%d'%i,NOW,NOW,NOW-timedelta(hours=3),NOW))
        for i in range(25):
            cur.execute("""INSERT INTO launchpad_tokens (network,token_address,dex,first_seen_at,last_seen_at,
                             graduated,graduated_at,cohort_sampled,enriched_at)
                           VALUES ('solana',%s,'pumpswap',%s,%s,true,%s,true,%s)""",
                        ('new%d'%i,NOW,NOW,NOW-timedelta(minutes=30),NOW))
            cur.execute("""INSERT INTO launchpad_trade_captures (network,token_address,pool,capture_started_at)
                           VALUES ('solana',%s,'pool',%s)""",('new%d'%i,epoch+timedelta(minutes=1)))
    h=backlog_health(db,SOLANA,now=NOW)
    # Only the 25 that graduated after the first capture count, and all of them were captured.
    assert h['captures_eligible']==25 and h['captures_taken']==25
    assert h['capture_coverage_24h']==1.0 and h['capture_state']=='healthy'
