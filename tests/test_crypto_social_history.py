from datetime import timedelta
import pytest
from test_crypto_social_pilot import db
from crypto_social_history import START,END,LIMIT,setup,reserve,collect_history


def test_late_july_range_cost():
    assert (END-START).days==51
    assert (END-START).days*4*300==61200<LIMIT


def test_db_history_keeps_live_budget_and_reuses_windows(db):
    setup(db);setup(db)
    with db,db.cursor() as c:
        c.execute('SELECT count(*) FROM crypto_social_history_windows');assert c.fetchone()[0]==204
        c.execute('SELECT reserved_credits FROM crypto_social_pilot');assert c.fetchone()[0]==0
    rid,w=reserve(db)
    assert w[1]==START
    with db,db.cursor() as c:
        c.execute('SELECT reserved_credits FROM crypto_social_history_campaign');assert c.fetchone()[0]==300
        c.execute('SELECT reserved_credits FROM crypto_social_pilot');assert c.fetchone()[0]==0
    with pytest.raises(RuntimeError):reserve(db)


def test_db_history_budget_survives_restart(db):
    setup(db)
    with db,db.cursor() as c:c.execute('UPDATE crypto_social_history_campaign SET reserved_credits=75000')
    setup(db)
    assert reserve(db) is None


def test_db_live_pilot_never_selects_historical_windows(db):
    from crypto_social_pilot import reserve as live_reserve
    setup(db)
    _,w=live_reserve(db)
    assert w[1]>=END-timedelta(days=14)  # fixture's initial Sep 1 date is retained


def test_db_history_timeout_is_not_retried(db):
    setup(db);calls=[]
    def fail(*args,**kwargs):calls.append(1);raise TimeoutError()
    with pytest.raises(RuntimeError):collect_history(db,'fake',40,fail)
    assert len(calls)==1
    with pytest.raises(RuntimeError):reserve(db)


def test_db_history_saves_evidence_and_skips_completed_coverage(db):
    setup(db)
    class Response:
        status_code=200
        def json(self):return {'tweets':[{'id':'123','author':{'id':'456','userName':'early'},
            'createdAt':(START+timedelta(hours=1)).isoformat(),'text':'$ZCAT'}],'has_next_page':False}
    assert collect_history(db,'fake',1,lambda *a,**k:Response())==1
    with db,db.cursor() as c:
        c.execute('SELECT count(*) FROM crypto_social_posts');assert c.fetchone()[0]==1
    _,w=reserve(db)
    assert w[1]==START+timedelta(hours=6)


def test_db_targeted_slots_keep_baseline_coverage_and_page_limit(db):
    setup(db)
    focus={'start':START+timedelta(days=30),'end':START+timedelta(days=32)}
    calls=[]
    class Response:
        status_code=200
        def json(self):return {'tweets':[],'has_next_page':False}
    def fetch(url,params,**kwargs):calls.append(params['query']);return Response()
    assert collect_history(db,'fake',8,fetch,focus=focus)==8
    assert f'since_time:{int(focus["start"].timestamp())}' in calls[0]
    assert f'since_time:{int(START.timestamp())}' in calls[3]
    assert f'since_time:{int((START+timedelta(hours=6)).timestamp())}' in calls[7]
    with db,db.cursor() as c:
        c.execute('UPDATE crypto_social_windows SET pages=3')
        c.execute('SELECT reserved_credits FROM crypto_social_history_campaign');assert c.fetchone()[0]==2400
    assert reserve(db,focus=focus) is None


def test_db_focus_uses_complete_consecutive_prices_and_keeps_market_reference(db):
    from crypto_market_history import setup as market_setup,save,normalize
    from test_crypto_market_history import source,candles,NOW
    from crypto_social_history import choose_focus,start_batch
    setup(db);market_setup(db)
    raw=candles('2026-09-01',1)
    raw['data']['attributes']['ohlcv_list']+=candles('2026-09-03',4)['data']['attributes']['ohlcv_list']
    save(db,source(),raw,normalize(raw,'ohlcv',NOW),'https://example.test',NOW)
    with db,db.cursor() as c:c.execute('UPDATE crypto_market_sources SET is_default=true')
    assert choose_focus(db) is None  # no consecutive observations
    raw=candles('2026-09-02',2)
    fid=save(db,source(),raw,normalize(raw,'ohlcv',NOW),'https://example.test',NOW)
    focus=choose_focus(db)
    assert focus['start'].isoformat().startswith('2026-08-26')
    assert focus['end'].isoformat().startswith('2026-09-05')
    assert focus['fetch_id']==fid
    batch=start_batch(db,focus,8)
    with db,db.cursor() as c:
        c.execute('SELECT market_fetch_id,max_requests FROM crypto_social_history_batches WHERE id=%s',(batch,))
        assert c.fetchone()==(fid,8)
