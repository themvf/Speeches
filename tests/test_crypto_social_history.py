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
    with db,db.cursor() as c:c.execute('UPDATE crypto_social_history_campaign SET reserved_credits=150000')
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


def test_db_expanded_priority_fills_focus_then_other_gaps_and_respects_page_limit(db):
    setup(db)
    focus={'start':START+timedelta(days=30),'end':START+timedelta(days=32)}
    calls=[]
    class Response:
        status_code=200
        def json(self):return {'tweets':[],'has_next_page':False}
    def fetch(url,params,**kwargs):calls.append(params['query']);return Response()
    assert collect_history(db,'fake',8,fetch,focus=focus)==8
    assert f'since_time:{int(focus["start"].timestamp())}' in calls[0]
    assert f'since_time:{int((focus["start"]+timedelta(hours=18)).timestamp())}' in calls[3]
    assert f'since_time:{int((focus["start"]+timedelta(hours=42)).timestamp())}' in calls[7]
    _,window=reserve(db,focus=focus)
    assert window[1]==START
    with db,db.cursor() as c:c.execute("UPDATE crypto_social_requests SET status='saved' WHERE status='reserved'")
    with db,db.cursor() as c:
        c.execute('UPDATE crypto_social_windows SET pages=8')
        c.execute('SELECT reserved_credits FROM crypto_social_history_campaign');assert c.fetchone()[0]==2700
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
    assert focus['start'].isoformat().startswith('2026-08-28')
    assert focus['end'].isoformat().startswith('2026-09-05')
    assert focus['fetch_id']==fid
    batch=start_batch(db,focus,8)
    with db,db.cursor() as c:
        c.execute('SELECT market_fetch_id,max_requests FROM crypto_social_history_batches WHERE id=%s',(batch,))
        assert c.fetchone()==(fid,8)


def test_db_unsearched_periods_precede_deeper_focus_pages(db):
    setup(db)
    focus={'start':START+timedelta(days=30),'end':START+timedelta(days=32)}
    with db,db.cursor() as c:
        c.execute("UPDATE crypto_social_windows SET pages=1,status='partial' WHERE start_at>=%s AND start_at<%s",(focus['start'],focus['end']))
    _,window=reserve(db,focus=focus)
    assert window[1]==START


def test_db_expanded_batch_stops_at_existing_campaign_ceiling(db):
    setup(db)
    with db,db.cursor() as c:c.execute('UPDATE crypto_social_history_campaign SET reserved_credits=149700')
    class Response:
        status_code=200
        def json(self):return {'tweets':[],'has_next_page':False}
    assert collect_history(db,'fake',80,lambda *a,**k:Response())==1
    with db,db.cursor() as c:
        c.execute('SELECT reserved_credits FROM crypto_social_history_campaign');assert c.fetchone()[0]==150000


def test_db_upgrade_preserves_spend_and_coverage(db):
    setup(db)
    with db,db.cursor() as c:
        c.execute("""ALTER TABLE crypto_social_history_campaign
            DROP CONSTRAINT crypto_social_history_campaign_credit_limit_check,
            DROP CONSTRAINT crypto_social_history_campaign_reserved_credits_check;
            UPDATE crypto_social_history_campaign SET credit_limit=75000,reserved_credits=38400;
            ALTER TABLE crypto_social_history_campaign ALTER COLUMN credit_limit SET DEFAULT 75000;
            ALTER TABLE crypto_social_history_campaign
            ADD CONSTRAINT crypto_social_history_campaign_credit_limit_check CHECK(credit_limit=75000),
            ADD CONSTRAINT crypto_social_history_campaign_reserved_credits_check CHECK(reserved_credits BETWEEN 0 AND 75000);
            UPDATE crypto_social_windows SET pages=2,cursor='saved-cursor',status='partial';""")
    setup(db);setup(db)
    with db,db.cursor() as c:
        c.execute('SELECT credit_limit,reserved_credits FROM crypto_social_history_campaign')
        assert c.fetchone()==(150000,38400)
        c.execute("SELECT count(*) FROM crypto_social_history_windows h JOIN crypto_social_windows w ON w.id=h.window_id WHERE w.pages=2 AND w.cursor='saved-cursor'")
        assert c.fetchone()[0]==204
        c.execute('SELECT reserved_credits FROM crypto_social_pilot');assert c.fetchone()[0]==0
    import psycopg2
    with pytest.raises(psycopg2.IntegrityError):
        with db,db.cursor() as c:c.execute('UPDATE crypto_social_history_campaign SET reserved_credits=150001')
    reserve(db)
    with db,db.cursor() as c:
        c.execute('SELECT reserved_credits FROM crypto_social_history_campaign');assert c.fetchone()[0]==38700


def test_db_pons_budget_scope_and_restarts_are_independent(db):
    from crypto_social_history import settings
    setup(db)
    with db,db.cursor() as c:
        c.execute("UPDATE crypto_social_history_campaign SET reserved_credits=150000 WHERE id='zcat-july-2026'")
    campaign,start,end,_=settings('PONS')
    setup(db,'PONS',now=end);setup(db,'PONS',now=end)
    with db,db.cursor() as c:
        c.execute('SELECT count(*) FROM crypto_social_history_windows WHERE campaign_id=%s',(campaign,));assert c.fetchone()[0]==304
        c.execute('SELECT reserved_credits FROM crypto_social_history_campaign WHERE id=%s',(campaign,));assert c.fetchone()[0]==0
    assert reserve(db) is None
    rid,window=reserve(db,coin='PONS')
    assert window[1]==start and '$PONS' in window[3] and '$ZCAT' not in window[3]
    with db,db.cursor() as c:
        c.execute('SELECT parameters FROM crypto_social_requests WHERE id=%s',(rid,));assert c.fetchone()[0]['campaign']==campaign
        c.execute("UPDATE crypto_social_requests SET status='saved' WHERE id=%s",(rid,))
        c.execute('SELECT reserved_credits FROM crypto_social_pilot');assert c.fetchone()[0]==0
        c.execute('UPDATE crypto_social_history_campaign SET reserved_credits=150000 WHERE id=%s',(campaign,))
    setup(db,'PONS',now=end)
    assert reserve(db,coin='PONS') is None


def test_db_pons_does_not_search_future_intervals(db):
    from crypto_social_history import settings
    _,start,_,_=settings('PONS')
    setup(db,'PONS',now=start+timedelta(hours=14))
    with db,db.cursor() as c:
        c.execute("SELECT count(*),max(end_at) FROM crypto_social_windows WHERE coin='PONS'")
        assert c.fetchone()==(2,start+timedelta(hours=12))


def test_db_dpons_daily_windows_independent_cap_and_restart(db):
    from crypto_social_history import settings, DPONS_ADDRESS
    setup(db)
    campaign,start,end,query=settings('DPONS')
    assert DPONS_ADDRESS in query and '$DPONS' in query and 'Diamond Pons' in query
    setup(db,'DPONS',now=end)
    with db,db.cursor() as c:
        c.execute('SELECT count(*) FROM crypto_social_history_windows WHERE campaign_id=%s',(campaign,))
        assert c.fetchone()[0]==52
        c.execute('UPDATE crypto_social_history_campaign SET reserved_credits=149700 WHERE id=%s',(campaign,))
    class Response:
        status_code=200
        def json(self):return {'tweets':[],'has_next_page':False}
    assert collect_history(db,'fake',80,lambda *a,**k:Response(),coin='DPONS')==1
    setup(db,'DPONS',now=end)
    assert reserve(db,coin='DPONS') is None
    with db,db.cursor() as c:
        c.execute('SELECT reserved_credits FROM crypto_social_history_campaign WHERE id=%s',(campaign,))
        assert c.fetchone()[0]==150000
        c.execute("SELECT reserved_credits FROM crypto_social_history_campaign WHERE id='zcat-july-2026'")
        assert c.fetchone()[0]==0
        c.execute('SELECT reserved_credits FROM crypto_social_pilot')
        assert c.fetchone()[0]==0
