from datetime import datetime,timedelta,timezone
import pytest
from test_crypto_social_pilot import db
from crypto_social_rolling import setup,reserve,collect_one,slot,cadence,ceiling,PAGES_PER_RUN,CAMPAIGN,TRACKED,REGISTRY
def windows(hours):return sum(hours//cadence(c)+1 for c in TRACKED)  # windows from anchor-hours to the anchor inclusive

NOW=datetime(2026,9,15,12,17,tzinfo=timezone.utc)
# setup_origin searches from originFrom to the first live window, so a coin whose originFrom is later (ASKR,
# 2026-09-15) gets none at this fixed clock. Counting it anyway failed this gate from 2026-09-19 and kept the
# rolling collector from running for two weeks.
FIRST_LIVE=slot(NOW)-timedelta(hours=42)
ORIGINS=sum(1 for c in REGISTRY.values()
            if c.get('originFrom') and datetime.fromisoformat(c['originFrom']).replace(tzinfo=timezone.utc)<FIRST_LIVE)

class Response:
    status_code=200
    def __init__(self,tweets=None,more=False,cursor=''):
        self.data={'tweets':tweets or [],'has_next_page':more,'next_cursor':cursor}
    def json(self):return self.data


def test_db_rolling_initial_overlap_and_gap_filling(db):
    assert setup(db,NOW)
    setup(db,NOW)
    with db,db.cursor() as c:
        c.execute('SELECT count(*) FROM crypto_rolling_windows');assert c.fetchone()[0]==windows(42)+ORIGINS
        c.execute('SELECT sum(credit_limit) FROM crypto_rolling_coins');assert c.fetchone()[0]==sum(ceiling(c) for c in TRACKED)
    setup(db,NOW+timedelta(hours=12))
    with db,db.cursor() as c:
        c.execute('SELECT count(*) FROM crypto_rolling_windows');assert c.fetchone()[0]==windows(54)+ORIGINS
    _,window,_,_=reserve(db,'STANDARD',NOW+timedelta(hours=12))
    assert window[2]==slot(NOW)+timedelta(hours=12)
    assert window[2]-window[1]==timedelta(hours=7)
    assert '0x88ad8ddf1e3898412146a534538d418c6f8a9062' in window[3]


def test_db_rolling_reconciles_known_charges_and_deduplicates_overlap(db):
    setup(db,NOW)
    tweet={'id':'123','author':{'id':'456','userName':'tester'},'text':'$STANDARD',
           'createdAt':(slot(NOW)-timedelta(hours=6,minutes=30)).isoformat()}
    for _ in range(2):assert collect_one(db,'fake','STANDARD',NOW,fetch=lambda *a,**k:Response([tweet]))
    with db,db.cursor() as c:
        c.execute("SELECT used_credits FROM crypto_rolling_coins WHERE coin='STANDARD'");assert c.fetchone()[0]==30
        c.execute('SELECT count(*) FROM crypto_social_posts');assert c.fetchone()[0]==1
        c.execute('SELECT count(*) FROM crypto_social_matches');assert c.fetchone()[0]==2
        c.execute('SELECT reserved_credits FROM crypto_social_pilot');assert c.fetchone()[0]==0


def test_db_rolling_uncertain_retains_charge_and_blocks_other_coins(db):
    setup(db,NOW)
    def fail(*a,**k):raise TimeoutError()
    with pytest.raises(RuntimeError):collect_one(db,'fake','DPONS',NOW,fetch=fail)
    setup(db,NOW)
    with db,db.cursor() as c:
        c.execute("SELECT used_credits FROM crypto_rolling_coins WHERE coin='DPONS'");assert c.fetchone()[0]==300
    with pytest.raises(RuntimeError):reserve(db,'ZEC',NOW)


def test_db_rolling_run_bound_budget_and_expiry_survive_restart(db):
    setup(db,NOW)
    for _ in range(4):assert collect_one(db,'fake','PONS',NOW,fetch=lambda *a,**k:Response())
    setup(db,NOW)
    assert reserve(db,'PONS',NOW) is None
    with db,db.cursor() as c:
        c.execute("UPDATE crypto_rolling_coins SET used_credits=30000 WHERE coin='PONS'")
    assert reserve(db,'PONS',NOW+timedelta(hours=6)) is None
    assert not setup(db,NOW+timedelta(days=31))
    assert reserve(db,'ZCAT',NOW+timedelta(days=31)) is None


def test_db_rolling_profiles_only_once_per_day(db):
    setup(db,NOW)
    tweet={'id':'123','author':{'id':'456','userName':'tester'},'text':'$DPONS','createdAt':(slot(NOW)-timedelta(hours=1)).isoformat()}
    assert collect_one(db,'fake','DPONS',NOW,fetch=lambda *a,**k:Response([tweet]))
    class Profiles:
        status_code=200
        def json(self):return {'users':[{'id':'456','userName':'tester','followers':500,'description':'DPONS'}]}
    from crypto_voice_research import snapshot
    # A cohort requires exact asset identity, not a bare ticker in an empty test database.
    with db,db.cursor() as c:c.execute("UPDATE crypto_social_posts SET text=%s",('0x0e6d1ebb33f3b8f2d09bacf3b1a1d5c581110c33',))
    snapshot(db,'DPONS',NOW)
    assert collect_one(db,'fake','DPONS',NOW,'profiles',lambda *a,**k:Profiles())
    assert reserve(db,'DPONS',NOW+timedelta(hours=6),'profiles') is None
    with db,db.cursor() as c:
        c.execute("SELECT used_credits FROM crypto_rolling_coins WHERE coin='DPONS'");assert c.fetchone()[0]==33


def test_db_continuation_gets_budget_before_more_fresh_windows(db):
    setup(db,NOW)
    with db,db.cursor() as c:
        c.execute("UPDATE crypto_social_windows SET pages=1,status='partial',cursor='next' WHERE coin='ZCAT'")
    # First call can resume when no untouched window remains.
    rid,w,_,_=reserve(db,'ZCAT',NOW)
    with db,db.cursor() as c:c.execute("UPDATE crypto_social_requests SET status='saved' WHERE id=%s",(rid,))
    _,next_window,_,_=reserve(db,'ZCAT',NOW)
    assert next_window[0]==w[0]


def test_db_focus_windows_are_bounded_contract_searches_and_do_not_advance_live_watermark(db):
    from crypto_social_rolling import setup_focus
    setup(db,NOW)
    with db,db.cursor() as c:
        c.execute("INSERT INTO crypto_social_accounts(id,handle) VALUES ('123','early')")
        c.execute("INSERT INTO crypto_social_posts(id,author_id,text,posted_at,kind,url) VALUES ('456','123',%s,%s,'original','https://x.com/i/status/456')",(TRACKED['PONS'][2],NOW-timedelta(days=10)))
    setup_focus(db,'PONS',NOW);setup_focus(db,'PONS',NOW)
    with db,db.cursor() as c:
        c.execute('SELECT count(*) FROM crypto_voice_focus');assert c.fetchone()[0]==30
        c.execute('SELECT min(end_at-start_at),max(end_at-start_at) FROM crypto_social_windows w JOIN crypto_voice_focus f ON f.window_id=w.id');assert c.fetchone()==(timedelta(hours=1),timedelta(hours=1))
    setup(db,NOW+timedelta(hours=6))
    with db,db.cursor() as c:
        c.execute("SELECT max(end_at) FROM crypto_social_windows WHERE coin='PONS'");assert c.fetchone()[0]==slot(NOW)+timedelta(hours=6)


def test_db_weekly_cohort_is_immutable_and_evaluation_keeps_inactive_accounts(db):
    from crypto_voice_research import snapshot,evaluate
    setup(db,NOW)
    with db,db.cursor() as c:
        c.execute("INSERT INTO crypto_social_accounts(id,handle,followers) VALUES ('123','early',500)")
        c.execute("INSERT INTO crypto_social_posts(id,author_id,text,posted_at,kind,url) VALUES ('456','123',%s,%s,'original','https://x.com/i/status/456')",(TRACKED['PONS'][2],NOW-timedelta(hours=1)))
        c.execute("INSERT INTO crypto_social_matches SELECT '456',min(id) FROM crypto_social_windows WHERE coin='PONS'")
    ids=snapshot(db,'PONS',NOW);assert ids==['123']
    assert snapshot(db,'PONS',NOW+timedelta(days=1))==ids
    evaluate(db,NOW+timedelta(days=8))
    with db,db.cursor() as c:
        c.execute("SELECT evaluation FROM crypto_voice_snapshots WHERE coin='PONS'");e=c.fetchone()[0]
        assert e['groups']['Mixed cohort']['accounts']==1
        assert e['groups']['Mixed cohort']['active']==0


def test_db_reviewed_profile_failure_keeps_maximum_charge_and_never_claims_data(db):
    from crypto_social_rolling import account_failed_profile
    setup(db,NOW)
    with db,db.cursor() as c:
        c.execute("INSERT INTO crypto_voice_snapshots VALUES ('ZCAT',%s,%s,'test',%s,'{}','{}',NULL)",(NOW.date(),NOW,'[{"id":"123","handle":"test","role":"Audience"}]'))
    rid,_,_,credits=reserve(db,'ZCAT',NOW,'profiles')
    with db,db.cursor() as c:c.execute("UPDATE crypto_social_requests SET status='uncertain' WHERE id=%s",(rid,))
    account_failed_profile(db,rid);account_failed_profile(db,rid)
    with db,db.cursor() as c:
        c.execute('SELECT status,estimated_credits FROM crypto_social_requests WHERE id=%s',(rid,));assert c.fetchone()==('failed_charged',credits)
        c.execute("SELECT used_credits FROM crypto_rolling_coins WHERE coin='ZCAT'");assert c.fetchone()[0]==credits


def test_db_recover_archived_profile_without_network_or_refund(db):
    import json
    from crypto_social_rolling import recover_profile
    setup(db,NOW)
    with db,db.cursor() as c:
        c.execute("INSERT INTO crypto_voice_snapshots VALUES ('ZCAT',%s,%s,'test',%s,'{}','{}',NULL)",(NOW.date(),NOW,'[{"id":"123","handle":"test","role":"Audience"}]'))
    rid,_,_,credits=reserve(db,'ZCAT',NOW,'profiles')
    with db,db.cursor() as c:
        c.execute("UPDATE crypto_social_requests SET status='uncertain',parameters=parameters || jsonb_build_object('provider_response',%s::jsonb) WHERE id=%s",(json.dumps({'users':[{'id':'123','followers':456}]}),rid))
    recover_profile(db,rid);recover_profile(db,rid)
    with db,db.cursor() as c:
        c.execute('SELECT status,accepted_count,estimated_credits FROM crypto_social_requests WHERE id=%s',(rid,))
        assert c.fetchone()==('saved',1,credits)
        c.execute('SELECT count(*) FROM crypto_social_profile_history WHERE request_id=%s',(rid,));assert c.fetchone()[0]==1
        c.execute("SELECT used_credits FROM crypto_rolling_coins WHERE coin='ZCAT'");assert c.fetchone()[0]==credits


def test_db_origin_window_is_one_bounded_contract_search_per_new_coin(db):
    assert setup(db,NOW);setup(db,NOW+timedelta(hours=12))
    with db,db.cursor() as c:
        c.execute("SELECT w.coin,w.start_at,w.end_at,w.query FROM crypto_origin_windows o JOIN crypto_social_windows w ON w.id=o.window_id ORDER BY w.coin")
        rows=c.fetchall()
    assert len(rows)==ORIGINS and all(r[0] in REGISTRY and REGISTRY[r[0]].get('originFrom') for r in rows)
    for coin,start,end,query in rows:
        assert start==datetime.fromisoformat(REGISTRY[coin]['originFrom']).replace(tzinfo=timezone.utc)
        assert end==FIRST_LIVE and query.startswith('"'+REGISTRY[coin]['address']+'"') and '$' not in query
    # Fresh live windows keep filling forward from the live start, never from the origin window's end.
    with db,db.cursor() as c:
        c.execute("SELECT count(*) FROM crypto_social_windows w JOIN crypto_rolling_windows r ON r.window_id=w.id WHERE w.coin=%s AND w.end_at>=%s AND NOT EXISTS(SELECT 1 FROM crypto_origin_windows o WHERE o.window_id=w.id)",(rows[0][0],slot(NOW)-timedelta(hours=42)))
        assert c.fetchone()[0]==54//cadence(rows[0][0])+1


def test_db_hourly_coins_get_hourly_windows_two_pages_and_the_larger_ceiling(db):
    hourly=[c for c in TRACKED if cadence(c)==1];assert set(hourly)=={'ZCAT','ZEC','KNOTS'}
    coin=hourly[0];assert ceiling(coin)==450000
    setup(db,NOW)
    with db,db.cursor() as c:
        c.execute('SELECT credit_limit FROM crypto_rolling_coins WHERE coin=%s',(coin,));assert c.fetchone()[0]==450000
        c.execute('SELECT max(end_at),max(end_at-start_at) FROM crypto_social_windows w JOIN crypto_rolling_windows r ON r.window_id=w.id WHERE w.coin=%s AND NOT EXISTS(SELECT 1 FROM crypto_origin_windows o WHERE o.window_id=w.id)',(coin,))
        assert c.fetchone()==(NOW.replace(minute=0),timedelta(hours=2))
    # Hourly coins take the same pages per run as everyone else; the hour-long slot is the only
    # difference. Fewer pages per run would cut throughput, because runs are the scarce resource.
    for _ in range(PAGES_PER_RUN):assert collect_one(db,'fake',coin,NOW,fetch=lambda *a,**k:Response())
    assert reserve(db,coin,NOW) is None, 'the slot cap still binds within one hour'
    assert reserve(db,coin,NOW+timedelta(hours=1)) is not None, 'the next hour is a new slot'
    # A six-hourly coin promoted mid-campaign keeps its spend and is raised, never lowered.
    with db,db.cursor() as c:
        c.execute("UPDATE crypto_rolling_coins SET credit_limit=30000,used_credits=20000 WHERE coin=%s",(coin,))
    setup(db,NOW)
    with db,db.cursor() as c:
        c.execute('SELECT credit_limit,used_credits FROM crypto_rolling_coins WHERE coin=%s',(coin,));assert c.fetchone()==(450000,20000)


def test_registry_credit_ceiling_raises_a_six_hour_coin(db):
    assert cadence('BACKPACK')==6 and ceiling('BACKPACK')==450000 and ceiling('PERSPAD')==30000
    setup(db,NOW)
    with db,db.cursor() as c:
        c.execute("SELECT coin,credit_limit FROM crypto_rolling_coins WHERE coin IN ('BACKPACK','PERSPAD') ORDER BY coin")
        assert c.fetchall()==[('BACKPACK',450000),('PERSPAD',30000)]


def test_db_changed_query_requeries_live_windows_since_the_registry_date(db,monkeypatch):
    import crypto_social_rolling as rolling
    coin='BACKPACK';name,query,address,note=rolling.TRACKED[coin]
    old='("$BACKPACK" OR "'+address+'")'
    monkeypatch.setitem(rolling.TRACKED,coin,(name,old,address,note))
    assert setup(db,NOW)
    live="""SELECT w.id,w.start_at,w.end_at,w.query,w.status FROM crypto_social_windows w JOIN crypto_rolling_windows r ON r.window_id=w.id
        WHERE w.coin=%s AND NOT EXISTS(SELECT 1 FROM crypto_origin_windows o WHERE o.window_id=w.id) ORDER BY w.end_at,w.start_at"""
    with db,db.cursor() as c:
        c.execute(live,(coin,));before=c.fetchall()
        c.execute("UPDATE crypto_social_windows SET status='search_exhausted',pages=1 WHERE id=%s",(before[-1][0],))
        c.execute('SELECT count(*) FROM crypto_origin_windows');origins=c.fetchone()[0]
    assert before and all(w[3].startswith(old+' ') for w in before)
    since=slot(NOW)-timedelta(hours=30)
    monkeypatch.setitem(rolling.TRACKED,coin,(name,query,address,note))
    monkeypatch.setitem(REGISTRY[coin],'requerySince',since.isoformat())
    setup(db,NOW);setup(db,NOW)  # a replay adds nothing
    with db,db.cursor() as c:
        c.execute(live,(coin,));after=c.fetchall()
        c.execute('SELECT count(*) FROM crypto_origin_windows');assert c.fetchone()[0]==origins
    by_id={w[0]:w for w in after}
    replaced=[w for w in before if w[2]>since]
    assert replaced and len(replaced)<len(before)
    for wid,start,end,_,_ in before:
        status=by_id[wid][4]
        if end<=since:assert status=='pending'  # before requerySince: untouched
        elif wid==before[-1][0]:assert status=='search_exhausted'  # finished evidence is kept
        else:assert status=='retired_query'
    new=[w for w in after if w[3].startswith(query+' ')]
    assert sorted((w[1],w[2]) for w in new)==sorted((s-timedelta(minutes=1),e) for _,s,e,_,_ in replaced)
    assert all(w[4]=='pending' for w in new)
    _,window,_,_=reserve(db,coin,NOW)
    assert window[3].startswith(query+' ')  # collection now uses the current query
