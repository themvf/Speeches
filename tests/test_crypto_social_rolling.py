from datetime import datetime,timedelta,timezone
import pytest
from test_crypto_social_pilot import db
from crypto_social_rolling import setup,reserve,collect_one,slot,CAMPAIGN,TRACKED

NOW=datetime(2026,9,15,12,17,tzinfo=timezone.utc)

class Response:
    status_code=200
    def __init__(self,tweets=None,more=False,cursor=''):
        self.data={'tweets':tweets or [],'has_next_page':more,'next_cursor':cursor}
    def json(self):return self.data


def test_db_rolling_initial_overlap_and_gap_filling(db):
    assert setup(db,NOW)
    setup(db,NOW)
    with db,db.cursor() as c:
        c.execute('SELECT count(*) FROM crypto_rolling_windows');assert c.fetchone()[0]==8*len(TRACKED)
        c.execute('SELECT sum(credit_limit) FROM crypto_rolling_coins');assert c.fetchone()[0]==30000*len(TRACKED)
    setup(db,NOW+timedelta(hours=12))
    with db,db.cursor() as c:
        c.execute('SELECT count(*) FROM crypto_rolling_windows');assert c.fetchone()[0]==10*len(TRACKED)
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
