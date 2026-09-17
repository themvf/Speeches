from datetime import datetime, timedelta, timezone
import pytest
from test_crypto_social_pilot import db, tweet, START, END
from crypto_social_profiles import matches, profile, save_profile
from crypto_social_tracking import (seed_candidates, reserve_request, request, save_profiles,
                                    daily_profiles, save_posts, discover)


def test_bio_terms_boundaries_unicode_and_phrases():
    assert matches('Builder | #ZCAT and $ZEC | Anonymous   Cat') == [
        ('ZCAT','zcat'),('ZCAT','anonymous cat'),('ZEC','zec')]
    assert matches('ＺＣＡＳＨ') == [('ZEC','zcash')]
    assert matches('zcatcher zcashier zecology') == []
    assert matches('Zcash critic') == [('ZEC','zcash')]  # matching isn't endorsement


def test_unavailable_profile_does_not_become_zero():
    p=profile({'id':'90071992547409930','followers':99,'description':'zcash','unavailable':True})
    assert p['followers'] is None and p['bio'] is None and not p['available']
    assert profile({'id':'2','description':''})['bio']==''
    assert profile({'id':'2'})['bio'] is None


def start_tracking(db):
    with db,db.cursor() as c:
        c.execute("INSERT INTO crypto_social_tracking(id) VALUES ('zcat-zec-v1')")


def test_db_shared_budget_deduplication_and_expiry(db):
    start_tracking(db)
    rid=reserve_request(db,'profile','p1',{},18,'profiles')
    with pytest.raises(RuntimeError): reserve_request(db,'profile','p2',{},18,'profiles')
    with db,db.cursor() as c:
        c.execute("UPDATE crypto_social_requests SET status='saved' WHERE id=%s",(rid,))
    assert reserve_request(db,'profile','p1',{},18,'profiles') is None
    with db,db.cursor() as c:
        c.execute('UPDATE crypto_social_pilot SET reserved_credits=49990')
    assert reserve_request(db,'profile','p2',{},18,'profiles') is None
    with db,db.cursor() as c:
        c.execute('UPDATE crypto_social_pilot SET reserved_credits=18')
        c.execute("UPDATE crypto_social_tracking SET end_at=now()-interval '1 day'")
    assert reserve_request(db,'profile','p2',{},18,'profiles') is None


def test_db_endpoint_allocation_cannot_take_profile_budget(db):
    start_tracking(db)
    for n in range(3):
        rid=reserve_request(db,'search',str(n),{},3000,'enrichment')
        assert rid
        with db,db.cursor() as c:
            c.execute("UPDATE crypto_social_requests SET status='saved' WHERE id=%s",(rid,))
    assert reserve_request(db,'search','4',{},300,'enrichment') is None
    assert reserve_request(db,'profile','5',{},18,'profiles') is not None


def test_db_failed_batch_never_retries(db):
    start_tracking(db);calls=[]
    def failure(*args,**kwargs): calls.append(1);raise TimeoutError()
    with pytest.raises(RuntimeError):
        request(db,'fake','profile','p1',{},18,'profiles',lambda *args:None,failure)
    with pytest.raises(RuntimeError):
        request(db,'fake','profile','p1',{},18,'profiles',lambda *args:None,failure)
    assert len(calls)==1


def test_db_history_rename_missing_and_bio_matches(db):
    start_tracking(db)
    for n,raw in enumerate([
        {'id':'99','userName':'old','followers':100,'description':'#ZCAT builder'},
        {'id':'99','userName':'new','followers':110,'description':'zcash'},
        {'id':'99','unavailable':True},
    ]):
        rid=reserve_request(db,'profile',str(n),{},18,'profiles')
        with db,db.cursor() as c:
            save_profiles(c,rid,{'users':[raw]},['99'])
            c.execute("UPDATE crypto_social_requests SET status='saved' WHERE id=%s",(rid,))
    with db,db.cursor() as c:
        c.execute('SELECT count(*) FROM crypto_social_accounts');assert c.fetchone()[0]==1
        c.execute('SELECT handle,followers FROM crypto_social_accounts');assert c.fetchone()==('new',None)
        c.execute('SELECT count(*) FROM crypto_social_profile_history');assert c.fetchone()[0]==3
        c.execute("SELECT count(*) FROM crypto_social_profile_matches WHERE field='bio'");assert c.fetchone()[0]==2
        c.execute('SELECT followers,bio,available FROM crypto_social_profile_history ORDER BY request_id DESC LIMIT 1')
        assert c.fetchone()==(None,None,False)


def test_db_targets_are_candidates_without_author_posts(db):
    from crypto_social_pilot import reserve, save_page
    rid,w=reserve(db)
    save_page(db,rid,w,{'tweets':[tweet()],'has_next_page':False})
    seed_candidates(db)
    with db,db.cursor() as c:
        c.execute('SELECT account_id,tracked FROM crypto_social_candidates ORDER BY account_id')
        assert c.fetchall()==[('22',True),('33',True)]


def test_db_missing_batch_ids_record_unknown_and_unexpected_roll_back(db):
    start_tracking(db)
    rid=reserve_request(db,'profile','p1',{},36,'profiles')
    with db,db.cursor() as c:
        save_profiles(c,rid,{'users':[{'id':'1','followers':0}]},['1','2'])
        c.execute('SELECT account_id,followers,available FROM crypto_social_profile_history ORDER BY account_id')
        assert c.fetchall()==[('1',0,True),('2',None,False)]
    with pytest.raises(ValueError):
        with db,db.cursor() as c:
            save_profiles(c,rid,{'users':[{'id':'3','followers':999}]},['1'])
    with db,db.cursor() as c:
        c.execute("SELECT count(*) FROM crypto_social_accounts WHERE id='3'");assert c.fetchone()[0]==0


def test_db_growth_uses_real_baselines_and_gaps_stay_unknown(db):
    start_tracking(db)
    now=datetime.now(timezone.utc)
    for n,(days,followers) in enumerate([(14,100),(7,110),(0,130)]):
        rid=reserve_request(db,'profile',str(n),{},18,'profiles')
        with db,db.cursor() as c:
            save_profiles(c,rid,{'users':[{'id':'1','followers':followers,'description':'zcash'}]},['1'])
            c.execute('UPDATE crypto_social_profile_history SET observed_at=%s WHERE request_id=%s',(now-timedelta(days=days),rid))
            c.execute("UPDATE crypto_social_requests SET status='saved' WHERE id=%s",(rid,))
    with db,db.cursor() as c:
        c.execute("SELECT growth_7d,growth_percent_7d,growth_acceleration FROM crypto_social_account_metrics WHERE id='1'")
        net,percent,acceleration=c.fetchone()
        assert net==20 and percent==18.18 and acceleration==pytest.approx(10/7)
        c.execute("DELETE FROM crypto_social_profile_matches WHERE request_id IN (SELECT request_id FROM crypto_social_profile_history WHERE followers=110)")
        c.execute('DELETE FROM crypto_social_profile_history WHERE followers=110')
        c.execute("SELECT growth_7d,growth_acceleration FROM crypto_social_account_metrics WHERE id='1'")
        assert c.fetchone()==(None,None)


def test_db_timeline_coverage_and_deduplication(db):
    start_tracking(db)
    rid=reserve_request(db,'timeline','t1',{},300,'enrichment')
    with db,db.cursor() as c:
        save_posts(c,rid,{'tweets':[tweet()],'has_next_page':True},START,END,account='22')
        c.execute('SELECT status,in_window_posts FROM crypto_social_account_coverage')
        assert c.fetchone()==('capped',1)
        c.execute('SELECT count(*) FROM crypto_social_matches');assert c.fetchone()[0]==1


def test_discovery_refuses_an_unverified_billing_bound():
    with pytest.raises(ValueError): discover(None,'fake',datetime.now(timezone.utc),0)


def test_db_unidentifiable_profile_is_unknown_without_discarding_valid_batch(db):
    start_tracking(db)
    rid=reserve_request(db,'profile','missing-id',{},36,'profiles')
    with db,db.cursor() as c:
        result=save_profiles(c,rid,{'users':[{'id':'1','followers':123},{'unavailable':True}]},['1','2'])
        assert result==(2,1,36)
        c.execute('SELECT account_id,followers,available FROM crypto_social_profile_history ORDER BY account_id')
        assert c.fetchall()==[('1',123,True),('2',None,False)]
