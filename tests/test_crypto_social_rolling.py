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
        c.execute('SELECT count(*) FROM crypto_rolling_windows');assert c.fetchone()[0]==40
        c.execute('SELECT sum(credit_limit) FROM crypto_rolling_coins');assert c.fetchone()[0]==150000
    setup(db,NOW+timedelta(hours=12))
    with db,db.cursor() as c:
        c.execute('SELECT count(*) FROM crypto_rolling_windows');assert c.fetchone()[0]==50
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
    assert collect_one(db,'fake','DPONS',NOW,'profiles',lambda *a,**k:Profiles())
    assert reserve(db,'DPONS',NOW+timedelta(hours=6),'profiles') is None
    with db,db.cursor() as c:
        c.execute("SELECT used_credits FROM crypto_rolling_coins WHERE coin='DPONS'");assert c.fetchone()[0]==33
