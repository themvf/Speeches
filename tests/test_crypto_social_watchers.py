from datetime import datetime,timedelta,timezone
import pytest
from test_crypto_social_pilot import db
from crypto_social_watchers import setup,reserve,collect_one,account_failed_request,ACCOUNTS,slot
NOW=datetime(2026,9,15,12,37,tzinfo=timezone.utc)
A=ACCOUNTS[0]['id']

def response(posts=None,more=False,cursor=''):
    class Response:
        status_code=200
        def json(self):return {'tweets':posts or [],'has_next_page':more,'next_cursor':cursor}
    return lambda *a,**k:Response()

def test_db_windows_resume_with_overlap_and_no_budget_reset(db):
    setup(db,NOW);setup(db,NOW+timedelta(hours=4))
    with db,db.cursor() as c:
        c.execute('SELECT count(*) FROM crypto_watcher_windows');assert c.fetchone()[0]==30
        c.execute('SELECT min(start_at),max(end_at) FROM crypto_watcher_windows');assert c.fetchone()==(slot(NOW)-timedelta(hours=24),slot(NOW)+timedelta(hours=4))
        c.execute('UPDATE crypto_watcher_campaign SET used_credits=150000')
    setup(db,NOW+timedelta(hours=4));assert reserve(db,A,NOW) is None

def test_db_empty_search_cost_and_stable_id_validation(db):
    setup(db,NOW)
    assert collect_one(db,'fake',A,NOW,response())
    assert reserve(db,A,NOW) is None
    with db,db.cursor() as c:
        c.execute('SELECT used_credits FROM crypto_watcher_campaign');assert c.fetchone()[0]==15
    setup(db,NOW+timedelta(hours=2))
    wrong=[{'id':'123','author':{'id':'999','userName':'wrong'},'createdAt':NOW.isoformat(),'text':'test'}]
    with pytest.raises(RuntimeError):collect_one(db,'fake',A,NOW+timedelta(hours=2),response(wrong))
    with pytest.raises(RuntimeError):reserve(db,A,NOW+timedelta(hours=2))

def test_db_pages_dedupe_and_two_request_cap(db):
    setup(db,NOW)
    def post(i):return {'id':str(i),'author':{'id':A,'userName':ACCOUNTS[0]['handle']},'createdAt':(slot(NOW)-timedelta(hours=1)).isoformat(),'text':'$ZCAT whale bought $1000'}
    assert collect_one(db,'fake',A,NOW,response([post(123)],True,'next'))
    assert collect_one(db,'fake',A,NOW,response([post(124)],True,'next2'))
    assert reserve(db,A,NOW) is None
    with db,db.cursor() as c:
        c.execute('SELECT count(*) FROM crypto_watcher_posts');assert c.fetchone()[0]==2
        c.execute('SELECT used_credits FROM crypto_watcher_campaign');assert c.fetchone()[0]==30
        c.execute('SELECT status FROM crypto_watcher_windows WHERE account_id=%s',(A,));assert c.fetchone()[0]=='partial'

def test_db_timeout_retains_reservation_and_expiry_stops(db):
    setup(db,NOW)
    def timeout(*a,**k):raise TimeoutError()
    with pytest.raises(RuntimeError):collect_one(db,'fake',A,NOW,timeout)
    with db,db.cursor() as c:
        c.execute('SELECT used_credits FROM crypto_watcher_campaign');assert c.fetchone()[0]==300
    assert not setup(db,NOW+timedelta(days=31))


def test_db_uncertain_search_records_why_and_operator_reconciles_without_refund(db,capsys):
    setup(db,NOW)
    wrong=[{'id':'123','author':{'id':'999','userName':'wrong'},'createdAt':(slot(NOW)-timedelta(hours=1)).isoformat(),'text':'test'}]
    with pytest.raises(RuntimeError,match='Unexpected author'):collect_one(db,'fake',A,NOW,response(wrong))
    with db,db.cursor() as c:
        c.execute("SELECT id,error FROM crypto_social_requests WHERE status='uncertain'");rid,error=c.fetchone()
    assert error=='watcher_search_failed: ValueError: Unexpected author; check saved handle'
    with pytest.raises(RuntimeError):reserve(db,ACCOUNTS[1]['id'],NOW)  # one uncertain request stops every endpoint
    account_failed_request(db,rid)
    review=capsys.readouterr().out
    assert '"refund": 0' in review and '"data_saved": false' in review and 'from:' in review and 'Unexpected author' in review
    with db,db.cursor() as c:
        c.execute('SELECT status,estimated_credits FROM crypto_social_requests WHERE id=%s',(rid,));assert c.fetchone()==('failed_charged',300)
        c.execute('SELECT used_credits FROM crypto_watcher_campaign');assert c.fetchone()[0]==300  # never refunded
        c.execute('SELECT w.status,w.pages FROM crypto_watcher_windows w JOIN crypto_watcher_calls c ON c.window_id=w.id WHERE c.request_id=%s',(rid,))
        assert c.fetchone()==('pending',0)  # searched again later
    assert reserve(db,ACCOUNTS[1]['id'],NOW)  # collection is unblocked
    account_failed_request(db,rid)  # replay is a no-op
    assert 'already_reconciled' in capsys.readouterr().out


def test_db_only_uncertain_watcher_searches_can_be_reconciled(db):
    setup(db,NOW)
    assert collect_one(db,'fake',A,NOW,response())
    with db,db.cursor() as c:
        c.execute("SELECT id FROM crypto_social_requests WHERE status='saved'");saved=c.fetchone()[0]
    with pytest.raises(ValueError,match='Only an uncertain'):account_failed_request(db,saved)
    with pytest.raises(ValueError,match='Not a watcher'):account_failed_request(db,saved+999)
