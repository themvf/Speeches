from datetime import timedelta
from test_crypto_social_pilot import db
from crypto_social_catchup import START,END,cutoff,setup
from crypto_social_pilot import collect,reserve

def test_cutoff_never_searches_future():
    now=START+timedelta(hours=14)
    assert cutoff(now)==now
    assert cutoff(END+timedelta(days=1))==END

def test_db_catchup_preserves_budgets_and_appends_without_overlap(db):
    end=START+timedelta(hours=14)
    setup(db,end);setup(db,end)
    with db,db.cursor() as c:
        c.execute("SELECT start_at,end_at FROM crypto_social_windows WHERE coin='ZCAT' AND start_at>=%s ORDER BY start_at",(START,))
        rows=c.fetchall();assert len(rows)==3 and rows[-1][1]==end
        c.execute('SELECT reserved_credits,end_at FROM crypto_social_pilot');before=c.fetchone()
    later=end+timedelta(hours=1)
    setup(db,later)
    with db,db.cursor() as c:
        c.execute("SELECT start_at,end_at FROM crypto_social_windows WHERE coin='ZCAT' AND start_at>=%s ORDER BY start_at",(START,))
        rows=c.fetchall();assert len(rows)==4 and rows[-1]==(end,later)
        c.execute('SELECT reserved_credits,end_at FROM crypto_social_pilot');assert c.fetchone()==before
    class Response:
        status_code=200
        def json(self):return {'tweets':[],'has_next_page':False}
    calls=[]
    def fetch(url,params,**kwargs):calls.append(params['query']);return Response()
    assert collect(db,'fake',40,fetch,max_pages=8,scope=(START,later,'ZCAT'))==4
    assert all('ZCAT' in q and 'Zcash' not in q for q in calls)
    with db,db.cursor() as c:
        c.execute('SELECT reserved_credits FROM crypto_social_pilot');assert c.fetchone()[0]==1200
        c.execute("SELECT count(*) FROM crypto_social_windows WHERE coin='ZEC' AND start_at>=%s",(START,));assert c.fetchone()[0]==0

def test_db_catchup_honors_existing_search_sublimit(db):
    setup(db,END)
    with db,db.cursor() as c:
        c.execute("INSERT INTO crypto_social_requests(endpoint,status,reserved_credits) VALUES ('search','saved',16800)")
        c.execute('UPDATE crypto_social_pilot SET reserved_credits=16800')
    assert reserve(db,8,(START,END,'ZCAT')) is None
