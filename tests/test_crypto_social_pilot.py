from datetime import datetime, timedelta, timezone
import pytest
from crypto_social_pilot import build_windows, normalize_post, validate_page, PAGE_RESERVE, LIMIT, collect

START = datetime(2026,9,1,tzinfo=timezone.utc)
END = START+timedelta(hours=6)

def tweet(**overrides):
    return dict({'id':'90071992547409930','author':{'id':'22','userName':'example'},
       'createdAt':'Tue Sep 01 01:00:00 +0000 2026','text':'$ZCAT',
       'likeCount':4,'isReply':True,'inReplyToUserId':'33',
       'entities':{'user_mentions':[{'id_str':'33'},{'id_str':'33'}]}},**overrides)

def test_identity_and_relationships():
    p=normalize_post(tweet(),START,END)
    assert p['id']=='90071992547409930'
    assert p['edges']==[('33','mention'),('33','reply')]
    assert p['metrics']==[4,None,None,None,None]
    assert not any(kind=='like' for _,kind in p['edges'])

def test_quote_repost_targets_do_not_require_extra_calls():
    p=normalize_post(tweet(quoted_tweet={'author':{'id':'44'}},retweeted_tweet={'author':{'id':'55'}}),START,END)
    assert ('44','quote') in p['edges'] and ('55','repost') in p['edges']

@pytest.mark.parametrize('raw',[tweet(createdAt='2026-09-02T01:00:00Z'),tweet(id='unsafe'),tweet(createdAt='2026-09-01T06:00:00Z')])
def test_rejects_wrong_windows_or_ids(raw):
    with pytest.raises(ValueError): normalize_post(raw,START,END)

@pytest.mark.parametrize('page',[
    {'tweets':[tweet()]*21,'has_next_page':False},
    {'tweets':[tweet(),tweet()],'has_next_page':False},
    {'tweets':[],'has_next_page':True,'next_cursor':'same'},
    {'tweets':[]}, {'error':'unauthorized'}])
def test_contract_and_cursor_validation(page):
    with pytest.raises(ValueError): validate_page(page,START,END,'same')

def test_empty_valid_page():
    assert validate_page({'tweets':[],'has_next_page':False},START,END,'')==([], '',False)

def test_plan_covers_full_week_equally_with_safe_page_bound():
    windows=list(build_windows(START,START+timedelta(days=7)))
    assert len(windows)==56
    assert [w[0] for w in windows[:4]]==['ZCAT','ZEC','ZCAT','ZEC']
    assert all('since_time:' in w[3] and 'until_time:' in w[3] for w in windows)
    assert 2*len(windows)*PAGE_RESERVE <= LIMIT

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
        cur.execute('DROP SCHEMA IF EXISTS crypto_test CASCADE; CREATE SCHEMA crypto_test; SET search_path TO crypto_test')
    conn.autocommit=False
    from crypto_social_pilot import initialize
    initialize(conn,START+timedelta(days=7))
    yield conn
    conn.close()

def test_db_budget_survives_reinitialization_and_blocks_overrun(db):
    from crypto_social_pilot import reserve, initialize
    with db,db.cursor() as c:
        c.execute("UPDATE crypto_social_pilot SET reserved_credits=49800")
    assert reserve(db) is None
    initialize(db,START+timedelta(days=14))
    with db,db.cursor() as c:
        c.execute('SELECT reserved_credits,start_at FROM crypto_social_pilot')
        assert c.fetchone()==(49800,START)

def test_db_atomic_save_duplicate_and_resume(db):
    from crypto_social_pilot import reserve,save_page
    rid,w=reserve(db)
    with pytest.raises(RuntimeError): reserve(db)  # concurrent collector cannot spend
    data={'tweets':[tweet()],'has_next_page':False,'next_cursor':''}
    save_page(db,rid,w,data)
    # Same post through overlapping query is stored once, separately linked.
    with db,db.cursor() as c:
        c.execute("UPDATE crypto_social_windows SET status='partial' WHERE id=%s",(w[0],))
        c.execute("UPDATE crypto_social_windows SET status='search_exhausted' WHERE id<>%s",(w[0],))
    rid2,w2=reserve(db)
    save_page(db,rid2,w2,data)
    with db,db.cursor() as c:
        c.execute('SELECT count(*) FROM crypto_social_posts'); assert c.fetchone()[0]==1
        c.execute('SELECT count(*) FROM crypto_social_snapshots'); assert c.fetchone()[0]==2
        c.execute('SELECT reserved_credits FROM crypto_social_pilot'); assert c.fetchone()[0]==600
    assert reserve(db) is None

def test_db_failed_call_is_not_retried_or_refunded(db):
    from crypto_social_pilot import reserve
    calls=[]
    def timeout(*args,**kwargs):
        calls.append(1)
        raise TimeoutError()
    with pytest.raises(RuntimeError): collect(db,'fake',14,timeout)
    assert len(calls)==1
    with pytest.raises(RuntimeError): reserve(db)
    with db,db.cursor() as c:
        c.execute('SELECT reserved_credits FROM crypto_social_pilot'); assert c.fetchone()[0]==300
        c.execute('SELECT status FROM crypto_social_requests'); assert c.fetchone()[0]=='uncertain'
