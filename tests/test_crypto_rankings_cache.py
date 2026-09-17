import json
from datetime import datetime,timedelta,timezone
from test_crypto_social_pilot import db
from crypto_rankings_cache import refresh,rank,QUERIES
from crypto_coins import SYMBOLS

NOW=datetime(2026,9,16,12,tzinfo=timezone.utc)

def fake_runner(calls):
    def run(payload):
        data=json.loads(payload);calls.append(data)
        if data['kind']=='voices':return json.dumps({'version':'voices-test','payload':{'voices':[{'id':p['author_id']} for p in data['posts']],'total':len(data['posts']),'unfinished':data['unfinished']}})
        return json.dumps({'version':'watchers-test','payload':{'accounts':[],'loaded':len(data['posts']),'total':len(data['posts']),'candidates':0,'coin':data['coin']}})
    return run

def test_queries_are_the_routes_queries():
    assert '__COINS__' in QUERIES['watchers'] and '__COIN__' in QUERIES['voices'] and '__LIMIT__' in QUERIES['voices']

def test_db_snapshots_cover_every_coin_and_are_replaced_in_place(db):
    calls=[]
    with db,db.cursor() as c:
        c.execute("INSERT INTO crypto_social_accounts(id,handle) VALUES ('7','alice')")
        c.execute("INSERT INTO crypto_social_windows(coin,start_at,end_at,query) VALUES ('ZCAT',%s,%s,'q') RETURNING id",(NOW-timedelta(days=1),NOW));wid=c.fetchone()[0]
        c.execute("INSERT INTO crypto_social_posts(id,author_id,text,posted_at,kind,url) VALUES ('1','7','$ZCAT',%s,'original','u')",(NOW-timedelta(hours=2),))
        c.execute('INSERT INTO crypto_social_matches VALUES (%s,%s)',('1',wid))
        c.execute("CREATE TABLE IF NOT EXISTS crypto_social_profile_history(account_id text,request_id bigint,observed_at timestamptz DEFAULT now(),handle text,name text,bio text,followers bigint,following bigint,available boolean DEFAULT true,source text)")
    written=refresh(db,run=fake_runner(calls),now=NOW)
    keys={w['key'] for w in written}
    assert 'voices:ZCAT' in keys and 'watchers:ALL' in keys and 'watchers:ZEC' in keys and len(keys)==2*len(SYMBOLS)+1
    assert next(w for w in written if w['key']=='voices:ZCAT')['rows']==1
    # Watcher rows are loaded once and reused for every coin scope.
    assert len([c for c in calls if c['kind']=='watchers'])==len(SYMBOLS)+1 and all(len(c['posts'])==1 for c in calls if c['kind']=='watchers')
    with db,db.cursor() as c:
        c.execute("SELECT payload,rows_loaded,computed_at FROM crypto_ranking_cache WHERE key='voices:ZCAT' AND version='voices-test'")
        payload,rows,computed=c.fetchone();assert payload['voices']==[{'id':'7'}] and rows==1 and computed==NOW
    refresh(db,kinds=('voices',),run=fake_runner([]),now=NOW+timedelta(hours=6))
    with db,db.cursor() as c:
        c.execute("SELECT count(*),max(computed_at) FROM crypto_ranking_cache WHERE key='voices:ZCAT'");n,latest=c.fetchone()
        assert n==1 and latest==NOW+timedelta(hours=6)

def test_node_ranker_matches_route_shape():
    out=rank('watchers','ALL',[])
    assert out['version']=='watchers-v1' and out['payload']['candidates']==0
    out=rank('voices','ZCAT',[],unfinished=3)
    assert out['version']=='voices-v1' and out['payload']['unfinished']==3 and out['payload']['voices']==[]
