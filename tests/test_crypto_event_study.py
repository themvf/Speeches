from datetime import datetime,timedelta,timezone
import pytest
from test_crypto_social_pilot import db
from test_crypto_market_history import candles,catalog,POOL
from crypto_market_history import normalize_hourly,save,setup,refresh
from crypto_event_study import mentions,build_events,compute,VERSION

NOW=datetime(2026,9,16,12,tzinfo=timezone.utc)
T0=datetime(2026,9,10,tzinfo=timezone.utc)

def series(hours=72,start=T0,price=lambda i:1.0+i*0.01,volume=lambda i:100.0):
    return {start+timedelta(hours=i):(price(i),volume(i)) for i in range(hours)}

def post(id,author,at,**extra):
    return dict(id=id,author_id=author,posted_at=at,**extra)

def test_mentions_follow_dashboard_evidence_rules():
    assert mentions('ZCAT','$ZCAT to the moon') and mentions('ZCAT','Anonymous Cat launch')
    assert mentions('ZCAT','HcRLc9VDgjLeK154xDawfb1dmVJ98DoSqcwTHGqiDeJR pumping')
    assert not mentions('ZCAT','zcat on TON is different') and not mentions('ZCAT','concatenate')
    assert mentions('PONS','pons is live on Robinhood') and not mentions('PONS','pons asinorum')
    assert mentions('STANDARD','The Standard Reserve') and not mentions('STANDARD','standard practice')
    assert mentions('ZEC','buy zec') and not mentions('ZEC','zecharia')
    assert mentions('DPONS','0X0E6D1EBB33F3B8F2D09BACF3B1A1D5C581110C33')

def test_events_use_post_hour_and_write_only_when_horizon_is_complete():
    s=series()
    rows=build_events([post('1','a',T0+timedelta(hours=10,minutes=42))],s,NOW)
    assert len(rows)==1 and rows[0]['hour']==T0+timedelta(hours=10)
    assert rows[0]['price_0']==pytest.approx(1.10) and rows[0]['price_after_24h']==pytest.approx(1.34)
    assert rows[0]['price_before_1h']==pytest.approx(1.09) and rows[0]['price_after_1h']==pytest.approx(1.11) and rows[0]['price_after_6h']==pytest.approx(1.16)
    assert rows[0]['volume_before_24h']==1000 and rows[0]['hours_before_24h']==10  # only ten archived hours precede the post
    assert rows[0]['volume_after_24h']==2400 and rows[0]['hours_after_24h']==24
    assert build_events([post('2','a',T0+timedelta(hours=60))],s,NOW)==[]  # 24h endpoint outside archive
    assert build_events([post('3','a',T0+timedelta(hours=10))],s,T0+timedelta(hours=34))==[]  # horizon candle incomplete

def test_missing_hours_leave_horizons_null_without_dropping_the_event():
    s=series();del s[T0+timedelta(hours=11)];del s[T0+timedelta(hours=9)]
    row=build_events([post('1','a',T0+timedelta(hours=10))],s,NOW)[0]
    assert row['price_after_1h'] is None and row['price_before_1h'] is None and row['hours_after_24h']==23

def test_episode_is_first_post_by_author_in_24_hours():
    s=series(120)
    posts=[post('1','a',T0+timedelta(hours=10)),post('2','a',T0+timedelta(hours=20)),post('3','a',T0+timedelta(hours=44)),post('4','b',T0+timedelta(hours=20))]
    rows={r['post_id']:r['episode'] for r in build_events(posts,s,NOW)}
    assert rows=={'1':True,'2':False,'3':True,'4':True}

def test_db_compute_is_incremental_and_never_rewrites(db):
    setup(db)
    with db,db.cursor() as c:
        c.execute("INSERT INTO crypto_social_accounts(id,handle) VALUES ('7','alice'),('8','bob')")
        c.execute("INSERT INTO crypto_social_windows(coin,start_at,end_at,query) VALUES ('ZCAT',%s,%s,'q') RETURNING id",(T0,T0+timedelta(hours=6)));wid=c.fetchone()[0]
        for pid,author,text,at in [('1','7','$ZCAT early',T0+timedelta(hours=10,minutes=5)),('2','8','unrelated post',T0+timedelta(hours=11)),('3','7','$ZCAT again',T0+timedelta(hours=12)),('4','7','$ZCAT',T0+timedelta(hours=70))]:
            c.execute("INSERT INTO crypto_social_posts(id,author_id,text,posted_at,kind,url) VALUES (%s,%s,%s,%s,'original','u')",(pid,author,text,at))
            c.execute('INSERT INTO crypto_social_matches VALUES (%s,%s)',(pid,wid))
    assert compute(db,NOW)['ZCAT']['status']=='no_default_source'
    src=dict(id='geckoterminal:'+POOL,coin='ZCAT',provider='GeckoTerminal',url='u',metadata={'id':POOL,'created':'2026-08-30'})
    data={'data':{'attributes':{'ohlcv_list':[[(T0+timedelta(hours=i)).timestamp(),1,2,.5,1+i*.01,100] for i in range(48)]}}}
    save(db,src,data,normalize_hourly(data,'ohlcv',NOW),'u',NOW,hourly=True)
    with db,db.cursor() as c:c.execute('UPDATE crypto_market_sources SET is_default=true')
    first=compute(db,NOW)['ZCAT']
    assert first['inserted']==2 and first['eligible_posts']==3 and first['pending']==1
    with db,db.cursor() as c:
        c.execute('SELECT post_id,episode,price_0,price_after_24h FROM crypto_price_events ORDER BY post_id');rows=c.fetchall()
    assert [r[0] for r in rows]==['1','3'] and rows[0][1] is True and rows[1][1] is False
    assert rows[0][2]==pytest.approx(1.10) and rows[0][3]==pytest.approx(1.34)
    # A later archive revision changes nothing already written; the pending post appears once coverage exists.
    later={'data':{'attributes':{'ohlcv_list':[[(T0+timedelta(hours=i)).timestamp(),1,20,.5,9.0,100] for i in range(100)]}}}
    save(db,src,later,normalize_hourly(later,'ohlcv',NOW),'u',NOW+timedelta(hours=1),hourly=True)
    second=compute(db,NOW)['ZCAT']
    assert second['inserted']==1 and second['pending']==0
    with db,db.cursor() as c:
        c.execute("SELECT price_0 FROM crypto_price_events WHERE post_id='1'");assert c.fetchone()[0]==pytest.approx(1.10)
        c.execute("SELECT price_0 FROM crypto_price_events WHERE post_id='4'");assert c.fetchone()[0]==9.0
        c.execute("SELECT version FROM crypto_price_events LIMIT 1");assert c.fetchone()[0]==VERSION

def test_hourly_normalization_buckets_and_completeness():
    t=T0+timedelta(hours=5,minutes=30)
    data={'prices':[[int((t-timedelta(minutes=20)).timestamp()*1000),1],[int(t.timestamp()*1000),2],[int((NOW-timedelta(minutes=10)).timestamp()*1000),3]],'total_volumes':[[int((t-timedelta(minutes=20)).timestamp()*1000),5],[int(t.timestamp()*1000),10],[int((NOW-timedelta(minutes=10)).timestamp()*1000),1]]}
    points=normalize_hourly(data,'price_observation',NOW)
    assert len(points)==2 and points[0]['hour']==T0+timedelta(hours=5) and points[0]['close']==2 and points[0]['complete']
    assert points[1]['hour']==NOW-timedelta(hours=1) and points[1]['complete']
    assert not normalize_hourly({'prices':[[int(NOW.timestamp()*1000),3]],'total_volumes':[[int(NOW.timestamp()*1000),1]]},'price_observation',NOW+timedelta(minutes=5))[0]['complete']

def test_db_refresh_archives_hourly_for_the_pinned_pool_only(db):
    class Response:
        status_code=200
        def __init__(self,data):self.data=data
        def json(self):return self.data
    calls=[]
    def fetch(url,**kwargs):
        calls.append(url)
        if 'coingecko.com' in url:raise ValueError('provider unavailable')
        if url.endswith('/pools'):
            data=catalog();second=dict(data['data'][0]);second['attributes']={**second['attributes'],'address':'2'*40,'pool_created_at':'2026-09-01T00:00:00Z'}
            data['data'].append(second);return Response(data)
        return Response(candles())
    result=refresh(db,fetch=fetch,now=NOW,wait=lambda _:None)
    hourly=[u for u in calls if '/ohlcv/hour' in u];daily=[u for u in calls if '/ohlcv/day' in u]
    assert len(daily)==2 and len(hourly)==1 and POOL in hourly[0] and 'limit=1000' in hourly[0]
    assert len([u for u in calls if u.endswith('/pools')])==4  # every contract coin is catalogued
    assert result['skipped']==[c+': no indexed pool for this contract' for c in ['PONS','DPONS','STANDARD']]
    with db,db.cursor() as c:
        c.execute('SELECT count(*) FROM crypto_market_hourly_latest');assert c.fetchone()[0]==1
        c.execute("SELECT id FROM crypto_market_sources WHERE is_default");assert c.fetchone()[0]=='geckoterminal:'+POOL

def test_db_daily_candles_are_limited_to_pinned_plus_two_liquid_pools(db):
    class Response:
        status_code=200
        def __init__(self,data):self.data=data
        def json(self):return self.data
    calls=[]
    def fetch(url,**kwargs):
        calls.append(url)
        if 'coingecko.com' in url:raise ValueError('provider unavailable')
        if url.endswith('/pools'):
            data=catalog();base=data['data'][0]
            for i in range(1,5):
                extra=dict(base);extra['attributes']={**base['attributes'],'address':str(i)*40,'pool_created_at':f'2026-09-0{i}T00:00:00Z','reserve_in_usd':str(1000*i)}
                data['data'].append(extra)
            return Response(data)
        return Response(candles())
    refresh(db,fetch=fetch,now=NOW,wait=lambda _:None)
    daily=[u for u in calls if '/ohlcv/day' in u]
    assert len(daily)==3 and any(POOL in u for u in daily) and any('4'*40 in u for u in daily) and any('3'*40 in u for u in daily)

def test_db_repeated_rate_limits_retry_within_bound_then_fail_softly(db):
    class Response:
        headers={'Retry-After':'5'}
        def __init__(self,status,data=None):self.status_code=status;self.data=data
        def json(self):return self.data
    calls=[]
    def fetch(url,**kwargs):
        calls.append(url)
        if url.endswith('/pools') and 'solana' in url:return Response(200,catalog())
        if '/ohlcv/day' in url:return Response(200,candles())
        return Response(429)
    result=refresh(db,fetch=fetch,now=NOW,wait=lambda _:None)
    assert len(result['saved'])==1  # the daily candles arrived; hourly and the other catalogs were rate-limited four times each
    assert calls.count([u for u in calls if '/ohlcv/hour' in u][0])==4
    assert any('429' in e for e in result['errors'])
