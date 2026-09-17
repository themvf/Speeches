"""Post-to-price event study from archived hourly candles. No provider calls, no X credits.

For every saved post that mentions a tracked coin (non-repost), record the pinned pool's
hourly close one hour before, at, and 1/6/24 hours after the post hour, plus hourly volume
summed over the 24 hours before and after. A row is written only once the 24-hour horizon
has passed and both endpoints exist, and it is never rewritten. An 'episode' is the
author's first eligible post on that coin in 24 hours; rankings use episodes so that
a burst of posts during one move counts once.

Association only: nothing here establishes that a post caused a price change.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from crypto_coins import SYMBOLS, archive_start, mentions as registry_mentions
from crypto_market_history import setup

VERSION='price-events-v1'
HORIZON=timedelta(hours=24)
COINS=SYMBOLS


def mentions(coin,text):
    return registry_mentions(text,coin,'words')


def coin_start(coin):
    return datetime.combine(archive_start(coin),datetime.min.time(),timezone.utc)


def floor_hour(stamp):
    return stamp.astimezone(timezone.utc).replace(minute=0,second=0,microsecond=0)


def window_sum(series,start,end):
    """Sum hourly volume for hours in [start,end); returns (sum or None, hours present)."""
    total=0.0;hours=0;hour=start
    while hour<end:
        point=series.get(hour)
        if point is not None:total+=point[1];hours+=1
        hour+=timedelta(hours=1)
    return (total if hours else None),hours


def build_events(posts,series,now):
    """posts: dicts with id, author_id, posted_at (UTC). series: {hour:(close,volume)}. Pure and deterministic."""
    by_author={}
    for post in sorted(posts,key=lambda p:(p['posted_at'],p['id'])):
        by_author.setdefault(post['author_id'],[]).append(post)
    rows=[]
    for author,items in by_author.items():
        previous=None
        for post in items:
            episode=previous is None or post['posted_at']-previous>=HORIZON
            previous=post['posted_at']
            hour=floor_hour(post['posted_at'])
            if hour+HORIZON+timedelta(hours=1)>now:continue  # the 24h candle is not complete yet
            at=series.get(hour);after=series.get(hour+HORIZON)
            if at is None or after is None:continue  # pending until archive coverage exists
            before=series.get(hour-timedelta(hours=1));h1=series.get(hour+timedelta(hours=1));h6=series.get(hour+timedelta(hours=6))
            vol_before,n_before=window_sum(series,hour-HORIZON,hour)
            vol_after,n_after=window_sum(series,hour+timedelta(hours=1),hour+HORIZON+timedelta(hours=1))
            rows.append(dict(post_id=post['id'],account_id=author,posted_at=post['posted_at'],hour=hour,episode=episode,
                price_before_1h=before and before[0],price_0=at[0],price_after_1h=h1 and h1[0],price_after_6h=h6 and h6[0],
                price_after_24h=after[0],volume_before_24h=vol_before,volume_after_24h=vol_after,
                hours_before_24h=n_before,hours_after_24h=n_after))
    return rows


def load_series(cur,source):
    cur.execute('SELECT hour,close,volume FROM crypto_market_hourly_latest WHERE source_id=%s',(source,))
    return {floor_hour(h):(c,v) for h,c,v in cur.fetchall()}


def compute(conn,now=None,coins=COINS):
    now=now or datetime.now(timezone.utc);setup(conn);summary={}
    for coin in coins:
        with conn,conn.cursor() as cur:
            cur.execute('SELECT id FROM crypto_market_sources WHERE coin=%s AND is_default',(coin,))
            row=cur.fetchone()
            if not row:summary[coin]={'status':'no_default_source','inserted':0};continue
            source=row[0];series=load_series(cur,source)
            if not series:summary[coin]={'status':'no_hourly_history','inserted':0};continue
            # Every eligible post is needed to decide episodes, not only the ones without rows.
            cur.execute('''SELECT DISTINCT p.id,p.author_id,p.text,p.posted_at FROM crypto_social_posts p
                JOIN crypto_social_matches m ON m.post_id=p.id JOIN crypto_social_windows w ON w.id=m.window_id
                WHERE w.coin=%s AND p.kind<>'repost' AND p.posted_at>=%s AND p.posted_at<%s ORDER BY p.posted_at,p.id''',
                (coin,coin_start(coin),now))
            posts=[dict(id=i,author_id=a,posted_at=t.astimezone(timezone.utc)) for i,a,text,t in cur.fetchall() if mentions(coin,text)]
            cur.execute('SELECT post_id FROM crypto_price_events WHERE coin=%s AND version=%s',(coin,VERSION))
            done={r[0] for r in cur.fetchall()}
            rows=[r for r in build_events(posts,series,now) if r['post_id'] not in done]
            for r in rows:
                cur.execute('''INSERT INTO crypto_price_events(post_id,coin,version,account_id,posted_at,source_id,hour,episode,
                    price_before_1h,price_0,price_after_1h,price_after_6h,price_after_24h,volume_before_24h,volume_after_24h,hours_before_24h,hours_after_24h)
                    VALUES (%(post_id)s,%(coin)s,%(version)s,%(account_id)s,%(posted_at)s,%(source_id)s,%(hour)s,%(episode)s,
                    %(price_before_1h)s,%(price_0)s,%(price_after_1h)s,%(price_after_6h)s,%(price_after_24h)s,%(volume_before_24h)s,%(volume_after_24h)s,%(hours_before_24h)s,%(hours_after_24h)s)
                    ON CONFLICT DO NOTHING''',{**r,'coin':coin,'version':VERSION,'source_id':source})
            summary[coin]={'status':'ok','source':source,'eligible_posts':len(posts),'inserted':len(rows),'pending':len(posts)-len(done)-len(rows),'hours':len(series)}
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'mode':'plan_only','version':VERSION,'coins':COINS,'network_requests':0,'twitter_credits':0}));return
    import psycopg2
    conn=psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
    try:print(json.dumps({'version':VERSION,'results':compute(conn)},default=str))
    finally:conn.close()

if __name__=='__main__':main()
