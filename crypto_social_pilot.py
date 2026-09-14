"""Manual, resumable TwitterAPI.io pilot. Default is a zero-network plan.

Every attempted search permanently reserves its documented worst case (20 * 15
credits). Unknown outcomes retain that reservation and block subsequent calls.
The singleton budget and frozen date range survive reruns; no CLI reset exists.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import json
import os
from pathlib import Path
import re

PILOT = 'zcat-zec-v1'
LIMIT = 50000
PAGE_RESERVE = 300
ENDPOINT = 'https://api.twitterapi.io/twitter/tweet/advanced_search'
ADDRESS = 'HcRLc9VDgjLeK154xDawfb1dmVJ98DoSqcwTHGqiDeJR'
COINS = {
    'ZCAT': ('Anonymous Cat', f'("$ZCAT" OR "Anonymous Cat" OR "{ADDRESS}")', ADDRESS, 'user-supplied address; not independently verified'),
    'ZEC': ('Zcash', '("$ZEC" OR "Zcash")', None, 'name and cashtag'),
}

def date_value(value):
    try:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        result = parsedate_to_datetime(str(value))
    if result.tzinfo is None:
        raise ValueError('Timestamp must include timezone')
    return result.astimezone(timezone.utc)

def count(value):
    try:
        return max(0, int(value)) if value is not None else None
    except (TypeError, ValueError):
        return None

def identity(value):
    value = str(value or '')
    return value if re.fullmatch(r'[0-9]+', value) else None

def normalize_post(raw, start, end):
    author = raw.get('author') or {}
    pid, aid = identity(raw.get('id')), identity(author.get('id'))
    if not pid or not aid:
        raise ValueError('Missing numeric post/author ID')
    posted = date_value(raw.get('createdAt'))
    if not start <= posted < end:
        raise ValueError('Search returned a post outside the requested window')
    kind = 'original'
    edges = set()
    if raw.get('isReply'):
        kind = 'reply'
        target = identity(raw.get('inReplyToUserId'))
        if target:
            edges.add((target, 'reply'))
    for field, label in [('quoted_tweet', 'quote'), ('retweeted_tweet', 'repost')]:
        nested = raw.get(field)
        if isinstance(nested, dict):
            kind = label
            target = identity((nested.get('author') or {}).get('id'))
            if target:
                edges.add((target, label))
    for mention in (raw.get('entities') or {}).get('user_mentions', []):
        target = identity(mention.get('id_str') or mention.get('id'))
        if target:
            edges.add((target, 'mention'))
    return dict(id=pid, author_id=aid, handle=str(author.get('userName') or aid),
                name=str(author.get('name') or ''), followers=count(author.get('followers')),
                text=str(raw.get('text') or ''), posted=posted, kind=kind,
                url=f'https://x.com/i/status/{pid}', edges=sorted(edges),
                metrics=[count(raw.get(k)) for k in ['likeCount','replyCount','quoteCount','retweetCount','viewCount']])

def validate_page(data, start, end, cursor):
    if not isinstance(data, dict) or not isinstance(data.get('tweets'), list):
        raise ValueError('Invalid search response')
    tweets = data['tweets']
    if len(tweets) > 20 or not isinstance(data.get('has_next_page'), bool):
        raise ValueError('Provider page contract changed; stop and review billing')
    more = data['has_next_page']
    next_cursor = data.get('next_cursor') or ''
    if more and (not isinstance(next_cursor, str) or not next_cursor or next_cursor == cursor):
        raise ValueError('Pagination did not advance')
    posts = [normalize_post(t, start, end) for t in tweets]
    if len({p['id'] for p in posts}) != len(posts):
        raise ValueError('Duplicate IDs in search page')
    return posts, next_cursor, more

def build_windows(start, end):
    current = start
    while current < end:
        until = min(current + timedelta(hours=6), end)
        for coin, (_, query, _, _) in COINS.items():
            yield coin, current, until, f'{query} since_time:{int(current.timestamp())} until_time:{int(until.timestamp())}'
        current = until

def initialize(conn, end):
    with conn, conn.cursor() as cur:
        cur.execute(Path(__file__).with_name('sql').joinpath('crypto_social.sql').read_text())
        cur.execute(Path(__file__).with_name('sql').joinpath('crypto_social_metrics.sql').read_text())
        cur.execute('INSERT INTO crypto_social_pilot(id,start_at,end_at) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING',
                    (PILOT, end-timedelta(days=7), end))
        cur.execute('SELECT start_at,end_at FROM crypto_social_pilot WHERE id=%s', (PILOT,))
        start, end = cur.fetchone()
        for symbol, (name, query, address, status) in COINS.items():
            cur.execute('INSERT INTO crypto_social_coins VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING',
                        (symbol,name,query,address,status))
        for row in build_windows(start,end):
            cur.execute('INSERT INTO crypto_social_windows(coin,start_at,end_at,query) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING', row)

def reserve(conn, max_pages=2):
    """Atomic reservation serializes collectors even when launched outside Actions."""
    with conn, conn.cursor() as cur:
        cur.execute('SELECT reserved_credits,credit_limit FROM crypto_social_pilot WHERE id=%s FOR UPDATE', (PILOT,))
        spent, limit = cur.fetchone()
        cur.execute("SELECT count(*) FROM crypto_social_requests WHERE status IN ('reserved','uncertain')")
        if cur.fetchone()[0]:
            raise RuntimeError('Outstanding or uncertain request; review ledger before continuing')
        if spent + PAGE_RESERVE > min(limit,LIMIT):
            return None
        cur.execute("SELECT coalesce(sum(reserved_credits),0) FROM crypto_social_requests WHERE endpoint='search'")
        if cur.fetchone()[0] + PAGE_RESERVE > 16800:
            return None
        # Preserve the initial discovery allocation for profile tracking.
        cur.execute("SELECT id,start_at,end_at,query,cursor FROM crypto_social_windows WHERE status IN ('pending','partial') AND start_at>=(SELECT start_at FROM crypto_social_pilot WHERE id='zcat-zec-v1') AND end_at<=(SELECT end_at FROM crypto_social_pilot WHERE id='zcat-zec-v1') AND pages<%s ORDER BY pages,start_at,coin LIMIT 1 FOR UPDATE", (max_pages,))
        window = cur.fetchone()
        if not window:
            return None
        cur.execute('UPDATE crypto_social_pilot SET reserved_credits=reserved_credits+%s WHERE id=%s', (PAGE_RESERVE,PILOT))
        cur.execute('INSERT INTO crypto_social_requests(window_id) VALUES (%s) RETURNING id', (window[0],))
        return cur.fetchone()[0],window

def save_page(conn, request_id, window, data):
    wid,start,end,_,cursor = window
    posts,next_cursor,more = validate_page(data,start,end,cursor)
    from crypto_social_profiles import save_profile
    with conn, conn.cursor() as cur:
        for raw in data['tweets']:
            save_profile(cur, raw['author'], request_id, 'search_author')
            for field in ('quoted_tweet','retweeted_tweet'):
                nested = raw.get(field)
                if isinstance(nested, dict) and identity((nested.get('author') or {}).get('id')):
                    save_profile(cur, nested['author'], request_id, 'embedded_target')
        for p in posts:
            cur.execute('''INSERT INTO crypto_social_accounts(id,handle,name,followers) VALUES (%s,%s,%s,%s)
                ON CONFLICT(id) DO UPDATE SET handle=EXCLUDED.handle,name=EXCLUDED.name,
                followers=EXCLUDED.followers,observed_at=now()''',
                (p['author_id'],p['handle'],p['name'],p['followers']))
            cur.execute('''INSERT INTO crypto_social_posts(id,author_id,text,posted_at,kind,url)
                VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING''',
                (p['id'],p['author_id'],p['text'],p['posted'],p['kind'],p['url']))
            cur.execute('INSERT INTO crypto_social_matches VALUES (%s,%s) ON CONFLICT DO NOTHING', (p['id'],wid))
            cur.execute('''INSERT INTO crypto_social_snapshots(post_id,request_id,likes,replies,quotes,reposts,views)
                VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING''', (p['id'],request_id,*p['metrics']))
            for target,kind in p['edges']:
                cur.execute('INSERT INTO crypto_social_edges VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING',
                            (p['id'],p['author_id'],target,kind))
        cur.execute('UPDATE crypto_social_windows SET cursor=%s,pages=pages+1,status=%s WHERE id=%s',
                    (next_cursor,'partial' if more else 'search_exhausted',wid))
        cur.execute("UPDATE crypto_social_requests SET status='saved',estimated_credits=%s,returned_count=%s,accepted_count=%s WHERE id=%s",
                    (max(1,len(posts))*15,len(posts),len(posts),request_id))

def collect(conn, key, max_requests, fetch=None, max_pages=2):
    import requests
    fetch = fetch or requests.get
    calls = 0
    while calls < max_requests:
        item = reserve(conn, max_pages)
        if item is None:
            break
        rid, window = item
        try:
            response = fetch(ENDPOINT, params={'query':window[3],'queryType':'Latest','cursor':window[4]},
                             headers={'X-API-Key':key},timeout=30,allow_redirects=False)
            if response.status_code != 200:
                raise ValueError(f'Provider HTTP {response.status_code}')
            save_page(conn,rid,window,response.json())
        except Exception as exc:
            # Never retry automatically or refund a possibly billed request.
            with conn, conn.cursor() as cur:
                cur.execute("UPDATE crypto_social_requests SET status='uncertain',error=%s WHERE id=%s",
                            (type(exc).__name__,rid))
            raise RuntimeError(f'Request {rid} stopped; reservation retained. Inspect provider and ledger.') from None
        calls += 1
    return calls

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Save to Neon and make paid calls; otherwise only print plan')
    parser.add_argument('--max-requests',type=int,default=14,help='Default smoke test: at most 4,200 reserved credits')
    args = parser.parse_args()
    if not 1 <= args.max_requests <= 166:
        parser.error('--max-requests must be between 1 and 166')
    end = datetime.now(timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0)
    if not args.execute:
        print(json.dumps({'mode':'plan_only','paid_calls':0,'credit_ceiling':LIMIT,
              'max_run_reservation':min(args.max_requests*PAGE_RESERVE,LIMIT),
              'new_pilot_range':[str(end-timedelta(days=7)),str(end)],
              'note':'Existing pilot retains its original range and credit ledger on reruns.',
              'queries':{k:v[1] for k,v in COINS.items()}},indent=2))
        return
    missing = [k for k in ['DATABASE_URL','TWITTERAPI_IO_API_KEY'] if not os.environ.get(k)]
    if missing:
        parser.error('Missing required secrets: '+', '.join(missing))
    import psycopg2
    conn = psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
    try:
        initialize(conn,end)
        calls = collect(conn,os.environ['TWITTERAPI_IO_API_KEY'],args.max_requests)
        with conn,conn.cursor() as cur:
            cur.execute('SELECT reserved_credits,credit_limit FROM crypto_social_pilot WHERE id=%s',(PILOT,))
            reserved,limit = cur.fetchone()
        print(json.dumps({'requests_saved':calls,'reserved_credits':reserved,'credit_limit':limit}))
    finally:
        conn.close()

if __name__ == '__main__':
    main()
