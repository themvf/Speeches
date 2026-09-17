"""Thirty-day, resumable profile/bio and social discovery pilot.

Default is a zero-network plan. All paid endpoints share the original 50,000
credit ledger. Uncertain calls are never retried or refunded automatically.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os

from crypto_social_pilot import (PILOT, LIMIT, COINS, ADDRESS, initialize, collect,
                                normalize_post, date_value, identity)
from crypto_social_profiles import save_profile, profile, matches

BASE = 'https://api.twitterapi.io'
ALLOCATIONS = {'profiles':21600, 'enrichment':9000, 'discovery':2600}


def seed_candidates(conn):
    """Include targets even when their own posts weren't returned by searches."""
    with conn, conn.cursor() as cur:
        cur.execute('''INSERT INTO crypto_social_accounts(id,handle)
            SELECT DISTINCT target_id,target_id FROM crypto_social_edges ON CONFLICT DO NOTHING''')
        cur.execute('''INSERT INTO crypto_social_candidates(coin,account_id,reason)
            SELECT DISTINCT w.coin,p.author_id,'observed coin post' FROM crypto_social_posts p
            JOIN crypto_social_matches m ON m.post_id=p.id JOIN crypto_social_windows w ON w.id=m.window_id
            ON CONFLICT DO NOTHING''')
        cur.execute('''INSERT INTO crypto_social_candidates(coin,account_id,reason)
            SELECT DISTINCT w.coin,e.target_id,'observed interaction target' FROM crypto_social_edges e
            JOIN crypto_social_matches m ON m.post_id=e.post_id JOIN crypto_social_windows w ON w.id=m.window_id
            ON CONFLICT DO NOTHING''')
        # Freeze a reproducible provisional cohort. Discovery continues in the queue.
        for coin in COINS:
            cur.execute('SELECT count(*) FROM crypto_social_candidates WHERE coin=%s AND tracked', (coin,))
            remaining = 20-cur.fetchone()[0]
            if remaining <= 0:
                continue
            cur.execute('''WITH matched AS (
                SELECT DISTINCT m.post_id FROM crypto_social_matches m JOIN crypto_social_windows w ON w.id=m.window_id
                WHERE w.coin=%s
            ), attention AS (
                SELECT e.target_id,count(DISTINCT e.source_id) AS participants,
                count(DISTINCT e.source_id) FILTER(WHERE e.kind IN ('quote','repost')) AS amplifiers
                FROM crypto_social_edges e JOIN matched m ON m.post_id=e.post_id
                WHERE e.source_id<>e.target_id GROUP BY e.target_id
            ), activity AS (
                SELECT p.author_id,count(*) AS posts FROM crypto_social_posts p JOIN matched m ON m.post_id=p.id GROUP BY p.author_id
            ) SELECT c.account_id FROM crypto_social_candidates c
              LEFT JOIN attention a ON a.target_id=c.account_id LEFT JOIN activity v ON v.author_id=c.account_id
              WHERE c.coin=%s AND NOT c.tracked
              ORDER BY coalesce(a.amplifiers,0) DESC,coalesce(a.participants,0) DESC,coalesce(v.posts,0) DESC,c.account_id
              LIMIT %s''', (coin,coin,remaining))
            ids=[r[0] for r in cur.fetchall()]
            for aid in ids:
                cur.execute('UPDATE crypto_social_candidates SET tracked=true,tracked_since=now() WHERE coin=%s AND account_id=%s', (coin,aid))


def reserve_request(conn, endpoint, key, params, credits, allocation):
    if allocation not in ALLOCATIONS or not 1 <= credits <= ALLOCATIONS[allocation]:
        raise ValueError('Invalid reservation')
    with conn, conn.cursor() as cur:
        cur.execute('SELECT reserved_credits,credit_limit FROM crypto_social_pilot WHERE id=%s FOR UPDATE', (PILOT,))
        spent,limit=cur.fetchone()
        cur.execute("SELECT count(*) FROM crypto_social_requests WHERE status IN ('reserved','uncertain')")
        if cur.fetchone()[0]:
            raise RuntimeError('Outstanding or uncertain request; review the ledger')
        cur.execute('SELECT count(*) FROM crypto_social_requests WHERE request_key=%s', (key,))
        if cur.fetchone()[0]:
            return None
        cur.execute("SELECT coalesce(sum(reserved_credits),0) FROM crypto_social_requests WHERE parameters->>'allocation'=%s", (allocation,))
        if cur.fetchone()[0]+credits>ALLOCATIONS[allocation] or spent+credits>min(limit,LIMIT):
            return None
        cur.execute('SELECT end_at FROM crypto_social_tracking WHERE id=%s', (PILOT,))
        end=cur.fetchone()
        if not end or datetime.now(timezone.utc)>=end[0]:
            return None
        cur.execute('UPDATE crypto_social_pilot SET reserved_credits=reserved_credits+%s WHERE id=%s', (credits,PILOT))
        metadata={'allocation':allocation, **params}
        cur.execute('''INSERT INTO crypto_social_requests(endpoint,request_key,parameters,reserved_credits)
            VALUES (%s,%s,%s,%s) RETURNING id''', (endpoint,key,json.dumps(metadata),credits))
        return cur.fetchone()[0]


def request(conn, key, endpoint, request_key, params, credits, allocation, save, fetch=None):
    import requests
    rid=reserve_request(conn,endpoint,request_key,params,credits,allocation)
    if rid is None:
        return 0
    try:
        response=(fetch or requests.get)(BASE+endpoint,params=params,headers={'X-API-Key':key},
                                       timeout=30,allow_redirects=False)
        if response.status_code!=200:
            raise ValueError(f'HTTP {response.status_code}')
        data=response.json()
        if not isinstance(data,dict) or data.get('status') not in (None,'success'):
            raise ValueError('Provider error')
        with conn,conn.cursor() as cur:
            returned,accepted,estimated=save(cur,rid,data)
            if estimated>credits:
                raise ValueError('Provider billing contract exceeded reservation')
            cur.execute("UPDATE crypto_social_requests SET status='saved',returned_count=%s,accepted_count=%s,estimated_credits=%s WHERE id=%s",
                        (returned,accepted,estimated,rid))
    except Exception as exc:
        with conn,conn.cursor() as cur:
            cur.execute("UPDATE crypto_social_requests SET status='uncertain',error=%s WHERE id=%s", (type(exc).__name__,rid))
        raise RuntimeError(f'Request {rid} stopped; reservation retained. Review provider and ledger.') from None
    return 1


def save_profiles(cur,rid,data,ids):
    users=data.get('users')
    if not isinstance(users,list) or len(users)>len(ids):
        raise ValueError('Invalid profile batch')
    seen=set()
    for raw in users:
        if not isinstance(raw,dict):raise ValueError('Invalid profile entry')
        # Unidentifiable entries cannot be attributed by response order.
        # Preserve requested-but-missing accounts as unknown, never zero.
        if not identity(raw.get('id')):continue
        p=profile(raw)
        if p['id'] not in ids or p['id'] in seen:
            raise ValueError('Unexpected or duplicate user ID')
        seen.add(p['id'])
        save_profile(cur,raw,rid,'daily_profile')
    # Missing profiles remain unknown, not zero. Record the failed observation.
    for aid in set(ids)-seen:
        save_profile(cur,{'id':aid,'unavailable':True},rid,'daily_profile_missing')
    return len(users),len(seen),max(15,len(users)*18)


def daily_profiles(conn,key,day):
    with conn,conn.cursor() as cur:
        cur.execute('SELECT DISTINCT account_id FROM crypto_social_candidates WHERE tracked ORDER BY account_id LIMIT 40')
        ids=[r[0] for r in cur.fetchall()]
    if not ids:
        return 0
    return request(conn,key,'/twitter/user/batch_info_by_ids',f'profiles:{day}',{'userIds':','.join(ids)},
                   len(ids)*18,'profiles',lambda c,r,d:save_profiles(c,r,d,ids))


def save_posts(cur,rid,data,start,end,coin=None,account=None,refresh_ids=None):
    tweets=data.get('tweets')
    if not isinstance(tweets,list) or len(tweets)>20:
        raise ValueError('Invalid post page')
    if refresh_ids is None and not isinstance(data.get('has_next_page'),bool):
        raise ValueError('Missing pagination metadata')
    seen=set();accepted=0;oldest=None
    for raw in tweets:
        # Validate identities/dates even when an account timeline includes older posts.
        p=normalize_post(raw,datetime(2006,1,1,tzinfo=timezone.utc),end+timedelta(days=1))
        if p['id'] in seen or (refresh_ids is not None and p['id'] not in refresh_ids):
            raise ValueError('Duplicate or unexpected post')
        seen.add(p['id'])
        if account and p['author_id']!=account:
            raise ValueError('Unexpected timeline author')
        oldest=min(oldest,p['posted']) if oldest else p['posted']
        if not start<=p['posted']<end:
            if coin: raise ValueError('Search returned post outside window')
            continue
        accepted+=1
        save_profile(cur,raw['author'],rid,'post_author')
        for field in ('quoted_tweet','retweeted_tweet'):
            nested=raw.get(field)
            if isinstance(nested,dict) and identity((nested.get('author') or {}).get('id')):
                save_profile(cur,nested['author'],rid,'embedded_target')
        cur.execute('''INSERT INTO crypto_social_posts(id,author_id,text,posted_at,kind,url)
            VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING''',
            (p['id'],p['author_id'],p['text'],p['posted'],p['kind'],p['url']))
        cur.execute('''INSERT INTO crypto_social_snapshots(post_id,request_id,likes,replies,quotes,reposts,views)
            VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING''',(p['id'],rid,*p['metrics']))
        for target,kind in p['edges']:
            cur.execute('INSERT INTO crypto_social_edges VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING',(p['id'],p['author_id'],target,kind))
        # Timeline results are coin-matched locally. Refreshes never invent coverage.
        coins={coin} if coin else {c for c,_ in matches(p['text'])} if account else set()
        for symbol in coins:
            query='bounded daily search' if coin else 'timeline text match'
            cur.execute('''INSERT INTO crypto_social_windows(coin,start_at,end_at,query,status)
                VALUES (%s,%s,%s,%s,'partial') ON CONFLICT DO NOTHING''',(symbol,start,end,query))
            cur.execute('SELECT id FROM crypto_social_windows WHERE coin=%s AND start_at=%s AND end_at=%s',(symbol,start,end))
            wid=cur.fetchone()[0]
            cur.execute('INSERT INTO crypto_social_matches VALUES (%s,%s) ON CONFLICT DO NOTHING',(p['id'],wid))
    if coin:
        cur.execute('''INSERT INTO crypto_social_windows(coin,start_at,end_at,query,pages,status)
            VALUES (%s,%s,%s,'bounded daily search',1,%s)
            ON CONFLICT(coin,start_at,end_at) DO UPDATE SET pages=1,status=EXCLUDED.status RETURNING id''',
            (coin,start,end,'partial' if data['has_next_page'] else 'search_exhausted'))
        wid=cur.fetchone()[0]
        cur.execute('UPDATE crypto_social_requests SET window_id=%s WHERE id=%s',(wid,rid))
    if account:
        more=data['has_next_page']
        status='window_reached' if oldest and oldest<=start else 'capped' if more else 'search_exhausted'
        cur.execute('''INSERT INTO crypto_social_account_coverage
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)''',(rid,account,start,end,oldest,len(tweets),accepted,more,status))
    return len(tweets),accepted,max(15,len(tweets)*15)


def enrich(conn,key,now,slot):
    """One bounded page daily; alternate coin searches and sample account activity."""
    end=now.replace(hour=0,minute=0,second=0,microsecond=0)
    start=end-timedelta(days=1)
    reqkey=f'enrichment:{end.date()}'
    if slot%7==5:
        with conn,conn.cursor() as cur:
            cur.execute('''SELECT DISTINCT c.account_id FROM crypto_social_candidates c WHERE tracked
                ORDER BY c.account_id''')
            ids=[r[0] for r in cur.fetchall()]
        if not ids:return 0
        aid=ids[(slot//7)%len(ids)]
        return request(conn,key,'/twitter/user/last_tweets',reqkey,{'userId':aid,'includeReplies':'true'},300,'enrichment',
                       lambda c,r,d:save_posts(c,r,d,start,end,account=aid))
    if slot%7==6:
        # Only refresh posts in a comparable age band. Do not claim exact 24h.
        with conn,conn.cursor() as cur:
            cur.execute('''SELECT p.id FROM crypto_social_posts p WHERE p.posted_at BETWEEN %s AND %s
                ORDER BY p.posted_at,p.id LIMIT 20''',(now-timedelta(hours=30),now-timedelta(hours=24)))
            ids=[r[0] for r in cur.fetchall()]
        if not ids:return 0
        return request(conn,key,'/twitter/tweets',reqkey,{'tweet_ids':','.join(ids)},300,'enrichment',
                       lambda c,r,d:save_posts(c,r,d,now-timedelta(hours=30),now,refresh_ids=ids))
    coin='ZCAT' if slot%2==0 else 'ZEC'
    query=f'{COINS[coin][1]} since_time:{int(start.timestamp())} until_time:{int(end.timestamp())}'
    return request(conn,key,'/twitter/tweet/advanced_search',reqkey,{'query':query,'queryType':'Latest','cursor':''},
                   300,'enrichment',lambda c,r,d:save_posts(c,r,d,start,end,coin=coin))


def discover(conn,key,now,max_results):
    """Optional first-page keyword search, only after its billing bound is verified.

Provider's public docs omit a maximum page size. No assumed bound is silently
used for paid calls. Bio scanning from collected profiles works independently.
"""
    if not 1<=max_results<=144:
        raise ValueError('User search needs a verified maximum page size (1..144)')
    week=now.strftime('%G-W%V')
    calls=0
    for term in ('zcat','zcash','Anonymous Cat'):
        def save(cur,rid,data):
            users=data.get('users')
            if not isinstance(users,list) or len(users)>max_results:
                raise ValueError('User-search page contract changed')
            seen=set()
            for raw in users:
                aid=profile(raw)['id']
                if aid in seen: raise ValueError('Duplicate profile')
                seen.add(aid)
                save_profile(cur,raw,rid,'keyword_discovery')
            return len(users),len(users),max(15,18*len(users))
        calls+=request(conn,key,'/twitter/user/search',f'discovery:{week}:{term}',{'query':term,'cursor':''},
                       18*max_results,'discovery',save)
    return calls


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--mode',choices=['daily','discover'],default='daily')
    parser.add_argument('--verified-user-search-max',type=int,default=0)
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'mode':'plan_only','paid_calls':0,'total_existing_ceiling':LIMIT,
          'initial_search_allocation':16800,'allocations':ALLOCATIONS,'duration_days':30,'max_unique_profiles':40,
          'bio_scanning':'included in saved profiles','keyword_search':'requires verified provider page-size bound'},indent=2))
        return
    if args.mode=='discover' and not 1<=args.verified_user_search_max<=144:
        parser.error('Verify provider user-search page-size bound before enabling paid keyword search')
    import psycopg2
    conn=psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
    now=datetime.now(timezone.utc)
    try:
        initialize(conn,now.replace(hour=0,minute=0,second=0,microsecond=0))
        with conn,conn.cursor() as cur:
            cur.execute('INSERT INTO crypto_social_tracking(id) VALUES (%s) ON CONFLICT DO NOTHING',(PILOT,))
            cur.execute('SELECT started_at,end_at FROM crypto_social_tracking WHERE id=%s',(PILOT,))
            started,until=cur.fetchone()
        if now>=until:
            print('Thirty-day pilot complete; no paid calls.');return
        key=os.environ['TWITTERAPI_IO_API_KEY']
        calls=0
        if args.mode=='daily':
            # Complete first-page coverage of original frozen windows, then stop.
            calls+=collect(conn,key,56,max_pages=1)
            seed_candidates(conn)
            calls+=daily_profiles(conn,key,now.date())
            slot=(now.date()-started.date()).days
            if slot<30:
                calls+=enrich(conn,key,now,slot)
        else:
            calls+=discover(conn,key,now,args.verified_user_search_max)
        seed_candidates(conn)
        with conn,conn.cursor() as cur:
            cur.execute('SELECT reserved_credits FROM crypto_social_pilot WHERE id=%s',(PILOT,))
            spent=cur.fetchone()[0]
        print(json.dumps({'requests_saved':calls,'reserved_credits':spent,'credit_limit':LIMIT,'tracking_end':str(until)}))
    finally:
        conn.close()

if __name__=='__main__':
    main()
