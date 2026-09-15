"""One-shot DPONS capture, bounded by its persistent 150,000-credit reservation ledger."""
import argparse
import json
import os
from crypto_social_history import setup, settings, start_batch, collect_history, LIMIT


def capture(conn, key):
    coin = 'DPONS'
    campaign, _, _, _ = settings(coin)
    setup(conn, coin)
    calls = 0
    # 500 maximum requests across this campaign, including previous runs.
    for _ in range(7):
        batch_id = start_batch(conn, None, 80, coin)
        try:
            saved = collect_history(conn, key, 80, batch_id=batch_id, coin=coin)
        except Exception:
            with conn, conn.cursor() as cur:
                cur.execute("UPDATE crypto_social_history_batches SET status='stopped',completed_at=now() WHERE id=%s", (batch_id,))
            raise
        with conn, conn.cursor() as cur:
            cur.execute("UPDATE crypto_social_history_batches SET status='completed',completed_at=now(),requests_saved=%s WHERE id=%s", (saved, batch_id))
        calls += saved
        if saved < 80:
            break
    with conn, conn.cursor() as cur:
        cur.execute('SELECT reserved_credits,credit_limit FROM crypto_social_history_campaign WHERE id=%s', (campaign,))
        reserved, ceiling = cur.fetchone()
        cur.execute("SELECT coalesce(sum(estimated_credits),0),count(*) FROM crypto_social_requests WHERE parameters->>'campaign'=%s", (campaign,))
        estimated, requests = cur.fetchone()
        cur.execute('''SELECT count(DISTINCT p.id),count(DISTINCT p.author_id),min(p.posted_at),max(p.posted_at)
            FROM crypto_social_posts p JOIN crypto_social_matches m ON m.post_id=p.id
            JOIN crypto_social_history_windows h ON h.window_id=m.window_id WHERE h.campaign_id=%s''', (campaign,))
        posts, authors, first, last = cur.fetchone()
        cur.execute('''SELECT count(*),count(*) FILTER(WHERE w.pages>0),count(*) FILTER(WHERE w.status='search_exhausted')
            FROM crypto_social_windows w JOIN crypto_social_history_windows h ON h.window_id=w.id WHERE h.campaign_id=%s''', (campaign,))
        windows, searched, exhausted = cur.fetchone()
    return dict(coin=coin,requests_this_run=calls,total_requests=requests,reserved_credits=reserved,
                estimated_credits=estimated,credit_ceiling=ceiling,unique_posts=posts,authors=authors,
                first_post=first,last_post=last,windows=windows,searched_windows=searched,
                exhausted_windows=exhausted,stop_reason='budget ceiling' if reserved>=ceiling else 'no eligible pages remain')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        campaign, start, end, query = settings('DPONS')
        print(json.dumps(dict(mode='plan_only',campaign=campaign,start=start,end_exclusive=end,
                              query=query,credit_ceiling=LIMIT,max_requests=500,paid_calls=0),default=str))
        return
    import psycopg2
    conn = psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
    try:
        print(json.dumps(capture(conn,os.environ['TWITTERAPI_IO_API_KEY']),default=str),flush=True)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
