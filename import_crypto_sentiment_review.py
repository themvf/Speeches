"""Import the user-requested direct assistant review. No model or X API calls."""
import collections,json,os,pathlib
import psycopg2
VERSION='assistant-zcat-50-v1'
def load_review():
    rows=json.loads((pathlib.Path(__file__).parent/'data/reviews/zcat-sentiment-50-2026-09-14.json').read_text())
    if len(rows)!=50 or len({(r['post_id'],r['coin']) for r in rows})!=50:raise ValueError('Expected 50 distinct post/coin assessments')
    for r in rows:
        if r['coin']!='ZCAT' or r['label'] not in {'bullish','bearish','neutral','mixed','unclear'} or r['confidence'] not in {'High','Medium','Low'}:raise ValueError('Invalid review label')
        if not isinstance(r['exclude_as_unrelated'],bool) or not r['reason']:raise ValueError('Invalid explanation or exclusion')
    return rows

def main():
    rows=load_review()
    with psycopg2.connect(os.environ['DATABASE_URL']) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT pg_advisory_xact_lock(7284362)')
            cur.execute('''CREATE TABLE IF NOT EXISTS crypto_assistant_sentiment_reviews(post_id text NOT NULL,coin text NOT NULL,version text NOT NULL,label text NOT NULL CHECK(label IN ('bullish','bearish','neutral','mixed','unclear')),confidence_label text NOT NULL CHECK(confidence_label IN ('High','Medium','Low')),explanation text NOT NULL,excluded boolean NOT NULL,model text NOT NULL,observed_at timestamptz NOT NULL DEFAULT now(),PRIMARY KEY(post_id,coin,version))''')
            cur.execute('SELECT id::text FROM crypto_social_posts WHERE id::text=ANY(%s)',([r['post_id'] for r in rows],))
            if {r[0] for r in cur.fetchall()}!={r['post_id'] for r in rows}:raise ValueError('Some reviewed posts are missing from the saved corpus')
            for r in rows:
                cur.execute('''INSERT INTO crypto_assistant_sentiment_reviews(post_id,coin,version,label,confidence_label,explanation,excluded,model) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING''',(r['post_id'],r['coin'],VERSION,r['label'],r['confidence'],r['reason'],r['exclude_as_unrelated'],'ChatGPT assistant — direct text review'))
            cur.execute('SELECT count(*) FROM crypto_assistant_sentiment_reviews WHERE version=%s',(VERSION,))
            if cur.fetchone()[0]!=50:raise ValueError('Import count mismatch')
    print(json.dumps({'saved_reviews':50,'labels':dict(collections.Counter(r['label'] for r in rows)),'excluded_from_daily_sentiment':sum(r['exclude_as_unrelated'] for r in rows),'version':VERSION}))
if __name__=='__main__':main()
