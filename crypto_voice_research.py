"""Freeze weekly research cohorts and score their next seven days from saved evidence only."""
import json,subprocess
from datetime import timedelta
from pathlib import Path
SCHEMA='''CREATE TABLE IF NOT EXISTS crypto_voice_snapshots (
 coin text NOT NULL,week date NOT NULL,created_at timestamptz NOT NULL,model text NOT NULL,
 selection jsonb NOT NULL,baseline jsonb NOT NULL,coverage jsonb NOT NULL,evaluation jsonb,
 PRIMARY KEY(coin,week));'''

def initialize(cur):cur.execute(SCHEMA)

def snapshot(conn,coin,now):
    week=(now-timedelta(days=now.weekday())).date()
    with conn,conn.cursor() as cur:
        initialize(cur)
        cur.execute('SELECT selection FROM crypto_voice_snapshots WHERE coin=%s AND week=%s',(coin,week))
        prior=cur.fetchone()
        if prior:return [a['id'] for a in prior[0]]
        cur.execute('''SELECT p.id,p.author_id,a.handle,p.text,p.posted_at,p.kind,p.url,a.followers,a.observed_at AS followers_observed_at,
         coalesce((SELECT jsonb_agg(jsonb_build_object('target_id',e.target_id,'target',e.target_id,'kind',e.kind)) FROM crypto_social_edges e WHERE e.post_id=p.id),'[]'::jsonb) AS edges
         FROM crypto_social_posts p JOIN crypto_social_accounts a ON a.id=p.author_id WHERE p.posted_at<%s AND EXISTS(
         SELECT 1 FROM crypto_social_matches m JOIN crypto_social_windows w ON w.id=m.window_id WHERE m.post_id=p.id AND w.coin=%s)
         ORDER BY p.posted_at,p.id LIMIT 50000''',(now,coin))
        names=[d[0] for d in cur.description];posts=[dict(zip(names,r)) for r in cur.fetchall()]
    output=subprocess.run(['node','--experimental-strip-types','scripts/crypto-voice-snapshot.mts'],cwd=Path(__file__).parent,input=json.dumps({'posts':posts,'coin':coin,'cutoff':now.isoformat()},default=str),text=True,capture_output=True,check=True)
    data=json.loads(output.stdout)
    with conn,conn.cursor() as cur:
        cur.execute('INSERT INTO crypto_voice_snapshots(coin,week,created_at,model,selection,baseline,coverage) VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING',(coin,week,now,data['model'],json.dumps(data['selection']),json.dumps(data['baseline']),json.dumps(data['coverage'])))
        cur.execute('SELECT selection FROM crypto_voice_snapshots WHERE coin=%s AND week=%s',(coin,week));return [a['id'] for a in cur.fetchone()[0]]

def evaluate(conn,now):
    with conn,conn.cursor() as cur:
        initialize(cur)
        cur.execute('SELECT coin,week,created_at,selection,baseline FROM crypto_voice_snapshots WHERE evaluation IS NULL AND created_at<=%s',(now-timedelta(days=7),));due=cur.fetchall()
        for coin,week,start,selection,baseline in due:
            end=start+timedelta(days=7);groups={'Mixed cohort':[v['id'] for v in selection],'Follower baseline':baseline['reach'],'Posting baseline':baseline['posting']};result={}
            for label,ids in groups.items():
                # Only posts first captured during the evaluation window count. Later backfills cannot rewrite success.
                cur.execute('''WITH heldout AS (SELECT DISTINCT p.* FROM crypto_social_posts p JOIN crypto_social_matches m ON m.post_id=p.id JOIN crypto_social_windows w ON w.id=m.window_id
                 WHERE w.coin=%s AND p.posted_at>=%s AND p.posted_at<%s AND p.first_seen_at<%s)
                 SELECT (SELECT count(DISTINCT author_id) FROM heldout WHERE author_id=ANY(%s)),
                 (SELECT count(*) FROM heldout WHERE author_id=ANY(%s)),
                 (SELECT count(DISTINCT (e.target_id,p.author_id)) FROM heldout p JOIN crypto_social_edges e ON e.post_id=p.id WHERE e.target_id=ANY(%s) AND p.author_id!=e.target_id)''',(coin,start,end,end,ids,ids,ids))
                active,posts,interactors=cur.fetchone();result[label]={'accounts':len(ids),'active':active,'newPosts':posts,'newInteractors':interactors}
                # Price-forward context from the immutable event study: median 24h move after members' episode posts in the window.
                cur.execute("SELECT to_regclass('crypto_price_events')")
                if cur.fetchone()[0]:
                    cur.execute('''SELECT count(*)::int,percentile_cont(0.5) WITHIN GROUP(ORDER BY price_after_24h/price_0-1)::float
                        FROM crypto_price_events WHERE coin=%s AND episode AND account_id=ANY(%s) AND posted_at>=%s AND posted_at<%s''',(coin,ids,start,end))
                    n,median=cur.fetchone();result[label].update({'priceEpisodes':n,'median24hReturn':median})
            cur.execute('UPDATE crypto_voice_snapshots SET evaluation=%s WHERE coin=%s AND week=%s AND evaluation IS NULL',(json.dumps({'evaluated_at':now.isoformat(),'days':7,'groups':result,'note':'Collected attention and archived pool prices only; uneven coverage and identity errors limit comparisons. Price moves after a post are association, not attribution.'}),coin,week))
