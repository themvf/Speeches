"""Read saved public-post evidence; no database mutations or provider calls."""
import json,os
from pathlib import Path
import psycopg2
conn=psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
conn.set_session(readonly=True)
try:
    query=Path('apps/web/lib/server/crypto-watchers-query.ts').read_text().split('`')[1]
    with conn,conn.cursor() as cur:
        cur.execute(query)
        keys=[c.name for c in cur.description]
        rows=[dict(zip(keys,row)) for row in cur.fetchall()]
    Path('/tmp/crypto-watcher-evidence.json').write_text(json.dumps(rows,default=lambda x:x.isoformat()))
finally:conn.close()
