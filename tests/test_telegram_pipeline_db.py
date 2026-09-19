"""The Telegram layer end to end against a real Postgres.

The pure tests pin the rules; these pin the SQL, which is the half a fake connection cannot check.
One of these caught a query whose parameters were never passed - it had passed every mocked test.
Skipped unless CRYPTO_SOCIAL_TEST_DATABASE_URL points at a disposable database, the same convention
as tests/test_launchpad_archive.py.
"""
from datetime import datetime,timedelta,timezone
import os
import pytest

import telegram_analytics as analytics
import telegram_collector as collector
import telegram_mentions as mentions

NOW=datetime(2026,9,19,18,0,tzinfo=timezone.utc)
GRAD=NOW-timedelta(hours=5)
FLEX='fvHLJUwsynVHJrssbZ8MLNyku9jt2izUspbBD4Spump'


@pytest.fixture
def db():
    import psycopg2
    url=os.environ.get('CRYPTO_SOCIAL_TEST_DATABASE_URL')
    if not url:pytest.skip('Disposable Postgres not configured')
    conn=psycopg2.connect(url);conn.autocommit=True
    with conn.cursor() as cur:
        cur.execute('DROP SCHEMA IF EXISTS telegram_test CASCADE; CREATE SCHEMA telegram_test; '
                    'SET search_path TO telegram_test')
    conn.autocommit=False
    from launchpad_archive import setup as archive_setup
    archive_setup(conn);collector.setup(conn);mentions.setup(conn)
    with conn,conn.cursor() as cur:
        cur.execute('''INSERT INTO launchpad_tokens (network,token_address,symbol,name,dex,first_seen_at,
              last_seen_at,graduated,graduated_at,measure_pool,measure_pool_timing,launchpad_family)
              VALUES ('solana',%s,'FLEX','FAIRLAUNCH','pump-fun',%s,%s,true,%s,'POOL1','at_graduation','pump.fun')''',
                    (FLEX,GRAD,NOW,GRAD))
        cur.execute('''INSERT INTO launchpad_observations (network,token_address,observed_at,phase,fdv_usd,liquidity_usd)
                       VALUES ('solana',%s,%s,'post',450000,32000)''',(FLEX,GRAD-timedelta(minutes=5)))
        for channel_id,username in ((1,'alpha'),(2,'beta')):
            cur.execute('INSERT INTO telegram_channels (channel_id,username,title,active) VALUES (%s,%s,%s,true)',
                        (channel_id,username,username))
    yield conn
    conn.close()


def message(message_id,posted_at,text,**over):
    row=dict(message_id=message_id,posted_at=posted_at,text=text,is_forward=False,urls=[],raw={})
    row.update(over);return row


def seed(conn):
    collector.upsert_messages(conn,1,[message(10,GRAD-timedelta(minutes=6),f'new call {FLEX} mc $450K, easy 10x'),
                                      message(11,GRAD-timedelta(minutes=2),'$FLEX still early')])
    collector.upsert_messages(conn,2,[message(5,GRAD-timedelta(minutes=5),f'fwd {FLEX}',is_forward=True,
                                              forward_from_channel_id=1,forward_from_name='alpha')])
    prices=[dict(minute=GRAD-timedelta(minutes=6)+timedelta(minutes=i),open=1.01**i,high=1.1*1.01**i,
                 low=0.9,close=1.01**i,volume_usd=10) for i in range(200)]
    mentions.store_prices(conn,'solana',FLEX,'POOL1','minute',prices)
    mentions.store_prices(conn,'solana',FLEX,'POOL1','hour',
                          [dict(minute=GRAD-timedelta(minutes=6)+timedelta(hours=h),open=2.0,high=2.2,
                                low=1.8,close=2.0,volume_usd=5) for h in range(30)])


def test_collection_is_idempotent(db):
    rows=[message(10,GRAD,'hello')]
    assert collector.upsert_messages(db,1,rows)==(1,0)
    # Telegram's own (channel_id, message_id) is the key, so a re-read of the same window updates
    # rather than duplicating. This is the property the whole backfill design rests on.
    assert collector.upsert_messages(db,1,rows)==(0,1)


def test_derivation_resolves_joins_the_archive_and_orders_the_channels(db):
    seed(db)
    summary=mentions.derive(db,now=NOW)
    assert summary['messages']==3 and summary['contracts']==2 and summary['tickers']==1
    with db,db.cursor() as cur:
        cur.execute('''SELECT channel_id,reference_kind,resolution,graduated,launchpad_family,
                              seconds_to_graduation,is_first_monitored_mention,monitored_sequence,
                              claimed_multiple
                       FROM telegram_token_mentions ORDER BY mentioned_at''')
        rows=cur.fetchall()
    first=rows[0]
    assert first[1:5]==('contract','contract',True,'pump.fun')
    # Positive means the channel posted before graduation, measured against graduated_at and never
    # against first_pool_created.
    assert first[5]==360 and first[6] is True and first[7]==1
    assert first[8]==10.0                                   # the claim is stored, not believed
    forward=[r for r in rows if r[0]==2][0]
    # A forward is not a discovery: it takes no sequence position, so the relay channel cannot be
    # credited with being early to something it re-transmitted.
    assert forward[6] is False and forward[7] is None


def test_outcomes_keep_pending_apart_from_observed(db):
    seed(db);mentions.derive(db,now=NOW)
    summary=mentions.compute_outcomes(db,now=NOW)
    assert summary['observed'] and summary['pending']
    with db,db.cursor() as cur:
        cur.execute('''SELECT rung_minutes,status,return_pct,position_value FROM telegram_mention_outcomes o
                       JOIN telegram_token_mentions m ON m.id=o.mention_id
                       WHERE m.channel_id=1 AND m.message_id=10 ORDER BY rung_minutes''')
        rungs={r[0]:r for r in cur.fetchall()}
        cur.execute('''SELECT market_cap_at_mention,peak_multiple FROM telegram_mention_summary s
                       JOIN telegram_token_mentions m ON m.id=s.mention_id
                       WHERE m.channel_id=1 AND m.message_id=10''')
        context=cur.fetchone()
    assert rungs[10][1]=='observed' and rungs[10][2]>0 and rungs[10][3]>100
    # 24h has not elapsed at NOW. Counting it as anything other than pending would make a channel
    # that posted recently look worse than one that posted last week.
    assert rungs[1440][1]=='pending' and rungs[1440][2] is None
    # Market cap comes from the archive's observation, not from the message's "$450K" claim.
    assert context[0]==450000.0 and context[1]>1


def test_price_targets_stop_asking_once_the_history_is_complete(db):
    seed(db);mentions.derive(db,now=NOW)
    assert mentions.price_targets(db,now=NOW)==[]


def test_analytics_case_study_and_health_run_against_real_sql(db):
    seed(db);mentions.derive(db,now=NOW);mentions.compute_outcomes(db,now=NOW)
    stats=analytics.channel_stats(db,now=NOW,store=True)
    alpha=[s for s in stats if s['channel_id']==1][0]
    assert alpha['graduated_tokens']==2 and alpha['pre_graduation']==2
    # Below the sample floor, so no label - four-for-four is the most impressive-looking record
    # there is and it means nothing.
    assert alpha['typology'] is None and 'below the' in alpha['typology_reason']
    assert analytics.pairs(db,now=NOW,store=True)[0]['forward_pairs']>0
    graph=analytics.propagation(db,FLEX)
    assert graph['original_posts']==2 and graph['forwards']==1
    case=analytics.case_study(db,FLEX)
    satisfied={i['item']:i['satisfied'] for i in case['acceptance']}
    assert satisfied['first observed Telegram mention'] and satisfied['graduation timing']
    # Only one channel posted originally, and the 24h rung has not elapsed, so the acceptance case
    # correctly does not pass yet. A case study that passed here would be the bug.
    assert case['passed'] is False
    health=analytics.health(db,now=NOW)
    assert health['resolution']['contract']==2 and health['coverage']['archive_share']==1.0
    assert set(health['outcome_status'])<= {'observed','pending','unobserved'}
