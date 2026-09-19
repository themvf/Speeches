"""Turn raw Telegram messages into archive-joined token mentions with measured outcomes.

Three stages, each re-runnable from the layer below it:

  --derive   messages -> telegram_token_mentions   (extraction + resolution + archive join)
  --prices   mentions -> telegram_price_points     (GeckoTerminal OHLCV, mention-anchored)
  --outcomes prices   -> telegram_mention_outcomes (per-rung returns, with a three-way status)

The archive's own ladder is anchored to GRADUATION and samples a quarter of Solana graduates, so it
cannot answer "what happened after this channel posted". That is why there is a separate,
mention-anchored price layer: OHLCV is historical and can be fetched after the fact, unlike the
opening trade window, which expires.

Methodology rules this module exists to enforce, from docs/telegram-osint-spec.md:
  * every resolved mention stays in the denominator, including tokens that went to zero
  * 'not yet eligible' (the horizon has not elapsed) is never counted as a loss
  * a channel's own claim of a multiple is never an input to its measured performance
  * forwards are flagged, so one post copied across five channels is not five discoveries

Default with no flags is a no-network plan.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os

import telegram_extract as extract
from launchpad_chains import CHAINS, SOLANA, family_of

GECKO='https://api.geckoterminal.com/api/v2/networks/'
REQUEST_WAIT=2.0                 # unkeyed GeckoTerminal ceiling is ~30/minute; never burst
RUNGS=(10,30,60,180,1440)        # minutes after the mention, as the brief specifies
# How far a candle may sit from the target instant and still be used. A minute candle is matched
# tightly; the 24h rung is matched against hourly candles, so its tolerance is an hour. The distance
# is stored per row (price_age_seconds), so a later analysis can tighten this without re-fetching.
TOLERANCE={'minute':300,'hour':3600}
MINUTE_RUNG_LIMIT=180            # rungs beyond this read hourly candles
SETTLE_SECONDS=120               # grace after a horizon before 'no price' counts as unobserved
# A ticker is only resolved when the archive holds exactly one token with that symbol near the
# message. Wider windows resolve more mentions and resolve some of them wrongly, and a wrong
# resolution is worse than an unresolved one: it puts another channel's token in this channel's
# denominator.
TICKER_WINDOW_HOURS=72
# Relay attribution. A Telegram forward announces itself in the message header; a copy-paste repost
# does not, and a repost is the commoner shape in call channels - which makes an unattributed copy
# the single easiest way for a relay to be credited as a discovery. Near-duplicate text about the
# same token, from a different channel, inside this window, is treated as a repost of the earlier
# post rather than as independent discovery.
REPOST_WINDOW_HOURS=24
REPOST_SIMILARITY=0.6            # Jaccard over word 5-grams; short texts fall back to exact match
REPOST_MIN_WORDS=8               # below this, wording is too generic for similarity to mean anything
PRICE_WINDOW_MINUTES=1500        # mention -> +25h, so the 24h rung and the peak are both covered


def setup(conn):
    from pathlib import Path
    with conn,conn.cursor() as cur:
        cur.execute(Path(__file__).with_name('sql').joinpath('telegram_osint.sql').read_text())


def number(value):
    try:
        out=float(value)
        return out if math.isfinite(out) else None
    except (TypeError,ValueError):return None


# ------------------------------------------------------------------ stage 1: derive

def resolve_reference(reference,archive_lookup):
    """One extracted reference -> (token_address, resolution, confidence).

    `archive_lookup` is (kind, value) -> list of candidate token rows. Kept as a callable so the
    resolution rules are testable without a database, and so the rules are in one place rather than
    spread through SQL.
    """
    if reference['reference_kind']=='contract':
        # The address is a fact about the message whether or not our archive holds the token. It is
        # resolved; archive membership is a separate question answered by the join below, and
        # conflating them would make "we have not archived this token" look like "the channel posted
        # something unreadable".
        return reference['raw_reference'],'contract',1.0
    matches=archive_lookup('ticker',reference['raw_reference'])
    if not matches:return None,'unresolved_not_in_archive',None
    if len(matches)>1:return None,'unresolved_ambiguous',None
    return matches[0]['token_address'],'ticker_unique',0.7


def message_references(text):
    """References worth resolving in one message.

    When a message carries a contract, its cashtags are dropped: the contract is what the message is
    about, and a ticker beside it resolving to some other token is how a channel ends up credited
    with a call it never made. This is the conservative half of "never silently assign an ambiguous
    ticker".
    """
    rows=extract.references(text)
    if any(r['reference_kind']=='contract' for r in rows):
        return [r for r in rows if r['reference_kind']=='contract']
    return rows


def shingles(text,size=5):
    """Normalised word 5-grams. Same device the crypto rings use: shared phrasing is what separates
    a copied post from two people independently saying 'new call' about the same token."""
    words=[w for w in ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in (text or '')).split()]
    if len(words)<size:return {' '.join(words)} if words else set()
    return {' '.join(words[i:i+size]) for i in range(len(words)-size+1)}


def similarity(left,right):
    a,b=shingles(left),shingles(right)
    if not a or not b:return 0.0
    return len(a&b)/len(a|b)


def classify_origin(text,is_forward,earlier):
    """'original' | 'forward' | 'repost', and what it relays, for one mention of one token.

    `earlier` is [(channel_id, text)] for mentions of the SAME token from OTHER channels inside the
    window, oldest first. Returning the relayed channel rather than a bare flag is what lets the
    propagation graph show 'original post -> reposted alert -> later independent mention' instead of
    three sightings that look alike.
    """
    if is_forward:return 'forward',None,'telegram forward header'
    words=len((text or '').split())
    for channel_id,other in earlier:
        if words<REPOST_MIN_WORDS or len((other or '').split())<REPOST_MIN_WORDS:
            # Too short for similarity to carry information, so only an exact copy counts. A bare
            # contract address posted twice is two posts, not a copy - that is how these channels
            # legitimately talk.
            if text and other and text.strip()==other.strip():
                return 'repost',channel_id,'identical text'
            continue
        score=similarity(text,other)
        if score>=REPOST_SIMILARITY:
            return 'repost',channel_id,f'{score:.2f} 5-gram overlap with an earlier post'
    return 'original',None,None


def _archive_lookup(conn,network,at):
    """Symbol -> candidate archive tokens near a message. Case-insensitive, time-scoped."""
    window=timedelta(hours=TICKER_WINDOW_HOURS)
    def lookup(kind,value):
        if kind!='ticker':return []
        with conn,conn.cursor() as cur:
            cur.execute('''SELECT token_address,symbol FROM launchpad_tokens
                           WHERE network=%s AND upper(symbol)=upper(%s)
                             AND coalesce(graduated_at,first_seen_at) BETWEEN %s AND %s''',
                        (network,value,at-window,at+window))
            return [dict(token_address=r[0],symbol=r[1]) for r in cur.fetchall()]
    return lookup


def derive(conn,network='solana',limit=5000,now=None):
    """Extract and resolve references for every message not yet derived."""
    now=now or datetime.now(timezone.utc)
    with conn,conn.cursor() as cur:
        cur.execute('''SELECT channel_id,message_id,posted_at,text,is_forward FROM telegram_messages
                       WHERE references_derived_at IS NULL ORDER BY posted_at LIMIT %s''',(limit,))
        messages=cur.fetchall()
    summary=dict(messages=len(messages),references=0,resolved=0,unresolved=0,contracts=0,tickers=0)
    for channel_id,message_id,posted_at,text,is_forward in messages:
        rows=message_references(text)
        claims=extract.claims(text)
        lookup=_archive_lookup(conn,network,posted_at)
        for reference in rows:
            address,resolution,confidence=resolve_reference(reference,lookup)
            summary['references']+=1
            summary['contracts' if reference['reference_kind']=='contract' else 'tickers']+=1
            summary['resolved' if address else 'unresolved']+=1
            joined=archive_row(conn,network,address) if address else {}
            with conn,conn.cursor() as cur:
                cur.execute('''INSERT INTO telegram_token_mentions
                      (channel_id,message_id,network,token_address,raw_reference,reference_kind,
                       resolution,confidence,mentioned_at,is_forward,graduated,graduated_at,
                       launchpad_family,measure_pool,measure_pool_timing,seconds_to_graduation,
                       claimed_multiple,claimed_market_cap)
                      VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                      ON CONFLICT (channel_id,message_id,network,raw_reference) DO UPDATE SET
                        token_address=EXCLUDED.token_address,resolution=EXCLUDED.resolution,
                        confidence=EXCLUDED.confidence,graduated=EXCLUDED.graduated,
                        graduated_at=EXCLUDED.graduated_at,launchpad_family=EXCLUDED.launchpad_family,
                        measure_pool=EXCLUDED.measure_pool,measure_pool_timing=EXCLUDED.measure_pool_timing,
                        seconds_to_graduation=EXCLUDED.seconds_to_graduation,derived_at=now()''',
                            (channel_id,message_id,network,address,reference['raw_reference'],
                             reference['reference_kind'],resolution,confidence,posted_at,is_forward,
                             joined.get('graduated'),joined.get('graduated_at'),
                             joined.get('launchpad_family'),joined.get('measure_pool'),
                             joined.get('measure_pool_timing'),
                             (joined['graduated_at']-posted_at).total_seconds()
                             if joined.get('graduated_at') else None,
                             claims['claimed_multiple'],claims['claimed_market_cap']))
        with conn,conn.cursor() as cur:
            cur.execute('UPDATE telegram_messages SET references_derived_at=%s WHERE channel_id=%s AND message_id=%s',
                        (now,channel_id,message_id))
    summary['origins']=attribute_relays(conn,network,now)
    summary.update(sequence(conn,network))
    return summary


def archive_row(conn,network,address):
    """The archive's view of a token, copied onto the mention at derivation time."""
    with conn,conn.cursor() as cur:
        cur.execute('''SELECT graduated,graduated_at,launchpad_family,dex,measure_pool,measure_pool_timing
                       FROM launchpad_tokens WHERE network=%s AND token_address=%s''',(network,address))
        row=cur.fetchone()
    if not row:return {}
    chain=CHAINS.get(network,SOLANA)
    return dict(graduated=row[0],graduated_at=row[1],
                # launchpad_family is the column analysis filters on; `dex`/`launchpad` is only ever
                # one side of a pairing and silently undercounts fast graduators.
                launchpad_family=row[2] or family_of(row[3],chain),
                measure_pool=row[4],measure_pool_timing=row[5])


def attribute_relays(conn,network='solana',now=None):
    """Label every resolved mention original / forward / repost before the sequence is computed.

    This runs on text, per token, in time order, so a repost is only ever attributed to a post that
    came BEFORE it. Getting this wrong in the permissive direction is the expensive failure: an
    unattributed copy-paste becomes a second independent sighting, the token looks like it was
    discovered twice, and the relay channel's 'first among monitored' count is inflated by exactly
    the tokens it was slowest on.
    """
    with conn,conn.cursor() as cur:
        cur.execute('''SELECT m.id,m.token_address,m.channel_id,m.mentioned_at,m.is_forward,g.text
                       FROM telegram_token_mentions m
                       JOIN telegram_messages g ON g.channel_id=m.channel_id AND g.message_id=m.message_id
                       WHERE m.network=%s AND m.token_address IS NOT NULL
                       ORDER BY m.token_address,m.mentioned_at''',(network,))
        rows=cur.fetchall()
    counts=dict(original=0,forward=0,repost=0)
    history={}
    updates=[]
    for mention_id,token,channel_id,at,is_forward,text in rows:
        seen=history.setdefault(token,[])
        earlier=[(cid,body) for cid,when,body in seen
                 if cid!=channel_id and (at-when).total_seconds()<=REPOST_WINDOW_HOURS*3600]
        origin,relay_of,reason=classify_origin(text,is_forward,earlier)
        counts[origin]+=1
        updates.append((origin,relay_of,reason,mention_id))
        seen.append((channel_id,at,text))
    if updates:
        from psycopg2.extras import execute_values
        with conn,conn.cursor() as cur:
            execute_values(cur,'''UPDATE telegram_token_mentions m SET mention_origin=v.origin,
                                    relay_of_channel_id=v.relay_of,relay_reason=v.reason
                                  FROM (VALUES %s) AS v(origin,relay_of,reason,id)
                                  WHERE m.id=v.id''',updates,
                           template='(%s,%s::bigint,%s,%s::bigint)')
    return counts


def sequence(conn,network='solana'):
    """Order the monitored channels' ORIGINAL posts per token: first, position, lag behind first.

    Forwards AND reposts are excluded from the ordering. A relayed post is the same discovery as the
    post it copies, and letting it take a sequence position would credit a relay channel with being
    early to something it re-transmitted - which is the whole attribution question this layer exists
    to answer.
    """
    with conn,conn.cursor() as cur:
        cur.execute('''WITH ordered AS (
                         SELECT id,row_number() OVER w AS seq,
                                extract(epoch FROM mentioned_at-min(mentioned_at) OVER w2) AS lag
                         FROM telegram_token_mentions
                         WHERE network=%s AND token_address IS NOT NULL AND mention_origin='original'
                         WINDOW w AS (PARTITION BY token_address ORDER BY mentioned_at,channel_id),
                                w2 AS (PARTITION BY token_address))
                       UPDATE telegram_token_mentions m
                          SET monitored_sequence=o.seq,is_first_monitored_mention=(o.seq=1),
                              seconds_after_first_mention=o.lag
                         FROM ordered o WHERE o.id=m.id''',(network,))
        updated=cur.rowcount
        # Forwards carry an explicit false rather than NULL, so "is this the first mention" never
        # has to be read as a three-valued question downstream.
        cur.execute('''UPDATE telegram_token_mentions SET is_first_monitored_mention=false,
                              monitored_sequence=NULL,seconds_after_first_mention=NULL
                       WHERE network=%s AND mention_origin<>'original' ''',(network,))
    return dict(sequenced=updated)


# ------------------------------------------------------------------ stage 2: prices

def parse_ohlcv(payload):
    """GeckoTerminal OHLCV -> rows. [timestamp, open, high, low, close, volume]."""
    lists=(((payload or {}).get('data') or {}).get('attributes') or {}).get('ohlcv_list') or []
    rows=[]
    for entry in lists:
        if not entry or len(entry)<6:continue
        stamp=number(entry[0])
        if stamp is None:continue
        rows.append(dict(minute=datetime.fromtimestamp(int(stamp),timezone.utc),
                         open=number(entry[1]),high=number(entry[2]),low=number(entry[3]),
                         close=number(entry[4]),volume_usd=number(entry[5])))
    return rows


def price_targets(conn,network='solana',limit=200,now=None):
    """Mentions that still need price history, oldest first.

    A mention is a target while any of its rungs could still be filled: either it has no prices at
    all, or its newest candle has not yet reached the last rung. A token whose pool stopped trading
    keeps returning empty and is bounded by attempts, not by pretending its outcome is unknown.
    """
    now=now or datetime.now(timezone.utc)
    with conn,conn.cursor() as cur:
        cur.execute('''SELECT m.id,m.token_address,coalesce(m.measure_pool,t.measure_pool),m.mentioned_at
                       FROM telegram_token_mentions m
                       LEFT JOIN launchpad_tokens t
                         ON t.network=m.network AND t.token_address=m.token_address
                       WHERE m.network=%s AND m.token_address IS NOT NULL
                         AND coalesce(m.measure_pool,t.measure_pool) IS NOT NULL
                         AND (NOT EXISTS (SELECT 1 FROM telegram_price_points p
                                          WHERE p.network=m.network AND p.token_address=m.token_address)
                              OR (SELECT max(p.minute) FROM telegram_price_points p
                                  WHERE p.network=m.network AND p.token_address=m.token_address)
                                 < LEAST(%s,m.mentioned_at+make_interval(mins=>%s)))
                       ORDER BY m.mentioned_at LIMIT %s''',
                    (network,now,max(RUNGS),limit))
        return [dict(mention_id=r[0],token_address=r[1],pool=r[2],mentioned_at=r[3]) for r in cur.fetchall()]


def fetch_prices(conn,network='solana',limit=200,fetch=None,wait=None,now=None):
    """Fill mention-anchored OHLCV. Minute candles for the short rungs, hourly for the 24h rung."""
    import requests,time
    fetch=fetch or requests.get;wait=wait or time.sleep;now=now or datetime.now(timezone.utc)
    summary=dict(targets=0,requests=0,points=0,errors=[])
    def get(url):
        wait(REQUEST_WAIT)
        response=fetch(url,timeout=25,headers={'Accept':'application/json'})
        if getattr(response,'status_code',0)!=200:raise ValueError('HTTP '+str(getattr(response,'status_code',0)))
        return response.json()
    seen=set()
    for target in price_targets(conn,network,limit,now=now):
        key=(target['token_address'],target['pool'])
        if key in seen:continue
        seen.add(key);summary['targets']+=1
        base=GECKO+network+'/pools/'+target['pool']+'/ohlcv/'
        for resolution,timeframe,span,count in (('minute','minute',MINUTE_RUNG_LIMIT+30,1000),
                                                ('hour','hour',PRICE_WINDOW_MINUTES,48)):
            before=int((target['mentioned_at']+timedelta(minutes=span)).timestamp())
            try:
                payload=get(base+timeframe+'?aggregate=1&limit='+str(count)+'&currency=usd&before_timestamp='+str(before))
                summary['requests']+=1
            except (ValueError,requests.RequestException) as error:
                summary['errors'].append(f"{target['token_address']} {resolution}: {type(error).__name__} {str(error)[:120]}")
                continue
            rows=parse_ohlcv(payload)
            summary['points']+=store_prices(conn,network,target['token_address'],target['pool'],resolution,rows)
    return summary


def store_prices(conn,network,token,pool,resolution,rows):
    if not rows:return 0
    from psycopg2.extras import execute_values
    with conn,conn.cursor() as cur:
        execute_values(cur,'''INSERT INTO telegram_price_points
              (network,token_address,pool,minute,resolution,open,high,low,close,volume_usd)
              VALUES %s ON CONFLICT (network,token_address,pool,resolution,minute) DO NOTHING''',
                       [(network,token,pool,r['minute'],resolution,r['open'],r['high'],r['low'],
                         r['close'],r['volume_usd']) for r in rows])
    return len(rows)


# ------------------------------------------------------------------ stage 3: outcomes

def nearest(points,target,tolerance):
    """The candle closest to an instant, and how far off it was. None when nothing is close enough.

    Returning the distance rather than silently accepting the nearest row is what keeps a price
    stitched from a thin, gappy pool distinguishable from a real reading.
    """
    best=None
    for point in points:
        delta=abs((point['minute']-target).total_seconds())
        if delta<=tolerance and (best is None or delta<best[1]):best=(point,delta)
    return best if best else (None,None)


def mention_outcome(points,mentioned_at,now,rungs=RUNGS):
    """Every rung for one mention, from its candles. Pure, so the status rules are testable.

    `points` is [{minute, resolution, open, high, low, close}]. The three statuses are the whole
    point: `pending` means the horizon has not elapsed, and a pending rung must never be counted as
    a failure - doing so makes any channel that posted recently look worse than one that did not.
    """
    minute=[p for p in points if p['resolution']=='minute']
    hourly=[p for p in points if p['resolution']=='hour']
    base,base_age=nearest(minute or hourly,mentioned_at,TOLERANCE['minute' if minute else 'hour'])
    out=dict(base_price=(base or {}).get('open') or (base or {}).get('close'),
             base_price_age_seconds=base_age,rungs={},peak=None)
    for rung in rungs:
        target=mentioned_at+timedelta(minutes=rung)
        if now<target+timedelta(seconds=SETTLE_SECONDS):
            out['rungs'][rung]=dict(status='pending');continue
        series,resolution=(minute,'minute') if rung<=MINUTE_RUNG_LIMIT and minute else (hourly,'hour')
        point,age=nearest(series,target,TOLERANCE[resolution])
        if out['base_price'] is None:
            out['rungs'][rung]=dict(status='unobserved',unobserved_reason='no price at the mention');continue
        if point is None:
            # An elapsed rung with no candle is an outcome we do not have, not a zero. A pool that
            # stopped trading and a collection gap are indistinguishable here, so neither is guessed.
            out['rungs'][rung]=dict(status='unobserved',
                                    unobserved_reason=f'no {resolution} candle within {TOLERANCE[resolution]}s');continue
        price=point.get('close')
        out['rungs'][rung]=dict(status='observed',base_price=out['base_price'],rung_price=price,
                                return_pct=(price/out['base_price']-1)*100 if price and out['base_price'] else None,
                                position_value=100*price/out['base_price'] if price and out['base_price'] else None,
                                price_age_seconds=age)
    window=[p for p in minute or hourly if p['minute']>=mentioned_at]
    highs=[(p['high'],p['minute']) for p in window if p.get('high') is not None]
    if highs and out['base_price']:
        peak,at=max(highs)
        out['peak']=dict(peak_price=peak,peak_at=at,peak_multiple=peak/out['base_price'],
                         minutes_to_peak=(at-mentioned_at).total_seconds()/60,
                         price_points=len(window),
                         window_minutes=int((max(p['minute'] for p in window)-mentioned_at).total_seconds()/60))
    return out


def market_context(conn,network,token,mentioned_at,tolerance_minutes=30):
    """FDV and liquidity nearest the mention, from the archive's own observations.

    These come from launchpad_observations rather than from the message, because a channel quoting
    "$450K mcap" is making a claim. The archive samples a fraction of graduates, so this is often
    absent - which is why it is a nullable context column and never a filter.
    """
    with conn,conn.cursor() as cur:
        cur.execute('''SELECT fdv_usd,liquidity_usd,abs(extract(epoch FROM observed_at-%s)) AS delta
                       FROM launchpad_observations WHERE network=%s AND token_address=%s
                       ORDER BY delta LIMIT 1''',(mentioned_at,network,token))
        row=cur.fetchone()
    if not row or row[2] is None or row[2]>tolerance_minutes*60:return {}
    return dict(fdv_usd=row[0],liquidity_usd=row[1],age_seconds=row[2])


def compute_outcomes(conn,network='solana',limit=2000,now=None):
    """Write telegram_mention_outcomes + telegram_mention_summary for mentions with price history."""
    now=now or datetime.now(timezone.utc)
    with conn,conn.cursor() as cur:
        cur.execute('''SELECT id,token_address,mentioned_at FROM telegram_token_mentions
                       WHERE network=%s AND token_address IS NOT NULL
                       ORDER BY mentioned_at DESC LIMIT %s''',(network,limit))
        mentions=cur.fetchall()
    summary=dict(mentions=len(mentions),observed=0,pending=0,unobserved=0,with_peak=0)
    for mention_id,token,mentioned_at in mentions:
        with conn,conn.cursor() as cur:
            cur.execute('''SELECT minute,resolution,open,high,low,close FROM telegram_price_points
                           WHERE network=%s AND token_address=%s AND minute BETWEEN %s AND %s
                           ORDER BY minute''',
                        (network,token,mentioned_at-timedelta(minutes=30),
                         mentioned_at+timedelta(minutes=PRICE_WINDOW_MINUTES)))
            points=[dict(minute=r[0],resolution=r[1],open=r[2],high=r[3],low=r[4],close=r[5])
                    for r in cur.fetchall()]
        result=mention_outcome(points,mentioned_at,now)
        context=market_context(conn,network,token,mentioned_at)
        with conn,conn.cursor() as cur:
            for rung,row in result['rungs'].items():
                summary[row['status']]+=1
                cur.execute('''INSERT INTO telegram_mention_outcomes
                      (mention_id,rung_minutes,status,base_price,rung_price,return_pct,position_value,
                       price_age_seconds,unobserved_reason)
                      VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                      ON CONFLICT (mention_id,rung_minutes) DO UPDATE SET
                        status=EXCLUDED.status,base_price=EXCLUDED.base_price,
                        rung_price=EXCLUDED.rung_price,return_pct=EXCLUDED.return_pct,
                        position_value=EXCLUDED.position_value,
                        price_age_seconds=EXCLUDED.price_age_seconds,
                        unobserved_reason=EXCLUDED.unobserved_reason,computed_at=now()''',
                            (mention_id,rung,row['status'],row.get('base_price'),row.get('rung_price'),
                             row.get('return_pct'),row.get('position_value'),row.get('price_age_seconds'),
                             row.get('unobserved_reason')))
            peak=result.get('peak') or {}
            if peak:summary['with_peak']+=1
            cur.execute('''INSERT INTO telegram_mention_summary
                  (mention_id,base_price,base_price_age_seconds,peak_price,peak_at,peak_multiple,
                   minutes_to_peak,price_points,window_minutes,market_cap_at_mention,liquidity_at_mention)
                  VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                  ON CONFLICT (mention_id) DO UPDATE SET
                    base_price=EXCLUDED.base_price,base_price_age_seconds=EXCLUDED.base_price_age_seconds,
                    peak_price=EXCLUDED.peak_price,peak_at=EXCLUDED.peak_at,
                    peak_multiple=EXCLUDED.peak_multiple,minutes_to_peak=EXCLUDED.minutes_to_peak,
                    price_points=EXCLUDED.price_points,window_minutes=EXCLUDED.window_minutes,
                    market_cap_at_mention=EXCLUDED.market_cap_at_mention,
                    liquidity_at_mention=EXCLUDED.liquidity_at_mention,computed_at=now()''',
                        (mention_id,result['base_price'],result['base_price_age_seconds'],
                         peak.get('peak_price'),peak.get('peak_at'),peak.get('peak_multiple'),
                         peak.get('minutes_to_peak'),peak.get('price_points',0),peak.get('window_minutes'),
                         context.get('fdv_usd'),context.get('liquidity_usd')))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--derive',action='store_true',help='extract + resolve references (database only)')
    parser.add_argument('--prices',action='store_true',help='fetch mention-anchored OHLCV (network)')
    parser.add_argument('--outcomes',action='store_true',help='compute per-rung outcomes (database only)')
    parser.add_argument('--network',default='solana',choices=sorted(CHAINS))
    parser.add_argument('--limit',type=int,default=None)
    args=parser.parse_args()
    if not (args.derive or args.prices or args.outcomes):
        print(json.dumps(dict(mode='plan_only',network=args.network,rungs=list(RUNGS),
                              llm_calls=0,twitter_credits=0,database_writes=0)));return
    import psycopg2
    conn=psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
    try:
        setup(conn)
        out={}
        if args.derive:out['derive']=derive(conn,args.network,limit=args.limit or 5000)
        if args.prices:out['prices']=fetch_prices(conn,args.network,limit=args.limit or 200)
        if args.outcomes:out['outcomes']=compute_outcomes(conn,args.network,limit=args.limit or 2000)
        print(json.dumps(out,indent=1,default=str))
    finally:conn.close()


if __name__=='__main__':main()
