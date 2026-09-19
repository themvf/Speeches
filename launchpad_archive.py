"""Graduation Archive: record Robinhood Chain launchpad activity every few minutes.

Spec: docs/graduation-archive-spec.md. No X calls, no LLM calls; free public market data only.
Default is a no-network plan, same as crypto_market_history.py.

V1 is an archive, not a radar: it surfaces nothing and sets no thresholds. About one launch in
sixty graduates, so a rule picked before outcomes are labelled would be confidently wrong. What
this buys is the history that makes the threshold question answerable at all - the feed only
reaches back about ten minutes, so anything not written down at the time is gone for good.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path

from launchpad_chains import (CHAINS,ROBINHOOD,SOLANA,Chain,choose_measure_pool,classify,in_cohort,
                              parse_extended_info,parse_pool_list,parse_trades,summarize_trades)
GECKO='https://api.geckoterminal.com/api/v2/networks/'
PAGES=10                      # new_pools caps at page 10 (page 11 returns 401)
# Cadence is set by how far the feed reaches back, and that shrinks as the chain gets busier. A live
# sweep on 2026-09-19 fetched 149 unique pools spanning 11.1 minutes - 13.4 pools/minute, well above
# the 8/minute measured an hour earlier. A ten-minute cadence would have left about one minute of
# margin, so any delayed run would lose launches. Five minutes restores the margin; the sweep table
# records its reach every run, so the real figure should be re-read from the data, not assumed.
SWEEP_MINUTES=ROBINHOOD.sweep_minutes
# Chain-specific identity lives in launchpad_chains.py. These aliases keep the default chain's
# constants importable at module level, so nothing that already reads them has to change.
NETWORK=ROBINHOOD.network
CURVE_DEXES=set(ROBINHOOD.curve_dexes)
GRADUATE_DEXES=set(ROBINHOOD.graduate_dexes)
RUNGS=ROBINHOOD.rungs
MULTI_BATCH=30                # tokens/multi accepts 30 addresses per call
MAX_CANDIDATES=300            # live curve tokens re-read per sweep, newest first
MAX_INFO=20                   # per-token /info calls per sweep (graduates only: holders, X handle)
MAX_SNAPSHOTS=40              # pool snapshots per sweep; unfilled rungs are picked up next sweep
STALE_HOURS=24                # a curve token with no movement for this long stops being a candidate


def setup(conn):
    with conn,conn.cursor() as cur:
        cur.execute(Path(__file__).with_name('sql').joinpath('launchpad_archive.sql').read_text())


def iso(value):
    """Parse GeckoTerminal's Z-suffixed timestamps; None for anything unusable."""
    if not value:return None
    try:return datetime.fromisoformat(str(value).replace('Z','+00:00'))
    except ValueError:return None


def number(value):
    try:
        out=float(value)
        return out if math.isfinite(out) else None
    except (TypeError,ValueError):return None


def parse_pool(entry,chain=ROBINHOOD):
    """One new_pools/pool record -> the fields we store. Returns None if it is not usable."""
    fold=(lambda x:x.lower()) if chain.lowercase_addresses else (lambda x:x)
    attributes=entry.get('attributes') or {};relationships=entry.get('relationships') or {}
    address=attributes.get('address')
    dex=((relationships.get('dex') or {}).get('data') or {}).get('id')
    base=((relationships.get('base_token') or {}).get('data') or {}).get('id') or ''
    token=base.split('_',1)[1] if '_' in base else None
    if not address or not dex or not token:return None
    txns=attributes.get('transactions') or {};volume=attributes.get('volume_usd') or {}
    change=attributes.get('price_change_percentage') or {}
    return dict(pool=fold(address),dex=dex,token=fold(token),
                symbol=(attributes.get('name') or '').split('/')[0].strip() or None,
                name=attributes.get('name'),created=iso(attributes.get('pool_created_at')),
                price=number(attributes.get('base_token_price_usd')),fdv=number(attributes.get('fdv_usd')),
                liquidity=number(attributes.get('reserve_in_usd')),
                volume_m30=number(volume.get('m30')),volume_h1=number(volume.get('h1')),volume_h24=number(volume.get('h24')),
                buyers_m30=(txns.get('m30') or {}).get('buyers'),sellers_m30=(txns.get('m30') or {}).get('sellers'),
                buyers_h1=(txns.get('h1') or {}).get('buyers'),sellers_h1=(txns.get('h1') or {}).get('sellers'),
                txns_h1=((txns.get('h1') or {}).get('buys') or 0)+((txns.get('h1') or {}).get('sells') or 0),
                change_h1=number(change.get('h1')))


def gap_seconds(previous_newest,this_oldest):
    """Seconds of launches this sweep could not see.

    The windows must overlap: if the previous sweep's newest pool is older than the oldest pool this
    sweep could reach, tokens launched in between were never observed and never will be. 0 means the
    windows overlapped; None means there is no prior sweep to compare against.
    """
    if previous_newest is None or this_oldest is None:return None
    return max(0,int((this_oldest-previous_newest).total_seconds()))


def rungs_due(graduated_at,now,filled,rungs=RUNGS):
    """Ladder rungs whose time has passed and which have no observation yet."""
    if not graduated_at:return []
    elapsed=(now-graduated_at).total_seconds()/60
    return [r for r in rungs if elapsed>=r and r not in filled]


def multi_url(addresses,network=NETWORK):
    return GECKO+network+'/tokens/multi/'+','.join(addresses)


def parse_multi(payload,chain=ROBINHOOD):
    """tokens/multi -> {address: {pct, completed, completed_at, destination, symbol, name, price, fdv}}."""
    out={}
    fold=(lambda x:x.lower()) if chain.lowercase_addresses else (lambda x:x)
    for entry in (payload or {}).get('data') or []:
        attributes=entry.get('attributes') or {}
        address=fold(attributes.get('address') or '')
        if not address:continue
        launchpad=attributes.get('launchpad_details') or {}
        # A token with no launchpad_details is not a launchpad token (or is no longer described as one);
        # record what we can and let the caller decide.
        out[address]=dict(pct=number(launchpad.get('graduation_percentage')),
                          completed=bool(launchpad.get('completed')),
                          completed_at=iso(launchpad.get('completed_at')),
                          destination=fold(launchpad.get('migrated_destination_pool_address') or '') or None,
                          symbol=attributes.get('symbol'),name=attributes.get('name'),
                          price=number(attributes.get('price_usd')),fdv=number(attributes.get('fdv_usd')),
                          has_launchpad='launchpad_details' in attributes and attributes.get('launchpad_details') is not None)
    return out


def normalize_handle(value):
    """A usable X handle, or None.

    `twitter_handle` is supplied by whoever launched the token and is not validated upstream: a live
    sweep returned 'Na1_N1ako/status/2101346622135230543' as a handle. Storing that would make the
    social join match nothing, or worse, match wrongly. Keep only what is actually a handle.
    """
    if not value:return None
    text=str(value).strip()
    for prefix in ('https://','http://'):
        if text.lower().startswith(prefix):text=text[len(prefix):]
    for host in ('www.','twitter.com/','x.com/','mobile.twitter.com/'):
        if text.lower().startswith(host):text=text[len(host):]
    text=text.split('?')[0].split('/')[0].lstrip('@').strip()
    if not text or len(text)>15:return None
    return text if all(c.isascii() and (c.isalnum() or c=='_') for c in text) else None


def parse_info(payload):
    """tokens/<address>/info -> holders, top-10 concentration and the project's declared X handle."""
    attributes=((payload or {}).get('data') or {}).get('attributes') or {}
    holders=attributes.get('holders') or {}
    distribution=holders.get('distribution_percentage') or {}
    return dict(holders=holders.get('count'),
                top10=number(distribution.get('top_10')),
                twitter=normalize_handle(attributes.get('twitter_handle')))


def sweep(conn,chain=ROBINHOOD,fetch=None,now=None,wait=None):
    """One sweep of one chain. Gathers everything, then writes once. Returns a summary dict."""
    import requests,time
    wait=wait or time.sleep;fetch=fetch or requests.get;now=now or datetime.now(timezone.utc)
    setup(conn)
    # Wall clock, not the sweep's logical `now`, so a fixture-driven test is never bounded by it.
    started=time.monotonic()
    def budget_left():
        return time.monotonic()-started < chain.deadline_seconds
    with conn,conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(hashtext(%s))",('launchpad-archive:'+chain.network,))
        if not cur.fetchone()[0]:return {'status':'already_running','errors':['sweep_already_running']}
    errors=[]
    def get(url):
        for attempt in range(4):
            wait(chain.request_wait)  # ~30 requests/minute is the unkeyed ceiling; never burst.
            response=fetch(url,timeout=25,allow_redirects=False,headers={'Accept':'application/json'})
            status=getattr(response,'status_code',0)
            if status==429 and attempt<3:
                try:delay=float(getattr(response,'headers',{}).get('Retry-After','15'))
                except (ValueError,TypeError):delay=15
                if not math.isfinite(delay) or delay>30:raise ValueError('rate limit cooldown exceeds retry bound')
                wait(max(3,delay));continue
            if status!=200:raise ValueError('HTTP '+str(status))
            return response.json()
        raise ValueError('rate limited')
    try:
        with conn,conn.cursor() as cur:
            cur.execute('SELECT newest_pool_at FROM launchpad_sweeps WHERE complete AND newest_pool_at IS NOT NULL ORDER BY started_at DESC LIMIT 1')
            row=cur.fetchone();previous_newest=row[0] if row else None
        # 1. Discovery. Stop early once a page predates the last sweep - the rest is already recorded.
        pools=[];pages=0;oldest=None;newest=None
        for page in range(1,PAGES+1):
            try:payload=get(GECKO+chain.network+'/new_pools?page='+str(page))
            except (ValueError,requests.RequestException) as exc:
                errors.append('new_pools page '+str(page)+': '+type(exc).__name__+' '+str(exc)[:120]);break
            pages+=1
            entries=[p for p in (parse_pool(e,chain) for e in (payload.get('data') or [])) if p]
            if not entries:break
            pools+=entries
            created=[e['created'] for e in entries if e['created']]
            if created:
                oldest=min(created) if oldest is None else min(oldest,min(created))
                newest=max(created) if newest is None else max(newest,max(created))
            if previous_newest and oldest and oldest<=previous_newest:break
        by_pool={}
        for entry in pools:by_pool.setdefault(entry['pool'],entry)
        pools=list(by_pool.values())
        curve=[e for e in pools if classify(e['dex'],chain)=='curve']
        graduate_pools=[e for e in pools if classify(e['dex'],chain)=='graduate']

        # 2. Known state for everything this sweep touches, plus the live candidate set.
        touched=sorted({e['token'] for e in pools})
        with conn,conn.cursor() as cur:
            known={}
            if touched:
                cur.execute('SELECT token_address,graduated,first_seen_pct,state FROM launchpad_tokens WHERE network=%s AND token_address=ANY(%s)',(chain.network,touched))
                known={r[0]:dict(graduated=r[1],first_pct=r[2],state=r[3]) for r in cur.fetchall()}
            cur.execute('''SELECT token_address FROM launchpad_tokens
                           WHERE network=%s AND state='live' AND last_seen_at>%s
                           ORDER BY moved DESC,last_seen_at DESC LIMIT %s''',
                        (chain.network,now-timedelta(hours=STALE_HOURS),chain.max_candidates))
            candidates=[r[0] for r in cur.fetchall()]
            cur.execute('''SELECT t.token_address,t.graduated_at,array_remove(array_agg(o.rung_minutes),NULL)
                           FROM launchpad_tokens t LEFT JOIN launchpad_observations o
                             ON o.network=t.network AND o.token_address=t.token_address AND o.rung_minutes IS NOT NULL
                           WHERE t.network=%s AND t.graduated AND t.cohort_sampled AND t.graduated_at>%s
                           GROUP BY t.token_address,t.graduated_at''',(chain.network,now-timedelta(days=8)))
            ladder=[(r[0],r[1],set(r[2] or [])) for r in cur.fetchall()]

        # 3. Graduation by arrival: a graduate-DEX pool whose token we hold is a graduation event.
        arrivals={e['token']:e for e in graduate_pools}

        # 4. Batched state for new tokens and live candidates.
        ask=sorted({e['token'] for e in curve if e['token'] not in known}|set(candidates)|set(arrivals))
        state={}
        for start in range(0,len(ask),MULTI_BATCH):
            batch=ask[start:start+MULTI_BATCH]
            try:state.update(parse_multi(get(multi_url(batch,chain.network)),chain))
            except (ValueError,requests.RequestException) as exc:
                errors.append('tokens/multi: '+type(exc).__name__+' '+str(exc)[:120])

        # 5. Enrichment for graduates we have not enriched, and 6. the snapshot ladder.
        newly=[t for t,entry in arrivals.items() if not known.get(t,{}).get('graduated')]
        newly+=[t for t,s in state.items() if s['completed'] and not known.get(t,{}).get('graduated') and t not in arrivals]
        info={};measure={};captures=[]
        for token in sorted(set(newly))[:chain.max_info]:
            if not budget_left():break
            try:
                payload=get(GECKO+chain.network+'/tokens/'+token+'/info')
                info[token]=parse_info(payload)
                # Creator, authorities, socials and description - free, in the call we already make.
                if chain.extended_info:info[token].update(parse_extended_info(payload))
            except (ValueError,requests.RequestException) as exc:
                errors.append('info '+token[:10]+': '+type(exc).__name__)
            # Which pool the ladder will read. Never the launchpad's destination field on a chain
            # where that has been seen pointing at an empty pool.
            destination=(state.get(token) or {}).get('destination') or (arrivals[token]['pool'] if token in arrivals else None)
            sampled=in_cohort(chain,token)
            if not chain.deepest_pool_wins:
                measure[token]=choose_measure_pool(chain,[],destination)
            elif not sampled:
                # Outside the cohort there is no ladder to read, so the pool list is never fetched.
                # Recorded as its own reason: "we did not look" must not read as "there was nothing".
                measure[token]=(None,'outside ladder cohort')
            else:
                try:pool_list=parse_pool_list(get(GECKO+chain.network+'/tokens/'+token+'/pools'))
                except (ValueError,requests.RequestException) as exc:
                    pool_list=[];errors.append('pools '+token[:10]+': '+type(exc).__name__)
                measure[token]=choose_measure_pool(chain,pool_list,destination)
            # Opening trade capture: perishable. On a busy graduate 300 trades spanned 33 seconds, so
            # this is taken now or never - it cannot be reconstructed later at any price.
            address=measure[token][0]
            if chain.capture_trades and address and sampled and budget_left():
                started=datetime.now(timezone.utc);rows=[];fetched=0
                for page in range(1,chain.trade_pages+1):
                    try:
                        batch=parse_trades(get(GECKO+chain.network+'/pools/'+address+'/trades?page='+str(page)),address)
                    except (ValueError,requests.RequestException) as exc:
                        errors.append('trades '+address[:10]+': '+type(exc).__name__);break
                    fetched+=1
                    if not batch:break
                    rows+=batch
                # Pages overlap rather than extending backwards, so dedupe on the transaction itself.
                unique={(r['tx_hash'],r['wallet'],r['traded_at']):r for r in rows}
                ordered=sorted(unique.values(),key=lambda r:(r['traded_at'],r['tx_hash'] or ''))
                captures.append(dict(token=token,pool=address,started=started,
                                     finished=datetime.now(timezone.utc),pages=fetched,
                                     rows=ordered,summary=summarize_trades(ordered)))
        observations=[]
        for entry in curve:
            pct=(state.get(entry['token']) or {}).get('pct')
            observations.append(dict(token=entry['token'],at=now,phase='curve',rung=None,pct=pct,pool=entry,holders=None))
        snapshots=0
        with conn,conn.cursor() as cur:
            for token,graduated_at,filled in ladder:
                due=rungs_due(graduated_at,now,filled,chain.rungs)
                if not due or snapshots>=chain.max_snapshots or not budget_left():continue
                cur.execute('SELECT coalesce(measure_pool,graduation_pool) FROM launchpad_tokens WHERE network=%s AND token_address=%s',(chain.network,token))
                row=cur.fetchone();pool_address=row[0] if row else None
                if not pool_address:continue
                try:payload=get(GECKO+chain.network+'/pools/'+pool_address)
                except (ValueError,requests.RequestException) as exc:
                    errors.append('pool '+pool_address[:10]+': '+type(exc).__name__);continue
                parsed=parse_pool((payload or {}).get('data') or {},chain)
                if not parsed:continue
                snapshots+=1
                # Only the earliest due rung is filled per sweep: one row per rung, honestly timestamped.
                observations.append(dict(token=token,at=now,phase='post',rung=min(due),pct=None,pool=parsed,
                                         holders=(info.get(token) or {}).get('holders')))

        # 7. One write for the whole sweep.
        summary=_persist(conn,chain=chain,now=now,pools=pools,curve=curve,arrivals=arrivals,state=state,info=info,
                         known=known,observations=observations,pages=pages,oldest=oldest,newest=newest,
                         previous_newest=previous_newest,errors=errors,measure=measure,captures=captures)
        return summary
    finally:
        with conn,conn.cursor() as cur:cur.execute("SELECT pg_advisory_unlock(hashtext(%s))",('launchpad-archive:'+chain.network,))


def _extra(chain,token,dex,chosen,enrich):
    """Adapter-owned columns, in the order the INSERT lists them."""
    from psycopg2.extras import Json
    return (dex,chosen[0],chosen[1],in_cohort(chain,token),
            enrich.get('developer_address'),enrich.get('developer_holding'),enrich.get('is_honeypot'),
            enrich.get('mint_authority'),enrich.get('freeze_authority'),enrich.get('telegram_handle'),
            enrich.get('website'),enrich.get('description'),enrich.get('categories'),enrich.get('gt_score'),
            Json(enrich['info_raw']) if enrich.get('info_raw') else None)


def _persist(conn,*,chain,now,pools,curve,arrivals,state,info,known,observations,pages,oldest,newest,previous_newest,errors,measure=None,captures=None):
    from psycopg2.extras import execute_values,Json
    measure=measure or {};captures=captures or []
    gap=gap_seconds(previous_newest,oldest)
    window=int((newest-oldest).total_seconds()) if (newest and oldest) else None
    rows=[]
    for entry in curve:
        token=entry['token'];s=state.get(token) or {}
        graduated=bool(s.get('completed')) or token in arrivals
        graduated_at=s.get('completed_at') or (arrivals[token]['created'] if token in arrivals else None)
        detected=now if (graduated and not known.get(token,{}).get('graduated')) else None
        enrich=info.get(token) or {}
        chosen=measure.get(token) or (None,None)
        rows.append((chain.network,token,s.get('symbol') or entry['symbol'],s.get('name') or entry['name'],entry['dex'],
                     entry['pool'],entry['created'],now,s.get('pct'),graduated,graduated_at,detected,
                     s.get('destination') or (arrivals[token]['pool'] if token in arrivals else None),
                     s.get('pct'),now,'graduated' if graduated else 'live',
                     enrich.get('holders'),enrich.get('top10'),enrich.get('twitter'),now if enrich else None)
                    +_extra(chain,token,entry['dex'],chosen,enrich))
    # A graduate pool whose curve pool we never saw still belongs in the archive, flagged by its
    # missing first_pool_created - that is the fingerprint of a launch the sweep window missed.
    for token,entry in arrivals.items():
        if any(r[1]==token for r in rows):continue
        s=state.get(token) or {};enrich=info.get(token) or {}
        chosen=measure.get(token) or (None,None)
        rows.append((chain.network,token,s.get('symbol') or entry['symbol'],s.get('name') or entry['name'],entry['dex'],
                     None,None,now,s.get('pct'),True,s.get('completed_at') or entry['created'],
                     now if not known.get(token,{}).get('graduated') else None,
                     # The launchpad's declared destination is metadata; what we measure is measure_pool.
                     s.get('destination') or entry['pool'],s.get('pct'),now,'graduated',
                     enrich.get('holders'),enrich.get('top10'),enrich.get('twitter'),now if enrich else None)
                    +_extra(chain,token,entry['dex'],chosen,enrich))
    observation_rows=[]
    for o in observations:
        p=o['pool']
        observation_rows.append((chain.network,o['token'],o['at'],o['phase'],o['rung'],o['pct'],p['price'],p['fdv'],
                                 p['liquidity'],p['volume_m30'],p['volume_h1'],p['volume_h24'],
                                 p['buyers_m30'],p['sellers_m30'],p['buyers_h1'],p['sellers_h1'],p['txns_h1'],
                                 p['change_h1'],o['holders']))
    new_tokens=sum(1 for r in rows if r[1] not in known)
    graduations=sum(1 for r in rows if r[9] and not known.get(r[1],{}).get('graduated'))
    with conn,conn.cursor() as cur:
        if rows:
            execute_values(cur,'''INSERT INTO launchpad_tokens
              (network,token_address,symbol,name,dex,curve_pool,first_pool_created,first_seen_at,first_seen_pct,
               graduated,graduated_at,graduated_detected_at,graduation_pool,last_pct,last_seen_at,state,
               holders,top10_share,twitter_handle,enriched_at,
               launchpad,measure_pool,measure_pool_reason,cohort_sampled,
               developer_address,developer_holding,is_honeypot,mint_authority,freeze_authority,
               telegram_handle,website,description,categories,gt_score,info_raw) VALUES %s
              ON CONFLICT (network,token_address) DO UPDATE SET
               symbol=COALESCE(EXCLUDED.symbol,launchpad_tokens.symbol),
               name=COALESCE(EXCLUDED.name,launchpad_tokens.name),
               curve_pool=COALESCE(launchpad_tokens.curve_pool,EXCLUDED.curve_pool),
               first_pool_created=COALESCE(launchpad_tokens.first_pool_created,EXCLUDED.first_pool_created),
               graduated=launchpad_tokens.graduated OR EXCLUDED.graduated,
               -- First observation of each fact wins: graduation time and our detection time are
               -- evidence about when things happened and must never be overwritten by a later sweep.
               graduated_at=COALESCE(launchpad_tokens.graduated_at,EXCLUDED.graduated_at),
               graduated_detected_at=COALESCE(launchpad_tokens.graduated_detected_at,EXCLUDED.graduated_detected_at),
               graduation_pool=COALESCE(launchpad_tokens.graduation_pool,EXCLUDED.graduation_pool),
               last_pct=COALESCE(EXCLUDED.last_pct,launchpad_tokens.last_pct),
               last_seen_at=EXCLUDED.last_seen_at,
               moved=launchpad_tokens.moved OR (EXCLUDED.last_pct IS NOT NULL AND launchpad_tokens.first_seen_pct IS NOT NULL
                     AND EXCLUDED.last_pct>launchpad_tokens.first_seen_pct),
               state=CASE WHEN launchpad_tokens.graduated OR EXCLUDED.graduated THEN 'graduated' ELSE launchpad_tokens.state END,
               holders=COALESCE(EXCLUDED.holders,launchpad_tokens.holders),
               top10_share=COALESCE(EXCLUDED.top10_share,launchpad_tokens.top10_share),
               twitter_handle=COALESCE(EXCLUDED.twitter_handle,launchpad_tokens.twitter_handle),
               enriched_at=COALESCE(launchpad_tokens.enriched_at,EXCLUDED.enriched_at),
               launchpad=COALESCE(launchpad_tokens.launchpad,EXCLUDED.launchpad),
               -- First choice of measurement pool wins: switching it mid-ladder would silently mix
               -- two different measurement targets inside one token's history.
               measure_pool=COALESCE(launchpad_tokens.measure_pool,EXCLUDED.measure_pool),
               measure_pool_reason=COALESCE(launchpad_tokens.measure_pool_reason,EXCLUDED.measure_pool_reason),
               cohort_sampled=launchpad_tokens.cohort_sampled AND EXCLUDED.cohort_sampled,
               developer_address=COALESCE(launchpad_tokens.developer_address,EXCLUDED.developer_address),
               developer_holding=COALESCE(EXCLUDED.developer_holding,launchpad_tokens.developer_holding),
               is_honeypot=COALESCE(EXCLUDED.is_honeypot,launchpad_tokens.is_honeypot),
               mint_authority=COALESCE(EXCLUDED.mint_authority,launchpad_tokens.mint_authority),
               freeze_authority=COALESCE(EXCLUDED.freeze_authority,launchpad_tokens.freeze_authority),
               telegram_handle=COALESCE(launchpad_tokens.telegram_handle,EXCLUDED.telegram_handle),
               website=COALESCE(launchpad_tokens.website,EXCLUDED.website),
               description=COALESCE(launchpad_tokens.description,EXCLUDED.description),
               categories=COALESCE(launchpad_tokens.categories,EXCLUDED.categories),
               gt_score=COALESCE(EXCLUDED.gt_score,launchpad_tokens.gt_score),
               info_raw=COALESCE(launchpad_tokens.info_raw,EXCLUDED.info_raw)''',rows,page_size=500)
        if observation_rows:
            execute_values(cur,'''INSERT INTO launchpad_observations
              (network,token_address,observed_at,phase,rung_minutes,graduation_pct,price_usd,fdv_usd,liquidity_usd,
               volume_m30,volume_h1,volume_h24,buyers_m30,sellers_m30,buyers_h1,sellers_h1,txns_h1,
               price_change_h1,holders) VALUES %s ON CONFLICT DO NOTHING''',observation_rows,page_size=500)
        # Curve tokens that have sat still past the stale window stop being candidates. Published
        # analysis says a token still on the curve after 24h almost never graduates; our own data
        # will confirm or refute that, which is why the rows stay and only the state changes.
        cur.execute('''UPDATE launchpad_tokens SET state='dead'
                       WHERE network=%s AND state='live' AND last_seen_at<%s''',(chain.network,now-timedelta(hours=STALE_HOURS)))
        captured=0
        for capture in captures:
            summary=capture['summary']
            earliest=iso(summary['earliest']);latest=iso(summary['latest'])
            graduated_at=next((r[10] for r in rows if r[1]==capture['token']),None)
            cur.execute('''INSERT INTO launchpad_trade_captures
              (network,token_address,pool,graduated_at,capture_started_at,capture_finished_at,pages_fetched,
               trades,wallets,buyers,sellers,top_wallet_share,repeat_wallets,earliest_trade_at,latest_trade_at,
               window_seconds,lag_seconds)
              VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id''',
              (chain.network,capture['token'],capture['pool'],graduated_at,capture['started'],capture['finished'],
               capture['pages'],summary['trades'],summary['wallets'],summary['buyers'],summary['sellers'],
               summary['top_wallet_share'],summary['repeat_wallets'],earliest,latest,
               int((latest-earliest).total_seconds()) if earliest and latest else None,
               # How much of the opening window we missed, rather than assuming we caught it all.
               int((earliest-graduated_at).total_seconds()) if earliest and graduated_at else None))
            capture_id=cur.fetchone()[0]
            if capture['rows']:
                execute_values(cur,'''INSERT INTO launchpad_trades
                  (capture_id,sequence,wallet,traded_at,kind,token_amount,usd,tx_hash,block_number)
                  VALUES %s ON CONFLICT DO NOTHING''',
                  [(capture_id,i,r['wallet'],r['traded_at'],r['kind'],r['token_amount'],r['usd'],r['tx_hash'],r['block_number'])
                   for i,r in enumerate(capture['rows'])],page_size=500)
            captured+=1
        cur.execute('''INSERT INTO launchpad_sweeps
          (started_at,finished_at,pages_fetched,pools_seen,oldest_pool_at,newest_pool_at,window_seconds,
           new_tokens,graduations,observations,gap_seconds,complete,errors)
          VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id''',
          (now,datetime.now(timezone.utc),pages,len(pools),oldest,newest,window,new_tokens,graduations,
           len(observation_rows),gap,not errors and (gap==0 or gap is None),errors))
        sweep_id=cur.fetchone()[0]
    return {'status':'ok','network':chain.network,'sweep_id':sweep_id,'pages':pages,'pools':len(pools),'curve':len(curve),
            'trade_captures':len(captures),'trades':sum(c['summary']['trades'] for c in captures),
            'new_tokens':new_tokens,'graduations':graduations,'observations':len(observation_rows),
            'window_seconds':window,'gap_seconds':gap,'enriched':len(info),'errors':errors,
            'twitter_credits':0}


def daily(conn,now=None,days=7,chain=ROBINHOOD):
    """One row per UTC day: is the archive healthy, without reading raw rows.

    p95 alongside the median because the median hides the tail that matters here - a lag that is
    usually seconds but occasionally minutes is the shape that argues for a faster collector, and a
    median alone would never show it.
    """
    now=now or datetime.now(timezone.utc);since=(now-timedelta(days=days)).replace(hour=0,minute=0,second=0,microsecond=0)
    rows={}
    with conn,conn.cursor() as cur:
        cur.execute('''SELECT date_trunc('day',started_at),count(*),count(*) FILTER (WHERE NOT complete),
                              coalesce(max(gap_seconds),0),coalesce(sum(new_tokens),0),coalesce(sum(graduations),0),
                              coalesce(sum(observations),0)
                       FROM launchpad_sweeps WHERE started_at>=%s GROUP BY 1''',(since,))
        for day,sweeps,incomplete,max_gap,launches,graduates,observations in cur.fetchall():
            rows[day]=dict(day=day.date().isoformat(),sweeps=sweeps,expected=int(24*60/chain.sweep_minutes),
                           incomplete_sweeps=incomplete,max_gap_seconds=max_gap,launches_seen=launches,
                           graduates_detected=graduates,observations=observations,
                           detection_lag_median=None,detection_lag_p95=None,lag_measured=0)
        # Lag is keyed on when the graduation happened, not on which sweep noticed it.
        cur.execute('''SELECT date_trunc('day',graduated_at),count(*),
                              percentile_disc(0.5) WITHIN GROUP (ORDER BY lag),
                              percentile_disc(0.95) WITHIN GROUP (ORDER BY lag)
                       FROM (SELECT graduated_at,extract(epoch FROM graduated_detected_at-graduated_at) AS lag
                             FROM launchpad_tokens
                             WHERE graduated AND graduated_at>=%s AND graduated_detected_at IS NOT NULL) d
                       GROUP BY 1''',(since,))
        for day,measured,median,p95 in cur.fetchall():
            row=rows.setdefault(day,dict(day=day.date().isoformat(),sweeps=0,expected=int(24*60/chain.sweep_minutes),
                                         incomplete_sweeps=0,max_gap_seconds=0,launches_seen=0,
                                         graduates_detected=0,observations=0))
            row.update(lag_measured=measured,detection_lag_median=int(median) if median is not None else None,
                       detection_lag_p95=int(p95) if p95 is not None else None)
    return [rows[k] for k in sorted(rows)]


def report(conn,now=None,hours=48,chain=ROBINHOOD):
    """Is the archive continuous and internally consistent? Read-only; this is the V1 deliverable."""
    now=now or datetime.now(timezone.utc);since=now-timedelta(hours=hours)
    out={'window_hours':hours,'as_of':now.isoformat()}
    with conn,conn.cursor() as cur:
        # How deep the feed reaches is only observable from a sweep that exhausted all PAGES pages:
        # any other sweep stopped early because it met ground already recorded, so its shallow reach
        # is proof the archive is healthy, not evidence the feed shrank. Measuring the wrong one
        # raises an alarm precisely when everything is working.
        cur.execute('''SELECT count(*),count(*) FILTER (WHERE NOT complete),coalesce(max(gap_seconds),0),
                              min(extract(epoch FROM started_at-oldest_pool_at)) FILTER (WHERE pages_fetched>=%s),
                              coalesce(sum(new_tokens),0),coalesce(sum(graduations),0),
                              coalesce(sum(observations),0),min(started_at),max(started_at)
                       FROM launchpad_sweeps WHERE started_at>=%s''',(PAGES,since))
        row=cur.fetchone()
        expected=int(hours*60/chain.sweep_minutes)
        out['sweeps']=dict(recorded=row[0],expected=expected,incomplete=row[1],max_gap_seconds=row[2],
                           min_reach_seconds=int(row[3]) if row[3] is not None else None,
                           first=row[7].isoformat() if row[7] else None,
                           last=row[8].isoformat() if row[8] else None)
        out['totals']=dict(new_tokens=row[4],graduations=row[5],observations=row[6])
        # Longest silence between consecutive sweeps: the honest measure of delivery reliability.
        cur.execute('''SELECT coalesce(max(delta),0) FROM (
                         SELECT extract(epoch FROM started_at-lag(started_at) OVER (ORDER BY started_at)) AS delta
                         FROM launchpad_sweeps WHERE started_at>=%s) g''',(since,))
        out['sweeps']['longest_interval_seconds']=int(cur.fetchone()[0] or 0)
        cur.execute('''SELECT count(*),count(*) FILTER (WHERE graduated),
                              count(*) FILTER (WHERE graduated AND first_pool_created IS NULL)
                       FROM launchpad_tokens WHERE first_seen_at>=%s''',(since,))
        seen,graduated,orphan=cur.fetchone()
        out['tokens']=dict(discovered=seen,graduated=graduated,
                           graduation_rate=round(graduated/seen,4) if seen else None,
                           graduated_without_launch_observed=orphan)
        cur.execute('''SELECT count(*),percentile_disc(0.5) WITHIN GROUP (ORDER BY lag),max(lag) FROM (
                         SELECT extract(epoch FROM graduated_detected_at-graduated_at) AS lag
                         FROM launchpad_tokens WHERE graduated AND graduated_at>=%s
                           AND graduated_detected_at IS NOT NULL) d''',(since,))
        count,median,worst=cur.fetchone()
        # This pair is what decides whether a 60-second fast lane ever earns its complexity.
        out['detection_lag_seconds']=dict(measured=count,median=int(median) if median is not None else None,
                                          worst=int(worst) if worst is not None else None)
        cur.execute('''SELECT rung_minutes,count(*) FROM launchpad_observations
                       WHERE rung_minutes IS NOT NULL AND observed_at>=%s GROUP BY rung_minutes ORDER BY rung_minutes''',(since,))
        out['ladder']={str(r[0]):r[1] for r in cur.fetchall()}
    # Margin is the headroom between how far back the feed reaches and how often we sweep. It shrinks
    # as the chain gets busier, so it is measured rather than assumed: once the feed's depth
    # approaches the sweep interval, ordinary jitter starts costing launches. None means no sweep in
    # this window went full depth, so the feed's limit was never observed - unknown, not healthy.
    narrowest=out['sweeps']['min_reach_seconds']
    out['margin_seconds']=narrowest-chain.sweep_minutes*60 if narrowest else None
    out['margin_warning']=bool(narrowest and narrowest<chain.sweep_minutes*60*1.5)
    complete=out['sweeps']['incomplete']==0 and out['sweeps']['max_gap_seconds']==0
    out['continuous']=bool(complete and out['sweeps']['recorded']>=expected*0.9)
    out['verdict']=('continuous and internally consistent' if out['continuous']
                    else 'INCOMPLETE: '+', '.join(filter(None,[
                        f"{out['sweeps']['incomplete']} incomplete sweeps" if out['sweeps']['incomplete'] else '',
                        f"max feed gap {out['sweeps']['max_gap_seconds']}s" if out['sweeps']['max_gap_seconds'] else '',
                        f"{out['sweeps']['recorded']} of {expected} expected sweeps" if out['sweeps']['recorded']<expected*0.9 else ''])))
    out['daily']=daily(conn,now=now,days=max(1,-(-hours//24)),chain=chain)
    if out['margin_warning']:
        out['verdict']+=f"; shallowest reach {narrowest}s against a {chain.sweep_minutes}m sweep - shorten the interval"
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='run a sweep (network + database writes)')
    parser.add_argument('--report',action='store_true',help='read-only continuity report')
    parser.add_argument('--daily',action='store_true',help='read-only daily health summary')
    parser.add_argument('--hours',type=int,default=48)
    parser.add_argument('--days',type=int,default=7)
    parser.add_argument('--chain',default=ROBINHOOD.network,choices=sorted(CHAINS),help='which chain to sweep or report on')
    args=parser.parse_args()
    chain=CHAINS[args.chain]
    if not args.execute and not args.report and not args.daily:
        print(json.dumps({'mode':'plan_only','chain':chain.network,
                          'max_public_requests':PAGES+MULTI_BATCH//10+MAX_INFO+MAX_SNAPSHOTS,
                          'twitter_credits':0,'database_writes':0,'sweep_minutes':chain.sweep_minutes}));return
    import psycopg2
    conn=psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
    try:
        if args.daily:print(json.dumps(daily(conn,days=args.days,chain=chain),indent=1));return
        if args.report:print(json.dumps(report(conn,hours=args.hours,chain=chain),indent=1));return
        result=sweep(conn,chain);print(json.dumps(result))
        # A sweep that reached no pages is a failure; losing a page to rate limiting is not, because
        # the gap is recorded and the next sweep still overlaps.
        if result.get('status')=='ok' and not result['pools']:raise SystemExit(1)
    finally:conn.close()


if __name__=='__main__':main()
