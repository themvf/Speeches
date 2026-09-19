"""The live-smoke gate for the Telegram OSINT collector. One command, six checks, PR-ready evidence.

Nothing in this layer is merged on mocked tests. The Solana chain adapter passed a green suite while
every live enrichment call 404'd, because a mock answers whatever it is asked; the same class of
defect is available here in peer resolution, history paging and relay attribution. So this runs
against real Telegram and the real database, and prints what it saw rather than a boolean.

    python telegram_gate.py --run --token <mint>

Check 6 deliberately makes a channel fail: it polls a peer that does not exist, to prove the health
report tells 'we lost the channel' apart from 'the channel was quiet'. That distinction is the one
that decides whether a silent dashboard means everything is fine or that collection died.
"""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import json
import os

import telegram_analytics as analytics
import telegram_collector as collector
import telegram_extract as extract
import telegram_mentions as mentions

IMPOSSIBLE_PEER='this_channel_does_not_exist_'+'osintgate'


def check(name,passed,detail):
    return dict(check=name,passed=bool(passed),detail=detail)


async def resolve(conn,client,usernames):
    """1. Peer resolution: usernames -> ids, with identity confirmed and stability re-checked."""
    first=await collector.resolve_channels(conn,client,usernames)
    second=await collector.resolve_channels(conn,client,usernames)
    stable=[a['username'] for a,b in zip(first,second)
            if a.get('channel_id') and a.get('channel_id')==b.get('channel_id')]
    resolved=[r for r in first if r.get('channel_id')]
    return check('1. peers resolved from usernames and identity confirmed',
                 len(resolved)>=3 and len(stable)==len(resolved),
                 dict(requested=usernames,resolved=[dict(username=r['username'],channel_id=r['channel_id'],
                                                         title=r.get('title')) for r in first],
                      stable_on_recheck=stable))


async def backfill(conn,fetch,channel):
    """2. Backfill walks strictly older, and a re-read writes no new rows."""
    before=_count(conn,channel['channel_id'])
    first=await collector.collect_channel(conn,channel,fetch,'backfill')
    after_first=_count(conn,channel['channel_id'])
    with conn,conn.cursor() as cur:
        cur.execute('SELECT backfill_floor_id FROM telegram_channels WHERE channel_id=%s',(channel['channel_id'],))
        floor=cur.fetchone()[0]
    second=await collector.collect_channel(conn,dict(channel,backfill_floor_id=None),fetch,'backfill')
    after_second=_count(conn,channel['channel_id'])
    return check('2. backfill pages walk older history and re-reads are idempotent',
                 after_first>before and second['new']==0 and after_second==after_first,
                 dict(rows_before=before,rows_after_first_page=after_first,
                      rows_after_reread=after_second,floor_id=floor,
                      second_pass_new_rows=second['new'],second_pass_updated=second['updated'],
                      note='(channel_id, message_id) is Telegram own identity, so a re-read must '
                           'update in place and never duplicate'))


def _count(conn,channel_id):
    with conn,conn.cursor() as cur:
        cur.execute('SELECT count(*) FROM telegram_messages WHERE channel_id=%s',(channel_id,))
        return cur.fetchone()[0]


def contract_resolution(conn,network='solana'):
    """3. A real posted contract resolves to the right archive token."""
    with conn,conn.cursor() as cur:
        cur.execute('''SELECT m.token_address,t.symbol,t.launchpad_family,t.graduated_at,m.mentioned_at,
                              m.seconds_to_graduation,g.text
                       FROM telegram_token_mentions m
                       JOIN launchpad_tokens t ON t.network=m.network AND t.token_address=m.token_address
                       JOIN telegram_messages g ON g.channel_id=m.channel_id AND g.message_id=m.message_id
                       WHERE m.network=%s AND m.reference_kind='contract'
                       ORDER BY m.mentioned_at DESC LIMIT 1''',(network,))
        row=cur.fetchone()
    if not row:
        return check('3. a real contract resolves to the archive token',False,
                     'no collected message carried a contract the archive holds - collect longer, or '
                     'the monitored channels are not posting graduating Solana tokens')
    address,symbol,family,graduated_at,at,delta,text=row
    # The address in the stored text must be the address we resolved: this is the check that would
    # catch a normalisation or truncation bug between extraction and storage.
    in_text=address in (extract.solana_addresses(text) or [])
    return check('3. a real contract resolves to the archive token',
                 in_text and symbol is not None,
                 dict(token=address,symbol=symbol,launchpad_family=family,
                      graduated_at=str(graduated_at),mentioned_at=str(at),
                      seconds_to_graduation=delta,address_round_tripped_from_message_text=in_text))


def contract_beats_ticker(conn,network='solana'):
    """4. A message carrying both a contract and a cashtag credits only the contract."""
    with conn,conn.cursor() as cur:
        cur.execute('''SELECT g.channel_id,g.message_id,g.text FROM telegram_messages g
                       WHERE g.text ~ '\\$[A-Za-z]' AND g.references_derived_at IS NOT NULL
                       ORDER BY g.posted_at DESC LIMIT 400''')
        rows=cur.fetchall()
    both=[r for r in rows if extract.solana_addresses(r[2]) and extract.cashtags(r[2])]
    if not both:
        return check('4. contract wins over a ticker in the same message',False,
                     'no collected message carried both a contract and a cashtag yet')
    channel_id,message_id,text=both[0]
    with conn,conn.cursor() as cur:
        cur.execute('''SELECT reference_kind,raw_reference,resolution FROM telegram_token_mentions
                       WHERE channel_id=%s AND message_id=%s''',(channel_id,message_id))
        derived=cur.fetchall()
    kinds={d[0] for d in derived}
    return check('4. contract wins over a ticker in the same message',
                 kinds=={'contract'},
                 dict(message=text[:200],cashtags_in_text=extract.cashtags(text),
                      derived=[dict(kind=d[0],reference=d[1],resolution=d[2]) for d in derived],
                      note='a ticker beside a contract resolving to some other token is how a channel '
                           'gets credited with a call it never made'))


def reconstruction(conn,token,network='solana'):
    """5. One known case, end to end, in the line the acceptance asks for."""
    case=analytics.case_study(conn,token,network)
    originals=[h for h in case['hops'] if h['origin']=='original']
    first=originals[0] if originals else None
    line=None
    if first:
        rungs=first.get('outcomes') or {}
        parts=[f"{r}m {rungs[r]['status']}"+(f" {rungs[r]['return_pct']:+.1f}%" if rungs[r].get('return_pct') is not None else '')
               for r in ('30','60','180','1440') if r in rungs]
        line=(f"@{first['channel']} -> {token} -> first observed mention {first['mentioned_at']} -> "
              f"market cap at mention {first.get('market_cap')} -> "+', '.join(parts))
    return check('5. one known case reconstructs from the original call to its outcomes',
                 case['passed'],dict(acceptance_line=line,items=case['acceptance']))


async def health_distinguishes(conn,fetch):
    """6. 'We lost the channel' is not the same fact as 'the channel was quiet'."""
    quiet=[c for c in collector.active_channels(conn) if c.get('last_message_id')]
    quiet=quiet[:1]
    summary=await collector.run(conn,fetch,mode='poll',channels=quiet+[
        dict(channel_id=-1,username=IMPOSSIBLE_PEER)],pause=0)
    report=analytics.health(conn)
    states={c['channel']:c['access_state'] for c in report['channels']}
    broken=[c for c in report['channels_losing_access']]
    quiet_ok=all(states.get(c.get('username') or str(c['channel_id']))=='ok' for c in quiet) if quiet else None
    return check('6. health tells a lost channel apart from a quiet one',
                 bool(broken) and quiet_ok is not False,
                 dict(deliberately_broken_peer=IMPOSSIBLE_PEER,channels_losing_access=broken,
                      quiet_channels_still_ok=quiet_ok,verdict=report['verdict'],
                      run_errors=summary['errors'],
                      note='a quiet channel returns zero messages and stays access_state=ok; a lost '
                           'one is forbidden/not_found and appears in channels_losing_access'))


def relay_attribution(conn,network='solana'):
    """The manual inspection: original post -> relayed alert -> later independent mention."""
    with conn,conn.cursor() as cur:
        cur.execute('''SELECT token_address,count(DISTINCT channel_id) FROM telegram_token_mentions
                       WHERE network=%s AND token_address IS NOT NULL
                       GROUP BY token_address HAVING count(DISTINCT channel_id)>1
                       ORDER BY count(*) DESC LIMIT 1''',(network,))
        row=cur.fetchone()
    if not row:
        return check('7. relay attribution, for manual inspection',False,
                     'no token has been posted by more than one monitored channel yet')
    graph=analytics.propagation(conn,row[0],network)
    return check('7. relay attribution, for manual inspection',
                 graph['original_posts']>=1,
                 dict(token=row[0],channels=row[1],
                      chain=[dict(channel=h['channel'],at=str(h['mentioned_at']),origin=h['origin'],
                                  relays=h['relay_of_channel_id'],why=h['relay_reason'],
                                  sequence=h['sequence']) for h in graph['hops']],
                      original_posts=graph['original_posts'],forwards=graph['forwards'],
                      reposts=graph['reposts'],
                      note='READ THIS ONE BY EYE. Only original posts take a sequence position. A '
                           'repost credited as original inflates a relay channel first-among-monitored '
                           'count by exactly the tokens it was slowest on.'))


def markdown(results):
    lines=['## Live gate evidence','',
           f"Run {datetime.now(timezone.utc).isoformat(timespec='seconds')} against real Telegram peers "
           'and the production database.','']
    for result in results:
        lines.append(f"- {'PASS' if result['passed'] else 'FAIL'} — {result['check']}")
        detail=result['detail']
        if isinstance(detail,dict) and detail.get('acceptance_line'):
            lines.append(f"  - `{detail['acceptance_line']}`")
    lines+=['','<details><summary>Full evidence</summary>','','```json',
            json.dumps(results,indent=1,default=str),'```','','</details>']
    return '\n'.join(lines)


async def gate(conn,usernames,token,limit):
    client,fetch,close=await collector.open_client()
    results=[]
    try:
        results.append(await resolve(conn,client,usernames))
        channels=collector.active_channels(conn)
        if not channels:raise SystemExit('no active channels after resolution')
        # Poll first so there is something to derive from, then backfill for the idempotence check.
        await collector.run(conn,fetch,mode='poll',limit=limit)
        results.append(await backfill(conn,fetch,channels[0]))
        mentions.derive(conn)
        mentions.fetch_prices(conn)
        mentions.compute_outcomes(conn)
        results.append(contract_resolution(conn))
        results.append(contract_beats_ticker(conn))
        results.append(reconstruction(conn,token))
        results.append(await health_distinguishes(conn,fetch))
        results.append(relay_attribution(conn))
    finally:await close()
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='execute the gate (Telegram + database)')
    parser.add_argument('--channels',nargs='*',help='usernames to resolve; default: the config file')
    parser.add_argument('--token',default=analytics.FLEX,help='the mint to reconstruct')
    parser.add_argument('--limit',type=int,default=200)
    parser.add_argument('--markdown',action='store_true',help='print a PR-ready evidence block')
    args=parser.parse_args()
    usernames=args.channels or [e['username'] for e in json.loads(collector.CONFIG.read_text())
                                if e.get('username')]
    if not args.run:
        print(json.dumps(dict(mode='plan_only',channels=usernames,token=args.token,
                              checks=['peers resolved','backfill idempotence','contract resolves',
                                      'contract beats ticker','case reconstructs','health separates '
                                      'lost from quiet','relay attribution (by eye)'],
                              requires=['TELEGRAM_API_ID','TELEGRAM_API_HASH','TELEGRAM_SESSION',
                                        'DATABASE_URL'])));return
    import psycopg2
    conn=psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
    try:
        collector.setup(conn);mentions.setup(conn)
        results=asyncio.run(gate(conn,usernames,args.token,args.limit))
        print(markdown(results) if args.markdown else json.dumps(results,indent=1,default=str))
        if not all(r['passed'] for r in results):raise SystemExit(1)
    finally:conn.close()


if __name__=='__main__':main()
