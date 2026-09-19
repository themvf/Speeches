"""Telegram OSINT collector: raw message capture from monitored channels into Neon.

Uses an existing MTProto client library rather than speaking MTProto ourselves - Kurigram first
(maintained Pyrogram fork), Telethon as the alternative; see docs/telegram-osint-spec.md for why,
and why the original Pyrogram repository is not an option.

This module collects and nothing else. It does not resolve tokens, join the archive or judge a
channel: those are re-runnable derivations and live in telegram_mentions.py / telegram_analytics.py.
Keeping collection separate is what makes it safe to change a classification rule later without
re-reading Telegram, which for deleted or rate-limited history is not possible at any price.

Collection is idempotent by construction: (channel_id, message_id) is Telegram's own identity, so a
re-read of a window writes the same rows.

Default with no flags is a no-network plan, same convention as launchpad_archive.py.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import telegram_extract as extract

CONFIG=Path(__file__).with_name('telegram_channels.json')
SQL=Path(__file__).with_name('sql').joinpath('telegram_osint.sql')
POLL_LIMIT=200                 # messages per channel per poll; a busy channel is caught up over runs
BACKFILL_PAGE=200
BACKFILL_MAX=2000              # per channel per backfill run, so one loud channel cannot eat the run
MAX_FLOOD_WAIT=300             # a longer wait is recorded and skipped rather than blocking the run
PER_CHANNEL_PAUSE=1.0


def setup(conn):
    with conn,conn.cursor() as cur:cur.execute(SQL.read_text())


def utc(value):
    """Telegram hands back naive UTC datetimes in some paths and aware ones in others."""
    if value is None:return None
    if isinstance(value,(int,float)):return datetime.fromtimestamp(value,timezone.utc)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _int(value):
    try:return int(value)
    except (TypeError,ValueError):return None


def normalize_pyrogram(message):
    """A Kurigram/Pyrogram Message -> the row we store. Attribute access only, so a fake works."""
    forward_chat=getattr(message,'forward_from_chat',None)
    text=getattr(message,'text',None) or getattr(message,'caption',None)
    sender=getattr(message,'from_user',None)
    return _row(message_id=getattr(message,'id',None),
                posted_at=utc(getattr(message,'date',None)),
                text=str(text) if text is not None else None,
                sender_id=getattr(sender,'id',None),
                sender_username=getattr(sender,'username',None),
                author_signature=getattr(message,'author_signature',None),
                edited_at=utc(getattr(message,'edit_date',None)),
                views=_int(getattr(message,'views',None)),
                forwards=_int(getattr(message,'forwards',None)),
                reply_to_message_id=_int(getattr(message,'reply_to_message_id',None)),
                forward_from_channel_id=getattr(forward_chat,'id',None),
                forward_from_name=(getattr(forward_chat,'title',None) or getattr(forward_chat,'username',None)
                                   or getattr(message,'forward_sender_name',None)),
                forward_from_message_id=_int(getattr(message,'forward_from_message_id',None)),
                forward_origin_at=utc(getattr(message,'forward_date',None)))


def normalize_telethon(message):
    """A Telethon Message -> the same row. The two libraries name almost nothing the same way, so
    the normalisers are separate and the row shape is the only thing downstream depends on."""
    forward=getattr(message,'fwd_from',None)
    origin=getattr(forward,'from_id',None) if forward else None
    return _row(message_id=getattr(message,'id',None),
                posted_at=utc(getattr(message,'date',None)),
                text=getattr(message,'message',None) or None,
                sender_id=_int(getattr(message,'sender_id',None)),
                sender_username=None,
                author_signature=getattr(message,'post_author',None),
                edited_at=utc(getattr(message,'edit_date',None)),
                views=_int(getattr(message,'views',None)),
                forwards=_int(getattr(message,'forwards',None)),
                reply_to_message_id=_int(getattr(message,'reply_to_msg_id',None)),
                forward_from_channel_id=_int(getattr(origin,'channel_id',None)) if origin else None,
                forward_from_name=getattr(forward,'from_name',None) if forward else None,
                forward_from_message_id=_int(getattr(forward,'channel_post',None)) if forward else None,
                forward_origin_at=utc(getattr(forward,'date',None)) if forward else None)


def _row(**fields):
    fields['is_forward']=bool(fields.get('forward_from_channel_id') or fields.get('forward_from_name')
                              or fields.get('forward_origin_at'))
    fields['urls']=extract.urls(fields.get('text'))
    # The raw payload is the normalised row itself: the library objects are not JSON-serialisable and
    # storing a lossy dump of them would invite exactly the kind of "we have it somewhere" reasoning
    # this schema is built to avoid. What we can reconstruct from is what we store.
    fields['raw']={k:(v.isoformat() if isinstance(v,datetime) else v) for k,v in fields.items()}
    return fields


def upsert_messages(conn,channel_id,rows):
    """Write a batch. Returns (new, updated).

    A re-read writes the same primary key, so the UPDATE branch exists for the fields that genuinely
    change after posting - edits, view counts - and nothing else. posted_at and text are refreshed
    too, because an edited message's text is the current truth about what the channel is saying;
    `edited_at` is what records that it changed.
    """
    rows=[r for r in rows if r.get('message_id') and r.get('posted_at')]
    if not rows:return 0,0
    from psycopg2.extras import execute_values,Json
    values=[(channel_id,r['message_id'],r['posted_at'],r.get('sender_id'),r.get('sender_username'),
             r.get('author_signature'),r.get('text'),r.get('edited_at'),r.get('views'),r.get('forwards'),
             r.get('reply_to_message_id'),r.get('is_forward',False),r.get('forward_from_channel_id'),
             r.get('forward_from_name'),r.get('forward_from_message_id'),r.get('forward_origin_at'),
             r.get('urls') or [],Json(r.get('raw') or {})) for r in rows]
    with conn,conn.cursor() as cur:
        result=execute_values(cur,'''INSERT INTO telegram_messages
              (channel_id,message_id,posted_at,sender_id,sender_username,author_signature,text,
               edited_at,views,forwards,reply_to_message_id,is_forward,forward_from_channel_id,
               forward_from_name,forward_from_message_id,forward_origin_at,urls,raw)
              VALUES %s
              ON CONFLICT (channel_id,message_id) DO UPDATE SET
                text=EXCLUDED.text,edited_at=EXCLUDED.edited_at,views=EXCLUDED.views,
                forwards=EXCLUDED.forwards,urls=EXCLUDED.urls,raw=EXCLUDED.raw
              RETURNING (xmax=0) AS inserted''',values,fetch=True)
    new=sum(1 for row in result if row[0])
    return new,len(result)-new


def record_channel_state(conn,channel_id,*,rows=None,state='ok',note=None,floor_id=None,
                         backfill_complete=None,now=None):
    """Advance a channel's cursors. last_message_id only ever moves forward, so an out-of-order or
    partial read can never rewind the cursor and silently skip the window in between."""
    now=now or datetime.now(timezone.utc)
    rows=rows or []
    top=max((r['message_id'] for r in rows),default=None)
    top_at=max((r['posted_at'] for r in rows),default=None)
    ok=state=='ok'
    with conn,conn.cursor() as cur:
        cur.execute('''UPDATE telegram_channels SET
              last_message_id=GREATEST(coalesce(last_message_id,0),coalesce(%s,0)),
              last_message_at=GREATEST(coalesce(last_message_at,'epoch'::timestamptz),
                                       coalesce(%s,'epoch'::timestamptz)),
              backfill_floor_id=LEAST(coalesce(backfill_floor_id,9223372036854775807),
                                      coalesce(%s,9223372036854775807)),
              backfill_complete=coalesce(%s,backfill_complete),
              last_poll_at=%s,
              last_success_at=CASE WHEN %s THEN %s ELSE last_success_at END,
              access_state=%s,access_note=%s,
              consecutive_failures=CASE WHEN %s THEN 0 ELSE consecutive_failures+1 END
            WHERE channel_id=%s''',
                    (top,top_at,floor_id,backfill_complete,now,ok,now,state,note,ok,channel_id))


# ------------------------------------------------------------------ channel config

def sync_config(conn,path=CONFIG):
    """Seed telegram_channels from the committed JSON. The table is the source of truth once seeded:
    a channel paused in the database stays paused unless the config says otherwise.

    A Telegram peer id cannot be derived from a username - it has to be asked for - so an entry may
    carry a username and a null channel_id. Those are reported as pending and filled in by --resolve
    against a live client; they are never guessed, because a wrong id silently collects the wrong
    channel rather than failing.
    """
    entries=json.loads(Path(path).read_text())
    pending=[e['username'] for e in entries if not e.get('channel_id') and e.get('username')]
    with conn,conn.cursor() as cur:
        for entry in entries:
            if not entry.get('channel_id'):continue
            cur.execute('''INSERT INTO telegram_channels (channel_id,username,title,kind,active,notes)
                           VALUES (%s,%s,%s,%s,%s,%s)
                           ON CONFLICT (channel_id) DO UPDATE SET
                             username=coalesce(EXCLUDED.username,telegram_channels.username),
                             title=coalesce(EXCLUDED.title,telegram_channels.title),
                             kind=coalesce(EXCLUDED.kind,telegram_channels.kind),
                             active=EXCLUDED.active,notes=EXCLUDED.notes''',
                        (entry['channel_id'],entry.get('username'),entry.get('title'),
                         entry.get('kind'),entry.get('active',True),entry.get('notes')))
    return dict(configured=len(entries),seeded=len(entries)-len(pending),pending_resolution=pending)


async def resolve_channels(conn,client,usernames):
    """Ask Telegram for each username's peer id and record it. Failures are per username: one
    channel we have lost access to must not block the rest from being registered."""
    out=[]
    for username in usernames:
        try:
            chat=await client.get_chat(username) if hasattr(client,'get_chat') else await client.get_entity(username)
        except Exception as error:                                     # noqa: BLE001
            out.append(dict(username=username,error=f'{type(error).__name__}: {error}'));continue
        channel_id=getattr(chat,'id',None)
        with conn,conn.cursor() as cur:
            cur.execute('''INSERT INTO telegram_channels (channel_id,username,title,kind,active)
                           VALUES (%s,%s,%s,%s,true)
                           ON CONFLICT (channel_id) DO UPDATE SET username=EXCLUDED.username,
                             title=coalesce(EXCLUDED.title,telegram_channels.title)''',
                        (channel_id,username.lstrip('@'),getattr(chat,'title',None),
                         str(getattr(chat,'type',None) or getattr(chat,'kind','') or '') or None))
        out.append(dict(username=username,channel_id=channel_id,title=getattr(chat,'title',None)))
    return out


def active_channels(conn):
    with conn,conn.cursor() as cur:
        cur.execute('''SELECT channel_id,username,last_message_id,backfill_floor_id,backfill_complete
                       FROM telegram_channels WHERE active ORDER BY coalesce(last_poll_at,'epoch') ASC''')
        return [dict(channel_id=r[0],username=r[1],last_message_id=r[2],
                     backfill_floor_id=r[3],backfill_complete=r[4]) for r in cur.fetchall()]


# ------------------------------------------------------------------ the run

async def collect_channel(conn,channel,fetch,mode,limit=None):
    """Collect one channel through `fetch`, an async callable the backend supplies.

    `fetch(peer, min_id, offset_id, limit)` yields normalised rows newest-first. Keeping the backend
    behind one callable is what lets the whole run be exercised without a Telegram account: the live
    smoke test is the only thing that can prove the real client, and it is a merge gate, not a unit
    test (see the spec).
    """
    peer=channel.get('username') or channel['channel_id']
    if mode=='poll':
        rows=await fetch(peer,min_id=channel.get('last_message_id') or 0,offset_id=0,
                         limit=limit or POLL_LIMIT)
        floor=None;complete=None
    else:
        offset=channel.get('backfill_floor_id') or 0
        rows=await fetch(peer,min_id=0,offset_id=offset,limit=limit or BACKFILL_PAGE)
        floor=min((r['message_id'] for r in rows),default=None)
        # Nothing older came back: either history is exhausted or access ends here. Either way the
        # backfill for this channel is done, and the run that decided so is on the record.
        complete=not rows
    new,updated=upsert_messages(conn,channel['channel_id'],rows)
    record_channel_state(conn,channel['channel_id'],rows=rows,floor_id=floor,backfill_complete=complete)
    return dict(channel_id=channel['channel_id'],seen=len(rows),new=new,updated=updated,
                floor_id=floor,backfill_complete=complete)


async def run(conn,fetch,mode='poll',limit=None,channels=None,now=None,pause=PER_CHANNEL_PAUSE):
    """One collector run across every active channel. Errors are per channel: one dead channel must
    not cost the run, because the window a poll covers is not recoverable once the feed moves on."""
    started=now or datetime.now(timezone.utc)
    channels=channels if channels is not None else active_channels(conn)
    summary=dict(mode=mode,started_at=started,channels_attempted=len(channels),channels_ok=0,
                 messages_seen=0,messages_new=0,messages_updated=0,flood_waits=0,flood_wait_seconds=0,
                 reconnects=0,errors=[],per_channel=[])
    for index,channel in enumerate(channels):
        try:
            result=await collect_channel(conn,channel,fetch,mode,limit=limit)
            summary['channels_ok']+=1
            summary['messages_seen']+=result['seen']
            summary['messages_new']+=result['new']
            summary['messages_updated']+=result['updated']
            summary['per_channel'].append(result)
        except Exception as error:                                     # noqa: BLE001 - per-channel isolation
            state,seconds=classify_error(error)
            if state=='flood_wait':
                summary['flood_waits']+=1;summary['flood_wait_seconds']+=seconds or 0
            summary['errors'].append(f"{channel.get('username') or channel['channel_id']}: {state}: {error}")
            record_channel_state(conn,channel['channel_id'],state=state,note=str(error)[:500])
        if pause and index<len(channels)-1:await asyncio.sleep(pause)
    summary['complete']=not summary['errors']
    summary['finished_at']=datetime.now(timezone.utc)
    record_run(conn,summary)
    return summary


def classify_error(error):
    """(access_state, seconds). Telegram's failure modes mean different things and must not collapse
    into one 'error': a FloodWait is the API working correctly, a Forbidden is access we have lost."""
    name=type(error).__name__;text=str(error)
    if 'FloodWait' in name or 'FLOOD_WAIT' in text:
        # Both clients carry the wait as an attribute (Pyrogram .value, Telethon .seconds). The text
        # fallback reads "wait of N seconds" rather than the first digits in the string, which on a
        # Pyrogram message would be the 420 error code.
        import re
        seconds=getattr(error,'value',None) or getattr(error,'seconds',None)
        if seconds is None:
            match=re.search(r'(\d+)\s*second',text)
            seconds=int(match.group(1)) if match else None
        return 'flood_wait',seconds
    if 'Forbidden' in name or 'CHANNEL_PRIVATE' in text or 'ChatAdminRequired' in name:return 'forbidden',None
    if 'UsernameNotOccupied' in name or 'NotFound' in name or 'PEER_ID_INVALID' in text:return 'not_found',None
    return 'error',None


def record_run(conn,summary):
    with conn,conn.cursor() as cur:
        cur.execute('''INSERT INTO telegram_collection_runs
              (started_at,finished_at,mode,channels_attempted,channels_ok,messages_seen,messages_new,
               messages_updated,reconnects,flood_waits,flood_wait_seconds,complete,errors,note)
              VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id''',
                    (summary['started_at'],summary.get('finished_at'),summary['mode'],
                     summary['channels_attempted'],summary['channels_ok'],summary['messages_seen'],
                     summary['messages_new'],summary['messages_updated'],summary.get('reconnects',0),
                     summary['flood_waits'],summary['flood_wait_seconds'],summary['complete'],
                     summary['errors'],summary.get('note')))
        summary['run_id']=cur.fetchone()[0]
    return summary


# ------------------------------------------------------------------ backends

def backend_name():
    return os.environ.get('TELEGRAM_LIBRARY','kurigram').lower()


async def kurigram_fetch_factory(client):
    """Kurigram/Pyrogram: get_chat_history walks newest-first from offset_id.

    min_id is applied here rather than passed through, because Pyrogram's history iterator has no
    min_id parameter - stopping the walk ourselves is the difference between an incremental poll and
    re-reading a channel's whole history every run.
    """
    async def fetch(peer,min_id=0,offset_id=0,limit=POLL_LIMIT):
        rows=[]
        async for message in client.get_chat_history(peer,limit=limit,offset_id=offset_id or 0):
            if min_id and getattr(message,'id',0)<=min_id:break
            rows.append(normalize_pyrogram(message))
        return rows
    return fetch


async def telethon_fetch_factory(client):
    async def fetch(peer,min_id=0,offset_id=0,limit=POLL_LIMIT):
        rows=[]
        async for message in client.iter_messages(peer,limit=limit,offset_id=offset_id or 0,min_id=min_id or 0):
            rows.append(normalize_telethon(message))
        return rows
    return fetch


async def open_client():
    """Authenticate as a Telegram USER account, not a bot.

    messages.getHistory is a user-account method; a Bot API bot cannot read arbitrary channel
    history even where it can read new posts. The session string is a credential equivalent to the
    account, so it is read from the environment and never written to disk by this process - see the
    spec's authentication section.
    """
    api_id=int(os.environ['TELEGRAM_API_ID']);api_hash=os.environ['TELEGRAM_API_HASH']
    session=os.environ['TELEGRAM_SESSION']
    if backend_name()=='telethon':
        from telethon import TelegramClient
        from telethon.sessions import StringSession
        client=TelegramClient(StringSession(session),api_id,api_hash)
        await client.start()
        return client,await telethon_fetch_factory(client),client.disconnect
    from pyrogram import Client
    client=Client('telegram-osint',api_id=api_id,api_hash=api_hash,session_string=session,in_memory=True)
    await client.start()
    return client,await kurigram_fetch_factory(client),client.stop


async def inspect(client,fetch,peers,limit=20):
    """Print real messages beside what extraction made of them. Writes nothing, anywhere.

    This is the step before the cron is turned on. Real call-channel formatting is where
    deterministic extraction fails in ways no fixture anticipates: zero-width joiners between
    characters of an address, a contract split across two lines by the sending client, decorative
    punctuation glued to a cashtag, Cyrillic homoglyphs. A fixture asserts what we already thought
    of; twenty real messages show what we did not.

    So the output deliberately puts the raw repr next to the extraction, and flags the two shapes
    that most often mean a missed contract: a message that mentions a contract in words but yielded
    no address, and a message carrying invisible characters.
    """
    import telegram_extract as extract
    out=[]
    for peer in peers:
        rows=await fetch(peer,min_id=0,offset_id=0,limit=limit)
        for row in rows:
            text=row.get('text') or ''
            invisible=sorted({hex(ord(c)) for c in text if ord(c) in INVISIBLE or 0x200b<=ord(c)<=0x200f})
            addresses=extract.solana_addresses(text)
            out.append(dict(peer=str(peer),message_id=row['message_id'],posted_at=str(row['posted_at']),
                            is_forward=row['is_forward'],
                            text=text[:400],
                            text_repr=repr(text[:200]),
                            addresses=addresses,cashtags=extract.cashtags(text),
                            urls=row['urls'],claims=extract.claims(text),
                            invisible_characters=invisible,
                            # Not a failure by itself - plenty of messages legitimately talk about a
                            # token without pasting its mint - but it is where to look first.
                            mentions_a_contract_but_none_extracted=bool(
                                not addresses and any(word in text.lower() for word in CONTRACT_WORDS))))
    return out


# Zero-width and directional marks: invisible in every client, and fatal to a base58 match if one
# lands inside an address.
INVISIBLE={0x00ad,0x061c,0x2060,0xfeff}
CONTRACT_WORDS=('contract','ca:','ca ','mint','token address','address:')


async def execute(conn,mode,limit=None,resolve=None):
    client,fetch,close=await open_client()
    try:
        if resolve:return dict(resolved=await resolve_channels(conn,client,resolve))
        return await run(conn,fetch,mode=mode,limit=limit)
    finally:await close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='collect (Telegram + database writes)')
    parser.add_argument('--backfill',action='store_true',help='walk older history instead of polling')
    parser.add_argument('--sync-config',action='store_true',help='seed telegram_channels from JSON')
    parser.add_argument('--resolve',action='store_true',help='resolve configured usernames to peer ids (Telegram)')
    parser.add_argument('--inspect',nargs='*',metavar='USERNAME',
                        help='print real messages beside their extraction and write nothing (do this '
                             'before enabling the cron)')
    parser.add_argument('--limit',type=int,default=None)
    args=parser.parse_args()
    if args.inspect is not None:
        usernames=args.inspect or [e['username'] for e in json.loads(CONFIG.read_text()) if e.get('username')]
        async def run_inspect():
            client,fetch,close=await open_client()
            try:return await inspect(client,fetch,usernames,limit=args.limit or 20)
            finally:await close()
        print(json.dumps(asyncio.run(run_inspect()),indent=1,ensure_ascii=False,default=str));return
    if not args.execute and not args.sync_config and not args.resolve:
        entries=json.loads(CONFIG.read_text()) if CONFIG.exists() else []
        print(json.dumps(dict(mode='plan_only',backend=backend_name(),
                              configured_channels=len(entries),
                              active_channels=sum(1 for e in entries if e.get('active',True)),
                              telegram_requests_max=len(entries)*(BACKFILL_MAX//BACKFILL_PAGE if args.backfill else 1),
                              llm_calls=0,database_writes=0)));return
    import psycopg2
    conn=psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
    try:
        setup(conn)
        if args.sync_config:print(json.dumps(sync_config(conn)));return
        if args.resolve:
            pending=sync_config(conn)['pending_resolution']
            print(json.dumps(asyncio.run(execute(conn,'poll',resolve=pending)),default=str));return
        summary=asyncio.run(execute(conn,'backfill' if args.backfill else 'poll',limit=args.limit))
        print(json.dumps(summary,default=str))
        if not summary['channels_ok']:raise SystemExit(1)
    finally:conn.close()


if __name__=='__main__':main()
