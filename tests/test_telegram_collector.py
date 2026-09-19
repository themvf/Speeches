"""Collector behaviour that a live smoke test cannot check cheaply: normalisation across the two
client libraries, per-channel error isolation, and the cursor rules.

What these tests deliberately cannot prove is the real client - a fake answers whatever it is asked,
never floods, and never loses access mid-run. That is why the spec makes a live run a merge gate,
the same lesson the Solana chain adapter learned when a mocked suite passed against an address
format that resolved to nothing.
"""
import asyncio
from datetime import datetime,timezone
import types
import telegram_collector as collector

POSTED=datetime(2026,9,19,12,0,tzinfo=timezone.utc)


class FakeCursor:
    def __init__(self,log):self.log=log;self.rowcount=0
    def execute(self,sql,args=None):self.log.append((' '.join(sql.split()),args))
    def fetchone(self):return (1,)
    def __enter__(self):return self
    def __exit__(self,*a):return False


class FakeConn:
    """Enough of psycopg2 to record what the run wrote, and nothing more."""
    def __init__(self):self.log=[]
    def cursor(self):return FakeCursor(self.log)
    def __enter__(self):return self
    def __exit__(self,*a):return False


def pyrogram_message(**over):
    base=dict(id=42,date=POSTED,text='gm $FLEX',views=100,forwards=3,edit_date=None,
              reply_to_message_id=None,author_signature='admin',from_user=None,
              forward_from_chat=None,forward_from_message_id=None,forward_date=None)
    base.update(over)
    return types.SimpleNamespace(**base)


def telethon_message(**over):
    base=dict(id=42,date=POSTED,message='gm $FLEX',views=100,forwards=3,edit_date=None,
              reply_to_msg_id=None,post_author='admin',sender_id=7,fwd_from=None)
    base.update(over)
    return types.SimpleNamespace(**base)


def test_both_libraries_normalise_to_the_same_row():
    # The row shape is the only thing downstream depends on, which is what makes the library choice
    # reversible: Kurigram today, Telethon if it stops being maintained.
    a=collector.normalize_pyrogram(pyrogram_message())
    b=collector.normalize_telethon(telethon_message())
    for field in ('message_id','posted_at','text','views','forwards','is_forward','author_signature'):
        assert a[field]==b[field],field
    assert a['posted_at'].tzinfo is timezone.utc


def test_a_forward_is_flagged_and_keeps_its_origin():
    origin=types.SimpleNamespace(id=-100123,title='Alpha Calls')
    row=collector.normalize_pyrogram(pyrogram_message(forward_from_chat=origin,
                                                      forward_from_message_id=9,forward_date=POSTED))
    # One post copied across five channels is one discovery. Everything downstream that counts
    # "who was first" depends on this flag being right.
    assert row['is_forward'] and row['forward_from_channel_id']==-100123
    assert row['forward_from_name']=='Alpha Calls' and row['forward_from_message_id']==9
    assert collector.normalize_telethon(telethon_message())['is_forward'] is False


def test_urls_are_extracted_at_collection_time():
    row=collector.normalize_pyrogram(pyrogram_message(text='see https://t.me/x and https://dexscreener.com/a'))
    assert row['urls']==['https://t.me/x','https://dexscreener.com/a']


def test_a_naive_telegram_date_is_read_as_utc():
    row=collector.normalize_pyrogram(pyrogram_message(date=datetime(2026,9,19,12,0)))
    assert row['posted_at']==POSTED


def test_one_failing_channel_does_not_cost_the_run(monkeypatch):
    # The window a poll covers is not recoverable once the feed moves on, so a dead channel must
    # never abort the channels after it.
    monkeypatch.setattr(collector,'upsert_messages',lambda conn,cid,rows:(len(rows),0))
    states=[];monkeypatch.setattr(collector,'record_channel_state',
                                  lambda conn,cid,**kw:states.append((cid,kw.get('state','ok'))))
    monkeypatch.setattr(collector,'record_run',lambda conn,summary:summary)
    async def fetch(peer,min_id=0,offset_id=0,limit=0):
        if peer=='bad':raise RuntimeError('[420 FLOOD_WAIT_X] wait of 30 seconds is required')
        return [dict(message_id=1,posted_at=POSTED)]
    summary=asyncio.run(collector.run(FakeConn(),fetch,channels=[
        dict(channel_id=1,username='bad'),dict(channel_id=2,username='good')],pause=0))
    assert summary['channels_attempted']==2 and summary['channels_ok']==1
    assert summary['messages_new']==1 and summary['complete'] is False
    assert summary['flood_waits']==1 and summary['flood_wait_seconds']==30
    assert ('flood_wait' in states[0][1]) and states[1][1]=='ok'


def test_telegram_failure_modes_are_told_apart():
    # A FloodWait is the API working correctly; a Forbidden is access we have lost and needs a human.
    assert collector.classify_error(RuntimeError('[420 FLOOD_WAIT_X] wait of 12 seconds'))==('flood_wait',12)
    assert collector.classify_error(type('Forbidden',(Exception,),{})())[0]=='forbidden'
    assert collector.classify_error(RuntimeError('PEER_ID_INVALID'))[0]=='not_found'
    assert collector.classify_error(RuntimeError('connection reset'))==('error',None)


def test_a_poll_asks_only_for_messages_newer_than_the_cursor(monkeypatch):
    monkeypatch.setattr(collector,'upsert_messages',lambda conn,cid,rows:(0,0))
    monkeypatch.setattr(collector,'record_channel_state',lambda *a,**k:None)
    seen={}
    async def fetch(peer,min_id=0,offset_id=0,limit=0):
        seen.update(peer=peer,min_id=min_id,offset_id=offset_id);return []
    asyncio.run(collector.collect_channel(FakeConn(),dict(channel_id=1,username='alpha',
                                                          last_message_id=500),fetch,'poll'))
    assert seen=={'peer':'alpha','min_id':500,'offset_id':0}


def test_a_backfill_walks_down_from_the_floor_and_records_when_it_ends(monkeypatch):
    monkeypatch.setattr(collector,'upsert_messages',lambda conn,cid,rows:(len(rows),0))
    recorded={};monkeypatch.setattr(collector,'record_channel_state',lambda conn,cid,**kw:recorded.update(kw))
    async def fetch(peer,min_id=0,offset_id=0,limit=0):
        return [] if offset_id==100 else [dict(message_id=120,posted_at=POSTED),
                                          dict(message_id=100,posted_at=POSTED)]
    result=asyncio.run(collector.collect_channel(FakeConn(),dict(channel_id=1,backfill_floor_id=None),
                                                 fetch,'backfill'))
    assert result['floor_id']==100 and result['backfill_complete'] is False
    result=asyncio.run(collector.collect_channel(FakeConn(),dict(channel_id=1,backfill_floor_id=100),
                                                 fetch,'backfill'))
    # Nothing older came back: history is exhausted or access ends here, and the run that decided
    # so is on the record rather than being retried forever.
    assert result['backfill_complete'] is True


def test_the_cursor_only_moves_forward():
    # GREATEST/LEAST in SQL, asserted here so an "optimisation" that drops them is caught: a partial
    # or out-of-order read must never rewind the cursor and skip the window in between.
    conn=FakeConn()
    collector.record_channel_state(conn,1,rows=[dict(message_id=5,posted_at=POSTED)])
    sql=conn.log[0][0]
    assert 'GREATEST(coalesce(last_message_id,0)' in sql and 'LEAST(coalesce(backfill_floor_id' in sql
