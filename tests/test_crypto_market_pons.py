"""Cover the one scheduled crypto archiver no test imported.

It shares setup()/pools()/normalize()/save() with crypto_market_history, so the schema it writes
into now carries the price_observation volume constraint. Nothing verified that an ohlcv archiver
still passes that constraint, and a scheduled run was this module's first execution.
"""
from datetime import datetime,timezone,timedelta
from unittest.mock import MagicMock,patch
import pytest
from test_crypto_social_pilot import db
import crypto_market_pons as pons
from crypto_social_history import PONS_ADDRESS

NOW=datetime(2026,9,20,12,tzinfo=timezone.utc)
POOL='0x39dbed3a2bd333467115de45665cc57f813c45710000000000000000000000aa'
OTHER='0x10cc6bd38112cac182db90b6a71d8bb5939526ba'

def _pool(address,created,liquidity='5000'):
    return {'attributes':{'address':address,'name':'PONS / WETH','pool_created_at':created,'reserve_in_usd':liquidity},
            'relationships':{'base_token':{'data':{'id':'robinhood_'+PONS_ADDRESS.lower()}}}}

def _candles(days=6,volume=1234.5):
    start=NOW.replace(hour=0)-timedelta(days=days)
    return {'data':{'attributes':{'ohlcv_list':[
        [(start+timedelta(days=i)).timestamp(),1.0,1.2,0.9,1.1,volume+i] for i in range(days)]}}}

def _responses(catalog,candles):
    def get(url,**_):
        r=MagicMock();r.status_code=200
        r.json.return_value=catalog if '/tokens/' in url else candles
        return r
    return get

def test_db_archives_real_volume_and_pins_the_pool_once(db):
    catalog={'data':[_pool(POOL,'2026-07-13T20:42:21Z'),_pool(OTHER,'2026-08-01T00:00:00Z','9000')]}
    with patch.object(pons.time,'sleep'),patch.object(pons.requests,'get',_responses(catalog,_candles())):
        result=pons.refresh(db)
    assert result['coin']=='PONS' and result['points']>0
    assert result['pool']==POOL          # the oldest candidate, matching the archive's pin rule
    with db,db.cursor() as cur:
        cur.execute("SELECT count(*),count(volume) FROM crypto_market_observations WHERE kind='ohlcv'")
        rows,with_volume=cur.fetchone()
        assert rows>0 and with_volume==rows   # an ohlcv archiver still records real volume
        cur.execute("SELECT count(*) FROM crypto_market_sources WHERE coin='PONS' AND is_default")
        assert cur.fetchone()[0]==1
    # Re-running keeps the same pin even when a newer, more liquid pool exists.
    newer={'data':[_pool(OTHER,'2026-09-01T00:00:00Z','999999'),_pool(POOL,'2026-07-13T20:42:21Z')]}
    with patch.object(pons.time,'sleep'),patch.object(pons.requests,'get',_responses(newer,_candles())):
        again=pons.refresh(db)
    assert again['pool']==POOL
    with db,db.cursor() as cur:
        cur.execute("SELECT id FROM crypto_market_sources WHERE coin='PONS' AND is_default")
        assert cur.fetchall()==[('geckoterminal:robinhood:'+POOL,)]

def test_db_a_contract_with_no_indexed_pool_fails_loudly(db):
    with patch.object(pons.time,'sleep'),patch.object(pons.requests,'get',_responses({'data':[]},_candles())):
        with pytest.raises(ValueError,match='No verified PONS pool'):
            pons.refresh(db)

def test_db_a_provider_error_is_not_archived_as_an_empty_day(db):
    def failing(url,**_):
        r=MagicMock();r.status_code=502;return r
    with patch.object(pons.time,'sleep'),patch.object(pons.requests,'get',failing):
        with pytest.raises(ValueError,match='Market HTTP 502'):
            pons.refresh(db)
    with db,db.cursor() as cur:
        cur.execute('SELECT count(*) FROM crypto_market_observations')
        assert cur.fetchone()[0]==0
