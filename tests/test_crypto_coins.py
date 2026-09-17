import json
from pathlib import Path
import pytest
from crypto_coins import COINS,SYMBOLS,mentions,markets,archive_start,config

FIXTURES=json.loads(Path(__file__).with_name('fixtures').joinpath('crypto-coin-matches.json').read_text())

@pytest.mark.parametrize('text,coin,mode,expected',FIXTURES)
def test_shared_fixtures_match_typescript(text,coin,mode,expected):
    assert mentions(text,coin,mode)==expected

def test_registry_shape():
    assert len(SYMBOLS)==len(set(SYMBOLS))>=5 and 'ZEC' in SYMBOLS and 'ZEC' not in markets()
    assert archive_start('PONS').isoformat()=='2026-07-01'
    for c in COINS.values():
        assert c['official']==[h.lower() for h in c['official']] and c['searchQuery']
    with pytest.raises(KeyError):config('BTC')

def test_live_campaign_queries_are_pinned_to_saved_window_text():
    # Saved search windows store query text verbatim; changing it would orphan pagination cursors.
    from crypto_social_rolling import TRACKED
    from crypto_social_pilot import COINS as PILOT
    from crypto_social_history import PONS_ADDRESS,DPONS_ADDRESS
    assert TRACKED['PONS'][1]==f'("{PONS_ADDRESS}" OR (PONS Robinhood) OR from:ponsdotfamily)'
    assert TRACKED['DPONS'][1]==f'(DPONS OR "DiamondPons" OR "Diamond Pons" OR "{DPONS_ADDRESS}")'
    assert TRACKED['STANDARD'][1]=='("The Standard Reserve" OR from:standard_rsv OR to:standard_rsv OR "0x88ad8ddf1e3898412146a534538d418c6f8a9062")'
    assert PILOT['ZCAT'][1]==COINS['ZCAT']['searchQuery'] and PILOT['ZEC'][1]==COINS['ZEC']['searchQuery']
