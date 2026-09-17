"""Tracked-coin registry shared with the dashboard (apps/web/lib/crypto-coins.json). No network."""
from datetime import date
import json
from pathlib import Path
import re

_PATH=Path(__file__).with_name('apps').joinpath('web','lib','crypto-coins.json')
COINS={c['symbol']:c for c in json.loads(_PATH.read_text())['coins']}
SYMBOLS=list(COINS)
_compiled={}


def config(symbol):
    if symbol not in COINS:raise KeyError('Unknown coin '+symbol)
    return COINS[symbol]


def archive_start(symbol):
    return date.fromisoformat(config(symbol)['archiveStart'])


def _patterns(symbol):
    if symbol not in _compiled:
        c=config(symbol)
        _compiled[symbol]=dict(words=[re.compile(w,re.I) for w in c['words']],
            context=[(re.compile(a,re.I),re.compile(b,re.I)) for a,b in c['contextWords']],
            exclude=[re.compile(e,re.I) for e in c['exclude']])
    return _compiled[symbol]


def has_contract(text,symbol):
    address=config(symbol)['address']
    if not address:return False
    return address.lower() in text.lower() if address.startswith('0x') else address in text


def mentions(text,symbol,mode='words'):
    """Byte-for-byte port of matchesCoin in apps/web/lib/crypto-coins.ts."""
    if has_contract(text,symbol):return True
    if mode=='contract':return False
    p=_patterns(symbol)
    if any(r.search(text) for r in p['exclude']):return False
    return any(r.search(text) for r in p['words']) or any(a.search(text) and b.search(text) for a,b in p['context'])


def markets():
    """Contract coins with an on-chain pool: symbol -> (network, address, archive start)."""
    return {s:(c['network'],c['address'],archive_start(s)) for s,c in COINS.items() if c['address']}
