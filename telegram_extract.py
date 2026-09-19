"""Deterministic extraction from a Telegram message: Solana mints, cashtags, URLs, claim language.

Pure functions, no network and no database, so every rule here is testable on a string. Nothing in
this module resolves a token - resolution needs the archive and lives in telegram_mentions.py. The
split matters because extraction runs once at collection time while resolution is re-runnable, and
a rule that silently changes what counts as a mention must be re-runnable.

Spec: docs/telegram-osint-spec.md.
"""
import re

# base58: Bitcoin's alphabet, no 0 O I l. A Solana mint is 32 bytes, which is 43-44 characters;
# shorter strings decode to fewer bytes and are something else (a tx signature is 64 bytes / ~88
# chars, so the upper bound excludes those too).
B58='123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
B58_INDEX={c:i for i,c in enumerate(B58)}
ADDRESS_RE=re.compile(r'(?<![1-9A-HJ-NP-Za-km-z])([1-9A-HJ-NP-Za-km-z]{32,44})(?![1-9A-HJ-NP-Za-km-z])')
CASHTAG_RE=re.compile(r'(?<![0-9A-Za-z$])\$([A-Za-z][A-Za-z0-9_]{1,14})\b')
URL_RE=re.compile(r'https?://[^\s<>"\')]+',re.I)
# "10x", "x10", "10.5X". Deliberately NOT used as evidence of a result: see claims() below.
MULTIPLE_RE=re.compile(r'(?<![A-Za-z0-9.])(?:(\d+(?:\.\d+)?)\s*[xX]|[xX]\s*(\d+(?:\.\d+)?))(?![A-Za-z0-9.])')
MONEY_RE=re.compile(r'\$\s*(\d+(?:[.,]\d+)?)\s*([KkMmBb])?\b')
# Market-cap language, so a bare "$100 entry" is not read as a market cap.
MCAP_RE=re.compile(r'(?:\bmc\b|\bmcap\b|\bmarket\s*cap\b)[^0-9$]{0,12}\$?\s*(\d+(?:[.,]\d+)?)\s*([KkMmBb])?'
                   r'|\$\s*(\d+(?:[.,]\d+)?)\s*([KkMmBb])?\s*(?:\bmc\b|\bmcap\b|\bmarket\s*cap\b)',re.I)
SUFFIX={'k':1e3,'m':1e6,'b':1e9}
# Tokens whose cashtag is a currency or a major asset rather than a launchpad token. A cashtag is
# only ever a low-confidence reference; this list keeps the obvious noise out of the candidate set
# before resolution has to reason about it.
CASHTAG_STOPLIST={'USD','USDT','USDC','SOL','BTC','ETH','BNB','EUR','GBP','JPY','SPX','QQQ'}


def b58_decode_len(value):
    """Byte length of a base58 string, or None if it is not base58. No dependency: the alternative
    is trusting a regex, and a 43-character word in base58's alphabet is not necessarily an address."""
    number=0
    for char in value:
        index=B58_INDEX.get(char)
        if index is None:return None
        number=number*58+index
    body=number.to_bytes((number.bit_length()+7)//8,'big') if number else b''
    leading=len(value)-len(value.lstrip('1'))
    return leading+len(body)


def is_solana_address(value):
    """32-byte base58. This is the only test applied: a mint is not required to end in 'pump',
    because non-Pump.fun launchpads are in the archive too and filtering on the suffix would
    silently make this a Pump.fun-only collector."""
    return bool(value) and 32<=len(value)<=44 and b58_decode_len(value)==32


def solana_addresses(text):
    """Every distinct 32-byte base58 string in the message, in order of first appearance.

    Addresses inside URLs are included on purpose: a Dexscreener or Birdeye link is one of the most
    common ways a channel posts a contract, and dropping them would lose exactly the messages that
    are most clearly about one token.
    """
    out=[]
    for match in ADDRESS_RE.finditer(text or ''):
        candidate=match.group(1)
        if is_solana_address(candidate) and candidate not in out:out.append(candidate)
    return out


def cashtags(text):
    """$TICKER references, upper-cased and de-duplicated, minus currencies and majors."""
    out=[]
    for match in CASHTAG_RE.finditer(text or ''):
        symbol=match.group(1).upper()
        if symbol in CASHTAG_STOPLIST or symbol in out:continue
        out.append(symbol)
    return out


def urls(text):
    return list(dict.fromkeys(match.group(0).rstrip('.,;') for match in URL_RE.finditer(text or '')))


def _money(value,suffix):
    try:amount=float(str(value).replace(',',''))
    except (TypeError,ValueError):return None
    return amount*SUFFIX.get((suffix or '').lower(),1)


def claims(text):
    """What the message claimed about itself: {claimed_multiple, claimed_market_cap}.

    Stored so a channel's claims can be compared against measured outcomes. It is never an input to
    a performance number - "did it 10x" is answered from price history, not from the post that says
    it did. The largest multiple in a message wins, because a call post that mentions several is
    usually advertising its best one and that is the claim worth checking.
    """
    multiples=[]
    for match in MULTIPLE_RE.finditer(text or ''):
        raw=match.group(1) or match.group(2)
        try:multiples.append(float(raw))
        except (TypeError,ValueError):continue
    caps=[]
    for match in MCAP_RE.finditer(text or ''):
        value,suffix=(match.group(1),match.group(2)) if match.group(1) else (match.group(3),match.group(4))
        amount=_money(value,suffix)
        if amount:caps.append(amount)
    return dict(claimed_multiple=max(multiples) if multiples else None,
                claimed_market_cap=caps[0] if caps else None)


def references(text):
    """Every token reference in a message, as rows ready for telegram_token_mentions.

    A contract is a fact about the message; a cashtag is a candidate that resolution may or may not
    be able to turn into a token. Both are recorded, so "this channel posts tickers without
    contracts" stays visible as a property of the channel rather than disappearing as missing data.
    """
    rows=[dict(raw_reference=address,reference_kind='contract',confidence=1.0)
          for address in solana_addresses(text)]
    for symbol in cashtags(text):
        rows.append(dict(raw_reference=symbol,reference_kind='ticker',confidence=None))
    return rows
