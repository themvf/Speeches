"""Profile normalization and evidence matching; no network calls."""
import re
import unicodedata
from crypto_social_pilot import count, identity, ADDRESS

from crypto_coins import COINS as _REGISTRY
TERMS = {symbol: cfg['profileTerms'] for symbol, cfg in _REGISTRY.items()}


def matches(text):
    normalized = unicodedata.normalize('NFKC', text or '').casefold()
    result = []
    for coin, terms in TERMS.items():
        for term in terms:
            if re.search(r'(?<!\w)[#$]?' + re.escape(term).replace(r'\ ', r'\s+') + r'(?!\w)', normalized):
                result.append((coin, term))
    if ADDRESS in (text or ''):
        result.append(('ZCAT', ADDRESS))
    return result


def profile(raw):
    aid = identity(raw.get('id'))
    if not aid:
        raise ValueError('Missing stable user ID')
    available = not bool(raw.get('unavailable'))
    bio = (raw.get('profile_bio') or {}).get('description', raw.get('description'))
    return dict(id=aid, handle=str(raw.get('userName') or aid), name=str(raw.get('name') or ''),
                bio=str(bio) if bio is not None and available else None,
                followers=count(raw.get('followers')) if available else None,
                following=count(raw.get('following')) if available else None, available=available)


def save_profile(cur, raw, request_id, source):
    p = profile(raw)
    cur.execute('''INSERT INTO crypto_social_accounts(id,handle,name,followers)
        VALUES (%s,%s,%s,%s) ON CONFLICT(id) DO UPDATE SET
        handle=CASE WHEN EXCLUDED.handle=EXCLUDED.id THEN crypto_social_accounts.handle ELSE EXCLUDED.handle END,
        name=CASE WHEN EXCLUDED.name='' THEN crypto_social_accounts.name ELSE EXCLUDED.name END,
        followers=EXCLUDED.followers,observed_at=now()''', (p['id'],p['handle'],p['name'],p['followers']))
    cur.execute('''INSERT INTO crypto_social_profile_history
        (account_id,request_id,handle,name,bio,followers,following,available,source)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING''',
        (p['id'],request_id,p['handle'],p['name'],p['bio'],p['followers'],p['following'],p['available'],source))
    for field in ('bio', 'name', 'handle'):
        for coin, term in matches(p[field]):
            cur.execute('''INSERT INTO crypto_social_profile_matches VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT DO NOTHING''', (p['id'],request_id,coin,field,term))
            cur.execute('''INSERT INTO crypto_social_candidates(coin,account_id,reason)
                VALUES (%s,%s,%s) ON CONFLICT DO NOTHING''', (coin,p['id'],f'{field}: {term}'))
    return p
