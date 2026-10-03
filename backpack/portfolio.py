"""Stored portfolio reads for tracked BP holders (Python port of apps/web/lib/server/trader-watch.ts).

Raw integer quantities throughout. A failed program read makes a wallet partial or unavailable; it never
implies a token is absent. Missing and stale prices stay unpriced, never zero.
"""
from datetime import timedelta
from decimal import Decimal
import re
from .metrics import BP_MINT, USDC, number

TOKEN_PROGRAM = 'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA'
TOKEN_2022 = 'TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb'
NATIVE = 'native'
WSOL = 'So11111111111111111111111111111111111111112'
USDT = 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB'
STABLES = {USDC, USDT}
SOL_MINTS = {NATIVE, WSOL}
NFT_INTERFACES = {'V1_NFT', 'V2_NFT', 'LEGACY_NFT', 'ProgrammableNFT', 'MplCoreAsset', 'MplCoreCollection', 'V1_PRINT'}
FUNGIBLE_INTERFACES = {'FungibleToken', 'FungibleAsset'}
POSITION_PATTERN = re.compile(r'(\bLP\b|\bLP[ -]?token|liquidity|pool token|receipt token|\bvault share)', re.I)
SPAM_PATTERN = re.compile(r'(https?://|www\.|\.(com|io|xyz|net|org|app|fun)\b|t\.me/|\bclaim|\bairdrop|\bvisit\b|\bfree\b)', re.I)
PRICE_MAX_AGE = timedelta(hours=1)
# Slot-based price age. Slots are leader-schedule ticks of roughly 0.4s; bounds use 0.35-0.5s so a block is only
# called fresh or stale without an exact lookup when that holds at either extreme.
SLOT_SECONDS, FAST_SLOT, SLOW_SLOT = 0.4, 0.35, 0.5


def price_block_ages(blocks, ref_slot, ref_time, max_age=PRICE_MAX_AGE):
    """Split price blocks into slot-estimated times (clearly fresh or clearly stale) and blocks that need an exact
    getBlockTime because they sit near the one-hour boundary. Saves one credit per priced token on most reads."""
    estimated, exact_needed = {}, []
    limit = max_age.total_seconds()
    for block in sorted({int(b) for b in blocks if b is not None}):
        slots = max(ref_slot - block, 0)
        if slots * SLOW_SLOT <= limit or slots * FAST_SLOT > limit:
            estimated[block] = ref_time - timedelta(seconds=round(slots * SLOT_SECONDS))
        else:
            exact_needed.append(block)
    return estimated, exact_needed


def _camel_to_snake(name):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def _parsed_info(account):
    """The jsonParsed token-account info, or None when the provider returned the account unparsed. Helius returned
    raw ['<base64>', 'base64'] data for one cohort wallet's account on 2026-10-03 that the public RPC parses."""
    data = account.get('account', {}).get('data') if isinstance(account, dict) and isinstance(account.get('account'), dict) else None
    parsed = data.get('parsed') if isinstance(data, dict) else None
    info = parsed.get('info') if isinstance(parsed, dict) else None
    return info if isinstance(info, dict) else None


def parse_accounts(owner, accounts, program):
    """jsonParsed getTokenAccountsByOwner rows -> ({mint: holding}, anomalies, unread). Nonzero balances plus
    confidential accounts, whose encrypted balance cannot be read (visibility 'partial', never zero). `unread` counts
    accounts whose balance could not be read at all; the caller marks the read partial, never complete."""
    holdings, anomalies, unread = {}, [], 0
    for account in accounts:
        pubkey = account.get('pubkey') if isinstance(account, dict) else None
        info = _parsed_info(account)
        if info is None:
            unread += 1
            anomalies.append(f'{pubkey}: account returned unparsed; its balance is unknown')
            continue
        token_amount = info.get('tokenAmount') if isinstance(info.get('tokenAmount'), dict) else {}
        amount, decimals = token_amount.get('amount'), token_amount.get('decimals')
        mint = info.get('mint')
        if info.get('owner') != owner:
            anomalies.append(f"{pubkey}: parsed owner differs from requested owner")
            continue
        if not mint or not isinstance(amount, str) or not amount.isdigit() or type(decimals) is not int or not 0 <= decimals <= 18:
            unread += 1
            anomalies.append(f"{pubkey}: unparseable token amount; its balance is unknown")
            continue
        extensions = sorted(_camel_to_snake(str(e.get('extension'))) for e in info.get('extensions') or [] if isinstance(e, dict))
        confidential = 'confidential_transfer_account' in extensions
        raw = int(amount)
        if raw == 0 and not confidential: continue
        holding = holdings.setdefault(mint, dict(mint=mint, program=program, decimals=decimals, raw=0, ui=Decimal(0),
                                                 ui_complete=True, accounts=[], frozen=False, visibility='full'))
        if holding['decimals'] != decimals:
            anomalies.append(f'{mint}: token accounts disagree on decimals; later account ignored')
            continue
        holding['raw'] += raw
        ui = number(token_amount.get('uiAmountString'))
        if ui is None: holding['ui_complete'] = False
        else: holding['ui'] += ui
        holding['frozen'] = holding['frozen'] or info.get('state') == 'frozen'
        if confidential: holding['visibility'] = 'partial'
        holding['accounts'].append(dict(address=pubkey, amount=amount, state=info.get('state'),
                                        extensions=extensions))
    return holdings, anomalies, unread


def wallet_read(owner, sol, legacy, token2022, max_accounts=10000):
    """Combine one wallet's three reads. Each read is (result, error) from Providers.rpc_many."""
    statuses = {'sol_status': 'ok' if sol[1] is None and isinstance((sol[0] or {}).get('value'), int) else 'failed'}
    holdings, anomalies, slots, accounts = {}, [], [], 0
    for key, program, (result, error) in (('spl_status', TOKEN_PROGRAM, legacy), ('token2022_status', TOKEN_2022, token2022)):
        rows = (result or {}).get('value') if error is None else None
        if not isinstance(rows, list):
            statuses[key] = 'failed'
            continue
        accounts += len(rows)
        slots.append(((result or {}).get('context') or {}).get('slot'))
        parsed, notes, unread = parse_accounts(owner, rows, program)
        statuses[key] = 'partial' if unread else 'ok'  # an unreadable account is an unknown balance, never an absent one
        anomalies += notes
        for mint, holding in parsed.items():
            if mint in holdings:  # One mint belongs to one program; a repeat is a provider anomaly.
                anomalies.append(f'{mint}: returned under both token programs')
                continue
            holdings[mint] = holding
    if statuses['sol_status'] == 'ok':
        slots.append(((sol[0] or {}).get('context') or {}).get('slot'))
        lamports = sol[0]['value']
        if lamports > 0:
            holdings[NATIVE] = dict(mint=NATIVE, program=None, decimals=9, raw=lamports, ui=Decimal(lamports) / Decimal(10) ** 9,
                                    ui_complete=True, accounts=[dict(address=owner, amount=str(lamports), state='system')],
                                    frozen=False, visibility='full')
    token_reads = [statuses['spl_status'], statuses['token2022_status']]
    if accounts > max_accounts:
        status, holdings = 'oversized', {}
        anomalies.append(f'{accounts} token accounts exceed the {max_accounts} limit; balances not stored (exchange-like wallet)')
    elif token_reads == ['failed', 'failed']: status = 'unavailable'
    elif any(s != 'ok' for s in token_reads) or statuses['sol_status'] == 'failed': status = 'partial'
    else: status = 'complete'
    known = [s for s in slots if isinstance(s, int)]
    return dict(wallet_address=owner, status=status, holdings=holdings, token_accounts=accounts,
                slot_min=min(known) if known else None, slot_max=max(known) if known else None,
                detail='; '.join(anomalies[:20]), **statuses)


def classify_asset(mint, das=None, program=None, decimals=None):
    """Metadata + spam/position classification for bp_tracked_assets. Reasons are always recorded."""
    if mint == NATIVE:
        return dict(mint=mint, asset_class='native', token_program=None, decimals=9, symbol='SOL', name='Solana (native)',
                    metadata_source='protocol', extension_flags=[], extensions={}, is_sol=True, is_stable=False, is_bp=False,
                    spam_class='none', spam_reason=None, class_reason=None)
    das = das or {}
    info = das.get('token_info') or {}
    meta = ((das.get('content') or {}).get('metadata') or {})
    symbol = (info.get('symbol') or meta.get('symbol') or '').strip()[:32] or None
    name = (meta.get('name') or '').strip()[:120] or None
    extensions = das.get('mint_extensions') if isinstance(das.get('mint_extensions'), dict) else {}
    flags = sorted({_camel_to_snake(k) for k in extensions})
    interface = das.get('interface')
    decimals = info.get('decimals', decimals)
    reasons = []
    if interface in NFT_INTERFACES: asset_class = 'nft'
    elif das and interface not in FUNGIBLE_INTERFACES and decimals == 0: asset_class = 'nft'
    elif not das: asset_class = 'fungible' if decimals else 'unknown'
    else: asset_class = 'fungible'
    text = f'{name or ""} {symbol or ""}'
    if asset_class == 'fungible' and POSITION_PATTERN.search(text):
        asset_class = 'position'
        reasons.append('name/symbol matches a liquidity or receipt token pattern')
    spam = SPAM_PATTERN.search(text) if mint not in STABLES | SOL_MINTS | {BP_MINT} else None
    return dict(mint=mint, asset_class=asset_class, token_program=info.get('token_program') or program, decimals=decimals,
                symbol=symbol, name=name, metadata_source='Helius DAS getAssetBatch' if das else None,
                extension_flags=flags, extensions=extensions, is_sol=mint in SOL_MINTS, is_stable=mint in STABLES,
                is_bp=mint == BP_MINT, spam_class='suspected_spam' if spam else 'none',
                spam_reason=f'name/symbol advertises a link or claim ("{spam.group(0)}")' if spam else None,
                class_reason='; '.join(reasons) or None)


def price_for(mint, prices, observed_at):
    """(price, price_at, source, status). Native SOL uses the wrapped-SOL price."""
    row = prices.get(WSOL if mint == NATIVE else mint)
    if not row: return None, None, None, 'unpriced'
    price, price_at = row['price'], row.get('price_at')
    fresh = price_at is not None and observed_at - price_at <= PRICE_MAX_AGE
    return price, price_at, row['source'], 'priced' if fresh else 'stale'


def balance_rows(read, assets, prices, observed_at, dust_usd=Decimal(1)):
    """Rows for bp_portfolio_balances plus the wallet's priced in-scope total."""
    rows, priced_total, unpriced = [], Decimal(0), 0
    for mint, h in sorted(read['holdings'].items()):
        asset = assets.get(mint) or classify_asset(mint, None, h['program'], h['decimals'])
        price, price_at, source, status = price_for(mint, prices, observed_at)
        in_scope = asset['asset_class'] in ('native', 'fungible', 'position', 'unknown')
        value = (Decimal(h['raw']) / Decimal(10) ** h['decimals'] * price) if status == 'priced' and in_scope else None
        if value is None: status = status if in_scope else 'unpriced'
        reasons = [r for r in (asset.get('spam_reason'), asset.get('class_reason')) if r]
        if not in_scope: reasons.append('NFT: excluded from portfolio totals')
        if status == 'stale': reasons.append('price block older than one hour; value withheld')
        if h['visibility'] == 'partial': reasons.append('confidential balance unreadable; public balance only')
        dust = None if value is None else value < dust_usd
        if dust: reasons.append(f'priced below ${dust_usd}')
        if value is not None and in_scope: priced_total += value
        elif in_scope: unpriced += 1
        rows.append(dict(wallet_address=read['wallet_address'], mint=mint, raw_amount=h['raw'], decimals=h['decimals'],
                         ui_amount=str(h['ui']) if h['ui_complete'] else None, token_accounts=h['accounts'],
                         slot=read['slot_max'], observed_at=observed_at, price=price, price_at=price_at, price_source=source,
                         pricing_status=status, value_usd=value, frozen=h['frozen'], balance_visibility=h['visibility'],
                         holding_class=asset['asset_class'], spam_class=asset['spam_class'], dust=dust,
                         classification_reason='; '.join(reasons)))
    return rows, (priced_total if read['status'] in ('complete', 'partial') else None), unpriced
