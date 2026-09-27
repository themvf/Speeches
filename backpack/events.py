"""Economic-event classifier for tracked wallets (docs/bp-holder-intelligence-spec.md section 8.2).

Input is a raw RPC transaction (getTransaction / getTransactionsForAddress shape, json or jsonParsed).
Owner-level deltas come from pre/postTokenBalances and lamport balances; fees, rent and SOL wrapping are
separated before anything is called economic. The Helius Enhanced parse, when supplied, only decides the
parsed_swap tier. A positive balance change alone is never a purchase.
"""
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from .portfolio import NATIVE, SOL_MINTS, STABLES, WSOL

UTC = timezone.utc
PARSER_VERSION = 'bp-events-v1'
DEX_PROGRAMS = json.loads(Path(__file__).with_name('dex_programs.json').read_text())['programs']
ALERT_TIERS = ('parsed_swap', 'inferred_swap')
KIND_ORDER = {k: i for i, k in enumerate(('swap', 'liquidity', 'transfer_in', 'transfer_out', 'delegate_transfer',
                                          'wrap', 'unwrap', 'unclassified', 'failed'))}
SOL = 'SOL'  # Combined native + wrapped SOL exposure inside one transaction.


def account_keys(tx):
    """Full account list in balance-array order, plus the signer set."""
    message = (tx.get('transaction') or {}).get('message') or {}
    raw = message.get('accountKeys') or []
    parsed = bool(raw) and isinstance(raw[0], dict)
    keys = [k['pubkey'] if isinstance(k, dict) else k for k in raw]
    if parsed: signers = {k['pubkey'] for k in raw if k.get('signer')}
    else: signers = set(keys[:(message.get('header') or {}).get('numRequiredSignatures', 0)])
    meta = tx.get('meta') or {}
    loaded = meta.get('loadedAddresses') or {}
    if len(keys) < len(meta.get('preBalances') or []):
        keys += list(loaded.get('writable') or []) + list(loaded.get('readonly') or [])
    return keys, signers


def _instructions(tx):
    message = (tx.get('transaction') or {}).get('message') or {}
    for ix in message.get('instructions') or []: yield ix
    for group in (tx.get('meta') or {}).get('innerInstructions') or []:
        for ix in group.get('instructions') or []: yield ix


def programs_invoked(tx, keys):
    found = set()
    for ix in _instructions(tx):
        if ix.get('programId'): found.add(ix['programId'])
        elif isinstance(ix.get('programIdIndex'), int) and ix['programIdIndex'] < len(keys): found.add(keys[ix['programIdIndex']])
    return found


def ata_creations(tx):
    """(wallet, mint) pairs from parsed associated-token-account create instructions."""
    out = set()
    for ix in _instructions(tx):
        parsed = ix.get('parsed') if isinstance(ix.get('parsed'), dict) else {}
        if ix.get('program') == 'spl-associated-token-account' and parsed.get('type') in ('create', 'createIdempotent'):
            info = parsed.get('info') or {}
            out.add((info.get('wallet'), info.get('mint')))
    return out


def account_funding(tx):
    """Who paid for each account created in the transaction and where each closed token account's lamports
    went, from parsed system, associated-token-account and token instructions."""
    funded, refunded = {}, {}
    for ix in _instructions(tx):
        parsed = ix.get('parsed') if isinstance(ix.get('parsed'), dict) else {}
        info, kind, program = parsed.get('info') or {}, parsed.get('type'), ix.get('program')
        if program == 'system' and kind in ('createAccount', 'createAccountWithSeed') and info.get('newAccount'):
            funded[info['newAccount']] = info.get('source')
        elif program == 'spl-associated-token-account' and kind in ('create', 'createIdempotent') and info.get('source'):
            funded.setdefault(info.get('account'), info['source'])
        elif program in ('spl-token', 'spl-token-2022') and kind == 'closeAccount' and info.get('account'):
            refunded[info['account']] = info.get('destination')
    return funded, refunded


def token_rows(tx, keys):
    meta = tx.get('meta') or {}
    pre = {e['accountIndex']: e for e in meta.get('preTokenBalances') or []}
    post = {e['accountIndex']: e for e in meta.get('postTokenBalances') or []}
    pre_l, post_l = meta.get('preBalances') or [], meta.get('postBalances') or []
    rows = []
    for idx in sorted(set(pre) | set(post)):
        a, b = pre.get(idx), post.get(idx)
        ref = b or a
        amount = lambda e: int(e['uiTokenAmount']['amount']) if e else 0
        rows.append(dict(index=idx, address=keys[idx] if idx < len(keys) else None, mint=ref['mint'],
                         decimals=int(ref['uiTokenAmount']['decimals']), pre_owner=a.get('owner') if a else None,
                         post_owner=b.get('owner') if b else None, pre=amount(a), post=amount(b),
                         pre_lamports=pre_l[idx] if idx < len(pre_l) else None,
                         post_lamports=post_l[idx] if idx < len(post_l) else None, created=a is None))
    return rows


def owner_changes(owner, rows, keys, meta, funding=({}, {})):
    """Owner-level asset positions with fee and token-account rent removed from native SOL."""
    tokens, own_rent, unknown_rent = {}, 0, 0
    funded, refunded = funding
    for r in rows:
        mine_pre, mine_post = r['pre_owner'] == owner, r['post_owner'] == owner
        if not (mine_pre or mine_post): continue
        t = tokens.setdefault(r['mint'], dict(pre=0, post=0, decimals=r['decimals'], accounts=set(), created=False))
        t['pre'] += r['pre'] if mine_pre else 0
        t['post'] += r['post'] if mine_post else 0
        t['accounts'].add(r['address'])
        t['created'] = t['created'] or (r['created'] and mine_post)
        if r['pre_lamports'] is not None and r['post_lamports'] is not None:
            lamports = r['post_lamports'] - r['pre_lamports']
            wrapped = ((r['post'] if mine_post else 0) - (r['pre'] if mine_pre else 0)) if r['mint'] == WSOL else 0
            part = lamports - wrapped  # Lamports parked in (or released from) the owner's token accounts.
            if part:
                # Rent the wallet itself paid or received mirrors its own SOL change; rent a third party paid, or a
                # refund sent elsewhere, never touched the wallet. Unattributed rent falls back to a capped offset.
                counterparty = funded.get(r['address'], '?') if part > 0 else refunded.get(r['address'], '?')
                if counterparty == owner: own_rent += part
                elif counterparty == '?': unknown_rent += part
    native_pre = native_post = 0
    if owner in keys:
        i = keys.index(owner)
        native_pre, native_post = (meta.get('preBalances') or [0])[i], (meta.get('postBalances') or [0])[i]
    fee = int(meta.get('fee') or 0) if keys and keys[0] == owner else 0
    wallet = native_post - native_pre + fee + own_rent
    rent = unknown_rent
    # Without instruction evidence, rent only cancels the part of the wallet's own SOL change it can explain.
    offset = min(rent, -wallet) if rent > 0 and wallet < 0 else max(rent, -wallet) if rent < 0 and wallet > 0 else 0
    native = wallet + offset
    wsol = tokens.pop(WSOL, None)
    wsol_delta = (wsol['post'] - wsol['pre']) if wsol else 0
    sol = dict(native=native, wrapped=wsol_delta, delta=native + wsol_delta, decimals=9,
               pre=native_pre + (wsol['pre'] if wsol else 0), post=native_post + (wsol['post'] if wsol else 0),
               mint=NATIVE if abs(native) >= abs(wsol_delta) else WSOL, accounts=wsol['accounts'] if wsol else set(), created=False)
    return tokens, sol


def _norm(mint):
    return SOL if mint in SOL_MINTS or mint == SOL else mint


def swap_tier(owner, disposed, acquired, enhanced, programs):
    if enhanced and enhanced.get('type') == 'SWAP' and not enhanced.get('transactionError'):
        swap = (enhanced.get('events') or {}).get('swap') or {}
        ins = {_norm(x.get('mint')) for x in swap.get('tokenInputs') or [] if x.get('userAccount') == owner}
        outs = {_norm(x.get('mint')) for x in swap.get('tokenOutputs') or [] if x.get('userAccount') == owner}
        if (swap.get('nativeInput') or {}).get('account') == owner: ins.add(SOL)
        if (swap.get('nativeOutput') or {}).get('account') == owner: outs.add(SOL)
        if _norm(disposed) in ins and _norm(acquired) in outs:
            return 'parsed_swap', enhanced.get('source') or 'Helius Enhanced', None
    known = sorted(DEX_PROGRAMS[p] for p in programs if p in DEX_PROGRAMS)
    if known: return 'inferred_swap', ', '.join(known), None
    return 'unclassified', None, 'one asset out and one in, but no Helius swap parse and no known DEX program'


def valuation(disposed, acquired, block_time, sol_price):
    """Execution-time USD from the trade's own counter-asset: stablecoin at $1, else SOL at the nearest stored price."""
    for side in (disposed, acquired):
        if side['mint'] in STABLES:
            return Decimal(side['raw']) / Decimal(10) ** side['decimals'], 'stablecoin counter-asset at $1 (estimated)'
    for side in (disposed, acquired):
        if side['mint'] in SOL_MINTS and sol_price and block_time:
            found = sol_price(block_time)
            if found:
                price, at = found
                return Decimal(side['raw']) / Decimal(10) ** 9 * price, f'SOL counter-asset at Jupiter SOL price observed {at.isoformat()}'
    return None, None


def classify(tx, owners, assets=None, enhanced=None, snapshot=None, sol_price=None,
             incidental_lamports=3_000_000, finality='finalized'):
    """Economic events for every tracked owner touched by one transaction, in deterministic order."""
    assets = assets or {}
    meta = tx.get('meta') or {}
    keys, signers = account_keys(tx)
    signature = ((tx.get('transaction') or {}).get('signatures') or [None])[0]
    if not signature or tx.get('slot') is None: raise ValueError('Transaction lacks signature or slot')
    block_time = datetime.fromtimestamp(tx['blockTime'], UTC) if tx.get('blockTime') is not None else None
    rows = token_rows(tx, keys)
    programs = programs_invoked(tx, keys)
    creations = ata_creations(tx)
    funding = account_funding(tx)
    base = dict(signature=signature, slot=tx['slot'], block_time=block_time, programs=sorted(programs),
                parser_version=PARSER_VERSION, finality=finality)
    touched = {o for o in owners if o in signers or o in keys or any(o in (r['pre_owner'], r['post_owner']) for r in rows)}
    events = []
    for owner in sorted(touched):
        signed = owner in signers
        blank = dict(base, wallet_address=owner, owner_signed=signed, tier='not_applicable', venue=None, detail='',
                     input_mint=None, input_raw=None, input_decimals=None, output_mint=None, output_raw=None,
                     output_decimals=None, usd_value=None, valuation_source=None, valuation_status='not_applicable',
                     pre_input_raw=None, post_input_raw=None, pre_output_raw=None, post_output_raw=None,
                     pre_balance_source=None, new_position=None, ata_created=None)
        owned = []
        if meta.get('err') is not None:
            if signed: owned.append(dict(blank, kind='failed', detail='execution failed; no economic effect beyond fees'))
            events += owned
            continue
        tokens, sol = owner_changes(owner, rows, keys, meta, funding)
        positions = {m: t for m, t in tokens.items() if t['post'] != t['pre']}
        if abs(sol['delta']) > incidental_lamports: positions[SOL] = sol
        elif sol['native'] and sol['wrapped'] and (sol['native'] > 0) != (sol['wrapped'] > 0) and not positions:
            kind = 'wrap' if sol['wrapped'] > 0 else 'unwrap'
            owned.append(dict(blank, kind=kind, input_mint=NATIVE if kind == 'wrap' else WSOL, input_raw=abs(sol['wrapped']),
                              input_decimals=9, output_mint=WSOL if kind == 'wrap' else NATIVE, output_raw=abs(sol['wrapped']),
                              output_decimals=9, detail='SOL wrapping changes form, not exposure; never a purchase'))
        incidental = (f"incidental SOL change of {sol['delta']} lamports after fee/rent ignored (limit {incidental_lamports})"
                      if sol['delta'] and SOL not in positions and not owned else '')

        def side(mint):
            t = positions[mint]
            resolved = t['mint'] if mint == SOL else mint
            return dict(mint=resolved, raw=abs(t['post'] - t['pre']) if mint != SOL else abs(t['delta']),
                        decimals=t['decimals'], pre=t['pre'], post=t['post'], accounts=t['accounts'], created=t['created'])

        def balances(event, disposed=None, acquired=None):
            source = 'transaction'
            for prefix, s in (('input', disposed), ('output', acquired)):
                if not s: continue
                pre = s['pre']
                if snapshot and s['mint'] not in SOL_MINTS and block_time:
                    elsewhere = [a for a in (snapshot(owner, s['mint'], block_time) or []) if a.get('address') not in s['accounts']]
                    if elsewhere:
                        pre += sum(int(a['amount']) for a in elsewhere)
                        source = 'snapshot'
                event.update({f'{prefix}_mint': s['mint'], f'{prefix}_raw': s['raw'], f'{prefix}_decimals': s['decimals'],
                              f'pre_{prefix}_raw': pre, f'post_{prefix}_raw': s['post'] + (pre - s['pre'])})
            event['pre_balance_source'] = source
            if acquired:
                event['new_position'] = event['pre_output_raw'] == 0 and event['post_output_raw'] > 0
                event['ata_created'] = (owner, acquired['mint']) in creations or acquired['created']
            return event

        disposed = sorted(m for m, t in positions.items() if (t['delta'] if m == SOL else t['post'] - t['pre']) < 0)
        acquired = sorted(m for m, t in positions.items() if (t['delta'] if m == SOL else t['post'] - t['pre']) > 0)
        position_token = lambda m: (assets.get(m) or {}).get('asset_class') == 'position'
        if len(disposed) == 1 and len(acquired) == 1:
            d, a = side(disposed[0]), side(acquired[0])
            if position_token(d['mint']) or position_token(a['mint']):
                owned.append(balances(dict(blank, kind='liquidity', detail='liquidity position token on one side'), d, a))
            else:
                tier, venue, note = swap_tier(owner, d['mint'], a['mint'], enhanced, programs)
                usd, source = valuation(d, a, block_time, sol_price)
                owned.append(balances(dict(blank, kind='swap' if tier in ALERT_TIERS else 'unclassified', tier=tier, venue=venue,
                    usd_value=usd, valuation_source=source, valuation_status='estimated' if usd is not None else 'unpriced',
                    detail=note or ('inferred from owner deltas and a known DEX program; not a parsed swap' if tier == 'inferred_swap'
                                    else 'Helius parsed swap; route legs are not positions')), d, a))
        elif disposed and acquired:
            liquidity = any(position_token(side(m)['mint']) for m in disposed + acquired) or (
                min(len(disposed), len(acquired)) == 1 and bool(programs & DEX_PROGRAMS.keys()))
            kind, tier = ('liquidity', 'not_applicable') if liquidity else ('unclassified', 'unclassified')
            note = ('multi-asset liquidity operation' if liquidity else
                    'several assets moved in each direction; independent swaps cannot be paired from balances alone')
            for m in disposed: owned.append(balances(dict(blank, kind=kind, tier=tier, detail=note), disposed=side(m)))
            for m in acquired: owned.append(balances(dict(blank, kind=kind, tier=tier, detail=note), acquired=side(m)))
        else:
            for m in acquired:
                owned.append(balances(dict(blank, kind='transfer_in',
                    detail='received without signing (possible airdrop or unsolicited transfer)' if not signed
                    else 'received; a balance increase alone is never a purchase'), acquired=side(m)))
            for m in disposed:
                mint = side(m)['mint']
                delegated = 'permanent_delegate' in ((assets.get(mint) or {}).get('extension_flags') or []) and not signed
                owned.append(balances(dict(blank, kind='delegate_transfer' if delegated else 'transfer_out',
                    detail='moved by the permanent delegate without the owner signing; never a sale' if delegated
                    else 'sent; a transfer is not a sale'), disposed=side(m)))
        events += [dict(e, detail='; '.join(x for x in (e['detail'], incidental) if x)) for e in owned]
    ordered = []
    for owner in sorted({e['wallet_address'] for e in events}):
        mine = sorted((e for e in events if e['wallet_address'] == owner),
                      key=lambda e: (KIND_ORDER[e['kind']], e['input_mint'] or '', e['output_mint'] or ''))
        ordered += [dict(e, event_index=i) for i, e in enumerate(mine)]
    return ordered


def swap_candidates(tx, owners, incidental_lamports=3_000_000):
    """Signatures worth an Enhanced parse: some tracked owner has an asset out and a different asset in."""
    meta = tx.get('meta') or {}
    if meta.get('err') is not None: return False
    keys, _ = account_keys(tx)
    rows = token_rows(tx, keys)
    funding = account_funding(tx)
    for owner in owners:
        tokens, sol = owner_changes(owner, rows, keys, meta, funding)
        signs = {(t['post'] - t['pre']) > 0 for t in tokens.values() if t['post'] != t['pre']}
        if abs(sol['delta']) > incidental_lamports: signs.add(sol['delta'] > 0)
        if signs == {True, False}: return True
    return False
