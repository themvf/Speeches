"""Pure, Decimal-based methodology. Missing data never becomes zero."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from statistics import median
from zoneinfo import ZoneInfo

BP_MINT = 'BPxxfRCXkUVhig4HS1Lh7kZqV6SPJhzfEk4x6fVBjPCy'
USDC = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'
SYSTEM_LABELS = {'Backpack', 'Treasury', 'Custody', 'DEX', 'Liquidity Pool', 'Lending Protocol', 'Bridge', 'Known Exchange', 'Protocol'}
THRESHOLDS = (100, 1000, 10000, 100000)


def number(value):
    if value is None or isinstance(value, bool): return None
    try:
        n = Decimal(str(value))
        return n if n.is_finite() else None
    except (InvalidOperation, ValueError): return None


def multiply(a, b):
    a, b = number(a), number(b)
    return a * b if a is not None and b is not None else None


def ratio(a, b, percent=False):
    a, b = number(a), number(b)
    return a / b * (100 if percent else 1) if a is not None and b is not None and b > 0 else None


def excluded(label):
    return bool(label and label.get('confidence') in {'confirmed', 'high'} and label.get('label') in SYSTEM_LABELS)


def holders(accounts, decimals, price, labels=None):
    """Raw = nonzero token accounts; economic cohorts = owner-aggregated balances."""
    labels = labels or {}
    balances = defaultdict(Decimal)
    raw = 0
    seen = set()
    for account in accounts:
        address = account['address']
        if address in seen: continue
        seen.add(address)
        amount = number(account['amount'])
        if amount is None or amount < 0 or not account.get('owner'): raise ValueError('Invalid token account')
        if amount > 0:
            raw += 1
            balances[account['owner']] += amount / Decimal(10) ** decimals
    rows = [{'wallet_address': w, 'balance_tokens': b, 'value_usd': multiply(b, price),
             'excluded': excluded(labels.get(w)), 'label': labels.get(w, {}).get('label', 'Unknown')}
            for w, b in balances.items()]
    economic = [r for r in rows if not r['excluded']]
    result = {'holder_count': raw, 'unique_holders': len(rows), 'rows': rows}
    for threshold in THRESHOLDS:
        result[f'holders_over_{threshold}'] = None if number(price) is None else sum(r['value_usd'] >= threshold for r in economic)
    for prefix, cohort in [('', rows), ('economic_', economic)]:
        ordered = sorted((r['balance_tokens'] for r in cohort), reverse=True)
        for n in (10, 20, 50, 100):
            result[f'{prefix}top_{n}_holder_pct'] = ratio(sum(ordered[:n]), sum(ordered), True)
    return result


def issuance(supply, previous, day, price):
    if not previous or previous['date'] != day - timedelta(days=1): return None, None
    before = number(previous.get('token_supply'))
    current = number(supply)
    delta = current - before if current is not None and before is not None else None
    return delta, multiply(delta, price)


def ecosystem(wallet_rows):
    """Each included asset must be >=$100; dust in a second asset adds no breadth."""
    values, assets = defaultdict(Decimal), defaultdict(set)
    for r in wallet_rows:
        if r.get('excluded') or r.get('value_usd') is None: continue
        value = number(r['value_usd'])
        values[r['wallet_address']] += value
        if value >= 100: assets[r['wallet_address']].add(r['asset_id'])
    meaningful = sum(v >= 100 for v in values.values())
    multi = {n: sum(len(a) >= n for a in assets.values()) for n in (1, 2, 3, 5)}
    return {'meaningful_holders': meaningful, 'multi_asset_1': multi[1], 'multi_asset_2': multi[2],
            'multi_asset_3': multi[3], 'multi_asset_5': multi[5], 'multi_asset_adoption_pct': ratio(multi[2], meaningful, True)}


def session(timestamp, calendar):
    """Calendar maps local date to official UTC open/close; holidays and early closes explicit."""
    t = datetime.fromtimestamp(timestamp, timezone.utc) if isinstance(timestamp, (int, float)) else timestamp
    local = t.astimezone(ZoneInfo('America/New_York'))
    day = local.date().isoformat()
    if day not in calendar: return 'unknown'  # No calendar coverage is not a holiday.
    hours = calendar[day]
    if hours is None: return 'weekend' if local.weekday() >= 5 else 'closed'
    opening, closing = hours
    if opening <= t < closing: return 'regular'
    if local.hour >= 4 and t < opening: return 'premarket'
    return 'after_hours'


def parity(token, equity, token_at, equity_at, market_open, max_age_seconds=300):
    discount = ratio(token, equity)
    pct = (discount - 1) * 100 if discount is not None else None
    contemporaneous = bool(market_open and token_at and equity_at and abs((token_at-equity_at).total_seconds()) <= max_age_seconds)
    return {'premium_discount_pct': pct, 'parity_alert_eligible': contemporaneous,
            'reference_status': 'contemporaneous' if contemporaneous else 'last_available'}


def normalize_swap(tx, mint):
    """One outer Helius swap per signature/asset. Never sum inner route legs or transfers.

    Only USDC-paired outer swaps have a USD proxy here. Other swaps retain a null USD
    amount rather than valuing historical trades at today's price. USDC at $1 is Estimated.
    """
    event = (tx.get('events') or {}).get('swap')
    if tx.get('transactionError') or tx.get('type') != 'SWAP' or not event: return None
    inputs, outputs = event.get('tokenInputs') or [], event.get('tokenOutputs') or []
    side = 'sell' if any(x.get('mint') == mint for x in inputs) else 'buy' if any(x.get('mint') == mint for x in outputs) else None
    if not side or not tx.get('signature') or tx.get('slot') is None or tx.get('timestamp') is None: return None
    security = [x for x in (inputs if side == 'sell' else outputs) if x.get('mint') == mint]
    counterpart = outputs if side == 'sell' else inputs
    stable = [x for x in counterpart if x.get('mint') == USDC]
    def amount(items):
        total = Decimal(0)
        for x in items:
            raw = x.get('rawTokenAmount') or {}
            if raw.get('tokenAmount') is None or raw.get('decimals') is None: return None
            n = number(raw['tokenAmount'])
            if n is None: return None
            total += n / Decimal(10) ** int(raw['decimals'])
        return total
    # userAccount belongs to swap legs; feePayer can be a relayer and is not a trader.
    owners = {x.get('userAccount') for x in security if x.get('userAccount')}
    return {'signature': tx['signature'], 'slot': tx['slot'], 'timestamp': tx['timestamp'],
            'wallet_address': next(iter(owners)) if len(owners) == 1 else None,
            'side': side, 'tokens': amount(security),
            'volume_usd': amount(stable) if stable and len(stable) == len(counterpart) else None,
            'venue': tx.get('source', 'Unknown'), 'source': 'Helius Enhanced Transactions'}


def trading(swaps, calendar, complete=False):
    unique = {s['signature']: s for s in swaps}
    rows = list(unique.values())
    amounts = sorted(s['volume_usd'] for s in rows if s['volume_usd'] is not None)
    fully_priced = len(amounts) == len(rows)
    wallets = {s['wallet_address'] for s in rows if s['wallet_address']}
    observed = sum(amounts) if amounts or complete else None
    sessions = defaultdict(Decimal)
    for s in rows:
        if s['volume_usd'] is not None: sessions[session(s['timestamp'], calendar)] += s['volume_usd']
    closed = sum(sessions[k] for k in ('premarket', 'after_hours', 'weekend', 'closed'))
    total = observed if complete and fully_priced else None
    return {'observed_swap_volume_usd': observed, 'daily_swap_volume_usd': total,
            'trades': len(rows), 'unique_traders': len(wallets) if complete else None,
            'observed_unique_traders': len(wallets), 'median_trade_size': median(amounts) if amounts else None,
            'average_trade_size': sum(amounts)/len(amounts) if amounts else None,
            'max_trade_size': max(amounts) if amounts else None,
            'p95_trade_size': amounts[max(0, (95*len(amounts)+99)//100-1)] if amounts else None,
            'after_hours_volume_pct': ratio(closed, total, True) if not sessions['unknown'] else None,
            'regular_session_volume_usd': sessions['regular'] if total is not None else None,
            'after_hours_volume_usd': closed if total is not None and not sessions['unknown'] else None}


def whale_cohorts(current, previous=None, thresholds=(100000, 500000, 1000000), labels=None):
    """Yesterday's whale cohort measures token accumulation without price-driven entries.

    None means incomplete/unpriced enumeration. An empty list is a complete empty
    population. Relabeled wallets are excluded on either date from comparisons.
    """
    thresholds = tuple(number(t) for t in thresholds)
    if any(t is None or t <= 0 for t in thresholds):
        raise ValueError('Whale thresholds must be positive finite USD values')
    result = []
    for threshold in sorted(set(thresholds)):
        row = dict(threshold_usd=threshold, whale_count=None, new_whales=None,
                   exited_whales=None, whale_net_accumulation_tokens=None)
        if current is None or any(number(r.get('value_usd')) is None for r in current):
            result.append(row)
            continue
        today = {r['wallet_address']: r for r in current if not r['excluded']}
        whales = {w for w,r in today.items() if number(r['value_usd']) >= threshold}
        row['whale_count'] = len(whales)
        if previous is not None and all(number(r.get('value_usd')) is not None for r in previous):
            blocked = {r['wallet_address'] for r in current + previous if r['excluded']}
            blocked.update(w for w,label in (labels or {}).items() if excluded(label))
            before = {r['wallet_address']: r for r in previous if r['wallet_address'] not in blocked}
            prior = {w for w,r in before.items() if number(r['value_usd']) >= threshold}
            comparable = whales - blocked
            row.update(new_whales=len(comparable-prior), exited_whales=len(prior-comparable),
                       whale_net_accumulation_tokens=sum((number(today[w]['balance_tokens']) if w in today else Decimal(0))
                           - number(before[w]['balance_tokens']) for w in prior))
        result.append(row)
    return result
