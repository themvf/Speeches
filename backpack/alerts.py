"""Alert rules over classified cohort activity (docs/bp-holder-intelligence-spec.md section 10).

Every classified purchase is recorded; thresholds apply only here. Only finalized, alert-eligible swaps by
members of the evaluated cohort version, after they joined tracking, can contribute. USD thresholds need an
execution-time valuation; unpriced events never qualify for a USD rule.
"""
from datetime import timedelta
from decimal import Decimal
import json
from .events import ALERT_TIERS
from .metrics import number


def eligible(e, members):
    return (e['kind'] == 'swap' and e['tier'] in ALERT_TIERS and not e['pre_membership']
            and e['finality'] == 'finalized' and e['wallet_address'] in members and e['block_time'] is not None)


def _window(items, start, width):
    return [e for e in items if start <= e['block_time'] < start + width]


def _alert(rule, version, mint, start, width, contributing, params, detail, data_through, decimals_side):
    wallets = {}
    for e in contributing:
        w = wallets.setdefault(e['wallet_address'], dict(address=e['wallet_address'], raw=0, usd=Decimal(0), priced=True,
                                                          tiers=set(), signatures=[]))
        w['raw'] += int(e[f'{decimals_side}_raw'])
        if e['usd_value'] is None: w['priced'] = False
        else: w['usd'] += number(e['usd_value'])
        w['tiers'].add(e['tier'])
        w['signatures'].append(e['signature'])
    priced = [w['priced'] for w in wallets.values()]
    status = 'priced' if all(priced) else 'partially_priced' if any(priced) else 'unpriced'
    end = max(e['block_time'] for e in contributing)
    return dict(
        alert_key=f"{rule}|{mint}|{start.isoformat()}|{version}", rule=rule, cohort_version_id=version, mint=mint,
        window_start=start, window_end=max(end, start + width) if rule in ('multiple_buyers', 'accumulation') else end,
        wallet_count=len(wallets), inferred_wallets=sum('inferred_swap' in w['tiers'] for w in wallets.values()),
        lowest_tier='inferred_swap' if any('inferred_swap' in w['tiers'] for w in wallets.values()) else 'parsed_swap',
        quantity_raw=sum(w['raw'] for w in wallets.values()), decimals=contributing[0][f'{decimals_side}_decimals'],
        usd_value=sum((w['usd'] for w in wallets.values() if w['priced']), Decimal(0)) if status != 'unpriced' else None,
        valuation_status=status, signatures=sorted({e['signature'] for e in contributing}),
        wallets=[dict(address=w['address'], raw=str(w['raw']), usd=str(w['usd']) if w['priced'] else None,
                      tier='inferred_swap' if 'inferred_swap' in w['tiers'] else 'parsed_swap', signatures=sorted(set(w['signatures'])))
                 for w in sorted(wallets.values(), key=lambda w: w['address'])],
        finality='finalized', data_through=data_through, params=params, detail=detail)


def evaluate(events, rules, version, members, data_through=None):
    """Pure evaluation. events: stored economic events (dicts); rules: {rule: {'enabled', 'params'}}."""
    pool = sorted((e for e in events if eligible(e, members)), key=lambda e: (e['block_time'], e['signature'], e['event_index']))
    out = []
    for rule, config in sorted(rules.items()):
        if not config.get('enabled', True): continue
        p = config['params']
        width = timedelta(minutes=int(p.get('window_minutes', 60)))
        exclude = set(p.get('exclude_mints') or [])
        if rule == 'new_position':
            minimum = Decimal(str(p['min_usd']))
            items = [e for e in pool if e['new_position'] and e['usd_value'] is not None and number(e['usd_value']) >= minimum]
            for mint in sorted({e['output_mint'] for e in items}):
                mine, cursor = [e for e in items if e['output_mint'] == mint], None
                for e in mine:
                    if cursor and e['block_time'] < cursor + width: continue
                    cursor = e['block_time']
                    group = _window(mine, cursor, width)
                    out.append(_alert(rule, version, mint, cursor, width, group, p,
                        f'{len({g["wallet_address"] for g in group})} wallet(s) opened a position of at least ${minimum} in a previously zero balance',
                        data_through, 'output'))
        elif rule in ('multiple_buyers', 'accumulation'):
            per_wallet, needed = Decimal(str(p['min_usd_per_wallet'])), int(p['min_wallets'])
            buys = [e for e in pool if e['output_mint'] not in exclude]
            for mint in sorted({e['output_mint'] for e in buys}):
                mine, blocked_until = [e for e in buys if e['output_mint'] == mint], None
                for anchor in mine:
                    if blocked_until and anchor['block_time'] < blocked_until: continue
                    group = _window(mine, anchor['block_time'], width)
                    spend = {}
                    for e in group:
                        if e['usd_value'] is not None:
                            spend[e['wallet_address']] = spend.get(e['wallet_address'], Decimal(0)) + number(e['usd_value'])
                    qualifying = {w for w, usd in spend.items() if usd >= per_wallet}
                    if len(qualifying) < needed: continue
                    detail = f'{len(qualifying)} tracked wallets each bought at least ${per_wallet} within {width}'
                    if rule == 'accumulation':
                        sold = sum(int(e['input_raw']) for e in _window([s for s in pool if s['input_mint'] == mint], anchor['block_time'], width))
                        bought = sum(int(e['output_raw']) for e in group)
                        if bought - sold <= 0: continue
                        detail += f'; net purchase flow {bought - sold} raw units (bought {bought}, sold {sold})'
                    contributing = [e for e in group if e['wallet_address'] in qualifying]
                    out.append(_alert(rule, version, mint, anchor['block_time'], width, contributing, p, detail, data_through, 'output'))
                    blocked_until = anchor['block_time'] + width
        elif rule == 'major_sale':
            fraction = Decimal(str(p['min_fraction']))
            sales = [e for e in pool if e['input_mint'] not in exclude and number(e['pre_input_raw'] or 0) > 0
                     and Decimal(int(e['input_raw'])) >= fraction * Decimal(int(e['pre_input_raw']))]
            for mint in sorted({e['input_mint'] for e in sales}):
                mine, cursor = [e for e in sales if e['input_mint'] == mint], None
                for e in mine:
                    if cursor and e['block_time'] < cursor + width: continue
                    cursor = e['block_time']
                    group = _window(mine, cursor, width)
                    out.append(_alert(rule, version, mint, cursor, width, group, p,
                        f'{len({g["wallet_address"] for g in group})} wallet(s) sold at least {fraction * 100:g}% of the pre-sale holding (from preTokenBalances)',
                        data_through, 'input'))
    return out


def store(cur, alerts, cooldowns, names=None):
    """Upsert by stable key; an alert overlapping an existing one for the same rule/mint/version (within its
    cooldown) updates that alert instead of emitting a new one."""
    names = names or {}
    written = 0
    for a in alerts:
        cooldown = timedelta(minutes=int(cooldowns.get(a['rule'], 0)))
        cur.execute('''SELECT alert_key,wallets,signatures,window_start,window_end FROM bp_alerts
            WHERE rule=%s AND mint=%s AND cohort_version_id=%s AND window_start<=%s AND window_end>=%s
            ORDER BY window_start LIMIT 1''', (a['rule'], a['mint'], a['cohort_version_id'], a['window_end'] + cooldown, a['window_start'] - cooldown))
        found = cur.fetchone()
        key, wallets, signatures = a['alert_key'], a['wallets'], a['signatures']
        start, end = a['window_start'], a['window_end']
        if found:
            key, old_wallets, old_signatures, old_start, old_end = found
            merged = {w['address']: w for w in old_wallets}
            merged.update({w['address']: w for w in wallets})
            wallets = [merged[k] for k in sorted(merged)]
            signatures = sorted(set(old_signatures) | set(signatures))
            start, end = min(start, old_start), max(end, old_end)
        priced = [w['usd'] is not None for w in wallets]
        status = 'priced' if all(priced) else 'partially_priced' if any(priced) else 'unpriced'
        usd = sum((Decimal(w['usd']) for w in wallets if w['usd'] is not None), Decimal(0)) if status != 'unpriced' else None
        cur.execute('''INSERT INTO bp_alerts(alert_key,rule,cohort_version_id,mint,display_name,window_start,window_end,wallet_count,
            wallets,inferred_wallets,lowest_tier,quantity_raw,decimals,usd_value,valuation_status,signatures,finality,data_through,
            params,detail) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT(alert_key) DO UPDATE SET window_start=excluded.window_start,window_end=excluded.window_end,
            wallet_count=excluded.wallet_count,wallets=excluded.wallets,inferred_wallets=excluded.inferred_wallets,
            lowest_tier=excluded.lowest_tier,quantity_raw=excluded.quantity_raw,usd_value=excluded.usd_value,
            valuation_status=excluded.valuation_status,signatures=excluded.signatures,data_through=excluded.data_through,
            detail=excluded.detail,display_name=excluded.display_name,updated_at=now()
            WHERE bp_alerts.wallets IS DISTINCT FROM excluded.wallets OR bp_alerts.signatures IS DISTINCT FROM excluded.signatures''',
            (key, a['rule'], a['cohort_version_id'], a['mint'], names.get(a['mint']), start, end, len(wallets), json.dumps(wallets),
             sum(w['tier'] == 'inferred_swap' for w in wallets),
             'inferred_swap' if any(w['tier'] == 'inferred_swap' for w in wallets) else 'parsed_swap',
             sum(int(w['raw']) for w in wallets), a['decimals'], usd, status, json.dumps(signatures), a['finality'],
             a['data_through'], json.dumps(a['params']), a['detail']))
        written += cur.rowcount
    return written
