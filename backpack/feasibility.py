"""Milestone 1 data-feasibility probe (docs/bp-holder-intelligence-spec.md section 14). Read-only.

Makes bounded real provider calls against current top BP holders and writes a JSON report: history endpoint
coverage of token-account and closed-account activity, portfolio response sizes, DAS metadata and Token-2022
coverage, Jupiter price availability, programs seen in swap-shaped transactions and an API usage estimate.
Never writes to the database and never prints credentials.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import time
from . import events as ev, portfolio as pf
from .collector import fetch_all
from .metrics import BP_MINT
from .providers import Providers, SourceError

UTC = timezone.utc


def _timed(function):
    start = time.monotonic()
    try: return function(), round(time.monotonic() - start, 3), None
    except SourceError as error: return None, round(time.monotonic() - start, 3), str(error)


def top_holders(conn, limit):
    rows = fetch_all(conn, '''SELECT wallet_address FROM bp_holder_rankings WHERE ranking='filtered'
        AND source_date=(SELECT max(source_date) FROM bp_holder_rankings) ORDER BY rank LIMIT %s''', (limit,)) \
        if fetch_all(conn, "SELECT to_regclass('bp_holder_rankings') IS NOT NULL AS ok")[0]['ok'] else []
    if rows: return [r['wallet_address'] for r in rows], 'bp_holder_rankings (filtered)'
    rows = fetch_all(conn, '''SELECT h.wallet_address FROM backpack_current_holders h JOIN backpack_assets a ON a.id=h.asset_id
        WHERE a.solana_mint=%s AND NOT h.excluded ORDER BY h.balance_tokens DESC LIMIT %s''', (BP_MINT, limit))
    return [r['wallet_address'] for r in rows], 'backpack_current_holders (not excluded)'


def _signatures(p, address, token_accounts, since, pages=3):
    found, token, count = {}, None, 0
    for _ in range(pages):
        rows, token = p.transactions_for_address(address, {'tokenAccounts': token_accounts, 'blockTime': {'gte': since}}, token,
                                                 limit=1000, details='signatures')
        count += 1
        for r in rows: found[r['signature']] = r
        if not token: return found, True, count
    return found, False, count


def compare_history(p, address, now):
    """Enhanced address history vs getTransactionsForAddress with and without token-account inclusion."""
    key = p.env.get('HELIUS_API_KEY')
    enhanced, before = {}, None
    for _ in range(2):
        params = {'api-key': key, 'limit': 100}
        if before: params['before'] = before
        page = p.request('Helius Enhanced', 'GET', f'https://api-mainnet.helius-rpc.com/v0/addresses/{address}/transactions', params=params)
        if not isinstance(page, list) or not page: break
        for tx in page: enhanced[tx['signature']] = tx
        before = page[-1]['signature']
    window = min((tx.get('timestamp') for tx in enhanced.values() if tx.get('timestamp')), default=int((now - timedelta(days=2)).timestamp()))
    direct, direct_done, _ = _signatures(p, address, 'none', window, 2)
    changed, changed_done, _ = _signatures(p, address, 'balanceChanged', window, 2)
    token_only = sorted(set(changed) - set(direct))
    evidence = []
    for sig in token_only[:5]:
        tx = p.rpc('getTransaction', [sig, {'encoding': 'jsonParsed', 'maxSupportedTransactionVersion': 1, 'commitment': 'finalized'}])
        keys, _ = ev.account_keys(tx or {})
        evidence.append(dict(signature=sig, owner_in_account_keys=address in keys, in_enhanced_history=sig in enhanced))
    closed = closed_account_check(p, address, window)
    return dict(wallet=address, window_start=datetime.fromtimestamp(window, UTC).isoformat(), enhanced=len(enhanced),
                gtfa_direct=len(direct), gtfa_balance_changed=len(changed), direct_complete=direct_done, balance_changed_complete=changed_done,
                token_account_only=len(token_only), token_account_only_in_enhanced=sum(s in enhanced for s in token_only),
                enhanced_missing_from_balance_changed=len(set(enhanced) - set(changed)), token_account_evidence=evidence, closed_accounts=closed)


def closed_account_check(p, address, since):
    """Find the owner's token accounts closed in recent history and test whether their earlier activity is
    returned when querying the owner (not the closed account)."""
    rows, _ = p.transactions_for_address(address, {'tokenAccounts': 'balanceChanged', 'blockTime': {'gte': since}}, None, 100)
    closed = []
    for tx in rows:
        keys, _ = ev.account_keys(tx)
        for r in ev.token_rows(tx, keys):
            if r['pre_owner'] == address and r['post_lamports'] == 0 and r['address']: closed.append((r['address'], tx['slot']))
    results = []
    owner_sigs = None
    for account, slot in closed[:3]:
        own, _, _ = _signatures(p, account, 'none', since, 1)
        earlier = {s for s, r in own.items() if r.get('slot', 0) < slot}
        if owner_sigs is None: owner_sigs, _, _ = _signatures(p, address, 'balanceChanged', since, 2)
        results.append(dict(token_account=account, closed_at_slot=slot, account_history=len(own), earlier_activity=len(earlier),
                            earlier_activity_in_owner_query=len(earlier & set(owner_sigs))))
    return dict(closed_accounts_found=len(closed), checked=results)


def run(conn, env, sample=3, portfolios=10, sizes=20, p=None):
    p = p or Providers(env, budget_key='BP_FEASIBILITY_MAX_REQUESTS', default_budget='400', deadline_seconds=1200)
    now = datetime.now(UTC)
    wallets, source = top_holders(conn, max(sample, portfolios, sizes))
    report = dict(generated_at=now.isoformat(), holder_source=source, wallets_available=len(wallets), items={}, limitations=[])
    if not wallets:
        report['status'] = 'unavailable'
        report['limitations'].append('No stored BP holder observation; run the daily capture first.')
        return report
    # Portfolio response sizes and latency.
    sizes_out, mints, programs = [], Counter(), Counter()
    for wallet in wallets[:sizes]:
        result, seconds, error = _timed(lambda: p.rpc_many([
            ('getBalance', [wallet, {'commitment': 'finalized'}]),
            ('getTokenAccountsByOwner', [wallet, {'programId': pf.TOKEN_PROGRAM}, {'encoding': 'jsonParsed', 'commitment': 'finalized'}]),
            ('getTokenAccountsByOwner', [wallet, {'programId': pf.TOKEN_2022}, {'encoding': 'jsonParsed', 'commitment': 'finalized'}])]))
        if error:
            sizes_out.append(dict(wallet=wallet, error=error, seconds=seconds))
            continue
        read = pf.wallet_read(wallet, *result)
        legacy, t22 = [len(((r or {}).get('value') or [])) for r, _ in result[1:]]
        sizes_out.append(dict(wallet=wallet, status=read['status'], spl_accounts=legacy, token2022_accounts=t22,
                              response_bytes=len(json.dumps([r for r, _ in result])), seconds=seconds, nonzero_mints=len(read['holdings'])))
        if len(mints) < 5000 and wallet in wallets[:portfolios]:
            for mint, h in read['holdings'].items():
                if mint != pf.NATIVE: mints[mint] += 1; programs[h['program']] += 1
    report['items']['portfolio_sizes'] = dict(wallets=sizes_out, max_token_accounts=max((w.get('spl_accounts', 0) + w.get('token2022_accounts', 0) for w in sizes_out), default=None))
    # DAS metadata and Token-2022 extension coverage for the distinct mints of the sample portfolios.
    distinct = sorted(mints)
    assets, _, error = _timed(lambda: p.assets(distinct))
    assets = assets or {}
    extensions = Counter(k for a in assets.values() for k in (a.get('mint_extensions') or {}))
    report['items']['metadata'] = dict(distinct_mints=len(distinct), with_metadata=len(assets), error=error,
        interfaces=dict(Counter(a.get('interface') for a in assets.values())), token2022_mints=programs.get(pf.TOKEN_2022, 0),
        extension_kinds=dict(extensions), classified=dict(Counter(pf.classify_asset(m, assets.get(m))['asset_class'] for m in distinct)))
    # Jupiter price availability and freshness.
    prices, _, error = _timed(lambda: p.prices(distinct + [pf.WSOL]))
    prices = prices or {}
    blocks = sorted({q.get('blockId') for q in prices.values() if q.get('blockId') is not None})[:100]
    times, _, time_error = _timed(lambda: p.block_times(blocks))
    ages = sorted((now - t).total_seconds() for t in (times or {}).values())
    report['items']['prices'] = dict(priced=len([m for m in distinct if m in prices]), of=len(distinct), error=error or time_error,
        sampled_block_times=len(ages), stale_over_1h=sum(a > 3600 for a in ages), median_age_seconds=ages[len(ages) // 2] if ages else None)
    # History endpoints, closed accounts and swap-shaped program usage.
    comparisons, seen_programs, volumes = [], Counter(), []
    for wallet in wallets[:sample]:
        try: comparisons.append(compare_history(p, wallet, now))
        except SourceError as error: comparisons.append(dict(wallet=wallet, error=str(error)))
        try:
            since = int((now - timedelta(days=30)).timestamp())
            sigs, complete, pages = _signatures(p, wallet, 'balanceChanged', since, 10)
            volumes.append(dict(wallet=wallet, transactions_30d=len(sigs), complete=complete, pages=pages))
            rows, _ = p.transactions_for_address(wallet, {'tokenAccounts': 'balanceChanged', 'status': 'succeeded'}, None, 100)
            for tx in rows:
                if ev.swap_candidates(tx, {wallet}):
                    for program in ev.programs_invoked(tx, ev.account_keys(tx)[0]): seen_programs[program] += 1
        except SourceError as error: volumes.append(dict(wallet=wallet, error=str(error)))
    report['items']['history_endpoints'] = comparisons
    report['items']['transaction_volume'] = volumes
    report['items']['swap_programs'] = dict(known={ev.DEX_PROGRAMS[k]: v for k, v in seen_programs.items() if k in ev.DEX_PROGRAMS},
        unknown={k: v for k, v in seen_programs.most_common(30) if k not in ev.DEX_PROGRAMS})
    report['items']['webhooks'] = dict(status='not_tested', detail='Requires registering a Helius webhook (a configuration change '
        'outside this read-only probe). Published documentation says deliveries cover transactions involving the monitored addresses '
        'but does not state whether owner-owned token accounts match; the design assumes they do not (spec section 9).')
    report['items']['usage_estimate'] = estimate(env, sizes_out, len(distinct), volumes)
    report['requests'] = dict(p.usage)
    report['json_rpc_calls'] = dict(p.calls)
    report['limitations'] = [
        'Samples the current top holders only; activity of other cohort members can differ by orders of magnitude.',
        'Credits are only stated where Helius documents them (getTransactionsForAddress); other figures are call counts.',
        'Webhook matching is documented, not observed.',
        'Program lists come from swap-shaped transactions of the sample wallets only.']
    report['status'] = 'completed'
    return report


def estimate(env, sizes, distinct_mints, volumes, cohort=None):
    cohort = cohort or int(env.get('BP_COHORT_SIZE', '200'))
    ok = [v for v in volumes if 'transactions_30d' in v]
    per_wallet_30d = sorted(v['transactions_30d'] for v in ok)
    median_30d = per_wallet_30d[len(per_wallet_30d) // 2] if per_wallet_30d else None
    hourly_runs = 24 * 30
    return dict(
        cohort=cohort, assumptions='Hourly portfolio refresh and history poll; metadata refreshed weekly; one poll page per wallet per run.',
        portfolio_calls_month=3 * cohort * hourly_runs, portfolio_http_requests_month=cohort * hourly_runs,
        price_requests_month=(distinct_mints // 50 + 1) * hourly_runs, block_time_calls_month=distinct_mints * hourly_runs,
        metadata_calls_month=(distinct_mints // 1000 + 1) * 5,
        history_poll_requests_month=cohort * hourly_runs,
        history_poll_credits_month=cohort * hourly_runs * 10,
        backfill_credits_once=None if median_30d is None else cohort * max(10, (median_30d // 100 + 1) * 10),
        median_sample_transactions_30d=median_30d,
        note='getTransactionsForAddress full mode: 10 credits per 100 returned transactions (10 minimum), per Helius documentation.')
