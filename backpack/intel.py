"""BP holder intelligence worker: stored portfolios, 30-day history, flags, alerts, reconciliation, retention.

Never runs DDL: the daily collector's --migrate owns the schema, and an unmigrated database returns
schema_pending. Worker exclusion uses the expiring, owner-checked lease row shared with the daily job.
"""
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import os
import uuid
from . import alerts, cohorts, events as ev, portfolio as pf
from .collector import fetch_all
from .providers import Providers, SourceError

UTC = timezone.utc
LEASE = 'bp_intel'
REQUIRED_TABLES = ('bp_cohorts', 'bp_cohort_members', 'bp_tracked_wallets', 'bp_tracked_assets', 'bp_portfolio_wallets',
                   'bp_portfolio_balances', 'bp_price_observations', 'bp_raw_transactions', 'bp_economic_events',
                   'bp_history_coverage', 'bp_alert_rules', 'bp_alerts', 'bp_holder_rankings', 'bp_reconciliation')


def providers(env):
    return Providers(env, budget_key='BP_INTEL_MAX_REQUESTS', default_budget='2500',
                     deadline_seconds=int(env.get('BP_INTEL_DEADLINE_SECONDS', '1500')))


def schema_ready(conn):
    rows = fetch_all(conn, 'SELECT ' + ','.join(f"to_regclass('{t}') IS NOT NULL AS {t}" for t in REQUIRED_TABLES))
    missing = [t for t, ok in rows[0].items() if not ok]
    if missing: return False
    return bool(fetch_all(conn, "SELECT 1 FROM information_schema.columns WHERE table_name='backpack_ingestion_runs' AND column_name='job' AND table_schema=current_schema()"))


def _lease(conn, run_id, minutes):
    with conn, conn.cursor() as cur:
        cur.execute('''INSERT INTO backpack_job_leases VALUES(%s,%s,now()+make_interval(mins=>%s))
            ON CONFLICT(name) DO UPDATE SET owner=excluded.owner,expires_at=excluded.expires_at
            WHERE backpack_job_leases.expires_at<now() RETURNING owner''', (LEASE, run_id, minutes))
        return bool(cur.fetchone())


def _release(conn, run_id):
    with conn, conn.cursor() as cur:
        cur.execute('DELETE FROM backpack_job_leases WHERE name=%s AND owner=%s', (LEASE, run_id))


def _safe(error):
    return str(error) if isinstance(error, SourceError) else type(error).__name__


# Portfolio ------------------------------------------------------------------------------------------------------

def collect_portfolios(conn, p, env, run_id, wallets, now=None):
    from psycopg2.extras import Json, execute_values
    observed = now or datetime.now(UTC)
    max_accounts = int(env.get('BP_MAX_TOKEN_ACCOUNTS', '10000'))
    reads, skipped = [], 0
    for w in wallets:
        owner = w['wallet_address']
        blank = dict(wallet_address=owner, holdings={}, token_accounts=None, slot_min=None, slot_max=None,
                     sol_status='failed', spl_status='failed', token2022_status='failed')
        if p.remaining_seconds() < 180:
            skipped += 1
            reads.append(dict(blank, status='unavailable', detail='run deadline reached before this wallet was read'))
            continue
        try:
            reads.append(pf.wallet_read(owner, *p.rpc_many([
                ('getBalance', [owner, {'commitment': 'finalized'}]),
                ('getTokenAccountsByOwner', [owner, {'programId': pf.TOKEN_PROGRAM}, {'encoding': 'jsonParsed', 'commitment': 'finalized'}]),
                ('getTokenAccountsByOwner', [owner, {'programId': pf.TOKEN_2022}, {'encoding': 'jsonParsed', 'commitment': 'finalized'}])]),
                max_accounts))
        except SourceError as error:
            reads.append(dict(blank, status='unavailable', detail=_safe(error)))
    mints = sorted({m for r in reads for m in r['holdings'] if m != pf.NATIVE})
    known = {r['mint']: r for r in fetch_all(conn, 'SELECT * FROM bp_tracked_assets WHERE mint=ANY(%s)', (mints,))} if mints else {}
    refresh_before = observed - timedelta(days=int(env.get('BP_METADATA_REFRESH_DAYS', '7')))
    stale = [m for m in mints if m not in known or not known[m]['metadata_at'] or known[m]['metadata_at'] < refresh_before]
    notes, das = [], {}
    if stale:
        try: das = p.assets(stale)
        except SourceError as error: notes.append('metadata: ' + _safe(error))
    sample = {m: h for r in reads for m, h in r['holdings'].items()}
    assets = {m: pf.classify_asset(m, das.get(m), sample[m]['program'], sample[m]['decimals']) if m in stale else known[m] for m in mints}
    assets[pf.NATIVE] = pf.classify_asset(pf.NATIVE)
    quotes, prices = {}, {}
    try:
        quotes = p.prices(mints + [pf.WSOL])
        times = p.block_times([q.get('blockId') for q in quotes.values()])
        for mint, q in quotes.items():
            prices[mint] = dict(price=Decimal(str(q['usdPrice'])), price_at=times.get(int(q['blockId'])) if q.get('blockId') is not None else None,
                                source='Jupiter Price V3', block_id=q.get('blockId'))
    except SourceError as error:
        notes.append('prices: ' + _safe(error))
    wallet_rows, balance_rows = [], []
    for read in reads:
        rows, total, unpriced = pf.balance_rows(read, assets, prices, observed, Decimal(env.get('BP_DUST_USD', '1')))
        balance_rows += rows
        wallet_rows.append((run_id, read['wallet_address'], read['status'], read['sol_status'], read['spl_status'], read['token2022_status'],
                            read['token_accounts'], read['slot_min'], read['slot_max'], observed, total, unpriced, read.get('detail') or ''))
    with conn, conn.cursor() as cur:
        for m in stale:
            a = assets[m]
            cur.execute('''INSERT INTO bp_tracked_assets(mint,asset_class,token_program,decimals,symbol,name,metadata_source,
                extension_flags,extensions,is_sol,is_stable,is_bp,spam_class,spam_reason,class_reason,metadata_at)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(network,mint) DO UPDATE SET
                asset_class=excluded.asset_class,token_program=excluded.token_program,decimals=excluded.decimals,symbol=excluded.symbol,
                name=excluded.name,metadata_source=excluded.metadata_source,extension_flags=excluded.extension_flags,
                extensions=excluded.extensions,spam_class=excluded.spam_class,spam_reason=excluded.spam_reason,
                class_reason=excluded.class_reason,metadata_at=excluded.metadata_at,updated_at=now()''',
                (m, a['asset_class'], a['token_program'], a['decimals'], a['symbol'], a['name'], a['metadata_source'], a['extension_flags'],
                 Json(a['extensions']), a['is_sol'], a['is_stable'], a['is_bp'], a['spam_class'], a['spam_reason'], a['class_reason'],
                 observed if a['metadata_source'] else None))
        cur.execute('''INSERT INTO bp_tracked_assets(mint,asset_class,decimals,symbol,name,metadata_source,is_sol,metadata_at)
            VALUES('native','native',9,'SOL','Solana (native)','protocol',true,now()) ON CONFLICT DO NOTHING''')
        for mint, q in quotes.items():
            created = q.get('createdAt')
            cur.execute('''UPDATE bp_tracked_assets SET liquidity_usd=%s,market_created_at=COALESCE(%s::timestamptz,market_created_at),
                market_observed_at=%s WHERE network='solana-mainnet' AND mint=%s''',
                (Decimal(str(q['liquidity'])) if q.get('liquidity') is not None else None, created, observed, mint))
        if pf.WSOL in prices and prices[pf.WSOL]['price_at']:
            s = prices[pf.WSOL]
            cur.execute('''INSERT INTO bp_price_observations(mint,price_at,source,price,block_id) VALUES(%s,%s,%s,%s,%s)
                ON CONFLICT DO NOTHING''', (pf.WSOL, s['price_at'], s['source'], s['price'], s['block_id']))
        execute_values(cur, '''INSERT INTO bp_portfolio_wallets(run_id,wallet_address,status,sol_status,spl_status,token2022_status,
            token_accounts,slot_min,slot_max,observed_at,priced_value_usd,unpriced_holdings,detail) VALUES %s''', wallet_rows)
        if balance_rows:
            execute_values(cur, '''INSERT INTO bp_portfolio_balances(run_id,wallet_address,mint,raw_amount,decimals,ui_amount,token_accounts,
                slot,observed_at,price,price_at,price_source,pricing_status,value_usd,frozen,balance_visibility,holding_class,spam_class,
                dust,classification_reason) VALUES %s''',
                [(run_id, r['wallet_address'], r['mint'], r['raw_amount'], r['decimals'], r['ui_amount'], Json(r['token_accounts']), r['slot'],
                  r['observed_at'], r['price'], r['price_at'], r['price_source'], r['pricing_status'], r['value_usd'], r['frozen'],
                  r['balance_visibility'], r['holding_class'], r['spam_class'], r['dust'], r['classification_reason']) for r in balance_rows],
                page_size=1000)
    counts = {s: sum(r['status'] == s for r in reads) for s in ('complete', 'partial', 'unavailable', 'oversized')}
    return dict(wallets=len(reads), deadline_skipped=skipped, mints=len(mints), priced_mints=len([m for m in mints if m in prices]),
                balances=len(balance_rows), notes=notes, **counts)


# History --------------------------------------------------------------------------------------------------------

def _sol_prices(conn, since):
    rows = fetch_all(conn, 'SELECT price_at,price FROM bp_price_observations WHERE mint=%s AND price_at>=%s ORDER BY price_at',
                     (pf.WSOL, since - timedelta(hours=2)))
    stamps = [r['price_at'] for r in rows]
    def nearest(at, max_gap=timedelta(hours=1)):
        i = bisect_left(stamps, at)
        best = min((rows[j] for j in (i - 1, i) if 0 <= j < len(rows)), key=lambda r: abs(r['price_at'] - at), default=None)
        return (best['price'], best['price_at']) if best and abs(best['price_at'] - at) <= max_gap else None
    return nearest


def _snapshots(conn):
    cache = {}
    def lookup(owner, mint, before):
        if (owner, mint) not in cache:
            cache[(owner, mint)] = fetch_all(conn, '''SELECT b.observed_at,b.token_accounts FROM bp_portfolio_balances b
                JOIN bp_portfolio_wallets w USING(run_id,wallet_address) WHERE b.wallet_address=%s AND b.mint=%s
                AND w.status IN ('complete','partial') ORDER BY b.observed_at DESC LIMIT 500''', (owner, mint))
        for row in cache[(owner, mint)]:
            if row['observed_at'] < before: return row['token_accounts']
        return None
    return lookup


def poll_wallet(conn, p, env, wallet, owners, tracked_since, assets, sol_price, snapshot, now):
    """Fetch new and backfill history for one wallet, classify for every tracked owner, commit atomically."""
    from psycopg2.extras import Json, execute_values
    owner = wallet['wallet_address']
    coverage = fetch_all(conn, 'SELECT * FROM bp_history_coverage WHERE wallet_address=%s', (owner,))
    cov = coverage[0] if coverage else dict(wallet_address=owner, requested_start=wallet['history_start'], earliest_retrieved=None,
        earliest_slot=None, newest_slot=None, newest_retrieved=None, backfill_status='pending', backfill_pages=0, gaps=[], transactions=0)
    # Separate page budgets: polling newer activity must never starve the backfill, and vice versa.
    poll_pages = int(env.get('BP_HISTORY_POLL_PAGES', '5'))
    backfill_pages = int(env.get('BP_HISTORY_BACKFILL_PAGES_PER_RUN', '5'))
    backfill_cap = int(env.get('BP_HISTORY_MAX_BACKFILL_PAGES', '30'))
    size = int(env.get('BP_HISTORY_PAGE_SIZE', '100'))
    base = {'status': env.get('BP_HISTORY_STATUS', 'succeeded'), 'tokenAccounts': 'balanceChanged'}
    fetched, pages, gaps, poll_status = {}, 0, list(cov['gaps'] or []), 'complete'
    start_ts = int(cov['requested_start'].timestamp())
    def take(rows):
        for tx in rows:
            sig = ((tx.get('transaction') or {}).get('signatures') or [None])[0]
            if sig and tx.get('blockTime') is not None and tx['blockTime'] >= start_ts: fetched[sig] = tx
    # Newer activity: from the newest slot already read, or from the top once a backfill found nothing at all.
    if cov['newest_slot'] is not None or cov['backfill_status'] in ('complete', 'capped'):
        token = None
        newer = dict(base, slot={'gte': cov['newest_slot']}) if cov['newest_slot'] is not None else base
        while pages < poll_pages:
            rows, token = p.transactions_for_address(owner, newer, token, size)
            pages += 1
            take(rows)
            if not token: break
        if token:
            oldest = min((tx['slot'] for tx in fetched.values()), default=None)
            gaps.append(dict(from_slot=cov['newest_slot'], to_slot=oldest, recorded_at=now.isoformat(),
                             reason='per-run page cap reached while polling newer activity; slots between were not read'))
            poll_status = 'capped'
    newest_before = cov['newest_slot']
    if cov['backfill_status'] in ('pending', 'incomplete'):
        # Within a run, follow the provider's pagination token; across runs, resume below the earliest slot read.
        filters = dict(base, slot={'lte': cov['earliest_slot']}) if cov['earliest_slot'] is not None else base
        token, done, used = None, False, 0
        while used < backfill_pages and cov['backfill_pages'] < backfill_cap:
            rows, token = p.transactions_for_address(owner, filters, token, size)
            pages += 1
            used += 1
            cov['backfill_pages'] += 1
            take(rows)
            stamped = [tx for tx in rows if tx.get('blockTime') is not None]
            if stamped:
                oldest = min(stamped, key=lambda tx: (tx['slot'], tx['blockTime']))
                if cov['earliest_slot'] is None or oldest['slot'] < cov['earliest_slot']:
                    cov['earliest_slot'], cov['earliest_retrieved'] = oldest['slot'], datetime.fromtimestamp(oldest['blockTime'], UTC)
                if newest_before is None:
                    top = max(stamped, key=lambda tx: tx['slot'])
                    if cov['newest_slot'] is None or top['slot'] > cov['newest_slot']:
                        cov['newest_slot'], cov['newest_retrieved'] = top['slot'], datetime.fromtimestamp(top['blockTime'], UTC)
                if min(tx['blockTime'] for tx in stamped) < start_ts: done = True
            if not token: done = True  # The provider reports no older history for this address.
            if done: break
        cov['backfill_status'] = 'complete' if done else 'capped' if cov['backfill_pages'] >= backfill_cap else 'incomplete'
        if cov['newest_slot'] is None: cov['newest_retrieved'] = now
    newest = max((tx['slot'] for tx in fetched.values()), default=None)
    if newest is not None and (cov['newest_slot'] is None or newest > cov['newest_slot']):
        cov['newest_slot'] = newest
        cov['newest_retrieved'] = datetime.fromtimestamp(max(tx['blockTime'] for tx in fetched.values() if tx['slot'] == newest), UTC)
    candidates = [s for s, tx in fetched.items() if ev.swap_candidates(tx, owners)]
    enhanced, note = {}, ''
    if candidates and env.get('BP_ENHANCED_PARSE', '1') == '1':
        try: enhanced = p.enhanced_parse(candidates)
        except SourceError as error: note = 'enhanced parse unavailable: ' + _safe(error)
    records, unparsed = [], []
    for sig, tx in fetched.items():
        try:
            found = ev.classify(tx, owners, assets, enhanced.get(sig), snapshot, sol_price,
                                int(env.get('BP_INCIDENTAL_LAMPORTS', '3000000')))
        except (KeyError, TypeError, ValueError, IndexError):
            unparsed.append(sig)  # Raw payload is kept for reprocessing; no event is invented.
            continue
        for e in found:
            records.append(dict(e, pre_membership=e['block_time'] is None or e['block_time'] < tracked_since[e['wallet_address']],
                                detail='; '.join(x for x in (e['detail'], note if sig in candidates else '') if x)))
    columns = ('signature', 'wallet_address', 'event_index', 'slot', 'block_time', 'kind', 'tier', 'input_mint', 'input_raw',
               'input_decimals', 'output_mint', 'output_raw', 'output_decimals', 'usd_value', 'valuation_source', 'valuation_status',
               'pre_input_raw', 'post_input_raw', 'pre_output_raw', 'post_output_raw', 'pre_balance_source', 'new_position',
               'ata_created', 'pre_membership', 'owner_signed', 'venue', 'programs', 'detail', 'parser_version', 'finality')
    with conn, conn.cursor() as cur:
        new = 0
        if fetched:
            new = len(execute_values(cur, '''INSERT INTO bp_raw_transactions(signature,slot,block_time,finality,source,payload,enhanced,
                parser_version) VALUES %s ON CONFLICT DO NOTHING RETURNING signature''',
                [(sig, tx['slot'], datetime.fromtimestamp(tx['blockTime'], UTC), 'finalized', 'Helius getTransactionsForAddress', Json(tx),
                  Json(enhanced[sig]) if sig in enhanced else None, ev.PARSER_VERSION) for sig, tx in fetched.items()],
                page_size=200, fetch=True))
        if records:
            execute_values(cur, f'''INSERT INTO bp_economic_events({','.join(columns)}) VALUES %s ON CONFLICT DO NOTHING''',
                           [tuple(e[k] for k in columns) for e in records], page_size=500)
        cur.execute('''INSERT INTO bp_history_coverage(wallet_address,source,requested_start,earliest_retrieved,earliest_slot,newest_slot,
            newest_retrieved,backfill_status,backfill_pages,poll_status,last_poll_at,gaps,transactions)
            VALUES(%s,'Helius getTransactionsForAddress (tokenAccounts=balanceChanged)',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT(wallet_address) DO UPDATE SET earliest_retrieved=excluded.earliest_retrieved,earliest_slot=excluded.earliest_slot,
            newest_slot=excluded.newest_slot,newest_retrieved=excluded.newest_retrieved,backfill_status=excluded.backfill_status,
            backfill_pages=excluded.backfill_pages,poll_status=excluded.poll_status,last_poll_at=excluded.last_poll_at,
            gaps=excluded.gaps,transactions=bp_history_coverage.transactions+%s,updated_at=now()''',
            (owner, cov['requested_start'], cov['earliest_retrieved'], cov['earliest_slot'], cov['newest_slot'], cov['newest_retrieved'],
             cov['backfill_status'], cov['backfill_pages'], poll_status, now, Json(gaps[-50:]), new, new))
    if unparsed:
        gaps.append(dict(recorded_at=now.isoformat(), reason=f'{len(unparsed)} transaction(s) could not be classified; raw payload retained',
                         signatures=unparsed[:10]))
        with conn, conn.cursor() as cur:
            cur.execute('UPDATE bp_history_coverage SET gaps=%s WHERE wallet_address=%s', (Json(gaps[-50:]), owner))
    return dict(transactions=len(fetched), events=len(records), pages=pages, backfill=cov['backfill_status'], poll=poll_status,
                enhanced=len(enhanced), candidates=len(candidates), unparsed=len(unparsed))


def poll_history(conn, p, env, wallets, now=None):
    now = now or datetime.now(UTC)
    owners = {w['wallet_address'] for w in wallets}
    tracked_since = {w['wallet_address']: w['tracked_since'] for w in fetch_all(conn, 'SELECT * FROM bp_tracked_wallets')}
    assets = {r['mint']: r for r in fetch_all(conn, 'SELECT mint,asset_class,extension_flags FROM bp_tracked_assets')}
    sol_price = _sol_prices(conn, min((w['history_start'] for w in wallets), default=now))
    snapshot = _snapshots(conn)
    order = {r['wallet_address']: r['last_poll_at'] for r in fetch_all(conn, 'SELECT wallet_address,last_poll_at FROM bp_history_coverage')}
    queue = sorted(wallets, key=lambda w: (order.get(w['wallet_address']) is not None, order.get(w['wallet_address']) or now, w['wallet_address']))
    summary = dict(wallets=len(queue), polled=0, transactions=0, events=0, failed=0, deadline_skipped=0, capped=0, errors=[])
    for w in queue:
        if p.remaining_seconds() < 120:
            summary['deadline_skipped'] += 1
            continue
        try:
            result = poll_wallet(conn, p, env, w, owners, tracked_since, assets, sol_price, snapshot, now)
            summary['polled'] += 1
            summary['transactions'] += result['transactions']
            summary['events'] += result['events']
            summary['capped'] += result['poll'] == 'capped' or result['backfill'] == 'capped'
        except SourceError as error:
            conn.rollback()
            summary['failed'] += 1
            summary['errors'].append(f"{w['wallet_address'][:6]}…: {_safe(error)}")
            if 'budget exhausted' in str(error): break
    summary['errors'] = summary['errors'][:20]
    return summary


# Derived flags --------------------------------------------------------------------------------------------------

def refresh_flags(conn, env):
    """Recompute history-dependent flags so out-of-order and duplicate deliveries converge."""
    days = int(env.get('BP_RECENT_LAUNCH_DAYS', '14'))
    with conn, conn.cursor() as cur:
        cur.execute('''UPDATE bp_economic_events e SET first_observed_purchase=NOT EXISTS(
              SELECT 1 FROM bp_economic_events x WHERE x.wallet_address=e.wallet_address AND x.kind='swap'
              AND x.output_mint=e.output_mint AND (x.block_time,x.signature,x.event_index)<(e.block_time,e.signature,e.event_index)),
            re_entry=EXISTS(
              SELECT 1 FROM bp_economic_events z WHERE z.wallet_address=e.wallet_address AND z.input_mint=e.output_mint
              AND z.pre_input_raw>0 AND z.post_input_raw=0 AND z.block_time<e.block_time),
            first_group_purchase=NOT EXISTS(
              SELECT 1 FROM bp_economic_events x WHERE x.kind='swap' AND x.output_mint=e.output_mint
              AND (x.block_time,x.signature,x.wallet_address,x.event_index)<(e.block_time,e.signature,e.wallet_address,e.event_index)),
            recently_launched=(SELECT CASE WHEN a.market_created_at IS NULL THEN NULL
              ELSE a.market_created_at>e.block_time-make_interval(days=>%s) END FROM bp_tracked_assets a
              WHERE a.network=e.network AND a.mint=e.output_mint)
            WHERE e.kind='swap' AND e.block_time IS NOT NULL''', (days,))


# Alerts ---------------------------------------------------------------------------------------------------------

def evaluate_alerts(conn, env, now=None):
    now = now or datetime.now(UTC)
    rules = {r['rule']: r for r in fetch_all(conn, 'SELECT * FROM bp_alert_rules')}
    if not rules: return dict(status='no_rules')
    lookback = now - timedelta(hours=int(env.get('BP_ALERT_LOOKBACK_HOURS', '48')))
    events = fetch_all(conn, "SELECT * FROM bp_economic_events WHERE kind='swap' AND block_time>=%s", (lookback,))
    names = {r['mint']: r['symbol'] or r['name'] for r in fetch_all(conn, 'SELECT mint,symbol,name FROM bp_tracked_assets')}
    written = raised = 0
    for version in cohorts.latest_versions(conn):
        members = cohorts.members(conn, version['version_id'])
        coverage = fetch_all(conn, 'SELECT min(last_poll_at) AS through, count(*) AS polled FROM bp_history_coverage WHERE wallet_address=ANY(%s)',
                             (sorted(members),))[0]
        through = coverage['through'] if coverage['polled'] == len(members) else None
        found = alerts.evaluate(events, rules, version['version_id'], members, through)
        raised += len(found)
        with conn, conn.cursor() as cur:
            written += alerts.store(cur, found, {k: r['params'].get('cooldown_minutes', 0) for k, r in rules.items()}, names)
    return dict(evaluated=raised, written=written)


# Reconciliation and retention -----------------------------------------------------------------------------------

def reconcile(conn, day):
    """Once per UTC day: compare token balance changes between two complete portfolio reads ~24h apart with the
    net of classified events between their slots, for wallets whose history has no recorded gap."""
    if fetch_all(conn, 'SELECT 1 FROM bp_reconciliation WHERE date=%s LIMIT 1', (day,)): return dict(status='already_reconciled')
    runs = fetch_all(conn, '''SELECT run_id,started_at FROM backpack_ingestion_runs WHERE job='bp_intel' AND started_at<%s
        AND EXISTS(SELECT 1 FROM bp_portfolio_wallets w WHERE w.run_id=backpack_ingestion_runs.run_id) ORDER BY started_at DESC LIMIT 60''',
        (datetime.combine(day + timedelta(days=1), datetime.min.time(), UTC),))
    if len(runs) < 2: return dict(status='insufficient_history')
    later = runs[0]
    earlier = next((r for r in runs[1:] if later['started_at'] - r['started_at'] >= timedelta(hours=20)), None)
    if not earlier: return dict(status='insufficient_history')
    rows = fetch_all(conn, '''WITH a AS (SELECT w.wallet_address,w.slot_max FROM bp_portfolio_wallets w WHERE w.run_id=%(early)s AND w.status='complete'),
        b AS (SELECT w.wallet_address,w.slot_max FROM bp_portfolio_wallets w WHERE w.run_id=%(late)s AND w.status='complete'),
        pairs AS (SELECT a.wallet_address,a.slot_max AS s0,b.slot_max AS s1 FROM a JOIN b USING(wallet_address)
            JOIN bp_history_coverage c USING(wallet_address) WHERE c.newest_slot>=b.slot_max AND jsonb_array_length(c.gaps)=0
            AND (c.backfill_status='complete' OR c.earliest_slot<=a.slot_max)),
        mints AS (SELECT p.wallet_address,x.mint FROM pairs p JOIN bp_portfolio_balances x ON x.wallet_address=p.wallet_address
            AND x.run_id IN (%(early)s,%(late)s) WHERE x.mint NOT IN ('native','So11111111111111111111111111111111111111112')
            UNION SELECT p.wallet_address,m FROM pairs p JOIN bp_economic_events e ON e.wallet_address=p.wallet_address
            AND e.slot>p.s0 AND e.slot<=p.s1 CROSS JOIN LATERAL unnest(ARRAY[e.input_mint,e.output_mint]) m
            WHERE m IS NOT NULL AND m NOT IN ('native','So11111111111111111111111111111111111111112'))
        SELECT m.wallet_address,m.mint,p.s0,p.s1,
          COALESCE((SELECT raw_amount FROM bp_portfolio_balances WHERE run_id=%(late)s AND wallet_address=m.wallet_address AND mint=m.mint),0)
          -COALESCE((SELECT raw_amount FROM bp_portfolio_balances WHERE run_id=%(early)s AND wallet_address=m.wallet_address AND mint=m.mint),0) AS snapshot_delta,
          COALESCE((SELECT sum(CASE WHEN e.output_mint=m.mint THEN e.output_raw ELSE 0 END)-sum(CASE WHEN e.input_mint=m.mint THEN e.input_raw ELSE 0 END)
            FROM bp_economic_events e WHERE e.wallet_address=m.wallet_address AND e.slot>p.s0 AND e.slot<=p.s1 AND e.kind<>'failed'),0) AS event_delta
        FROM mints m JOIN pairs p USING(wallet_address)''', dict(early=earlier['run_id'], late=later['run_id']))
    with conn, conn.cursor() as cur:
        for r in rows:
            difference = r['snapshot_delta'] - r['event_delta']
            cur.execute('''INSERT INTO bp_reconciliation(date,wallet_address,mint,from_run_id,to_run_id,from_slot,to_slot,
                snapshot_delta,event_delta,difference,status) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING''',
                (day, r['wallet_address'], r['mint'], earlier['run_id'], later['run_id'], r['s0'], r['s1'], r['snapshot_delta'],
                 r['event_delta'], difference, 'matched' if difference == 0 else 'discrepancy'))
    return dict(status='reconciled', rows=len(rows), discrepancies=sum(r['snapshot_delta'] != r['event_delta'] for r in rows))


def maintain(conn, env, now=None):
    now = now or datetime.now(UTC)
    raw_days = int(env.get('BP_RAW_TX_RETENTION_DAYS', '45'))
    keep_days = int(env.get('BP_PORTFOLIO_RETENTION_DAYS', '90'))
    if raw_days < 30 or keep_days < 30: raise ValueError('BP retention must be at least 30 days')
    with conn, conn.cursor() as cur:
        cur.execute('''DELETE FROM bp_raw_transactions WHERE ctid IN (SELECT ctid FROM bp_raw_transactions
            WHERE block_time<%s LIMIT 10000)''', (now - timedelta(days=raw_days),))
        raw = cur.rowcount
        # Hourly reads older than 48h thin to the first read of each UTC day; beyond retention, reads are dropped.
        cur.execute('''WITH runs AS (SELECT run_id,started_at,row_number() OVER (PARTITION BY (started_at AT TIME ZONE 'UTC')::date
                ORDER BY started_at) AS n FROM backpack_ingestion_runs WHERE job='bp_intel'),
            doomed AS (SELECT run_id FROM runs WHERE (started_at<%s AND n>1) OR started_at<%s)
            DELETE FROM bp_portfolio_balances WHERE ctid IN (SELECT b.ctid FROM bp_portfolio_balances b JOIN doomed USING(run_id) LIMIT 50000)''',
            (now - timedelta(hours=48), now - timedelta(days=keep_days)))
        balances = cur.rowcount
        cur.execute('''DELETE FROM bp_portfolio_wallets w WHERE NOT EXISTS(SELECT 1 FROM bp_portfolio_balances b
            WHERE b.run_id=w.run_id AND b.wallet_address=w.wallet_address) AND w.run_id IN (
            SELECT run_id FROM (SELECT run_id,started_at,row_number() OVER (PARTITION BY (started_at AT TIME ZONE 'UTC')::date
                ORDER BY started_at) AS n FROM backpack_ingestion_runs WHERE job='bp_intel') r
            WHERE (started_at<%s AND n>1) OR started_at<%s)''', (now - timedelta(hours=48), now - timedelta(days=keep_days)))
        cur.execute('DELETE FROM bp_price_observations WHERE price_at<%s', (now - timedelta(days=keep_days),))
    return dict(raw_transactions_deleted=raw, balances_deleted=balances)


# Orchestration ----------------------------------------------------------------------------------------------------

def run(conn, p=None, env=None, steps=('portfolio', 'history', 'alerts', 'reconcile', 'maintenance'), now=None):
    env = os.environ if env is None else env
    if not schema_ready(conn): return dict(status='schema_pending', detail='Run python backpack_monitor.py --migrate first')
    p = p or providers(env)
    now = now or datetime.now(UTC)
    try: cohort = cohorts.refresh_cohort(conn, env)
    except Exception as error:
        conn.rollback()
        cohort = dict(status='failed', detail=_safe(error))
    run_id = str(uuid.uuid4())
    if not _lease(conn, run_id, int(env.get('BP_INTEL_LEASE_MINUTES', '35'))): return dict(status='already_running', cohort=cohort)
    with conn, conn.cursor() as cur:
        cur.execute("INSERT INTO backpack_ingestion_runs(run_id,snapshot_date,job) VALUES(%s,%s,'bp_intel')", (run_id, now.date()))
    result, errors = dict(run_id=run_id, cohort=cohort), []
    try:
        wallets = cohorts.tracked_wallets(conn)
        result['tracked_wallets'] = len(wallets)
        if not wallets: errors.append('No tracked wallets: no current cohort version exists yet')
        for step, function in (('portfolio', lambda: collect_portfolios(conn, p, env, run_id, wallets, now)),
                               ('history', lambda: poll_history(conn, p, env, wallets, now))):
            if step in steps and wallets:
                try: result[step] = function()
                except (SourceError, KeyError, TypeError, ValueError) as error:
                    conn.rollback()
                    errors.append(f'{step}: {_safe(error)}')
        if 'history' in steps: refresh_flags(conn, env)
        if 'alerts' in steps: result['alerts'] = evaluate_alerts(conn, env, now)
        if 'reconcile' in steps: result['reconcile'] = reconcile(conn, now.date())
    except Exception as error:
        conn.rollback()
        errors.append('run: ' + _safe(error))
    finally:
        portfolio_result = result.get('portfolio') or {}
        status = 'failed' if errors and not portfolio_result else 'partial' if errors or portfolio_result.get('partial') or \
            portfolio_result.get('unavailable') or (result.get('history') or {}).get('failed') else 'completed'
        with conn, conn.cursor() as cur:
            for provider, requests in p.usage.items():
                cur.execute('''INSERT INTO backpack_provider_usage(run_id,provider,requests,methodology) VALUES(%s,%s,%s,%s)
                    ON CONFLICT DO NOTHING''', (run_id, provider, requests,
                    f"HTTP attempts incl. retries; {p.calls.get(provider, 0)} JSON-RPC calls carried in batches. Credits not inferred."))
            cur.execute('''UPDATE backpack_ingestion_runs SET completed_at=now(),status=%s,assets_attempted=%s,assets_succeeded=%s,
                assets_failed=%s,assets_skipped=%s,source_errors=%s,data_quality_warnings=%s WHERE run_id=%s''',
                (status, portfolio_result.get('wallets', 0), portfolio_result.get('complete', 0),
                 portfolio_result.get('unavailable', 0) + portfolio_result.get('partial', 0), portfolio_result.get('deadline_skipped', 0),
                 '; '.join(errors), 'BP intelligence run: asset counters hold wallet reads (attempted/complete/partial-or-unavailable/deadline-skipped).',
                 run_id))
        _release(conn, run_id)
    if 'maintenance' in steps:
        try: result['maintenance'] = maintain(conn, env, now)
        except Exception as error:
            conn.rollback()
            result['maintenance'] = 'failed: ' + _safe(error)
    return dict(result, status=status, errors=errors)
