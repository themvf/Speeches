"""BP holder intelligence against disposable PostgreSQL. Set BACKPACK_TEST_DATABASE_URL explicitly to run."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
import os
from pathlib import Path
import re
import uuid
import pytest
from backpack import cohorts, intel, portfolio as pf
from backpack.collector import fetch_all, setup
from backpack.metrics import BP_MINT, USDC
from test_bp_intel import JUP, account, enhanced_swap, make_tx, token

URL = os.environ.get('BACKPACK_TEST_DATABASE_URL')
pytestmark = pytest.mark.skipif(not URL, reason='Requires disposable BACKPACK_TEST_DATABASE_URL')
UTC = timezone.utc
NOW = datetime.now(UTC).replace(microsecond=0)
TOKEN, UNPRICED = 'TokenMintAAAA', 'UnpricedMintBBBB'
ENV = dict(BP_COHORT_SIZE='3', BP_COHORT_EXIT_RANK='4', BP_COHORT_ENTRANT_CAP='1')


@pytest.fixture(scope='session')
def database():
    import psycopg2
    connection = psycopg2.connect(URL)
    yield connection
    connection.close()


@pytest.fixture
def conn(database):
    schema = 'bp_intel_test_' + uuid.uuid4().hex
    with database, database.cursor() as cur:
        cur.execute('CREATE SCHEMA ' + schema)
        cur.execute('SET search_path TO ' + schema)
    setup(database)
    try: yield database
    finally:
        database.rollback()
        with database, database.cursor() as cur: cur.execute('DROP SCHEMA ' + schema + ' CASCADE')


def bp_capture(conn, day, balances, excluded=(), captured_at=None, retained=None, labels=None):
    """A complete, checkpointed BP holder capture as the daily collector stores it."""
    asset = fetch_all(conn, "SELECT id FROM backpack_assets WHERE asset_type='bp'")[0]['id']
    run_id = str(uuid.uuid4())
    rows = [(w, D(b)) for w, b in balances.items()]
    with conn, conn.cursor() as cur:
        cur.execute('INSERT INTO backpack_ingestion_runs(run_id,snapshot_date) VALUES(%s,%s)', (run_id, day))
        cur.execute('''INSERT INTO backpack_asset_daily_snapshots(asset_id,date,run_id,captured_at,slot,source,token_supply,decimals,
            holders_complete,unique_holders,holder_start_slot,holder_end_slot,data_quality_score,quality_status)
            VALUES(%s,%s,%s,%s,100,'fixture',%s,9,true,%s,90,110,60,'Estimated')''',
            (asset, day, run_id, captured_at or NOW - timedelta(hours=2), sum(b for _, b in rows), len(rows)))
        for w, b in rows[:retained]:
            label = ('Treasury', 'confirmed') if w in excluded else (labels or {}).get(w, ('Unknown', None))
            cur.execute('''INSERT INTO backpack_asset_holder_daily_snapshots(asset_id,date,wallet_address,balance_tokens,excluded,label,
                label_confidence,source,slot) VALUES(%s,%s,%s,%s,%s,%s,%s,'fixture',110)''', (asset, day, w, b, w in excluded, *label))
        cur.execute('''INSERT INTO backpack_holder_checkpoints(asset_id,date,owner_count,balance_tokens,aggregates_validated,methodology)
            VALUES(%s,%s,%s,%s,true,'fixture')''', (asset, day, len(rows), sum(b for _, b in rows)))


def keys_of(tx):
    return {k['pubkey'] for k in tx['transaction']['message']['accountKeys']}


def involved(tx):
    keys = [k['pubkey'] for k in tx['transaction']['message']['accountKeys']]
    owners = [b['owner'] for b in tx['meta']['preTokenBalances'] + tx['meta']['postTokenBalances']]
    return set(keys) | set(owners)


class FakeIntel:
    """Deterministic provider double with the same method contract as backpack.providers.Providers."""
    def __init__(self, holdings, history=(), parses=None, price_table=None, now=NOW, slot=900):
        self.env, self.usage, self.calls = {}, defaultdict(int), defaultdict(int)
        self.holdings, self.history, self.parses = holdings, list(history), parses or {}
        self.price_table = price_table if price_table is not None else {BP_MINT: '1.3', TOKEN: '2', USDC: '1', pf.WSOL: '150'}
        self.now, self.slot, self.pages = now, slot, 0
    def remaining_seconds(self): return 10_000
    def rpc_many(self, calls):
        self.usage['Helius RPC'] += 1
        self.calls['Helius RPC'] += len(calls)
        out = []
        for method, params in calls:
            owner = params[0]
            if method == 'getBalance': out.append((dict(context=dict(slot=self.slot), value=2 * 10**9), None))
            else:
                rows = [account(f'{owner}-{mint}', mint, owner, raw, dec) for mint, raw, dec, prog in self.holdings.get(owner, [])
                        if prog == params[1]['programId']]
                out.append((dict(context=dict(slot=self.slot), value=rows), None))
        return out
    def assets(self, mints):
        self.usage['Helius RPC'] += 1
        return {m: dict(id=m, interface='FungibleToken', token_info=dict(symbol=m[:6], decimals=6), content=dict(metadata=dict(name=m)))
                for m in mints}
    def prices(self, mints):
        self.usage['Jupiter'] += 1
        return {m: dict(usdPrice=self.price_table[m], blockId=801 if m == pf.WSOL else 800, liquidity=12345.5, createdAt='2026-09-24T00:00:00Z')
                for m in mints if m in self.price_table}
    def block_times(self, slots): return {int(s): self.now - timedelta(minutes=3) for s in slots}
    def transactions_for_address(self, address, filters, token=None, limit=100, details='full'):
        self.usage['Helius RPC'] += 1
        self.pages += 1
        # tokenAccounts='none' sees only transactions naming the address; 'balanceChanged' adds owned token accounts.
        match = keys_of if filters.get('tokenAccounts') == 'none' else involved
        txs = sorted((tx for tx in self.history if address in match(tx)), key=lambda t: -t['slot'])
        bounds = filters.get('slot') or {}
        if 'gte' in bounds: txs = [t for t in txs if t['slot'] >= bounds['gte']]
        if 'lte' in bounds: txs = [t for t in txs if t['slot'] <= bounds['lte']]
        if 'gte' in (filters.get('blockTime') or {}): txs = [t for t in txs if t['blockTime'] >= filters['blockTime']['gte']]
        start = int(token or 0)
        page = txs[start:start + limit]
        if details == 'signatures':
            page = [dict(signature=t['transaction']['signatures'][0], slot=t['slot'], blockTime=t['blockTime']) for t in page]
        return page, (str(start + limit) if start + limit < len(txs) else None)
    def request(self, provider, method, url, **kwargs):
        self.usage[provider] += 1
        address = url.split('/addresses/')[1].split('/')[0]  # Enhanced address history: account-key matches only
        rows = sorted((t for t in self.history if address in keys_of(t)), key=lambda t: -t['slot'])
        if kwargs['params'].get('before'): return []
        return [dict(signature=t['transaction']['signatures'][0], timestamp=t['blockTime']) for t in rows]
    def rpc_paced(self, method, params): return self.rpc(method, params)
    def rpc(self, method, params, independent=False):
        if method == 'getSlot': return self.slot + 100  # price blocks (800) sit a few hundred slots behind the tip
        if method == 'getBlockTime': return int(self.now.timestamp())
        return next(t for t in self.history if t['transaction']['signatures'][0] == params[0])
    def enhanced_parse(self, signatures):
        self.usage['Helius Enhanced'] += 1
        return {s: self.parses[s] for s in signatures if s in self.parses}


def web_templates(name='apps/web/lib/server/bp-intel-store.ts'):
    return re.findall(r'sql`([^`]+)`', Path(name).read_text())


def run_template(conn, template, values):
    params = []
    def bind(match):
        params.append(values[match.group(1)])
        return '%s'
    return fetch_all(conn, re.sub(r'\$\{([^}]+)\}', bind, template.replace('%', '%%')), tuple(params))


def values(version=None, run=None, mint=None, wallet=None):
    return dict(v=version, r=run, mint=mint, wallet=wallet, days=7, threshold=100, cohortId=version, runId=run)


BUYS = [('w1', 90), ('w2', 80), ('w3', 70)]


def history():
    txs = [make_tx(sig=f'buy-{w}', signers=(w,), natives={w: (10**10, 10**10 - 5000)}, programs=[JUP], slot=1000 + i,
                   block_time=NOW - timedelta(minutes=m),
                   tokens=[token(f'{w}-usdc', USDC, w, 1000_000000, 700_000000), token(f'{w}-tok', TOKEN, w, 0, 150_000000)])
           for i, (w, m) in enumerate(BUYS)]
    early = make_tx(sig='before-tracking', signers=('w1',), natives={'w1': (10**10, 10**10 - 5000)}, programs=[JUP], slot=950,
                    block_time=NOW - timedelta(hours=3), tokens=[token('w1-usdc', USDC, 'w1', 2000_000000, 1000_000000),
                                                              token('w1-other', UNPRICED, 'w1', 0, 10)])
    gift = make_tx(sig='gift', signers=('sender',), natives={'sender': (10**9, 10**9 - 5000)}, slot=1010,
                   block_time=NOW - timedelta(minutes=30), tokens=[token('w1-gift', UNPRICED, 'w1', 0, 99)])
    return txs + [early, gift]


def holdings():
    t = pf.TOKEN_PROGRAM
    return {'w1': [(BP_MINT, 5000 * 10**9, 9, t), (TOKEN, 10**18 + 7, 6, t), (USDC, 700_000000, 6, t)],
            'w2': [(TOKEN, 150_000000, 6, pf.TOKEN_2022)], 'w3': [(UNPRICED, 100, 6, t)]}


def test_cohort_versions_are_reproducible_and_hysteresis_persists(conn):
    day = NOW.date()
    bp_capture(conn, day - timedelta(days=1), dict(w1=5000, w2=4000, w3=3000, sys=9000, w4=10), excluded={'sys'})
    first = cohorts.refresh_cohort(conn, ENV)
    assert first['status'] == 'created' and first['bootstrap'] and first['size'] == 3
    assert cohorts.refresh_cohort(conn, ENV)['status'] == 'already_current'
    raw = fetch_all(conn, "SELECT wallet_address,raw_balance FROM bp_holder_rankings WHERE ranking='raw' ORDER BY rank")
    assert [r['wallet_address'] for r in raw][:2] == ['sys', 'w1'] and raw[1]['raw_balance'] == 5000 * 10**9
    assert 'sys' not in {r['wallet_address'] for r in fetch_all(conn, "SELECT * FROM bp_holder_rankings WHERE ranking='filtered'")}
    # Next capture: w2 slips to rank 4 (inside the exit band), w3 to rank 5 (first strike); two newcomers compete for one slot.
    bp_capture(conn, day, dict(w1=5000, n1=4500, n2=4200, w2=4000, w3=3000, sys=9000), excluded={'sys'})
    second = cohorts.refresh_cohort(conn, ENV)
    assert (second['entered'], second['queued'], second['entrant_cap_bound'], second['size']) == (1, 1, True, 4)
    state = {r['wallet_address']: r for r in fetch_all(conn, 'SELECT * FROM bp_cohort_members WHERE version_id=%s', (second['version_id'],))}
    assert state['w3']['member'] and state['w3']['below_exit_runs'] == 1 and state['n2']['event'] == 'queued'
    tracked = {r['wallet_address'] for r in fetch_all(conn, 'SELECT * FROM bp_tracked_wallets WHERE active')}
    assert tracked == {'w1', 'w2', 'w3', 'n1'}


def test_labelled_market_maker_leaves_the_cohort_at_the_next_capture(conn):
    day = NOW.date()
    bp_capture(conn, day - timedelta(days=1), dict(w1=5000, mm=4500, w2=4000, w3=3000))
    assert cohorts.refresh_cohort(conn, ENV)['size'] == 3
    bp_capture(conn, day, dict(w1=5000, mm=4500, w2=4000, w3=3000), labels=dict(mm=('Market Maker', 'high')))
    second = cohorts.refresh_cohort(conn, ENV)
    state = {r['wallet_address']: r for r in fetch_all(conn, 'SELECT * FROM bp_cohort_members WHERE version_id=%s', (second['version_id'],))}
    assert state['mm']['exit_reason'] == 'excluded_by_label' and state['w3']['event'] == 'entered'
    ranked = {(r['ranking'], r['wallet_address']): r for r in fetch_all(conn, 'SELECT * FROM bp_holder_rankings WHERE source_date=%s', (day,))}
    assert ranked[('raw', 'mm')]['excluded'] and ranked[('raw', 'mm')]['label'] == 'Market Maker' and ('filtered', 'mm') not in ranked
    versions = fetch_all(conn, "SELECT excluded_count,exclusion_fingerprint FROM bp_cohorts WHERE kind='current' ORDER BY source_date")
    assert [v['excluded_count'] for v in versions] == [0, 1] and versions[0]['exclusion_fingerprint'] != versions[1]['exclusion_fingerprint']


def test_committed_labels_never_overwrite_an_admin_label(conn, tmp_path):
    from backpack.registry import seed_wallet_labels
    evidence = tmp_path / 'labels.json'
    evidence.write_text("""{"reviewed_at":"2026-09-27T16:00:00Z","labels":[
      {"wallet_address":"GySFHFS5ZiN4Z5YnyPZcjjxpYcGvD7qHZYVjE9QzMHVH","label":"Treasury","entity":"BP vault","confidence":"high",
       "source":"https://solscan.io/account/GySFHFS5ZiN4Z5YnyPZcjjxpYcGvD7qHZYVjE9QzMHVH","notes":"Squads vault"},
      {"wallet_address":"BM9CcyErJcu2mjrFvUsRRrD3snGeHDDVirJLvL6EjvMN","label":"Market Maker","entity":"Operator","confidence":"high",
       "source":"https://solscan.io/account/BM9CcyErJcu2mjrFvUsRRrD3snGeHDDVirJLvL6EjvMN","notes":"Automated trading"}]}""")
    with conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO backpack_wallet_labels VALUES('BM9CcyErJcu2mjrFvUsRRrD3snGeHDDVirJLvL6EjvMN','Unknown','Reviewed by admin',
            'low','https://example.test/admin',now(),'admin decision')""")
    first = seed_wallet_labels(conn, evidence)
    assert (first['labels_seeded'], first['already_labelled']) == (1, 1)
    assert seed_wallet_labels(conn, evidence)['labels_seeded'] == 0  # replays change nothing
    labels = {r['wallet_address']: r for r in fetch_all(conn, 'SELECT * FROM backpack_wallet_labels')}
    assert labels['BM9CcyErJcu2mjrFvUsRRrD3snGeHDDVirJLvL6EjvMN']['label'] == 'Unknown'
    assert labels['GySFHFS5ZiN4Z5YnyPZcjjxpYcGvD7qHZYVjE9QzMHVH']['verified_at'] == datetime(2026, 9, 27, 16, tzinfo=UTC)
    revisions = fetch_all(conn, 'SELECT wallet_address,actor FROM backpack_wallet_label_revisions')
    assert revisions == [dict(wallet_address='GySFHFS5ZiN4Z5YnyPZcjjxpYcGvD7qHZYVjE9QzMHVH', actor='committed_evidence')]


def test_partial_holder_rows_withhold_the_ranking(conn):
    bp_capture(conn, NOW.date(), dict(w1=5, w2=4, w3=3), retained=2)
    result = cohorts.refresh_cohort(conn, ENV)
    assert result['status'] == 'unavailable' and 'partial list' in result['detail']
    assert not fetch_all(conn, 'SELECT * FROM bp_cohorts')


def test_original_cohort_is_approved_once_and_stays_tracked(conn):
    bp_capture(conn, NOW.date() - timedelta(days=2), dict(w1=5, w2=4, w3=3))
    version = cohorts.refresh_cohort(conn, ENV)['version_id']
    with pytest.raises(ValueError): cohorts.approve_original(conn, version, 'admin', '')
    original = cohorts.approve_original(conn, version, 'admin', 'Reviewed top holders against labels')
    with pytest.raises(ValueError): cohorts.approve_original(conn, version, 'admin', 'again')
    assert cohorts.members(conn, original) == {'w1', 'w2', 'w3'}
    # w3 sells out twice; it leaves the current cohort but the original keeps it tracked.
    for offset in (1, 0):
        bp_capture(conn, NOW.date() - timedelta(days=offset), dict(w1=5, w2=4, x=3))
        cohorts.refresh_cohort(conn, ENV)
    current = fetch_all(conn, "SELECT version_id FROM bp_cohorts WHERE kind='current' ORDER BY source_date DESC LIMIT 1")[0]['version_id']
    assert 'w3' not in cohorts.members(conn, current)
    assert 'w3' in {r['wallet_address'] for r in cohorts.tracked_wallets(conn)}


def test_admin_approval_sql_matches_the_collector(conn):
    bp_capture(conn, NOW.date(), dict(w1=5, w2=4, w3=3))
    version = cohorts.refresh_cohort(conn, ENV)['version_id']
    source = Path('apps/web/lib/server/bp-intel-admin.ts').read_text()
    [statement] = re.findall(r'sql`(WITH original[^`]+)`', source)
    run_template(conn, statement, dict(version=version, actor='authenticated_admin', notes='Reviewed'))
    conn.commit()
    original = fetch_all(conn, "SELECT * FROM bp_cohorts WHERE kind='original'")
    assert len(original) == 1 and original[0]['derived_from_version'] == version and original[0]['approval_notes'] == 'Reviewed'
    assert cohorts.members(conn, original[0]['version_id']) == {'w1', 'w2', 'w3'}
    assert run_template(conn, statement, dict(version=version, actor='a', notes='again')) == []  # never a second original


def test_worker_end_to_end_is_idempotent(conn):
    bp_capture(conn, NOW.date(), dict(w1=5000, w2=4000, w3=3000, w4=10))
    parses = {'buy-w1': enhanced_swap('w1', [USDC], [TOKEN])}
    p = FakeIntel(holdings(), history(), parses)
    result = intel.run(conn, p, ENV, now=NOW)
    assert result['status'] == 'completed', result
    assert result['portfolio']['complete'] == 3 and result['history']['polled'] == 3
    balances = {(r['wallet_address'], r['mint']): r for r in fetch_all(conn, 'SELECT * FROM bp_portfolio_balances')}
    assert balances[('w1', TOKEN)]['raw_amount'] == 10**18 + 7  # exact, never a float
    assert balances[('w1', TOKEN)]['value_usd'] == D(10**18 + 7) / D(10**6) * 2
    assert balances[('w3', UNPRICED)]['value_usd'] is None and balances[('w3', UNPRICED)]['pricing_status'] == 'unpriced'
    assert balances[('w1', pf.NATIVE)]['value_usd'] == D(300)
    # Token price times are estimated from slots; SOL's stays exact because stored SOL prices value swaps.
    assert 'estimated from slot' in balances[('w1', TOKEN)]['price_source']
    assert 'estimated' not in balances[('w1', pf.NATIVE)]['price_source']
    assert len(fetch_all(conn, 'SELECT * FROM bp_price_observations WHERE mint=%s', (pf.WSOL,))) == 1
    events = {(r['signature'], r['wallet_address']): r for r in fetch_all(conn, 'SELECT * FROM bp_economic_events')}
    assert events[('buy-w1', 'w1')]['tier'] == 'parsed_swap' and events[('buy-w2', 'w2')]['tier'] == 'inferred_swap'
    assert events[('buy-w1', 'w1')]['usd_value'] == D(300) and events[('buy-w1', 'w1')]['first_observed_purchase']
    assert events[('before-tracking', 'w1')]['pre_membership'] and not events[('buy-w1', 'w1')]['pre_membership']
    assert events[('gift', 'w1')]['kind'] == 'transfer_in'
    alerts = {r['rule']: r for r in fetch_all(conn, 'SELECT * FROM bp_alerts')}
    assert alerts['multiple_buyers']['wallet_count'] == 3 and alerts['multiple_buyers']['inferred_wallets'] == 2
    assert alerts['multiple_buyers']['lowest_tier'] == 'inferred_swap' and alerts['new_position']['wallet_count'] == 3
    assert 'accumulation' not in alerts  # three buyers is below the five-wallet rule
    coverage = fetch_all(conn, 'SELECT * FROM bp_history_coverage ORDER BY wallet_address')
    assert {c['backfill_status'] for c in coverage} == {'complete'}
    counts = [len(fetch_all(conn, f'SELECT * FROM {t}')) for t in ('bp_economic_events', 'bp_raw_transactions', 'bp_alerts')]
    usage = fetch_all(conn, 'SELECT * FROM backpack_provider_usage u JOIN backpack_ingestion_runs r USING(run_id) WHERE r.job=%s', ('bp_intel',))
    assert {u['provider'] for u in usage} >= {'Helius RPC', 'Jupiter'}
    # Replaying the same deliveries and re-polling creates nothing new; the alert is updated in place.
    again = intel.run(conn, FakeIntel(holdings(), history(), parses), ENV, now=NOW + timedelta(minutes=5))
    assert again['status'] == 'completed'
    assert [len(fetch_all(conn, f'SELECT * FROM {t}')) for t in ('bp_economic_events', 'bp_raw_transactions', 'bp_alerts')] == counts
    # The daily capture's readers stay scoped to daily runs.
    assert {r['job'] for r in fetch_all(conn, 'SELECT job FROM backpack_ingestion_runs')} == {'daily', 'bp_intel'}
    # Web templates execute against the populated schema, and overlap traces to balances.
    version = fetch_all(conn, "SELECT version_id FROM bp_cohorts WHERE kind='current'")[0]['version_id']
    run = fetch_all(conn, "SELECT run_id FROM backpack_ingestion_runs WHERE job='bp_intel' ORDER BY started_at DESC LIMIT 1")[0]['run_id']
    for template in web_templates():
        for scope in (values(version, run), values(version, run, TOKEN, 'w1')):
            run_template(conn, template, scope)
    [overlap] = [t for t in web_templates() if 'WITH members AS' in t]
    rows = {r['mint']: r for r in run_template(conn, overlap, values(version, run))}
    assert rows[TOKEN]['holders'] == 2 and rows[TOKEN]['meaningful_holders'] == 2 and rows[TOKEN]['cohort_size'] == 3
    assert rows[TOKEN]['new_buyers_24h'] == 3 and rows[TOKEN]['buyers_7d'] == 3 and rows[TOKEN]['inferred_purchases_7d'] == 2
    assert rows[UNPRICED]['unpriced_holders'] == 1 and rows[UNPRICED]['combined_value_usd'] is None
    assert abs(rows[TOKEN]['largest_owner_share'] - D(10**18 + 7) / D(10**18 + 7 + 150_000000)) < D('1e-15')


def test_web_templates_execute_on_an_empty_schema(conn):
    for template in web_templates():
        run_template(conn, template, values())
    # Explicit ids outside the listed window are looked up directly; unknown ids find nothing (-> 404 not_found).
    [by_version] = [t for t in web_templates() if 'version_id=${cohortId}' in t]
    [by_run] = [t for t in web_templates() if 'WHERE r.run_id=${runId}::uuid' in t]
    assert run_template(conn, by_version, dict(cohortId=999)) == []
    assert run_template(conn, by_run, dict(runId=str(uuid.uuid4()))) == []


def test_history_page_cap_is_recorded_not_silent(conn):
    bp_capture(conn, NOW.date(), dict(w1=5, w2=4, w3=3))
    txs = [make_tx(sig=f'noise-{i}', signers=('w1',), natives={'w1': (10**9, 10**9 - 5000)}, slot=2000 + i, block_time=NOW - timedelta(minutes=100 - i),
                   tokens=[token('w1-a', TOKEN, 'w1', i, i + 1)]) for i in range(6)]
    env = dict(ENV, BP_HISTORY_POLL_PAGES='1', BP_HISTORY_BACKFILL_PAGES_PER_RUN='1', BP_HISTORY_PAGE_SIZE='2', BP_HISTORY_MAX_BACKFILL_PAGES='2')
    intel.run(conn, FakeIntel({}, txs), env, steps=('history',), now=NOW)
    first = fetch_all(conn, "SELECT * FROM bp_history_coverage WHERE wallet_address='w1'")[0]
    assert first['backfill_status'] == 'incomplete' and first['earliest_slot'] == 2004 and first['newest_slot'] == 2005
    intel.run(conn, FakeIntel({}, txs + [make_tx(sig=f'new-{i}', signers=('w1',), natives={'w1': (10**9, 10**9 - 5000)}, slot=3000 + i,
              block_time=NOW - timedelta(minutes=5), tokens=[token('w1-a', TOKEN, 'w1', 10, 11)]) for i in range(3)]), env,
              steps=('history',), now=NOW + timedelta(hours=1))
    second = fetch_all(conn, "SELECT * FROM bp_history_coverage WHERE wallet_address='w1'")[0]
    assert second['backfill_status'] == 'capped' and second['poll_status'] == 'capped'
    assert second['gaps'] and 'page cap' in second['gaps'][0]['reason']


def test_reconciliation_flags_unexplained_balance_changes(conn):
    bp_capture(conn, NOW.date(), dict(w1=5, w2=4, w3=3))
    cohorts.refresh_cohort(conn, ENV)
    wallets = cohorts.tracked_wallets(conn)
    for when, slot, token_raw, other_raw in ((NOW - timedelta(hours=25), 900, 10, 7), (NOW, 1000, 15, 9)):
        run_id = str(uuid.uuid4())
        with conn, conn.cursor() as cur:
            cur.execute("INSERT INTO backpack_ingestion_runs(run_id,snapshot_date,started_at,job) VALUES(%s,%s,%s,'bp_intel')", (run_id, when.date(), when))
        hold = {'w1': [(TOKEN, token_raw, 6, pf.TOKEN_PROGRAM), (UNPRICED, other_raw, 6, pf.TOKEN_PROGRAM)]}
        intel.collect_portfolios(conn, FakeIntel(hold, slot=slot), ENV, run_id, wallets, when)
    with conn, conn.cursor() as cur:
        cur.execute('''INSERT INTO bp_history_coverage(wallet_address,source,requested_start,newest_slot,backfill_status)
            VALUES('w1','fixture',%s,1000,'complete')''', (NOW - timedelta(days=30),))
        cur.execute('''INSERT INTO bp_economic_events(signature,wallet_address,event_index,slot,block_time,kind,tier,output_mint,output_raw,
            output_decimals,valuation_status,pre_membership,owner_signed,parser_version,finality)
            VALUES('rx',  'w1',0,950,%s,'transfer_in','not_applicable',%s,5,6,'not_applicable',false,false,'fixture','finalized')''',
            (NOW - timedelta(hours=12), TOKEN))
    result = intel.reconcile(conn, NOW.date())
    rows = {r['mint']: r for r in fetch_all(conn, 'SELECT * FROM bp_reconciliation')}
    assert result['discrepancies'] == 1
    assert rows[TOKEN]['status'] == 'matched' and rows[UNPRICED]['status'] == 'discrepancy' and rows[UNPRICED]['difference'] == 2
    assert intel.reconcile(conn, NOW.date())['status'] == 'already_reconciled'


def test_retention_thins_hourly_reads_but_keeps_one_per_day(conn):
    bp_capture(conn, NOW.date(), dict(w1=5, w2=4, w3=3))
    cohorts.refresh_cohort(conn, ENV)
    wallets = cohorts.tracked_wallets(conn)
    day = datetime.combine(NOW.date() - timedelta(days=5), datetime.min.time(), UTC)
    stamps = [day + timedelta(hours=1), day + timedelta(hours=2), NOW - timedelta(hours=2), NOW - timedelta(hours=1)]
    for when in stamps:
        run_id = str(uuid.uuid4())
        with conn, conn.cursor() as cur:
            cur.execute("INSERT INTO backpack_ingestion_runs(run_id,snapshot_date,started_at,job) VALUES(%s,%s,%s,'bp_intel')", (run_id, when.date(), when))
        intel.collect_portfolios(conn, FakeIntel(holdings()), ENV, run_id, wallets, when)
    intel.maintain(conn, {}, NOW)
    kept = fetch_all(conn, '''SELECT DISTINCT r.started_at FROM bp_portfolio_balances b JOIN backpack_ingestion_runs r USING(run_id)
        ORDER BY 1''')
    # Equally complete reads: the latest of the old day survives, and both recent reads are untouched.
    assert [k['started_at'] for k in kept] == [stamps[1], stamps[2], stamps[3]]
    with pytest.raises(ValueError): intel.maintain(conn, {'BP_RAW_TX_RETENTION_DAYS': '7'}, NOW)


def read_at(conn, wallets, when, succeeded, provider=None):
    run_id = str(uuid.uuid4())
    with conn, conn.cursor() as cur:
        cur.execute('''INSERT INTO backpack_ingestion_runs(run_id,snapshot_date,started_at,job,assets_succeeded)
            VALUES(%s,%s,%s,'bp_intel',%s)''', (run_id, when.date(), when, succeeded))
    intel.collect_portfolios(conn, provider or FakeIntel(holdings()), ENV, run_id, wallets, when)
    return run_id


def test_retention_keeps_the_most_complete_read_and_never_the_page_s_last_read(conn):
    bp_capture(conn, NOW.date(), dict(w1=5, w2=4, w3=3))
    cohorts.refresh_cohort(conn, ENV)
    wallets = cohorts.tracked_wallets(conn)
    day = datetime.combine(NOW.date() - timedelta(days=5), datetime.min.time(), UTC)
    # 2026-09-27 in production: the day's first read was the poor one. Then refreshes stopped for five days.
    poor = read_at(conn, wallets, day + timedelta(hours=1), 1)
    best = read_at(conn, wallets, day + timedelta(hours=2), 3)
    last = read_at(conn, wallets, day + timedelta(hours=3), 2)
    intel.maintain(conn, {}, NOW)
    kept = {r['run_id'] for r in fetch_all(conn, 'SELECT DISTINCT run_id::text FROM bp_portfolio_balances')}
    assert kept == {best, last}, 'the most complete read of the day plus the newest read, which the page shows'
    assert {r['run_id'] for r in fetch_all(conn, 'SELECT DISTINCT run_id::text FROM bp_portfolio_wallets')} == {best, last}
    intel.maintain(conn, {'BP_PORTFOLIO_RETENTION_DAYS': '30'}, NOW + timedelta(days=60))
    kept = {r['run_id'] for r in fetch_all(conn, 'SELECT DISTINCT run_id::text FROM bp_portfolio_balances')}
    assert kept == {last}, 'beyond retention only the newest read survives, so a stalled refresh never empties the page'
    assert poor


class MalformedFor(FakeIntel):
    """Answers one wallet's token-account read with a list where the RPC returns an object."""
    def __init__(self, bad, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bad = bad
    def rpc_many(self, calls):
        out = super().rpc_many(calls)
        return [(['unexpected'], None) if params[0] == self.bad and method == 'getTokenAccountsByOwner' else row
                for (method, params), row in zip(calls, out)]


def test_one_malformed_wallet_response_is_that_wallet_s_read_not_the_run(conn):
    bp_capture(conn, NOW.date(), dict(w1=5, w2=4, w3=3))
    cohorts.refresh_cohort(conn, ENV)
    wallets = cohorts.tracked_wallets(conn)
    run_id = read_at(conn, wallets, NOW, 0, MalformedFor('w2', holdings()))
    reads = {r['wallet_address']: r for r in fetch_all(conn, 'SELECT * FROM bp_portfolio_wallets WHERE run_id=%s', (run_id,))}
    assert reads['w2']['status'] == 'unavailable'
    assert reads['w2']['detail'].startswith("unreadable provider response: AttributeError ('list' object has no attribute 'get') at portfolio.py:")
    assert reads['w1']['status'] == 'complete' and reads['w3']['status'] == 'complete'


def test_worker_never_runs_ddl_on_an_unmigrated_database(database):
    schema = 'bp_intel_bare_' + uuid.uuid4().hex
    with database, database.cursor() as cur:
        cur.execute('CREATE SCHEMA ' + schema)
        cur.execute('SET search_path TO ' + schema)
    try:
        assert intel.run(database, FakeIntel({}), ENV)['status'] == 'schema_pending'
        assert fetch_all(database, "SELECT count(*) AS n FROM information_schema.tables WHERE table_schema=%s", (schema,))[0]['n'] == 0
    finally:
        database.rollback()
        with database, database.cursor() as cur: cur.execute('DROP SCHEMA ' + schema + ' CASCADE')


def test_feasibility_probe_reports_token_account_coverage_without_writing(conn):
    from backpack import feasibility
    bp_capture(conn, NOW.date(), dict(w1=5000, w2=4000, w3=3000))
    cohorts.refresh_cohort(conn, ENV)
    before = {t: len(fetch_all(conn, f'SELECT * FROM {t}')) for t in ('bp_portfolio_balances', 'bp_economic_events', 'bp_raw_transactions')}
    report = feasibility.run(conn, ENV, sample=1, portfolios=3, sizes=3, p=FakeIntel(holdings(), history()))
    assert report['status'] == 'completed' and report['holder_source'].startswith('bp_holder_rankings')
    [compare] = report['items']['history_endpoints']
    # The gift reached w1's token account without naming w1: only the token-account query returns it.
    assert compare['token_account_only'] == 1 and compare['token_account_only_in_enhanced'] == 0
    assert compare['token_account_evidence'][0] == dict(signature='gift', owner_in_account_keys=False, in_enhanced_history=False)
    assert report['items']['swap_programs']['known'] == {'Jupiter Aggregator v6': 2}
    assert report['items']['prices']['priced'] == 3 and report['items']['metadata']['distinct_mints'] == 4
    assert report['items']['webhooks']['status'] == 'not_tested' and report['items']['usage_estimate']['cohort'] == 3
    assert {t: len(fetch_all(conn, f'SELECT * FROM {t}')) for t in before} == before


def test_wallet_without_history_keeps_being_polled(conn):
    bp_capture(conn, NOW.date(), dict(w1=5, w2=4, w3=3))
    intel.run(conn, FakeIntel({}, []), ENV, steps=('history',), now=NOW)
    first = fetch_all(conn, "SELECT * FROM bp_history_coverage WHERE wallet_address='w1'")[0]
    assert first['backfill_status'] == 'complete' and first['newest_slot'] is None and first['transactions'] == 0
    later = make_tx(sig='later', signers=('w1',), natives={'w1': (10**9, 10**9 - 5000)}, slot=5000, block_time=NOW - timedelta(minutes=10),
                    tokens=[token('w1-a', TOKEN, 'w1', 0, 5)])
    for offset in (1, 2):  # the second poll re-reads the newest slot; the counter counts new transactions only
        intel.run(conn, FakeIntel({}, [later]), ENV, steps=('history',), now=NOW + timedelta(hours=offset))
    after = fetch_all(conn, "SELECT * FROM bp_history_coverage WHERE wallet_address='w1'")[0]
    assert after['newest_slot'] == 5000 and after['transactions'] == 1
    assert [e['kind'] for e in fetch_all(conn, "SELECT kind FROM bp_economic_events WHERE signature='later'")] == ['transfer_in']


def test_metadata_is_marked_checked_only_after_a_successful_lookup(conn):
    from backpack.providers import SourceError
    bp_capture(conn, NOW.date(), dict(w1=5, w2=4, w3=3))
    cohorts.refresh_cohort(conn, ENV)
    wallets = cohorts.tracked_wallets(conn)
    class Partial(FakeIntel):  # DAS knows every mint except UNPRICED; records what it is asked for
        asked = []
        def assets(self, mints):
            Partial.asked.append(sorted(mints))
            return {m: a for m, a in FakeIntel.assets(self, mints).items() if m != UNPRICED}
    class Down(FakeIntel):
        def assets(self, mints): raise SourceError('Helius DAS: HTTP 429')
    def read(provider, when):
        run_id = str(uuid.uuid4())
        with conn, conn.cursor() as cur:
            cur.execute("INSERT INTO backpack_ingestion_runs(run_id,snapshot_date,started_at,job) VALUES(%s,%s,%s,'bp_intel')", (run_id, when.date(), when))
        return intel.collect_portfolios(conn, provider, ENV, run_id, wallets, when)
    read(Down(holdings()), NOW - timedelta(hours=3))  # a failed lookup marks nothing as checked
    assert all(r['metadata_at'] is None for r in fetch_all(conn, "SELECT metadata_at FROM bp_tracked_assets WHERE mint<>'native'"))
    read(Partial(holdings()), NOW - timedelta(hours=2))
    unknown = fetch_all(conn, 'SELECT * FROM bp_tracked_assets WHERE mint=%s', (UNPRICED,))[0]
    assert unknown['metadata_at'] is not None and 'no asset returned' in unknown['metadata_source']
    read(Partial(holdings()), NOW - timedelta(hours=1))
    assert len(Partial.asked) == 1  # nothing is stale an hour later, so DAS is not asked again
