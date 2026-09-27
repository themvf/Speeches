"""BP holder cohorts: rankings from stored holder snapshots, hysteresis membership, immutable versions.

Rankings come only from a complete, supply-reconciled BP holder capture already in the database, so every
version is reproducible from stored observations. Pure functions first; persistence at the bottom.
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json
from .metrics import BP_MINT, number

UTC = timezone.utc
METHODOLOGY = ('cohort-v1: owner-aggregated raw BP balances from one complete, supply-reconciled capture; '
               'ties broken by owner address. Filtered ranking excludes confirmed/high system labels recorded '
               'with that capture and, unless disabled, confirmed/high Market Maker labels. Enter at filtered rank <= size; leave after exit_runs consecutive versions '
               'ranked below exit_rank or absent; labelled exclusions leave immediately. Entrants beyond the '
               'cap are queued, oldest first. An observation window, not a point-in-time state.')


def config(env):
    values = {key: int(env.get(name, default)) for key, name, default in (
        ('size', 'BP_COHORT_SIZE', '200'), ('exit_rank', 'BP_COHORT_EXIT_RANK', '220'),
        ('exit_runs', 'BP_COHORT_EXIT_RUNS', '2'), ('entrant_cap', 'BP_COHORT_ENTRANT_CAP', '20'),
        ('history_days', 'BP_HISTORY_DAYS', '30'))}
    if values['size'] < 1 or values['exit_rank'] < values['size'] or values['exit_runs'] < 1 \
            or values['entrant_cap'] < 1 or not 1 <= values['history_days'] <= 90:
        raise ValueError('Invalid cohort configuration')
    # Market makers stay in the securities adoption metrics; they are not investors, so the cohort drops them.
    values['exclude_market_makers'] = env.get('BP_COHORT_EXCLUDE_MARKET_MAKERS', '1').strip().lower() not in ('0', 'false', 'no')
    return values


def cohort_excluded(row, cfg):
    """System labels excluded at capture, plus confirmed/high Market Maker labels recorded with the same capture."""
    return bool(row['excluded'] or (cfg.get('exclude_market_makers') and row.get('label') == 'Market Maker'
                                    and row.get('label_confidence') in ('confirmed', 'high')))


def raw_units(balance_tokens, decimals):
    """Stored balances are exact decimals of raw integers; refuse anything that is not."""
    value = number(balance_tokens)
    if value is None or value < 0: raise ValueError('Invalid stored holder balance')
    raw = value * Decimal(10) ** int(decimals)
    if raw != raw.to_integral_value(): raise ValueError('Stored holder balance is not a whole raw amount')
    return int(raw)


def rank_owners(rows, decimals):
    """Raw ranking over every nonzero owner and filtered ranking over eligible owners."""
    owners = {}
    for r in rows:
        wallet = r['wallet_address']
        if wallet in owners: raise ValueError('Duplicate owner in holder snapshot')
        raw = raw_units(r['balance_tokens'], decimals)
        if raw > 0: owners[wallet] = dict(r, raw_balance=raw)
    ordered = sorted(owners.values(), key=lambda r: (-r['raw_balance'], r['wallet_address']))
    raw_ranking = [dict(r, rank=i + 1) for i, r in enumerate(ordered)]
    filtered = [dict(r, rank=i + 1) for i, r in enumerate(x for x in ordered if not x['excluded'])]
    return raw_ranking, filtered


def exclusion_fingerprint(rows):
    excluded = sorted([r['wallet_address'], r.get('label'), r.get('label_confidence')] for r in rows if r['excluded'])
    return sha256(json.dumps(excluded, separators=(',', ':')).encode()).hexdigest()


def next_membership(filtered, excluded_wallets, previous, cfg, now):
    """Return member/queued/left rows for the next version.

    previous maps wallet -> prior version row (member or queued) or is None for the first version.
    """
    ranks = {r['wallet_address']: r for r in filtered}
    rows = []
    if previous is None:
        for r in filtered[:cfg['size']]:
            rows.append(dict(wallet_address=r['wallet_address'], member=True, rank=r['rank'], raw_balance=r['raw_balance'],
                             previous_rank=None, event='bootstrap', below_exit_runs=0, queued_since=None, exit_reason=None))
        return rows, dict(bootstrap=True, entrant_cap_bound=False)
    members = {w: p for w, p in previous.items() if p['member']}
    for wallet in sorted(members):
        prior, current = members[wallet], ranks.get(wallet)
        rank = current['rank'] if current else None
        base = dict(wallet_address=wallet, rank=rank, raw_balance=current['raw_balance'] if current else None,
                    previous_rank=prior['rank'], queued_since=None)
        if wallet in excluded_wallets:
            rows.append(dict(base, member=False, event='left', below_exit_runs=0, exit_reason='excluded_by_label'))
            continue
        runs = 0 if rank is not None and rank <= cfg['exit_rank'] else prior['below_exit_runs'] + 1
        if runs >= cfg['exit_runs']:
            rows.append(dict(base, member=False, event='left', below_exit_runs=runs,
                             exit_reason='no_positive_balance' if rank is None else 'below_exit_rank'))
        else:
            rows.append(dict(base, member=True, below_exit_runs=runs, exit_reason=None,
                             event='rank_changed' if rank != prior['rank'] else 'unchanged'))
    queued = {w: p['queued_since'] for w, p in previous.items() if not p['member'] and p.get('queued_since')}
    candidates = [r for r in filtered if r['rank'] <= cfg['size'] and r['wallet_address'] not in members]
    # Queued wallets keep their place; everyone else follows by rank.
    candidates.sort(key=lambda r: (r['wallet_address'] not in queued, queued.get(r['wallet_address']) or now, r['rank']))
    for i, r in enumerate(candidates):
        admitted = i < cfg['entrant_cap']
        rows.append(dict(wallet_address=r['wallet_address'], member=admitted, rank=r['rank'], raw_balance=r['raw_balance'],
                         previous_rank=None, event='entered' if admitted else 'queued', below_exit_runs=0,
                         queued_since=None if admitted else queued.get(r['wallet_address'], now), exit_reason=None))
    return rows, dict(bootstrap=False, entrant_cap_bound=len(candidates) > cfg['entrant_cap'])


def summarize(rows):
    count = lambda event: sum(r['event'] == event for r in rows)
    return dict(size=sum(r['member'] for r in rows), entered=count('entered'), left_count=count('left'),
                rank_changed=count('rank_changed'), queued=count('queued'))


# Persistence ---------------------------------------------------------------------------------------------------

def _bp_asset(conn):
    from .collector import fetch_all
    rows = fetch_all(conn, "SELECT id FROM backpack_assets WHERE asset_type='bp' AND solana_mint=%s", (BP_MINT,))
    return rows[0]['id'] if rows else None


def refresh_cohort(conn, env, day=None):
    """Create the current-cohort version for the latest complete BP capture. Idempotent per capture date."""
    from psycopg2.extras import execute_values
    from .collector import fetch_all
    cfg = config(env)
    asset_id = _bp_asset(conn)
    if asset_id is None: return dict(status='unavailable', detail='BP asset is not registered')
    snapshots = fetch_all(conn, '''SELECT s.date,s.run_id,s.captured_at,s.decimals,s.unique_holders,s.holder_start_slot,
        s.holder_end_slot,c.aggregates_validated FROM backpack_asset_daily_snapshots s
        JOIN backpack_holder_checkpoints c USING(asset_id,date)
        WHERE s.asset_id=%s AND s.holders_complete AND (%s::date IS NULL OR s.date=%s::date)
        ORDER BY s.date DESC LIMIT 1''', (asset_id, day, day))
    if not snapshots:
        return dict(status='unavailable', detail='No complete, supply-reconciled BP holder capture is stored')
    snap = snapshots[0]
    if not snap['aggregates_validated']:
        return dict(status='unavailable', detail=f"BP capture {snap['date']} failed checkpoint validation; ranking withheld")
    latest = fetch_all(conn, "SELECT version_id,source_date FROM bp_cohorts WHERE kind='current' ORDER BY source_date DESC LIMIT 1")
    if latest and latest[0]['source_date'] >= snap['date']:
        return dict(status='already_current', version_id=latest[0]['version_id'], source_date=str(latest[0]['source_date']))
    holders = fetch_all(conn, '''SELECT wallet_address,balance_tokens,excluded,label,label_confidence,label_entity,label_source
        FROM backpack_asset_holder_daily_snapshots WHERE asset_id=%s AND date=%s''', (asset_id, snap['date']))
    if len(holders) != snap['unique_holders']:
        return dict(status='unavailable', detail=f"Retained holder rows ({len(holders)}) do not match capture owners "
                    f"({snap['unique_holders']}); ranking withheld rather than published from a partial list")
    holders = [dict(r, excluded=cohort_excluded(r, cfg)) for r in holders]
    raw_ranking, filtered = rank_owners(holders, snap['decimals'])
    excluded_wallets = {r['wallet_address'] for r in holders if r['excluded']}
    now = snap['captured_at']
    stored_rank = cfg['exit_rank'] + 30
    with conn, conn.cursor() as cur:
        cur.execute('SELECT pg_advisory_xact_lock(98274032)')
        cur.execute("SELECT version_id FROM bp_cohorts WHERE kind='current' AND source_date>=%s", (snap['date'],))
        if cur.fetchone(): return dict(status='already_current', source_date=str(snap['date']))
        cur.execute('''SELECT m.wallet_address,m.member,m.rank,m.below_exit_runs,m.queued_since FROM bp_cohort_members m
            WHERE m.version_id=(SELECT version_id FROM bp_cohorts WHERE kind='current' ORDER BY source_date DESC LIMIT 1)
            AND (m.member OR m.event='queued')''')
        names = [d[0] for d in cur.description]
        prior_rows = [dict(zip(names, r)) for r in cur.fetchall()]
        previous = {r['wallet_address']: r for r in prior_rows} if latest else None
        rows, flags = next_membership(filtered, excluded_wallets, previous, cfg, now)
        counts = summarize(rows)
        cur.execute('''INSERT INTO bp_cohorts(kind,effective_at,source_date,source_run_id,source_slot_start,source_slot_end,
            exclusion_fingerprint,excluded_count,eligible_count,target_size,exit_rank,exit_runs,entrant_cap,size,entered,
            left_count,rank_changed,queued,entrant_cap_bound,bootstrap,status,methodology)
            VALUES('current',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING version_id''',
            (now, snap['date'], snap['run_id'], snap['holder_start_slot'], snap['holder_end_slot'],
             exclusion_fingerprint(holders), len(excluded_wallets), len(filtered), cfg['size'], cfg['exit_rank'],
             cfg['exit_runs'], cfg['entrant_cap'], counts['size'], counts['entered'], counts['left_count'],
             counts['rank_changed'], counts['queued'], flags['entrant_cap_bound'], flags['bootstrap'],
             'Estimated' if counts['size'] >= min(cfg['size'], len(filtered)) else 'Partial', METHODOLOGY))
        version = cur.fetchone()[0]
        execute_values(cur, '''INSERT INTO bp_cohort_members(version_id,wallet_address,member,rank,raw_balance,previous_rank,
            event,below_exit_runs,queued_since,exit_reason) VALUES %s''',
            [(version, r['wallet_address'], r['member'], r['rank'], r['raw_balance'], r['previous_rank'], r['event'],
              r['below_exit_runs'], r['queued_since'], r['exit_reason']) for r in rows], page_size=1000)
        execute_values(cur, '''INSERT INTO bp_holder_rankings(source_date,ranking,rank,wallet_address,raw_balance,decimals,
            label,label_confidence,label_entity,label_source,excluded) VALUES %s ON CONFLICT DO NOTHING''',
            [(snap['date'], name, r['rank'], r['wallet_address'], r['raw_balance'], snap['decimals'], r.get('label') or 'Unknown',
              r.get('label_confidence'), r.get('label_entity'), r.get('label_source'), r['excluded'])
             for name, ranking in (('raw', raw_ranking), ('filtered', filtered)) for r in ranking[:stored_rank]], page_size=1000)
        entrants = [r['wallet_address'] for r in rows if r['member'] and r['event'] in ('bootstrap', 'entered')]
        if entrants:
            execute_values(cur, '''INSERT INTO bp_tracked_wallets(wallet_address,tracked_since,history_start,first_version_id,active)
                VALUES %s ON CONFLICT(wallet_address) DO NOTHING''',
                [(w, now, now - timedelta(days=cfg['history_days']), version, True) for w in entrants])
        _refresh_active(cur, version)
    return dict(status='created', version_id=version, source_date=str(snap['date']), **counts, **flags)


def _refresh_active(cur, current_version):
    """Tracked = member of the latest current version or of the approved original cohort."""
    cur.execute('''UPDATE bp_tracked_wallets t SET active=x.active,updated_at=now() FROM (
        SELECT w.wallet_address, EXISTS(SELECT 1 FROM bp_cohort_members m JOIN bp_cohorts c USING(version_id)
            WHERE m.wallet_address=w.wallet_address AND m.member AND (m.version_id=%s OR c.kind='original')) AS active
        FROM bp_tracked_wallets w) x WHERE t.wallet_address=x.wallet_address AND t.active<>x.active''', (current_version,))


def approve_original(conn, version_id, actor, notes):
    """Freeze an existing current version as the one original cohort. Never replaces an existing original."""
    if not str(notes or '').strip(): raise ValueError('Approval notes are required')
    with conn, conn.cursor() as cur:
        cur.execute('SELECT pg_advisory_xact_lock(98274032)')
        cur.execute("SELECT 1 FROM bp_cohorts WHERE kind='original'")
        if cur.fetchone(): raise ValueError('The original cohort is already approved and immutable')
        cur.execute(ORIGINAL_INSERT, dict(version=version_id, actor=actor, notes=notes))
        row = cur.fetchone()
        if not row: raise ValueError('Only an existing current version can be approved')
        cur.execute(ORIGINAL_MEMBERS, dict(original=row[0], version=version_id))
        cur.execute("SELECT version_id FROM bp_cohorts WHERE kind='current' ORDER BY source_date DESC LIMIT 1")
        _refresh_active(cur, cur.fetchone()[0])
        return row[0]


# Shared with the admin route (apps/web/lib/server/bp-intel-admin.ts carries the same statements).
ORIGINAL_INSERT = '''INSERT INTO bp_cohorts(kind,effective_at,source_date,source_run_id,source_slot_start,source_slot_end,
    derived_from_version,exclusion_fingerprint,excluded_count,eligible_count,target_size,exit_rank,exit_runs,entrant_cap,
    size,entered,left_count,rank_changed,queued,entrant_cap_bound,bootstrap,status,methodology,approved_at,approved_by,approval_notes)
    SELECT 'original',effective_at,source_date,source_run_id,source_slot_start,source_slot_end,version_id,exclusion_fingerprint,
    excluded_count,eligible_count,target_size,exit_rank,exit_runs,entrant_cap,size,size,0,0,0,false,true,status,
    'Original cohort: frozen copy of an approved current version; members stay tracked after selling BP. '||methodology,
    now(),%(actor)s,%(notes)s FROM bp_cohorts WHERE version_id=%(version)s AND kind='current' RETURNING version_id'''
ORIGINAL_MEMBERS = '''INSERT INTO bp_cohort_members(version_id,wallet_address,member,rank,raw_balance,previous_rank,event,below_exit_runs)
    SELECT %(original)s,wallet_address,true,rank,raw_balance,NULL,'bootstrap',0 FROM bp_cohort_members
    WHERE version_id=%(version)s AND member'''


def tracked_wallets(conn):
    from .collector import fetch_all
    return fetch_all(conn, 'SELECT * FROM bp_tracked_wallets WHERE active ORDER BY wallet_address')


def latest_versions(conn):
    """The versions alerts and flows are evaluated against: the original (if approved) and the latest current."""
    from .collector import fetch_all
    return fetch_all(conn, '''SELECT * FROM bp_cohorts WHERE kind='original'
        UNION ALL (SELECT * FROM bp_cohorts WHERE kind='current' ORDER BY source_date DESC LIMIT 1)''')


def members(conn, version_id):
    from .collector import fetch_all
    return {r['wallet_address'] for r in fetch_all(conn, 'SELECT wallet_address FROM bp_cohort_members WHERE version_id=%s AND member', (version_id,))}


def now_utc():
    return datetime.now(UTC)
