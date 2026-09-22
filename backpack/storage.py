"""Bounded raw retention. Permanent aggregates, provenance and material evidence survive."""
from datetime import timedelta
from decimal import Decimal


def holder_changes(before, after):
    """Price-only moves emit threshold crossings, never token accumulation events."""
    events = []
    if before is None:
        events.append('NEW_HOLDER')
    elif after is None:
        events.append('EXITED_HOLDER')
    elif after['balance_tokens'] != before['balance_tokens']:
        events.append('BALANCE_INCREASE' if after['balance_tokens'] > before['balance_tokens'] else 'BALANCE_DECREASE')
    if before is not None and after is not None:
        if (before.get('label'), before.get('label_confidence'), before.get('excluded')) != (after.get('label'), after.get('label_confidence'), after.get('excluded')):
            events.append('SYSTEM_LABEL_CHANGED')
    old = before.get('value_usd') if before else Decimal(0)
    new = after.get('value_usd') if after else Decimal(0)
    if old is not None and new is not None:
        for threshold in (100, 1000, 10000, 100000):
            if (old >= threshold) != (new >= threshold):
                events.append(f'CROSSED_{threshold}')
    return events


def persist_holders(cur, asset_id, day, rows, labels):
    from psycopg2.extras import execute_values
    cur.execute('SELECT wallet_address,balance_tokens,value_usd,label,label_confidence,excluded,last_seen_at FROM backpack_current_holders WHERE asset_id=%s', (asset_id,))
    names = [d[0] for d in cur.description]
    previous = {r[0]: dict(zip(names, r)) for r in cur.fetchall()}
    current = {r['wallet_address']: dict(r, label_confidence=labels.get(r['wallet_address'], {}).get('confidence')) for r in rows}
    cur.execute('SELECT max(date) FROM backpack_holder_checkpoints WHERE asset_id=%s', (asset_id,))
    prior_date = cur.fetchone()[0]
    # Initial state is a baseline, not evidence that every wallet just arrived.
    events = []
    if prior_date is not None and day <= prior_date:
        raise ValueError('Current holder state cannot move backwards or replay a date')
    if prior_date is not None:
        for wallet in sorted(previous.keys() | current.keys()):
            before, after = previous.get(wallet), current.get(wallet)
            old_balance = before['balance_tokens'] if before else Decimal(0)
            new_balance = after['balance_tokens'] if after else Decimal(0)
            old_value = before['value_usd'] if before else Decimal(0)
            new_value = after['value_usd'] if after else Decimal(0)
            for kind in holder_changes(before, after):
                events.append((asset_id, wallet, day, kind, old_balance, new_balance, old_value, new_value,
                    new_balance-old_balance, new_value-old_value if new_value is not None and old_value is not None else None,
                    (before or {}).get('label'), (after or {}).get('label'), (before or {}).get('label_confidence'),
                    (after or {}).get('label_confidence'), prior_date, 'Helius DAS; changes between complete observations; USD crossings may be price-driven'))
    if events:
        execute_values(cur, '''INSERT INTO backpack_holder_events(asset_id,wallet_address,date,event_type,
            previous_balance,new_balance,previous_value_usd,new_value_usd,change_tokens,change_usd,
            previous_label,new_label,previous_confidence,new_confidence,previous_observation_date,source)
            VALUES %s ON CONFLICT DO NOTHING''', events, page_size=1000)
    if rows:
        execute_values(cur, '''INSERT INTO backpack_current_holders(asset_id,wallet_address,balance_tokens,value_usd,
            excluded,label,label_confidence,first_seen_at,last_seen_at,label_entity,label_source,label_verified_at) VALUES %s
            ON CONFLICT(asset_id,wallet_address) DO UPDATE SET balance_tokens=excluded.balance_tokens,
            value_usd=excluded.value_usd,excluded=excluded.excluded,label=excluded.label,
            label_confidence=excluded.label_confidence,last_seen_at=excluded.last_seen_at,updated_at=now(),
            label_entity=excluded.label_entity,label_source=excluded.label_source,label_verified_at=excluded.label_verified_at''',
            [(asset_id,r['wallet_address'],r['balance_tokens'],r['value_usd'],r['excluded'],r.get('label'),
              labels.get(r['wallet_address'],{}).get('confidence'),day,day,
              labels.get(r['wallet_address'],{}).get('entity'),labels.get(r['wallet_address'],{}).get('source'),
              labels.get(r['wallet_address'],{}).get('verified_at')) for r in rows], page_size=1000)
    cur.execute('DELETE FROM backpack_current_holders WHERE asset_id=%s AND last_seen_at<>%s', (asset_id,day))
    cur.execute('''INSERT INTO backpack_holder_checkpoints(asset_id,date,owner_count,balance_tokens,aggregates_validated,methodology)
        SELECT asset_id,date,%s,%s,holders_complete AND unique_holders=%s,
        'v1: complete supply-reconciled enumeration; daily aggregate and wallet state committed atomically. First observation is a baseline.'
        FROM backpack_asset_daily_snapshots WHERE asset_id=%s AND date=%s ON CONFLICT DO NOTHING''',
        (len(rows),sum((r['balance_tokens'] for r in rows), Decimal(0)),len(rows),asset_id,day))


def retention_days(env, key):
    value = int(env.get(key, '30'))
    if not 30 <= value <= 90:
        raise ValueError('Raw retention must be between 30 and 90 days')
    return value


def maintain(conn, day, env):
    """Bounded deletes only after persistent validation; pinned transactions never deleted."""
    holder_cutoff = day-timedelta(days=retention_days(env,'BACKPACK_HOLDER_RETENTION_DAYS'))
    swap_cutoff = day-timedelta(days=retention_days(env,'BACKPACK_SWAP_RETENTION_DAYS'))
    with conn, conn.cursor() as cur:
        cur.execute('''DELETE FROM backpack_asset_holder_daily_snapshots WHERE ctid IN (
            SELECT h.ctid FROM backpack_asset_holder_daily_snapshots h
            JOIN backpack_holder_checkpoints c USING(asset_id,date)
            WHERE h.date<%s AND c.aggregates_validated LIMIT 10000)''', (holder_cutoff,))
        cur.execute('''DELETE FROM backpack_transactions WHERE ctid IN (
            SELECT t.ctid FROM backpack_transactions t
            JOIN backpack_transaction_retention_checks c ON c.asset_id=t.asset_id
                AND c.activity_date=(t.timestamp AT TIME ZONE 'UTC')::date
            JOIN backpack_asset_daily_snapshots s ON s.asset_id=t.asset_id AND s.date=c.activity_date+1
            WHERE t.timestamp<(%s::date::timestamp AT TIME ZONE 'UTC') AND c.aggregates_validated
            AND s.daily_swap_volume_usd IS NOT NULL AND s.unique_traders IS NOT NULL
            AND EXISTS(SELECT 1 FROM backpack_asset_dex_daily_snapshots d
                WHERE d.asset_id=s.asset_id AND d.date=s.date AND d.coverage_status IN ('Verified','Estimated')
                AND d.trades IS NOT NULL AND d.median_trade_size IS NOT NULL AND d.average_trade_size IS NOT NULL
                AND d.p95_trade_size IS NOT NULL AND d.max_trade_size IS NOT NULL)
            AND NOT EXISTS(SELECT 1 FROM backpack_transaction_evidence e
                WHERE e.asset_id=t.asset_id AND e.signature=t.signature AND e.event_kind=t.event_kind)
            LIMIT 10000)''', (swap_cutoff,))
        cur.execute('''INSERT INTO backpack_storage_observations(date,relation_name,table_bytes,index_bytes,
            total_bytes,estimated_rows,database_bytes,database_connections)
            SELECT %s,c.relname,pg_table_size(c.oid),pg_indexes_size(c.oid),pg_total_relation_size(c.oid),
                greatest(c.reltuples,0)::bigint,pg_database_size(current_database()),
                (SELECT numbackends FROM pg_stat_database WHERE datname=current_database())
            FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname=current_schema() AND c.relkind='r' AND (c.relname LIKE 'backpack_%%' OR c.relname LIKE 'tokenized_security_%%')
            ON CONFLICT DO NOTHING''', (day,))
        cur.execute('''INSERT INTO backpack_index_observations(date,index_name,table_name,bytes)
            SELECT %s,i.relname,t.relname,pg_relation_size(i.oid) FROM pg_index x
            JOIN pg_class i ON i.oid=x.indexrelid JOIN pg_class t ON t.oid=x.indrelid
            JOIN pg_namespace n ON n.oid=t.relnamespace
            WHERE n.nspname=current_schema() AND (t.relname LIKE 'backpack_%%' OR t.relname LIKE 'tokenized_security_%%') ON CONFLICT DO NOTHING''',(day,))
        allowance = env.get('BACKPACK_DATABASE_ALLOWANCE_BYTES')
        if allowance:
            cur.execute('''INSERT INTO backpack_operational_alerts(date,metric,detail)
                SELECT %s,'database_storage','Database storage exceeds 50%% of configured plan allowance; includes other applications'
                WHERE pg_database_size(current_database())>%s::bigint*0.5 ON CONFLICT DO NOTHING''', (day,int(allowance)))
        cur.execute('''INSERT INTO backpack_operational_alerts(date,metric,detail)
            SELECT %s,'storage_growth','Backpack table allocation grew more than 20%% versus 30 days earlier'
            WHERE (SELECT sum(total_bytes) FROM backpack_storage_observations WHERE date=%s)>
                  1.2*(SELECT nullif(sum(total_bytes),0) FROM backpack_storage_observations WHERE date=%s)
            ON CONFLICT DO NOTHING''', (day,day,day-timedelta(days=30)))

        for relation,limit in (
            ('backpack_transactions',int(env.get('BACKPACK_RAW_TRANSACTION_ROW_ALERT','1000000'))),
            ('backpack_asset_holder_daily_snapshots',int(env.get('BACKPACK_HOLDER_ROW_ALERT','2000000')))):
            cur.execute("""INSERT INTO backpack_operational_alerts(date,metric,detail)
                SELECT %s,%s,'Estimated raw row count exceeds configured threshold; inspect retention validation and backlog'
                WHERE EXISTS(SELECT 1 FROM backpack_storage_observations WHERE date=%s AND relation_name=%s AND estimated_rows>%s)
                ON CONFLICT DO NOTHING""",(day,relation,day,relation,limit))
        cur.execute("""INSERT INTO backpack_operational_alerts(date,metric,detail)
            SELECT %s,'provider_request_budget','Capture request count reached configured daily threshold'
            WHERE (SELECT sum(u.requests) FROM backpack_provider_usage u JOIN backpack_ingestion_runs r USING(run_id)
                   WHERE r.snapshot_date=%s)>=%s ON CONFLICT DO NOTHING""",(day,day,int(env.get('BACKPACK_MAX_REQUESTS','500'))))


def cost_report(conn):
    from .collector import fetch_all
    from .cost_review import cost_review
    reviews=cost_review(conn)
    projections=[r['projected_30d_infrastructure_cost_usd'] for r in reversed(reviews['windows']) if r['projected_30d_infrastructure_cost_usd'] is not None]
    return dict(status='Observed storage and request counts; consult attributed billing coverage',reviews=reviews,
        storage=fetch_all(conn,'SELECT * FROM backpack_storage_observations ORDER BY date DESC,total_bytes DESC LIMIT 1500'),
        indexes=fetch_all(conn,'SELECT * FROM backpack_index_observations WHERE date=(SELECT max(date) FROM backpack_index_observations) ORDER BY bytes DESC'),
        providers=fetch_all(conn,'''SELECT r.snapshot_date,u.provider,sum(u.requests) requests,
            sum(u.credits) credits,sum(u.estimated_cost_usd) estimated_cost_usd FROM backpack_provider_usage u
            JOIN backpack_ingestion_runs r USING(run_id) GROUP BY r.snapshot_date,u.provider ORDER BY r.snapshot_date DESC LIMIT 1000'''),
        readiness_requests=fetch_all(conn,'''SELECT c.checked_at::date date,u.provider,sum(u.requests) requests
            FROM backpack_readiness_usage u JOIN (SELECT run_id,min(checked_at) checked_at FROM backpack_readiness_checks GROUP BY run_id) c USING(run_id)
            GROUP BY c.checked_at::date,u.provider ORDER BY date DESC LIMIT 1000'''),
        storage_growth=fetch_all(conn,'''WITH daily AS (SELECT date,sum(total_bytes) bytes FROM backpack_storage_observations GROUP BY date),
            latest AS (SELECT * FROM daily ORDER BY date DESC LIMIT 1)
            SELECT l.date,l.bytes current_bytes,d.date baseline_date,d.bytes baseline_bytes,l.bytes-d.bytes growth_bytes,
                100.0*(l.bytes-d.bytes)/nullif(d.bytes,0) growth_pct
            FROM latest l LEFT JOIN daily d ON d.date IN (l.date-7,l.date-30)'''),
        alerts=fetch_all(conn,'SELECT * FROM backpack_operational_alerts ORDER BY date DESC LIMIT 100'),
        neon_cu_hours=None,neon_egress=None,vercel_invocations=None,vercel_active_cpu=None,
        vercel_cache_hit_rate=None,vercel_bandwidth=None,vercel_api_requests=None,average_api_response_bytes=None,projected_monthly_cost_usd=projections[0] if projections else None,
        limitations='Storage is allocated bytes, not billed usage. Connection count is a point-in-time observation. Billing exports required for a cost projection. Raw deletion does not immediately shrink allocated PostgreSQL files.')
