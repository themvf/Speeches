"""Read-only diagnostics for graduations on or after an explicit repair cutoff."""
from datetime import datetime, timezone
from launchpad_archive import ladder_health, MIN_HEALTH_SAMPLE


def cutoff(value):
    try:
        result=datetime.fromisoformat(value.replace('Z','+00:00'))
        if result.tzinfo is None:raise ValueError('timezone required')
        return result.astimezone(timezone.utc)
    except (ValueError,AttributeError) as exc:
        raise ValueError('cutoff must be an ISO timestamp with a timezone') from exc


def cohort_report(conn,chain,since,now=None):
    now=now or datetime.now(timezone.utc)
    if since>=now:raise ValueError('cutoff must precede report time')
    from psycopg2.extras import RealDictCursor
    with conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute('''SELECT t.token_address,t.launchpad_family,t.cohort_sampled,
                      t.graduated_at,t.graduated_detected_at,t.measure_pool,t.measure_pool_timing,
                      t.measure_pool_reason,t.enriched_at,
                      extract(epoch FROM t.graduated_detected_at-t.graduated_at) AS detection_lag_seconds,
                      c.captures,c.nonempty,c.started,c.opening_lag_seconds,
                      s.errors AS detecting_sweep_errors
               FROM launchpad_tokens t
               LEFT JOIN LATERAL (
                 SELECT count(*) AS captures,count(*) FILTER (WHERE trades>0) AS nonempty,
                        min(capture_started_at) AS started,min(lag_seconds) AS opening_lag_seconds
                 FROM launchpad_trade_captures c
                 WHERE c.network=t.network AND c.token_address=t.token_address
                   AND c.capture_started_at<=%s) c ON true
               LEFT JOIN LATERAL (
                 SELECT errors FROM launchpad_sweeps s
                 WHERE s.network=t.network AND s.started_at=t.graduated_detected_at
                 ORDER BY id LIMIT 1) s ON true
               WHERE t.network=%s AND t.graduated AND t.graduated_at>=%s AND t.graduated_at<=%s
               ORDER BY t.graduated_at,t.token_address''',(now,chain.network,since,now))
        rows=[dict(r) for r in cur.fetchall()]
        cur.execute('''SELECT started_at,finished_at,pages_fetched,gap_seconds,errors
                       FROM launchpad_sweeps WHERE network=%s AND started_at>=%s AND started_at<=%s
                       ORDER BY started_at''',(chain.network,since,now))
        sweeps=[dict(r) for r in cur.fetchall()]
        cur.execute('''SELECT started_at,finished_at,processed,rungs_filled,error_count,pending,
                              oldest_pending_age_seconds,arrival_rate_per_hour,service_rate_per_hour
                       FROM launchpad_enrich_runs WHERE network=%s AND started_at>=%s AND started_at<=%s
                       ORDER BY started_at''',(chain.network,since,now))
        runs=[dict(r) for r in cur.fetchall()]
        cur.execute('''SELECT count(*) AS pending,min(graduated_at) AS oldest
                       FROM launchpad_tokens WHERE network=%s AND graduated AND enriched_at IS NULL''',
                    (chain.network,))
        backlog=dict(cur.fetchone())
        cur.execute('''SELECT count(*) AS unknown_graduation_time FROM launchpad_tokens
                       WHERE network=%s AND graduated AND graduated_at IS NULL''',(chain.network,))
        unknown=cur.fetchone()['unknown_graduation_time']
    sample=[r for r in rows if r['cohort_sampled']]
    captured=sum(bool(r['captures']) for r in sample)
    nonempty=sum(bool(r['nonempty']) for r in sample)
    selected=sum(r['measure_pool'] is not None and r['measure_pool_timing']=='at_graduation' for r in sample)
    def median(values):
        values=sorted(float(v) for v in values if v is not None)
        n=len(values)
        return round((values[(n-1)//2]+values[n//2])/2,1) if n else None
    boundaries=[since]+[r['started_at'] for r in sweeps]+[now]
    errors=[e for r in sweeps for e in r['errors']]
    pending=[r for r in rows if r['enriched_at'] is None]
    return dict(network=chain.network,since=since.isoformat(),as_of=now.isoformat(),
        sample_state='sufficient' if len(sample)>=MIN_HEALTH_SAMPLE else 'too_small',
        denominator='Observed graduates with graduated_at in [since, as_of]; deterministic cohort unchanged. Undiscovered graduates cannot be counted.',
        graduates=len(rows),cohort_graduates=len(sample),unknown_graduation_time=unknown,
        captures=dict(applicable=chain.capture_trades,captured=captured,nonempty=nonempty,
                      missing=len(sample)-captured,empty_only=captured-nonempty,
                      coverage=round(captured/len(sample),3) if sample and chain.capture_trades else None,
                      median_opening_lag_seconds=median(r['opening_lag_seconds'] for r in sample)),
        pool_selection=dict(at_graduation=selected,
                            at_graduation_coverage=round(selected/len(sample),3) if sample else None,
                            late=sum(r['measure_pool_timing']=='late' for r in sample),
                            without_pool=sum(r['measure_pool'] is None for r in sample)),
        median_detection_lag_seconds=median(r['detection_lag_seconds'] for r in rows),
        cohort_backlog=dict(pending=len(pending),oldest_age_seconds=int((now-pending[0]['graduated_at']).total_seconds()) if pending else None),
        global_backlog=dict(pending=backlog['pending'],oldest_age_seconds=int((now-backlog['oldest']).total_seconds()) if backlog['oldest'] else None),
        sweeps=dict(recorded=len(sweeps),expected=int((now-since).total_seconds()/(chain.sweep_minutes*60)),
                    longest_silence_seconds=int(max((b-a).total_seconds() for a,b in zip(boundaries,boundaries[1:]))),
                    with_errors=sum(bool(r['errors']) for r in sweeps),
                    with_rate_limit=sum(any('429' in e or 'rate limit' in e for e in r['errors']) for r in sweeps),
                    with_budget_exhaustion=sum(any('budget' in e or 'deadline' in e for e in r['errors']) for r in sweeps),
                    max_feed_gap_seconds=max((r['gap_seconds'] or 0 for r in sweeps),default=0),
                    median_duration_seconds=median((r['finished_at']-r['started_at']).total_seconds() for r in sweeps if r['finished_at']),
                    error_examples=list(dict.fromkeys(errors))[:20]),
        ladder=ladder_health(conn,chain,now,since),enrichment_runs=runs,
        missed_capture_examples=[r for r in sample if not r['captures']][:20] if chain.capture_trades else [],
        attribution_caveat='Sweep errors are correlated evidence, not proof of why an individual capture failed. Failed jobs before persistence require workflow logs. Empty captures do not prove opening-window coverage.')
