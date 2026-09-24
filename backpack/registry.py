"""Curated approvals plus immutable reconciliation of Backpack's official security universe."""
import hashlib
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from .collector import fetch_all, insert, insert_many
from .providers import Providers

ASSET_SOURCE = 'https://api.backpack.exchange/api/v1/assets'
SECURITY_SOURCE = 'https://api.backpack.exchange/api/v1/securities'
REGISTRY_METHOD = 'registry-lifecycle-v1'


def identity_fingerprint(symbol, mint, decimals, issuer='Backpack Securities'):
    value = '|'.join((str(symbol), 'solana', str(mint), str(decimals), str(issuer)))
    return hashlib.sha256(value.encode()).hexdigest()


def lifecycle_state(previous, supply, deposit_enabled, withdraw_enabled, launch_date=None, active=True):
    """Classify product state without treating positive supply as the product count."""
    if not active:
        return 'inactive'
    prior_state = previous.get('lifecycle_state') if previous else None
    prior_supply = previous.get('token_supply') if previous else None
    if prior_supply is not None and prior_supply > 0 and supply == 0:
        return 'redeemed'
    if prior_state in {'launched', 'paused'} and supply > 0 and not (deposit_enabled or withdraw_enabled):
        return 'paused'
    if launch_date or deposit_enabled or withdraw_enabled or supply > 0:
        return 'launched'
    return 'registered'


def _official_candidates(assets, securities, approved_mints):
    by_security = {row['asset']: row for row in securities}
    rows = []
    for asset in assets:
        symbol = asset.get('symbol')
        security = by_security.get(symbol)
        if not symbol or security is None:
            continue
        for token in asset.get('tokens') or []:
            mint = token.get('contractAddress')
            if token.get('blockchain') != 'Solana' or not mint:
                continue
            enabled = bool(token.get('depositEnabled') or token.get('withdrawEnabled'))
            if not enabled and mint not in approved_mints:
                continue
            rows.append(dict(token_symbol=symbol, solana_mint=mint,
                token_name=asset.get('displayName') or security.get('name') or symbol,
                decimals=token.get('nativeDecimals'), deposit_enabled=bool(token.get('depositEnabled')),
                withdraw_enabled=bool(token.get('withdrawEnabled')), security_name=security.get('name'),
                cusip=security.get('cusip')))
    return rows


def seed_context(conn):
    """Load reviewed committed context. Replays are idempotent and never rewrite evidence."""
    from psycopg2.extras import Json
    evidence = json.loads(Path(__file__).with_name('context_evidence.json').read_text())
    with conn, conn.cursor() as cur:
        for row in evidence['events']:
            record = dict(row)
            record['asset_symbols'] = Json(record['asset_symbols'])
            insert(cur, 'backpack_economy_events', record)
        for row in evidence['external_observations']:
            insert(cur, 'backpack_external_observations', row)
    return dict(events=len(evidence['events']), external_observations=len(evidence['external_observations']))


def seed_starter(conn, provider=None, observed_at=None):
    p = provider or Providers()
    now = observed_at or datetime.now(timezone.utc)
    day = now.date()
    manifest = json.loads(Path(__file__).with_name('starter_universe.json').read_text())
    assets = p.request('Backpack primary registry', 'GET', ASSET_SOURCE)
    securities = p.request('Backpack primary registry', 'GET', SECURITY_SOURCE)
    by_asset = {row['symbol']: row for row in assets}
    by_security = {row['asset']: row for row in securities}
    approved_mints = {row['solana_mint'] for row in manifest['assets']}
    results = []
    run_id = str(uuid.uuid4())

    for candidate in manifest['assets']:
        symbol = candidate['token_symbol']
        try:
            tokens = [token for token in by_asset[symbol]['tokens'] if token['blockchain'] == 'Solana']
            assert len(tokens) == 1 and tokens[0]['contractAddress'] == candidate['solana_mint']
            assert by_security[symbol]['name'] == candidate['underlying_name']
            assert by_security[symbol].get('cusip') == candidate['cusip']
            token = tokens[0]
            supply, decimals, slot = p.supply(candidate['solana_mint'])
            assert decimals == candidate['decimals'] and supply >= 0
            existing = fetch_all(conn, 'SELECT * FROM backpack_assets WHERE solana_mint=%s', (candidate['solana_mint'],))
            if existing and existing[0]['token_symbol'] != symbol:
                raise ValueError('Official mint conflicts with an approved symbol')
            previous = fetch_all(conn, '''SELECT lifecycle_state,token_supply FROM backpack_asset_registry_daily
                WHERE asset_id=%s AND date<%s ORDER BY date DESC LIMIT 1''', (existing[0]['id'], day)) if existing else []
            state = lifecycle_state(previous[0] if previous else None, supply,
                bool(token.get('depositEnabled')), bool(token.get('withdrawEnabled')), candidate.get('launch_date'))
            fingerprint = identity_fingerprint(symbol, candidate['solana_mint'], decimals, candidate['issuer'])
            notes = ('Curated exact official asset/security identity, checked against committed exchange-listing evidence. '
                'Primary registry and live mint revalidated; mint decimals '+str(decimals)+' at slot '+str(slot)+'. '
                '1:1 redemption source: '+manifest['backing_source']+'. '
                'Security metadata: '+candidate['security_source']+'; exchange: '+candidate['exchange_source'])
            if existing:
                asset_id = existing[0]['id']
                with conn, conn.cursor() as cur:
                    cur.execute('''UPDATE backpack_assets SET last_official_seen_at=%s,
                        first_official_seen_at=COALESCE(first_official_seen_at,%s), deposit_enabled=%s,
                        withdraw_enabled=%s, registry_status=%s, identity_fingerprint=%s,
                        launch_evidence_source=COALESCE(launch_evidence_source,%s), updated_at=now()
                        WHERE id=%s''', (now, now, bool(token.get('depositEnabled')),
                        bool(token.get('withdrawEnabled')), state, fingerprint,
                        candidate['official_source'] if candidate.get('launch_date') else None, asset_id))
            else:
                record = {key:candidate[key] for key in ('token_symbol','token_name','solana_mint',
                    'underlying_symbol','underlying_name','underlying_exchange','asset_type','issuer',
                    'launch_date','official_source')}
                record.update(verification_status='official', source_verified_at=now,
                    approval_notes=notes, registry_status=state, first_official_seen_at=now,
                    last_official_seen_at=now, deposit_enabled=bool(token.get('depositEnabled')),
                    withdraw_enabled=bool(token.get('withdrawEnabled')), identity_fingerprint=fingerprint,
                    launch_evidence_source=candidate['official_source'] if candidate.get('launch_date') else None)
                with conn, conn.cursor() as cur:
                    insert(cur, 'backpack_assets', record)
                asset_id = fetch_all(conn, 'SELECT id FROM backpack_assets WHERE solana_mint=%s',
                    (candidate['solana_mint'],))[0]['id']
            with conn, conn.cursor() as cur:
                insert(cur, 'backpack_asset_registry_daily', dict(asset_id=asset_id, date=day,
                    official_present=True, deposit_enabled=bool(token.get('depositEnabled')),
                    withdraw_enabled=bool(token.get('withdrawEnabled')), token_supply=supply,
                    lifecycle_state=state, source=ASSET_SOURCE+'; '+SECURITY_SOURCE,
                    observed_at=now, methodology_version=REGISTRY_METHOD, quality_status='Verified',
                    limitation='Operational flags and supply establish observable lifecycle state, not user adoption.',
                    identity_fingerprint=fingerprint))
            results.append(dict(check_name=symbol, status='Verified',
                detail='Reconciled exact official Solana mint; '+state+'; chain slot '+str(slot)))
        except Exception as error:
            conn.rollback()
            results.append(dict(check_name=symbol, status='Unavailable', detail='Not reconciled: '+type(error).__name__))

    approved = fetch_all(conn, "SELECT id,token_symbol,solana_mint FROM backpack_assets WHERE asset_type<>'bp' AND verification_status<>'pending'")
    by_mint = {row['solana_mint']: row for row in approved}
    by_symbol = {row['token_symbol']: row for row in approved}
    candidate_records = []
    for row in _official_candidates(assets, securities, set(by_mint)):
        mint_match, symbol_match = by_mint.get(row['solana_mint']), by_symbol.get(row['token_symbol'])
        if mint_match and symbol_match and mint_match['id'] == symbol_match['id']:
            status, matched, detail = 'approved', mint_match['id'], 'Exact approved symbol and mint match.'
        elif mint_match or symbol_match:
            status, matched, detail = 'conflict', (mint_match or symbol_match)['id'], 'Official symbol or mint conflicts with the approved identity.'
        else:
            status, matched, detail = 'unresolved', None, 'Official enabled Solana security is not yet in the approved research universe.'
        candidate_records.append(dict(row, date=day, match_status=status, matched_asset_id=matched,
            source=ASSET_SOURCE+'; '+SECURITY_SOURCE, observed_at=now, detail=detail))
    with conn, conn.cursor() as cur:
        insert_many(cur, 'backpack_registry_candidates_daily', candidate_records)
        insert_many(cur, 'backpack_readiness_checks', [dict(row, run_id=run_id) for row in results])
        insert_many(cur, 'backpack_readiness_usage',
            [dict(run_id=run_id, provider=key, requests=value) for key,value in p.usage.items()])
    context = seed_context(conn)
    unresolved = sum(row['match_status'] == 'unresolved' for row in candidate_records)
    conflicts = sum(row['match_status'] == 'conflict' for row in candidate_records)
    return dict(run_id=run_id, registry_results=results, provider_requests=dict(p.usage),
        failed=sum(row['status'] == 'Unavailable' for row in results),
        coverage=dict(official_candidates=len(candidate_records),
            approved=sum(row['match_status'] == 'approved' for row in candidate_records),
            unresolved=unresolved, conflicts=conflicts), context=context)
