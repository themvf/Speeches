"""Daily immutable capture; snapshot date is capture UTC day, activity is prior UTC day."""
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import uuid
from .metrics import BP_MINT, ecosystem, holders, issuance, multiply, normalize_swap, number, parity, ratio, trading, whale_cohorts
from .providers import Providers, SourceError
from .storage import persist_holders, maintain

UTC = timezone.utc


def setup(conn):
    with conn, conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(98274031)")
        cur.execute(Path(__file__).resolve().parents[1].joinpath('sql/backpack.sql').read_text())
        # Exact mint verified against Backpack's primary-source article on 2026-09-22.
        cur.execute("""INSERT INTO backpack_assets(token_symbol,token_name,solana_mint,asset_type,issuer,
            official_source,source_verified_at,verification_status,approval_notes)
            VALUES ('BP','Backpack',%s,'bp','Backpack',
            'https://learn.backpack.exchange/articles/what-is-bp-backpack-token','2026-09-22T00:00:00Z',
            'official','Exact Solana contract in official BP Token Details article; staking/circulating figures are not independently inferred')
            ON CONFLICT(solana_mint) DO UPDATE SET official_source=excluded.official_source,
            source_verified_at=excluded.source_verified_at,verification_status=excluded.verification_status,
            approval_notes=excluded.approval_notes,updated_at=now()
            WHERE backpack_assets.official_source='User-provided specification, 2026-09-22'
              AND backpack_assets.asset_type='bp'""", (BP_MINT,))


def calendar_for(day):
    """Exchange calendar handles DST, observed holidays, exceptional closures, early closes."""
    import exchange_calendars as xcals
    cal = xcals.get_calendar('XNYS')
    result = {}
    for d in (day - timedelta(days=1), day):
        key = d.isoformat()
        result[key] = (cal.session_open(key).to_pydatetime(), cal.session_close(key).to_pydatetime()) if cal.is_session(key) else None
    return result


def fetch_all(conn, query, params=()):
    from psycopg2.extras import RealDictCursor
    with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        return [dict(r) for r in cur.fetchall()]


def insert(cur, table, record, conflict=True):
    from psycopg2 import sql
    columns = list(record)
    cur.execute(sql.SQL('INSERT INTO {} ({}) VALUES ({}){}').format(
        sql.Identifier(table), sql.SQL(',').join(map(sql.Identifier, columns)),
        sql.SQL(',').join(sql.Placeholder() for _ in columns), sql.SQL(' ON CONFLICT DO NOTHING' if conflict else '')),
        [record[c] for c in columns])
    return cur.rowcount


def insert_many(cur, table, records):
    if not records: return
    from psycopg2 import sql
    from psycopg2.extras import execute_values
    columns=list(records[0])
    query=sql.SQL('INSERT INTO {} ({}) VALUES %s ON CONFLICT DO NOTHING').format(
        sql.Identifier(table),sql.SQL(',').join(map(sql.Identifier,columns))).as_string(cur)
    execute_values(cur,query,[[r.get(c) for c in columns] for r in records],page_size=1000)


def upsert_current_snapshot(cur, record):
    """Insert once, except that today's incomplete holder evidence may be repaired."""
    from psycopg2 import sql
    columns = list(record)
    mutable = [column for column in columns if column not in {'asset_id', 'date'}]
    query = sql.SQL('''INSERT INTO backpack_asset_daily_snapshots ({columns}) VALUES ({values})
        ON CONFLICT(asset_id,date) DO UPDATE SET {updates}
        WHERE NOT backpack_asset_daily_snapshots.holders_complete AND excluded.holders_complete''').format(
        columns=sql.SQL(',').join(map(sql.Identifier, columns)),
        values=sql.SQL(',').join(sql.Placeholder() for _ in columns),
        updates=sql.SQL(',').join(
            sql.SQL('{}=excluded.{}').format(sql.Identifier(column), sql.Identifier(column))
            for column in mutable))
    cur.execute(query, [record[column] for column in columns])
    return cur.rowcount


def upsert_current_ecosystem(cur, record):
    """Recompute only the live UTC day's aggregate after a retry completes evidence."""
    from psycopg2 import sql
    columns = list(record)
    mutable = [column for column in columns if column != 'date']
    query = sql.SQL('''INSERT INTO backpack_ecosystem_daily_snapshots ({columns}) VALUES ({values})
        ON CONFLICT(date) DO UPDATE SET {updates}''').format(
        columns=sql.SQL(',').join(map(sql.Identifier, columns)),
        values=sql.SQL(',').join(sql.Placeholder() for _ in columns),
        updates=sql.SQL(',').join(
            sql.SQL('{}=excluded.{}').format(sql.Identifier(column), sql.Identifier(column))
            for column in mutable))
    cur.execute(query, [record[column] for column in columns])
    return cur.rowcount


def collect_asset(conn, p, asset, run_id, day, labels, calendar):
    from psycopg2.extras import Json
    now = datetime.now(UTC)
    asset_id, mint = asset['id'], asset['solana_mint']
    events = []
    def note(metric, status, source, calculation, limitation='', observed_at=None):
        events.append(dict(run_id=run_id, asset_id=asset_id, date=day, metric=metric,
            status=status, source=source, calculation=calculation, limitation=limitation, observed_at=observed_at or now))
    def optional(metric, source, function):
        try: return function()
        except (SourceError, KeyError, TypeError, ValueError) as error:
            note(metric, 'Unavailable', source, metric, str(error) if isinstance(error, SourceError) else 'Invalid provider payload')
            return None
    supply, decimals, slot = p.supply(mint)
    block_time = optional('block_timestamp','Solana RPC',lambda:p.rpc('getBlockTime',[slot]))
    reference = optional('underlying_price','Alpaca SIP',lambda:p.equity(asset['underlying_symbol'])) if asset['asset_type'] != 'bp' else None
    token = optional('onchain_price','Jupiter Price V3',lambda:p.price(mint))
    price, price_at = reference[:2] if reference else (None, None)
    token_price, token_at = token if token else (None, None)
    if price_at and (now-price_at).total_seconds() > 4*86400:
        note('underlying_price','Stale','Alpaca SIP','Latest underlying trade','Reference is over four days old; inspect holiday/halt or provider coverage',price_at)
        note('reference_aum_usd','Stale','Solana RPC + Alpaca SIP','Supply × last available reference','Underlying price is over four days old',price_at)
    token_stale = token_at is not None and (now-token_at).total_seconds() > 3600
    if token_stale:
        note('onchain_price','Stale','Jupiter Price V3','Last swapped price','Price block is over one hour old',token_at)
        note('onchain_market_value_usd','Stale','Solana RPC + Jupiter','Supply × last swapped price','Token price is over one hour old',token_at)
    us_exchange=str(asset.get('underlying_exchange') or '').upper() in {'XNYS','XNAS','NYSE','NASDAQ','NASDAQGS','NYSEARCA','NYSE ARCA','ARCX','NYSE AMERICAN','XASE'}
    market_open = None if not us_exchange else bool(calendar.get(day.isoformat()) and calendar[day.isoformat()][0] <= now < calendar[day.isoformat()][1])
    snap = dict(asset_id=asset_id, date=day, run_id=run_id, captured_at=now, slot=slot,
        block_timestamp=datetime.fromtimestamp(block_time,UTC) if block_time else None, source='Helius / Solana finalized RPC',
        token_supply=supply, decimals=decimals, underlying_price=price, underlying_price_timestamp=price_at,
        previous_official_close=reference[2] if reference else None, underlying_market_open=market_open,
        underlying_price_source='Alpaca SIP' if reference else None,
        onchain_price=token_price, onchain_price_timestamp=token_at,
        reference_aum_usd=multiply(supply,price), onchain_market_value_usd=multiply(supply,token_price),
        holders_complete=False, data_quality_score=0, quality_status='Partial')
    if price is not None and token_price is not None:
        snap['premium_discount_pct'] = parity(token_price,price,token_at,price_at,market_open)['premium_discount_pct']
        note('premium_discount_pct','Estimated','Jupiter + Alpaca SIP','(token price / reference price - 1) × 100',
             'Last-available equity reference while market closed; no arbitrage alert. Daily sample cannot measure deviation duration.')
    previous = fetch_all(conn,'SELECT * FROM backpack_asset_daily_snapshots WHERE asset_id=%s AND date=%s',(asset_id,day-timedelta(days=1)))
    wallet_rows = []
    holder_attempts = int(p.env.get('BACKPACK_HOLDER_RECONCILIATION_ATTEMPTS', '2'))
    if holder_attempts < 1 or holder_attempts > 3:
        raise ValueError('Invalid holder reconciliation attempts')
    last_mismatch = None
    for attempt in range(holder_attempts):
        attempt_supply, attempt_decimals, attempt_slot = (supply, decimals, slot) if attempt == 0 else p.supply(mint)
        holder_data = optional('holders','Helius DAS',lambda:p.holders(mint))
        if not holder_data:
            break
        accounts, first_slot, end_slot = holder_data
        hp = token_price if asset['asset_type']=='bp' and not token_stale else price
        analytics = holders(accounts,attempt_decimals,hp,labels)
        candidate_rows = analytics.pop('rows')
        observed_supply = sum(r['balance_tokens'] for r in candidate_rows)
        tolerance = Decimal(p.env.get('BACKPACK_SUPPLY_TOLERANCE','0.001'))
        if not tolerance.is_finite() or tolerance < 0:
            raise ValueError('Invalid supply tolerance')
        def matches(value):
            return abs(observed_supply-value) <= max(Decimal('0.000000001'),value*tolerance)
        chosen = (attempt_supply, attempt_decimals, attempt_slot) if matches(attempt_supply) else None
        ending = optional('holder_supply_recheck','Solana RPC',lambda:p.supply(mint))
        if chosen is None and ending and ending[1] == attempt_decimals and matches(ending[0]):
            chosen = ending
            note('holder_supply_alignment','Estimated','Helius DAS + finalized Solana RPC',
                 'Holder sum reconciled to finalized supply read after enumeration',
                 f'Initial supply {attempt_supply} at slot {attempt_slot}; aligned supply {ending[0]} at slot {ending[2]}.')
        if chosen is not None:
            supply, decimals, slot = chosen
            if slot != snap['slot']:
                aligned_time = optional('block_timestamp','Solana RPC',lambda:p.rpc('getBlockTime',[slot]))
                snap.update(slot=slot,block_timestamp=datetime.fromtimestamp(aligned_time,UTC) if aligned_time else None,
                            token_supply=supply,decimals=decimals,
                            reference_aum_usd=multiply(supply,price),onchain_market_value_usd=multiply(supply,token_price))
            wallet_rows = candidate_rows
            snap.update(holder_start_slot=first_slot,holder_end_slot=end_slot)
            reconciled = True
        else:
            reconciled = False
            last_supply = ending[0] if ending and ending[1] == attempt_decimals else attempt_supply
            last_slot = ending[2] if ending and ending[1] == attempt_decimals else attempt_slot
            supply, decimals, slot = last_supply, attempt_decimals, last_slot
            last_mismatch = (observed_supply, last_supply, first_slot, end_slot)
        if reconciled:
            snap.update(analytics,holders_complete=True)
            if previous and previous[0]['holders_complete']:
                old = fetch_all(conn,'SELECT wallet_address FROM backpack_asset_holder_daily_snapshots WHERE asset_id=%s AND date=%s',(asset_id,day-timedelta(days=1)))
                before, after = {r['wallet_address'] for r in old}, {r['wallet_address'] for r in wallet_rows}
                snap.update(new_holders=len(after-before),lost_holders=len(before-after))
            note('holders','Estimated','Helius DAS','Paginated token accounts; grouped by owner for economic cohorts',
                f'Enumeration spans slots {first_slot}–{end_slot}; not an atomic historic snapshot. Only verified system labels excluded.')
            break
    if not snap['holders_complete'] and last_mismatch:
        observed_supply, compared_supply, first_slot, end_slot = last_mismatch
        snap.update(token_supply=supply,decimals=decimals,slot=slot,holder_start_slot=first_slot,holder_end_slot=end_slot,
                    reference_aum_usd=multiply(supply,price),onchain_market_value_usd=multiply(supply,token_price))
        wallet_rows=[]
        note('holders','Unavailable','Helius DAS vs RPC','Sum owner balances / finalized supply',
             f'Holder sum {observed_supply}; finalized supply {compared_supply}; fractional tolerance {tolerance}; '
             f'{holder_attempts} bounded attempt(s). Does not reconcile; analytics withheld and current-day retry remains eligible.')
    delta, delta_usd = issuance(supply,previous[0] if previous else None,day,price)
    snap.update(net_supply_change_tokens=delta, net_supply_change_usd=delta_usd)
    # Bounded observed-wallet tape. Never claim mint-wide coverage from mint-address history.
    swaps, cursors = {}, []
    if wallet_rows and p.env.get('HELIUS_API_KEY'):
        start = datetime.combine(day-timedelta(days=1),datetime.min.time(),UTC)
        end = datetime.combine(day,datetime.min.time(),UTC)
        candidates = sorted(wallet_rows,key=lambda r:r['balance_tokens'],reverse=True)[:int(p.env.get('BACKPACK_MAX_ACTIVITY_WALLETS','10'))]
        for wallet in candidates:
            address = wallet['wallet_address']
            old = fetch_all(conn,'SELECT * FROM backpack_transaction_cursors WHERE asset_id=%s AND wallet_address=%s',(asset_id,address))
            # Continue from last fully processed time. Replay previous-day boundary for dedup.
            result = optional('swap_tape','Helius Enhanced Transactions',lambda:p.history(address,int(start.timestamp()),int(end.timestamp()),old[0]['last_signature'] if old and old[0]['through_at']==start else None))
            if result:
                tape, signature, complete = result
                for tx in tape:
                    s = normalize_swap(tx,mint)
                    if s: swaps[s['signature']] = s
                if signature and complete: cursors.append((address,signature,end))
        note('daily_swap_volume_usd','Partial','Helius Enhanced Transactions','One outer swap per signature/asset; USDC legs valued at $1',
             'Bounded current-holder wallet sample; closed accounts, unsampled wallets and non-USDC value may be missing. Total volume and total traders withheld.')
    stats = trading(list(swaps.values()),calendar if us_exchange else {},complete=False)
    for key in ('observed_swap_volume_usd','daily_swap_volume_usd','unique_traders','observed_unique_traders','after_hours_volume_pct','regular_session_volume_usd','after_hours_volume_usd'):
        snap[key] = stats[key] if swaps else None
    quotes=[]
    if p.env.get('BACKPACK_ENABLE_QUOTES','0') == '1':
        for size in (1000,10000,50000,100000):
            for direction in ('buy','sell'):
                q = optional(f'quote_{direction}_{size}','Jupiter Metis',lambda:p.quote(mint,decimals,None if token_stale else token_price,size,direction))
                quote = dict(asset_id=asset_id,date=day,direction=direction,notional_usd=size,quote_at=datetime.now(UTC),
                    source='Jupiter Metis /swap/v1/quote',status='Estimated' if q else 'Unavailable',
                    limitation='Indicative executable route, not a submitted trade; output subject to state changes; USDC assumed $1')
                if q:
                    out_decimals = decimals if direction=='buy' else 6
                    quote.update(input_amount=q['inAmount'],expected_output=Decimal(q['outAmount'])/Decimal(10)**out_decimals,
                        minimum_output=Decimal(q['otherAmountThreshold'])/Decimal(10)**out_decimals,
                        price_impact_pct=number(q.get('priceImpactPct')),route=Json(q['routePlan']),route_legs=len(q['routePlan']),slot=q.get('contextSlot'))
                quotes.append(quote)
            pair=[q.get('price_impact_pct') for q in quotes if q['notional_usd']==size]
            if all(x is not None for x in pair): snap[f'quote_{size//1000}k_price_impact']=max(pair)
    validation = optional('supply_validation','Independent Solana RPC',lambda:p.supply(mint,True))
    if validation:
        difference=abs(validation[0]-supply)
        independent_source=bool(p.env.get('HELIUS_API_KEY')) or bool(p.env.get('SOLANA_RPC_URL') and p.env.get('SOLANA_RPC_URL') != (p.env.get('SOLANA_VALIDATION_RPC_URL') or 'https://api.mainnet-beta.solana.com'))
        note('token_supply',('Verified' if independent_source else 'Estimated') if difference<=max(Decimal('0.000000001'),supply*Decimal('0.001')) else 'Partial',
             'Solana finalized RPC + independent RPC','Raw integer supply / 10^decimals',
             f'Primary supply {supply} at slot {slot}; validation supply {validation[0]} at slot {validation[2]}. Different slots may explain divergence. Separate source: {independent_source}', now)
    independent = optional('price_validation','DexScreener validation',lambda:p.validation_market(mint))
    if independent and token_price:
        difference=abs(number(independent['priceUsd'])/token_price-1)*100
        note('price_validation','Estimated' if difference<=Decimal(p.env.get('BACKPACK_PRICE_TOLERANCE_PCT','5')) else 'Partial',
             'DexScreener highest-liquidity exact-base pair','Absolute independent price difference / Jupiter price',
             f'Jupiter {token_price}; independent pair {number(independent["priceUsd"])}; deviation {difference:.2f}%; provider observation time only. Pair rolling volume cannot validate full UTC-day volume.')
    note('registry','Verified' if asset['verification_status']=='official' else 'Estimated',asset['official_source'],
         'Mint identity comes only from approved registry','Manual approval is not independent official-source verification' if asset['verification_status']!='official' else '',asset['source_verified_at'])
    # Per-field provenance, including future-phase columns which are explicitly unavailable.
    existing={e['metric'] for e in events}
    meanings={
        'reference_aum_usd':('Solana RPC + Alpaca SIP','Token supply × last available underlying security price','Economic exposure, not customer deposits; assumes one token represents one share'),
        'onchain_market_value_usd':('Solana RPC + Jupiter','Token supply × on-chain price','May differ from reference AUM'),
        'net_supply_change_tokens':('Stored daily snapshots',"Today supply − yesterday supply",'Requires consecutive captured dates; not customer deposits'),
        'net_supply_change_usd':('Stored snapshots + Alpaca SIP','Daily supply delta × current reference price','Issuance valued on each observation day; not customer deposits'),
    }
    for field in ('token_supply','underlying_price','onchain_price','reference_aum_usd','onchain_market_value_usd','net_supply_change_tokens','net_supply_change_usd',
                  'holder_count','unique_holders','holders_over_100','holders_over_1000','holders_over_10000','holders_over_100000',
                  'top_10_holder_pct','top_20_holder_pct','top_50_holder_pct','top_100_holder_pct',
                  'economic_top_10_holder_pct','economic_top_20_holder_pct','economic_top_50_holder_pct','economic_top_100_holder_pct',
                  'daily_swap_volume_usd','unique_traders','observed_swap_volume_usd','observed_unique_traders',
                  'daily_transfer_volume_usd','defi_utilization_pct','after_hours_volume_pct'):
        if field in existing: continue
        source,calc,limit=meanings.get(field,('Helius / stored observations',field.replace('_',' '),'Owner addresses are not identified individual investors'))
        if field.startswith('economic_top_'): calc='Top N eligible owner balances / sum eligible owner balances × 100'
        elif field.startswith('top_'): calc='Top N owner balances / sum all owner balances × 100'
        if field=='underlying_price': source,calc,limit='Alpaca SIP','Latest trade reference','Last available price; closed-market differences do not trigger parity alerts'
        if field=='onchain_price': source,calc,limit='Jupiter Price V3','Last swapped price','Timestamp is Solana price block time'
        note(field,'Estimated' if snap.get(field) is not None else 'Unavailable',source,calc,limit)
    cohorts=[]
    if asset['asset_type']=='bp':
        prior_wallets=None
        if previous and previous[0]['holders_complete']:
            prior_wallets=fetch_all(conn,'SELECT * FROM backpack_asset_holder_daily_snapshots WHERE asset_id=%s AND date=%s',(asset_id,day-timedelta(days=1)))
        thresholds=p.env.get('BACKPACK_WHALE_THRESHOLDS_USD','100000,500000,1000000').split(',')
        # The $100K baseline remains available for the BP overview.
        cohorts=whale_cohorts(wallet_rows if snap['holders_complete'] and not token_stale and token_price is not None else None,
                              prior_wallets,thresholds+['100000'],labels)
        note('whale_cohorts','Estimated' if cohorts[0]['whale_count'] is not None else 'Unavailable',
             'Stored complete holder snapshots + Jupiter price',
             "USD threshold cohorts; accumulation is token balance change of yesterday's economic whales",
             'Threshold entries can result from price changes. Consecutive complete priced snapshots required for changes; wallets excluded on either date omitted from changes.')
    scored=('token_supply','reference_aum_usd','holders_over_100','daily_swap_volume_usd','onchain_price')
    snap['data_quality_score']=int(100*sum(snap.get(k) is not None for k in scored)/len(scored))
    # Quality score measures field coverage, never thesis strength.
    with conn,conn.cursor() as cur:
        written = upsert_current_snapshot(cur,snap)
        if not written and snap['holders_complete']:
            return 'skipped'
        insert_many(cur,'backpack_asset_holder_daily_snapshots',[dict(r,asset_id=asset_id,date=day,source='Helius DAS',slot=snap['holder_end_slot'],label_entity=labels.get(r['wallet_address'],{}).get('entity'),label_confidence=labels.get(r['wallet_address'],{}).get('confidence'),label_source=labels.get(r['wallet_address'],{}).get('source'),label_verified_at=labels.get(r['wallet_address'],{}).get('verified_at')) for r in wallet_rows])
        if snap['holders_complete']:
            persist_holders(cur, asset_id, day, wallet_rows, labels)
        for s in swaps.values():
            insert(cur,'backpack_transactions',dict(asset_id=asset_id,signature=s['signature'],event_kind='swap',slot=s['slot'],
                timestamp=datetime.fromtimestamp(s['timestamp'],UTC),wallet_address=s['wallet_address'],side=s['side'],tokens=s['tokens'],
                volume_usd=s['volume_usd'],venue=s['venue'],source=s['source']))
        for address, signature, through in cursors:
            cur.execute('''INSERT INTO backpack_transaction_cursors VALUES(%s,%s,%s,%s)
                ON CONFLICT(asset_id,wallet_address) DO UPDATE SET last_signature=excluded.last_signature, through_at=excluded.through_at''',
                (asset_id,address,signature,through))
        for q in quotes: insert(cur,'backpack_asset_liquidity_daily_snapshots',q)
        insert_many(cur,'backpack_data_quality_events',events)
        for kind,pv,pt,source in [('underlying',price,price_at,'Alpaca SIP'),('onchain',token_price,token_at,'Jupiter Price V3')]:
            if pv is not None and pt: insert(cur,'backpack_market_prices',dict(asset_id=asset_id,price_at=pt,source=source,price=pv,price_kind=kind,market_open=market_open))
        if swaps:
            insert(cur,'backpack_asset_dex_daily_snapshots',dict(asset_id=asset_id,date=day,venue='All observed (deduplicated)',
                observed_volume_usd=stats['observed_swap_volume_usd'],trades=stats['trades'],unique_traders=stats['observed_unique_traders'],
                **{k:stats[k] for k in ('median_trade_size','average_trade_size','p95_trade_size','max_trade_size')},coverage_status='Partial',source='Helius Enhanced Transactions'))
        if asset['asset_type']=='bp':
            for cohort in cohorts:
                insert(cur,'backpack_bp_whale_daily_snapshots',dict(cohort,asset_id=asset_id,date=day,
                    status='Estimated' if cohort['whale_count'] is not None else 'Unavailable',source='Helius DAS + Jupiter + stored holder snapshots',
                    methodology="Counts exclude confirmed/high system labels. New/exited counts may reflect price changes. Accumulation measures yesterday's eligible whales in tokens; relabeled systems excluded on both dates."))
            baseline=next(c for c in cohorts if c['threshold_usd']==100000)
            insert(cur,'backpack_bp_daily_snapshots',dict(date=day,asset_id=asset_id,fdv_usd=multiply(supply,token_price),
                **{k:baseline[k] for k in ('whale_count','new_whales','whale_net_accumulation_tokens')}))
    return 'succeeded' if snap['holders_complete'] else 'incomplete'


def aggregate(conn,run_id,day,assets):
    ids=[a['id'] for a in assets if a['asset_type']!='bp']
    if not ids: return
    rows=fetch_all(conn,'SELECT * FROM backpack_asset_daily_snapshots WHERE date=%s AND asset_id=ANY(%s)',(day,ids))
    # Do not freeze an incomplete ecosystem day; a retry can fill failed assets first.
    if len(rows)!=len(ids): return
    wallet_rows=fetch_all(conn,'SELECT * FROM backpack_asset_holder_daily_snapshots WHERE date=%s AND asset_id=ANY(%s)',(day,ids))
    record=dict(date=day,run_id=run_id,assets_expected=len(ids),assets_captured=len(rows),quality_status='Partial')
    for key in ('reference_aum_usd','onchain_market_value_usd','net_supply_change_usd','daily_swap_volume_usd','observed_swap_volume_usd'):
        record[key]=sum(r[key] for r in rows) if all(r[key] is not None for r in rows) else None
    if all(r['holders_complete'] and r['underlying_price'] is not None for r in rows): record.update(ecosystem(wallet_rows))
    with conn,conn.cursor() as cur: upsert_current_ecosystem(cur,record)


def run(conn,p=None,day=None):
    p=p or Providers()
    day=day or datetime.now(UTC).date()
    if day!=datetime.now(UTC).date(): raise ValueError('Live RPC cannot backfill historical snapshots; capture only today')
    run_id=str(uuid.uuid4())
    with conn,conn.cursor() as cur:
        cur.execute("""INSERT INTO backpack_job_leases VALUES('daily',%s,now()+interval '30 minutes')
            ON CONFLICT(name) DO UPDATE SET owner=excluded.owner,expires_at=excluded.expires_at
            WHERE backpack_job_leases.expires_at<now() RETURNING owner""",(run_id,))
        if not cur.fetchone(): return {'status':'already_running'}
        insert(cur,'backpack_ingestion_runs',dict(run_id=run_id,snapshot_date=day))
    counts=defaultdict(int)
    errors=[]
    try:
        assets=fetch_all(conn,"SELECT * FROM backpack_assets WHERE active AND verification_status<>'pending' AND (launch_date IS NULL OR launch_date<=%s) ORDER BY asset_type='bp',id",(day,))
        labels={r['wallet_address']:r for r in fetch_all(conn,'SELECT * FROM backpack_wallet_labels')}
        calendar=calendar_for(day)
        for asset in assets:
            counts['attempted']+=1
            existing=fetch_all(conn,'SELECT holders_complete FROM backpack_asset_daily_snapshots WHERE asset_id=%s AND date=%s',(asset['id'],day))
            if existing and existing[0]['holders_complete']:
                counts['skipped']+=1
                continue
            try:
                outcome=collect_asset(conn,p,asset,run_id,day,labels,calendar)
                if outcome=='incomplete':
                    counts['failed']+=1
                    errors.append(f"asset {asset['id']}: holder reconciliation incomplete; current-day retry required")
                else:
                    counts[outcome]+=1
            except Exception as error:
                conn.rollback()
                counts['failed']+=1
                # Do not leak URLs, response bodies, DB DSNs, or provider credentials.
                safe=str(error) if isinstance(error,SourceError) else type(error).__name__
                errors.append(f"asset {asset['id']}: {safe}")
                with conn,conn.cursor() as cur:
                    insert(cur,'backpack_data_quality_events',dict(run_id=run_id,asset_id=asset['id'],date=day,metric='capture',
                        status='Unavailable',source='Daily collector',calculation='Daily capture',limitation=safe))
        aggregate(conn,run_id,day,assets)
        from .analytics import precompute
        precompute(conn,day)
        from .research import capture_research
        capture_research(conn,day,p.env)
        security_ids={a['id'] for a in assets if a['asset_type']!='bp'}
        adoption=fetch_all(conn,'SELECT data FROM backpack_adoption_daily WHERE date=%s',(day,))
        assessments=fetch_all(conn,'SELECT period_days FROM backpack_adoption_assessments WHERE date=%s',(day,))
        observed_ids=set(map(int,adoption[0]['data'].get('assets',{}))) if adoption else set()
        if observed_ids != security_ids or {r['period_days'] for r in assessments}!={7,30,90}:
            errors.append('Current-day adoption evidence is incomplete; no classification was published')
    except Exception as error:
        conn.rollback()
        errors.append('Run: '+type(error).__name__)
    finally:
        status='failed' if errors and not counts['succeeded'] and not counts['skipped'] else 'partial' if errors else 'completed'
        with conn,conn.cursor() as cur:
            for provider,requests in p.usage.items():
                # Unknown billing units/cost stay NULL; actual request counts are always recorded.
                insert(cur,'backpack_provider_usage',dict(run_id=run_id,provider=provider,requests=requests,
                    methodology='HTTP attempts including retries; credit/cost schedule not configured'))
            cur.execute('''UPDATE backpack_ingestion_runs SET completed_at=now(),status=%s,assets_attempted=%s,
                assets_succeeded=%s,assets_failed=%s,assets_skipped=%s,source_errors=%s,
                data_quality_warnings='See per-metric quality events. Completed means execution completed, not full metric coverage.' WHERE run_id=%s''',
                (status,counts['attempted'],counts['succeeded'],counts['failed'],counts['skipped'],'; '.join(errors),run_id))
            cur.execute("DELETE FROM backpack_job_leases WHERE name='daily' AND owner=%s",(run_id,))
    # Maintenance failure must not roll back successfully captured observations.
    try:
        maintain(conn, day, p.env)
    except Exception as error:
        conn.rollback()
        with conn, conn.cursor() as cur:
            cur.execute("INSERT INTO backpack_operational_alerts(date,metric,detail) VALUES(%s,'maintenance',%s) ON CONFLICT DO NOTHING", (day, type(error).__name__))
    return dict(run_id=run_id,status=status,**counts)
