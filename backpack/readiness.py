"""Bounded production probes and read-only reconciliation. Never log credentials."""
from datetime import datetime, timezone
import uuid
from .collector import fetch_all, insert_many
from .metrics import BP_MINT, USDC, number
from .providers import Providers


def preflight(conn, provider=None):
    p=provider or Providers()
    run_id=str(uuid.uuid4())
    checks=[]
    def check(name, function, configured=True):
        if not configured:
            checks.append(dict(check_name=name,status='Unavailable',detail='Required credential is not configured'))
            return
        try:
            detail=function()
            checks.append(dict(check_name=name,status='Verified',detail=detail))
        except Exception as error:
            # Even unexpected SDK/HTTP/DB exceptions can contain connection strings.
            checks.append(dict(check_name=name,status='Unavailable',detail='Probe failed: '+type(error).__name__))
    def db():
        fetch_all(conn,'SELECT 1')
        return 'Connection and additive schema available; no market-data capture performed'
    def slot(independent=False):
        n=p.rpc('getSlot',[{'commitment':'finalized'}],independent)
        if type(n) is not int or n<=0:raise ValueError('Invalid slot')
        return 'Finalized slot '+str(n)
    def das():
        value=p.rpc('getTokenAccounts',{'mint':USDC,'limit':1,'options':{'showZeroBalance':False}})
        if not isinstance(value.get('token_accounts'),list) or value.get('last_indexed_slot') is None:raise ValueError('Invalid DAS response')
        return 'USDC one-page DAS probe only; not a holder snapshot'
    def price():
        value,at=p.price(USDC)
        if value<=0:raise ValueError('Invalid price')
        return 'USDC price endpoint and block timestamp available at '+at.isoformat()
    def equity():
        value,at,_=p.equity('AAPL')
        return 'AAPL SIP reference '+str(value)+' at '+at.isoformat()+'; test instrument, not registry approval'
    def validation():
        value=p.validation_market(BP_MINT)
        if number(value.get('priceUsd')) is None:raise ValueError('Missing price')
        return 'Exact-base BP pair found; this is not issuer/mint verification'
    check('database',db)
    check('helius_finalized_rpc',slot,bool(p.env.get('HELIUS_API_KEY')))
    check('helius_das',das,bool(p.env.get('HELIUS_API_KEY')))
    check('jupiter_price',price,bool(p.env.get('JUPITER_API_KEY')))
    check('alpaca_sip',equity,bool(p.env.get('ALPACA_API_KEY') and p.env.get('ALPACA_SECRET_KEY')))
    check('secondary_solana_rpc',lambda:slot(True))
    check('dexscreener_bp_price',validation)
    assets=fetch_all(conn,"SELECT id FROM backpack_assets WHERE active AND asset_type<>'bp' AND verification_status IN ('official','manual_approved') AND (launch_date IS NULL OR launch_date<=current_date)")
    checks.append(dict(check_name='starter_universe',status='Verified' if assets else 'Unavailable',
                       detail=str(len(assets))+' approved securities; provider probes do not establish mint identity'))
    with conn,conn.cursor() as cur:
        insert_many(cur,'backpack_readiness_checks',[dict(r,run_id=run_id) for r in checks])
        insert_many(cur,'backpack_readiness_usage',[dict(run_id=run_id,provider=k,requests=v) for k,v in p.usage.items()])
    return dict(run_id=run_id,checked_at=datetime.now(timezone.utc).isoformat(),
                ready_for_security_capture=all(r['status']=='Verified' for r in checks),checks=checks,
                provider_requests=dict(p.usage),manual_signoff='Not performed')


def audit(conn,day=None):
    """Stored observations only. Audit output cannot mark human sign-off complete."""
    if day is None:
        day=fetch_all(conn,'SELECT max(date) AS date FROM backpack_asset_daily_snapshots')[0]['date']
    if day is None:return dict(date=None,assets=[],manual_signoff='No snapshots available')
    rows=fetch_all(conn,'''SELECT s.*,a.token_symbol,a.solana_mint,a.asset_type,a.underlying_symbol,a.underlying_exchange,a.verification_status,a.official_source
        FROM backpack_asset_daily_snapshots s JOIN backpack_assets a ON a.id=s.asset_id WHERE s.date=%s ORDER BY s.asset_id''',(day,))
    output=[]
    for r in rows:
        totals=fetch_all(conn,'''SELECT sum(balance_tokens) AS tokens,count(*) AS wallets FROM backpack_asset_holder_daily_snapshots
            WHERE asset_id=%s AND date=%s''',(r['asset_id'],day))[0]
        quality=fetch_all(conn,'''SELECT metric,status,calculation,limitation FROM backpack_data_quality_events
            WHERE asset_id=%s AND date=%s ORDER BY id''',(r['asset_id'],day))
        expected=r['token_supply']*r['underlying_price'] if r['underlying_price'] is not None else None
        output.append(dict(asset_id=r['asset_id'],symbol=r['token_symbol'],mint=r['solana_mint'],registry_status=r['verification_status'],
            underlying_symbol=r['underlying_symbol'],underlying_exchange=r['underlying_exchange'],
            supply=r['token_supply'],supply_slot=r['slot'],supply_timestamp=r['block_timestamp'],
            holders_complete=r['holders_complete'],holder_token_sum=totals['tokens'],
            holder_supply_difference=totals['tokens']-r['token_supply'] if totals['tokens'] is not None else None,
            holder_start_slot=r['holder_start_slot'],holder_end_slot=r['holder_end_slot'],
            underlying_price=r['underlying_price'],underlying_price_timestamp=r['underlying_price_timestamp'],
            onchain_price=r['onchain_price'],onchain_price_timestamp=r['onchain_price_timestamp'],
            stored_reference_aum=r['reference_aum_usd'],reproduced_reference_aum=expected,
            aum_reconciles=(r['reference_aum_usd']==expected) if expected is not None else None,
            observed_swap_volume=r['observed_swap_volume_usd'],total_swap_volume=r['daily_swap_volume_usd'],quality=quality))
    return dict(date=str(day),assets=output,manual_signoff='Required: review registry sources, raw provider evidence and these reconciliations; execution success is not sign-off')
