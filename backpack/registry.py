"""Curated starter approval only after exact primary-source and chain revalidation."""
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from .collector import fetch_all, insert, insert_many
from .providers import Providers


def seed_starter(conn,provider=None):
    p=provider or Providers()
    manifest=json.loads(Path(__file__).with_name('starter_universe.json').read_text())
    assets=p.request('Backpack primary registry','GET','https://api.backpack.exchange/api/v1/assets')
    securities=p.request('Backpack primary registry','GET','https://api.backpack.exchange/api/v1/securities')
    by_asset={r['symbol']:r for r in assets}
    by_security={r['asset']:r for r in securities}
    results=[];run_id=str(uuid.uuid4())
    for candidate in manifest['assets']:
        symbol=candidate['token_symbol']
        if fetch_all(conn,'SELECT 1 FROM backpack_assets WHERE solana_mint=%s',(candidate['solana_mint'],)):
            results.append(dict(check_name=symbol,status='Estimated',detail='Existing registry entry preserved; not reapproved or overwritten'))
            continue
        try:
            tokens=[t for t in by_asset[symbol]['tokens'] if t['blockchain']=='Solana']
            assert len(tokens)==1 and tokens[0]['contractAddress']==candidate['solana_mint']
            assert by_security[symbol]['name']==candidate['underlying_name']
            assert by_security[symbol].get('cusip')==candidate['cusip']
            supply,decimals,slot=p.supply(candidate['solana_mint'])
            assert decimals==candidate['decimals'] and supply>=0
            notes=('Curated exact official asset/security identity, checked against committed Nasdaq listing evidence. '
                   'Primary registry and live mint revalidated; mint decimals '+str(decimals)+' at slot '+str(slot)+'. '
                   '1:1 redemption source: '+manifest['backing_source']+'. Launch date unknown; deposit/withdraw flags do not establish launch or adoption. '
                   'Security metadata: '+candidate['security_source']+'; exchange: '+candidate['exchange_source'])
            record={k:candidate[k] for k in ('token_symbol','token_name','solana_mint','underlying_symbol','underlying_name','underlying_exchange','asset_type','issuer','launch_date','official_source')}
            record.update(verification_status='official',source_verified_at=datetime.now(timezone.utc),approval_notes=notes)
            with conn,conn.cursor() as cur:insert(cur,'backpack_assets',record)
            results.append(dict(check_name=symbol,status='Verified',detail='Registered exact official Solana mint; chain slot '+str(slot)))
        except Exception as error:
            conn.rollback()
            results.append(dict(check_name=symbol,status='Unavailable',detail='Not registered: '+type(error).__name__))
    with conn,conn.cursor() as cur:
        insert_many(cur,'backpack_readiness_checks',[dict(r,run_id=run_id) for r in results])
        insert_many(cur,'backpack_readiness_usage',[dict(run_id=run_id,provider=k,requests=v) for k,v in p.usage.items()])
    return dict(run_id=run_id,registry_results=results,provider_requests=dict(p.usage),failed=sum(r['status']=='Unavailable' for r in results))
