from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal as D
import pytest
from backpack.adoption import summarize, assess


def history(days=61, holder_step=1, supply_step=1):
    start=date(2026,1,1)
    return [dict(date=str(start+timedelta(days=i)),methodology='adoption-v1',holders=1000+i*holder_step,
        whole_token_holders=900+i*holder_step,multi_asset_holders=100,exclusion_fingerprint='stable',
        assets={str(a):dict(supply=str(D(1000)+D(i)*D(str(supply_step))),holders=1000+i*holder_step) for a in range(3)}) for i in range(days)]


@pytest.mark.parametrize('step,supply,state',[(10,1,'Growing rapidly'),(1,1,'Growing slowly'),(0,0,'Status quo'),(-1,-1,'Declining'),(1,-1,'Mixed')])
def test_states_without_prices(step,supply,state):
    rows=history(31,step,supply)
    assert assess(rows,date.fromisoformat(rows[-1]['date']),30)['state']==state


@pytest.mark.parametrize('period',[7,30,90])
def test_windows_and_price_independence(period):
    rows=history(period*2+1)
    day=date.fromisoformat(rows[-1]['date'])
    expected=assess(rows,day,period)
    assert expected['state']=='Growing slowly'
    for r in rows:r.update(reference_aum_usd=None,underlying_price=None,daily_swap_volume_usd=None,bp_price=99999999)
    assert assess(rows,day,period)==expected
    assert assess(rows[:-1],day,period)['state']=='Insufficient evidence'
    rows[period]['assets']['new']={'supply':'1000','holders':100}
    assert assess(rows,day,period)['state']=='Insufficient evidence'


def test_dust_breadth_and_exclusion_changes():
    rows=history(31,10,1);day=date.fromisoformat(rows[-1]['date'])
    rows[-1]['whole_token_holders']=900
    assert assess(rows,day,30)['state']=='Mixed'
    rows=history(31,10,1)
    for a in ('0','1'):rows[-1]['assets'][a]['holders']=1000
    assert assess(rows,day,30)['state']=='Mixed'
    rows=history(31)
    rows[10]['exclusion_fingerprint']='changed'
    assert assess(rows,day,30)['state']=='Insufficient evidence'


def test_baseline_and_invalid_settings():
    rows=history(1);day=date.fromisoformat(rows[0]['date'])
    r=assess(rows,day,7)
    assert r['state']=='Insufficient evidence' and r['observed_days']==1
    with pytest.raises(ValueError):assess(rows,day,7,{'BACKPACK_ADOPTION_RAPID_PCT_30D':'0'})
    with pytest.raises(ValueError):assess(rows,day,7,{'BACKPACK_ADOPTION_HOLDER_PCT_30D':'NaN'})
    rows=history(31);rows[0]['holders']=0
    assert assess(rows,date.fromisoformat(rows[-1]['date']),30)['state']=='Insufficient evidence'


def test_rapid_growth_can_slow_without_declining():
    rows=history(61,20,10)
    for i,r in enumerate(rows[31:],31):
        r['holders']=1600+(i-30)*10;r['whole_token_holders']=1500+(i-30)*10
        for a in r['assets'].values():a.update(supply=str(1300+(i-30)*2),holders=r['holders'])
    r=assess(rows,date.fromisoformat(rows[-1]['date']),30)
    assert r['state']=='Growing rapidly' and r['momentum']=='Slowing'


def test_ownership_deduplicates_and_globally_excludes_systems():
    snaps=[dict(asset_id=a,unique_holders=3,holders_complete=True,token_supply=D(100)) for a in (1,2)]
    holders=[dict(asset_id=a,wallet_address=w,balance_tokens=D(b),excluded=w=='system' and a==1)
             for a in (1,2) for w,b in [('shared','10'),('dust','0.001'),('system','89.999')]]
    r=summarize(date(2026,1,1),snaps,holders,2)
    assert r['holders']==2 and r['multi_asset_holders']==2 and r['whole_token_holders']==1
    assert r['assets']['1']['holders']==2
    assert summarize(date(2026,1,1),snaps,holders[:-1],2) is None
    snaps[0]['holders_complete']=False
    assert summarize(date(2026,1,1),snaps,holders,2) is None


def test_ecosystem_cohorts_deduplicate_wallets_without_persisting_identities():
    current_snaps=[dict(asset_id=a,unique_holders=2,holders_complete=True,token_supply=D(100)) for a in (1,2)]
    current_holders=[dict(asset_id=1,wallet_address='retained',balance_tokens=D(1),excluded=False),
                     dict(asset_id=1,wallet_address='entered',balance_tokens=D(1),excluded=False),
                     dict(asset_id=2,wallet_address='retained',balance_tokens=D(1),excluded=False),
                     dict(asset_id=2,wallet_address='entered',balance_tokens=D(1),excluded=False)]
    previous_snaps=[dict(asset_id=a,unique_holders=2,holders_complete=True,token_supply=D(100)) for a in (1,2)]
    previous_holders=[dict(asset_id=1,wallet_address='retained',balance_tokens=D(1),excluded=False),
                      dict(asset_id=1,wallet_address='departed',balance_tokens=D(1),excluded=False),
                      dict(asset_id=2,wallet_address='retained',balance_tokens=D(1),excluded=False),
                      dict(asset_id=2,wallet_address='departed',balance_tokens=D(1),excluded=False)]
    comparison={7:dict(day=date(2026,1,1),snapshots=previous_snaps,holders=previous_holders,expected=2)}
    result=summarize(date(2026,1,8),current_snaps,current_holders,2,comparison)
    assert result['cohorts']['7']==dict(window_days=7,baseline_holders=2,current_holders=2,
        retained_holders=1,entered_holders=1,departed_holders=1,retention_pct='50')
    assert 'wallet_address' not in str(result)  # Counts survive; wallet identities do not.


def test_assessment_exposes_matching_endpoint_cohort():
    rows=history(8)
    rows[-1]['cohorts']={'7':dict(retained_holders=900,entered_holders=107,departed_holders=100,retention_pct='90')}
    result=assess(rows,date.fromisoformat(rows[-1]['date']),7)
    assert result['retained_holders']==900 and result['entered_holders']==107
    assert result['departed_holders']==100 and result['retention_pct']==D(90)


def test_different_token_denominations_do_not_change_supply_growth():
    rows=history(31)
    before=assess(rows,date.fromisoformat(rows[-1]['date']),30)
    other=deepcopy(rows)
    for r in other:r['assets']['1']['supply']=str(D(r['assets']['1']['supply'])*1000000000)
    assert assess(other,date.fromisoformat(rows[-1]['date']),30)==before


def test_unissued_assets_do_not_block_growth_but_new_issuance_needs_baseline():
    rows=history(31);day=date.fromisoformat(rows[-1]['date'])
    for r in rows:r['assets']['dormant']=dict(supply='0',holders=0)
    result=assess(rows,day,30)
    assert result['state']=='Growing slowly' and result['unissued_securities']==1
    assert result['expanding_supply_pct']==100
    rows[-1]['assets']['dormant']['supply']='10'
    assert assess(rows,day,30)['state']=='Insufficient evidence'


def test_full_redemption_is_decline_not_missing_evidence():
    rows=history(31,-1,-1);day=date.fromisoformat(rows[-1]['date'])
    for a in rows[-1]['assets'].values():a.update(supply='0',holders=0)
    rows[-1].update(holders=0,whole_token_holders=0,multi_asset_holders=0)
    result=assess(rows,day,30)
    assert result['state']=='Declining' and result['median_supply_growth_pct']==-100
