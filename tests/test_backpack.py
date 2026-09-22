from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D
from unittest.mock import Mock
import pytest
from backpack.metrics import *
from backpack.providers import Providers, SourceError
from backpack.collector import calendar_for


def test_unknown_never_zero():
    assert number(None) is None
    assert number('NaN') is None
    assert number(True) is None
    assert multiply(None,12) is None
    assert ratio(1,0) is None
    assert multiply(0,12)==0


def test_owner_aggregation_dust_and_system_confidence():
    accounts=[{'address':'a','owner':'wallet','amount':60}, {'address':'b','owner':'wallet','amount':40},
              {'address':'c','owner':'dust','amount':1}, {'address':'d','owner':'pool','amount':1000}]
    result=holders(accounts,0,1,{'pool':{'label':'Liquidity Pool','confidence':'high'}})
    assert result['holder_count']==4
    assert result['unique_holders']==3
    assert result['holders_over_100']==1
    assert not next(r for r in result['rows'] if r['wallet_address']=='wallet')['excluded']
    assert holders(accounts,0,1,{'pool':{'label':'Liquidity Pool','confidence':'low'}})['holders_over_100']==2
    assert holders(accounts,0,None)['holders_over_100'] is None
    assert holders(accounts+accounts,0,1)['holder_count']==4


def test_concentration_denominators():
    accounts=[{'address':str(i),'owner':str(i),'amount':100} for i in range(20)]+[{'address':'p','owner':'p','amount':8000}]
    result=holders(accounts,0,1,{'p':{'label':'Custody','confidence':'confirmed'}})
    assert result['top_10_holder_pct']==D(89)
    assert result['economic_top_10_holder_pct']==D(50)
    assert excluded({'label':'Unknown','confidence':'confirmed'}) is False


def test_exact_decimal_supply_and_consecutive_issuance():
    day=date(2026,9,22)
    before={'date':day-timedelta(days=1),'token_supply':'1000000000000.000001'}
    delta,usd=issuance('1000000000000.000003',before,day,'125.50')
    assert delta==D('0.000002') and usd==D('0.00025100')
    assert issuance(100,before,day+timedelta(days=1),20)==(None,None)
    assert issuance(100,None,day,20)==(None,None)


def test_multi_asset_dedup_and_dust():
    rows=[dict(wallet_address='a',asset_id=1,value_usd=D(100)),dict(wallet_address='a',asset_id=2,value_usd=D(1)),
          dict(wallet_address='b',asset_id=1,value_usd=D(100)),dict(wallet_address='b',asset_id=2,value_usd=D(100)),
          dict(wallet_address='pool',asset_id=1,value_usd=D(10000),excluded=True)]
    result=ecosystem(rows)
    assert result['meaningful_holders']==2
    assert result['multi_asset_2']==1
    assert result['multi_asset_adoption_pct']==50


def tx(signature='a',kind='SWAP'):
    return {'signature':signature,'slot':100,'timestamp':1788888600,'type':kind,'feePayer':'RELAYER','source':'JUPITER',
        'events':{'swap':{'tokenInputs':[{'mint':USDC,'userAccount':'USER','rawTokenAmount':{'tokenAmount':'100000000','decimals':6}}],
        'tokenOutputs':[{'mint':'SECURITY','userAccount':'USER','rawTokenAmount':{'tokenAmount':'5000000','decimals':6}}],
        'innerSwaps':[{'duplicated_route_leg':True}]}}}


def test_transfers_not_trades_and_routed_swap_dedup():
    assert normalize_swap(tx(kind='TRANSFER'),'SECURITY') is None
    swap=normalize_swap(tx(),'SECURITY')
    assert swap['volume_usd']==100 and swap['tokens']==5 and swap['wallet_address']=='USER'
    result=trading([swap,swap],{},True)
    assert result['trades']==1 and result['daily_swap_volume_usd']==100
    assert result['after_hours_volume_pct'] is None
    assert trading([swap],{},False)['daily_swap_volume_usd'] is None
    assert trading([],{},False)['observed_swap_volume_usd'] is None
    assert trading([],{},True)['daily_swap_volume_usd']==0


def test_unpriced_swap_not_zero():
    t=tx();t['events']['swap']['tokenInputs'][0]['mint']='OTHER'
    s=normalize_swap(t,'SECURITY')
    assert s['volume_usd'] is None
    assert trading([s],{},True)['daily_swap_volume_usd'] is None


@pytest.mark.parametrize('stamp,expected',[
 ('2026-03-06T14:30:00+00:00','regular'), # EST
 ('2026-03-09T13:30:00+00:00','regular'), # EDT
 ('2026-03-09T13:29:00+00:00','premarket'),
 ('2026-03-09T20:00:00+00:00','after_hours'),
 ('2026-11-27T18:01:00+00:00','after_hours'), # Black Friday early close
 ('2026-12-25T15:00:00+00:00','closed'),
 ('2026-09-20T15:00:00+00:00','weekend')])
def test_exchange_calendar(stamp,expected):
    dt=datetime.fromisoformat(stamp)
    assert session(dt,calendar_for(dt.date()))==expected


def test_stale_equity_no_false_parity_alert():
    now=datetime(2026,9,22,0,30,tzinfo=timezone.utc)
    result=parity(110,100,now,now-timedelta(hours=5),False)
    assert result['premium_discount_pct']==10
    assert not result['parity_alert_eligible']
    assert not parity(110,100,now,now-timedelta(hours=5),True)['parity_alert_eligible']
    assert parity(110,100,now,now,True)['parity_alert_eligible']


def test_quote_failure_unavailable():
    p=Providers({'JUPITER_API_KEY':'test'})
    p.request=Mock(return_value={'error':'no route'})
    with pytest.raises(SourceError):p.quote('SECURITY',6,D(100),10000,'buy')


def test_holder_pagination_budget_does_not_publish_partial_totals():
    p=Providers({'HELIUS_API_KEY':'test','BACKPACK_MAX_HOLDER_PAGES':'1'})
    p.rpc=Mock(return_value={'token_accounts':[{'address':'x','owner':'w','amount':1}],'last_indexed_slot':100,'cursor':'more'})
    with pytest.raises(SourceError):p.holders('SECURITY')


def test_holder_pagination_slots_and_dedup():
    p=Providers({'HELIUS_API_KEY':'test'})
    p.rpc=Mock(side_effect=[{'token_accounts':[{'address':'a','owner':'w','amount':1}],'last_indexed_slot':100,'cursor':'two'},
      {'token_accounts':[{'address':'a','owner':'w','amount':1},{'address':'b','owner':'q','amount':2}],'last_indexed_slot':102}])
    rows,start,end=p.holders('SECURITY')
    assert len(rows)==2 and start==100 and end==102


def test_budget_enforced_and_no_secret_leaks():
    p=Providers({'BACKPACK_MAX_REQUESTS':'0'})
    with pytest.raises(SourceError,match='budget'):p.rpc('getSlot',[])
    assert not p.usage


def test_bp_exact_identity():
    assert BP_MINT=='BPxxfRCXkUVhig4HS1Lh7kZqV6SPJhzfEk4x6fVBjPCy'


def test_whales_distinguish_price_crossings_accumulation_and_exits():
    from backpack.metrics import whale_cohorts
    def row(w, tokens, value, excluded=False):
        return dict(wallet_address=w,balance_tokens=D(tokens),value_usd=D(value),excluded=excluded)
    before=[row('price-only',100,90000),row('seller',200,200000),row('treasury',1000,1000000)]
    today=[row('price-only',100,110000),row('seller',50,55000),row('treasury',1000,1100000,True)]
    result=whale_cohorts(today,before)[0]
    assert result['whale_count']==1 and result['new_whales']==1 and result['exited_whales']==1
    assert result['whale_net_accumulation_tokens']==-150 # Treasury relabel does not masquerade as outflow.
    assert whale_cohorts([],before)[0]['whale_net_accumulation_tokens']==-1200
    assert whale_cohorts([],before,labels={'treasury':{'label':'Treasury','confidence':'high'}})[0]['whale_net_accumulation_tokens']==-200


def test_whales_unknown_history_and_zero_population():
    from backpack.metrics import whale_cohorts
    assert whale_cohorts(None)[0]['whale_count'] is None
    assert whale_cohorts([])[0]['whale_count']==0
    assert whale_cohorts([])[0]['new_whales'] is None
    assert whale_cohorts([],[])[0]['new_whales']==0
    assert whale_cohorts([dict(wallet_address='a',balance_tokens=D(1),value_usd=None,excluded=False)])[0]['whale_count'] is None
    with pytest.raises(ValueError):whale_cohorts([],thresholds=['NaN'])
