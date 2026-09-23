from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D
from unittest.mock import Mock
import pytest
from backpack.metrics import *
from backpack.providers import Providers, SourceError
from backpack.collector import calendar_for


@pytest.mark.parametrize('mode,alpaca,helius,ready,full', [
    ('0','missing',True,True,False),
    ('0','failed',True,True,False),
    ('0','working',True,True,True),
    ('1','missing',True,False,False),
    (None,'missing',True,False,False),
    ('0','missing',False,False,False),
])
def test_readiness_optional_equity(monkeypatch,mode,alpaca,helius,ready,full):
    from unittest.mock import MagicMock
    from backpack import readiness
    env={'JUPITER_API_KEY':'test'}
    if mode is not None: env['BACKPACK_REQUIRE_EQUITY_REFERENCE']=mode
    if helius: env['HELIUS_API_KEY']='test'
    if alpaca!='missing': env.update(ALPACA_API_KEY='test',ALPACA_SECRET_KEY='test')
    p=Mock(env=env,usage={})
    p.rpc.side_effect=lambda method,*args: {'token_accounts':[], 'last_indexed_slot':1} if method=='getTokenAccounts' else 1
    p.price.return_value=(D(1),datetime.now(timezone.utc))
    p.validation_market.return_value={'priceUsd':'1'}
    if alpaca=='failed': p.equity.side_effect=SourceError('No SIP entitlement')
    else: p.equity.return_value=(D(100),datetime.now(timezone.utc),D(99))
    monkeypatch.setattr(readiness,'fetch_all',lambda *args:[{'id':1}])
    persisted=[]
    monkeypatch.setattr(readiness,'insert_many',lambda cur,table,rows: persisted.extend(rows) if table=='backpack_readiness_checks' else None)
    result=readiness.preflight(MagicMock(),p)
    assert result['ready_for_capture'] is ready
    assert result['ready_for_security_capture'] is full
    equity=next(r for r in persisted if r['check_name']=='alpaca_sip')
    assert equity['status']==('Verified' if alpaca=='working' else 'Unavailable')
    if mode=='0': assert 'optional for capture' in equity['detail']
    if alpaca=='missing': p.equity.assert_not_called()


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


def test_cli_database_failure_redacts_connection_secrets(monkeypatch,capsys):
    import backpack_monitor
    import psycopg2
    monkeypatch.setenv('DATABASE_URL','postgresql://user:VERY_SECRET_PASSWORD@example.test/db')
    monkeypatch.setattr('sys.argv',['backpack_monitor.py','--migrate'])
    def fail(*a,**k):raise RuntimeError('VERY_SECRET_PASSWORD')
    monkeypatch.setattr(psycopg2,'connect',fail)
    assert backpack_monitor.main()==1
    output=capsys.readouterr().out
    assert 'VERY_SECRET_PASSWORD' not in output and 'RuntimeError' in output


def test_holder_events_separate_price_changes_from_token_accumulation():
    from backpack.storage import holder_changes, retention_days
    old=dict(balance_tokens=D(10),value_usd=D(99),label='Unknown',label_confidence=None,excluded=False)
    assert holder_changes(old,dict(old))==[]
    assert holder_changes(old,dict(old,value_usd=D(101)))==['CROSSED_100']
    assert 'BALANCE_INCREASE' in holder_changes(old,dict(old,balance_tokens=D(11)))
    assert 'SYSTEM_LABEL_CHANGED' in holder_changes(old,dict(old,label='Custody',label_confidence='high',excluded=True))
    assert holder_changes(old,dict(old,value_usd=None))==[]
    assert 'EXITED_HOLDER' in holder_changes(old,None)
    assert retention_days({},'retention')==30
    with pytest.raises(ValueError):retention_days({'retention':'1'},'retention')


def test_cache_notification_never_forwards_to_redirect_or_logs_secrets(monkeypatch):
    import requests
    from backpack.revalidation import notify
    assert notify({})['cache_revalidation']=='Unavailable'
    mock=Mock(return_value=Mock(status_code=302))
    monkeypatch.setattr(requests,'post',mock)
    result=notify({'BACKPACK_REVALIDATE_SECRET':'private-secret'})
    assert result['cache_revalidation']=='Unavailable' and 'private-secret' not in str(result)
    assert mock.call_args.kwargs['allow_redirects'] is False
    mock.side_effect=requests.ConnectionError('private-secret')
    assert 'private-secret' not in str(notify({'BACKPACK_REVALIDATE_SECRET':'private-secret'}))


def research_history(period=7):
    today=date(2026,9,22)
    return [dict(date=today-timedelta(days=i),cohort={1,2},valuation_current=True,reference_aum_usd=D(110 if i<period else 100),
        meaningful_holders=110 if i<period else 100,net_supply_change_usd=D(1),daily_swap_volume_usd=D(200 if i<period else 100))
        for i in range(2*period)]


def test_environment_quadrants_and_incomplete_evidence():
    from backpack.research import classify
    rows=research_history();day=rows[0]['date']
    assert classify(rows,day)['state']=='Expansion'
    quiet=[dict(r,daily_swap_volume_usd=D(100)) for r in rows]
    assert classify(quiet,day)['state']=='Accumulation'
    churn=[dict(r,net_supply_change_usd=D(0),reference_aum_usd=D(100),meaningful_holders=100) for r in rows]
    assert classify(churn,day)['state']=='Churn'
    assert classify([dict(r,daily_swap_volume_usd=D(100)) for r in churn],day)['state']=='Stagnant'
    assert classify([dict(r,meaningful_holders=100) for r in rows],day)['state']=='Mixed'
    assert classify(rows[:-1],day)['state']=='Unavailable'
    assert classify([dict(r,daily_swap_volume_usd=None,observed_swap_volume_usd=500) for r in rows],day)['state']=='Unavailable'
    assert classify([dict(r,valuation_current=False) for r in rows],day)['state']=='Unavailable'
    rows[-1]['cohort']={1,3}
    assert classify(rows,day)['state']=='Unavailable'
    with pytest.raises(ValueError):classify(rows,day,1)


def test_cost_windows_do_not_infer_zero_or_attribute_shared_billing():
    from backpack.cost_review import review_window
    end=date(2026,9,22)
    storage=[dict(date=end-timedelta(days=i),total_bytes=1000-i*10) for i in range(8)]
    result=review_window(storage,[],7)
    assert result['storage_growth_bytes']==70 and result['projected_storage_bytes_30d']==1300
    assert result['infrastructure_cost_usd'] is None
    bills=[dict(date=end-timedelta(days=i),provider=p,metric='cost_usd',scope='shared',value=D('0.01')) for i in range(7) for p in ('Neon','Vercel')]
    assert review_window(storage,bills,7)['projected_30d_infrastructure_cost_usd'] is None
    attributed=[dict(b,scope='backpack') for b in bills]
    assert review_window(storage,attributed,7)['projected_30d_infrastructure_cost_usd']==D('0.6')
    assert review_window(storage,attributed[:-1],7)['infrastructure_cost_usd'] is None
    assert review_window(storage[:-1],[],7)['storage_growth_bytes'] is None
    assert review_window(storage,attributed,30)['infrastructure_cost_usd'] is None


def test_billing_export_validation(tmp_path):
    from backpack.cost_review import parse_billing
    p=tmp_path/'billing.csv'
    header='date,provider,scope,metric,value,unit,source\n'
    valid='2026-01-01,Neon,backpack,cost_usd,0,USD,https://example.test/invoice\n'
    p.write_text(header+valid)
    assert parse_billing(p)[0]['value']==0
    for bad in (valid.replace(',0,',',NaN,'),valid.replace(',0,',',-1,'),valid.replace(',USD,',',bytes,'),valid.replace('/invoice','/invoice?token=secret'),valid+valid):
        p.write_text(header+bad)
        with pytest.raises(ValueError):parse_billing(p)


def growth_history(period=7):
    start=date(2026,1,1)
    return [dict(date=start+timedelta(days=i),cohort={1},valuation_current=True,reference_aum_usd=D(1000),
        meaningful_holders=100+i if i<=period else 100+period+(i-period)*3,
        net_supply_change_usd=D(1 if i<=period else 5),daily_swap_volume_usd=None,
        components={1:dict(token_supply=D(100),underlying_price=D(10))}) for i in range(2*period+1)]


def test_growth_is_independent_of_trading_and_price_and_has_adjacent_windows():
    from backpack.growth import assess_growth
    for period in (7,30,90):
        rows=growth_history(period);day=rows[-1]['date']
        result=assess_growth(rows,day,period)
        assert result['state']=='Growing' and result['momentum']=='Accelerating'
        assert result['net_issuance_usd']==period*5 and result['previous_net_issuance_usd']==period
        assert assess_growth([dict(r,daily_swap_volume_usd=D(999999)) for r in rows],day,period)==result
        price_only=[dict(r,net_supply_change_usd=D(0),reference_aum_usd=D(1000+i*100)) for i,r in enumerate(rows)]
        assert assess_growth(price_only,day,period)['state']=='Mixed / flat'
        short=assess_growth(rows[-period-1:],day,period)
        assert short['state']=='Growing' and short['momentum']=='Insufficient evidence'


def test_growth_distinguishes_slowing_from_decline_and_rejects_bad_evidence():
    from backpack.growth import assess_growth
    rows=growth_history();day=rows[-1]['date']
    slowing=[dict(r,net_supply_change_usd=D(5 if i<=7 else 1),meaningful_holders=100+i*3 if i<=7 else 121+i-7) for i,r in enumerate(rows)]
    assert assess_growth(slowing,day,7)['state']=='Growing, but slowing'
    declining=[dict(r,net_supply_change_usd=D(-5),meaningful_holders=150-i*2) for i,r in enumerate(rows)]
    assert assess_growth(declining,day,7)['state']=='Declining'
    for broken in (rows[:-1], [dict(r,valuation_current=False) for r in rows], [dict(r,meaningful_holders=None) for r in rows],
                   [dict(r,cohort={2}) if i==10 else r for i,r in enumerate(rows)], [dict(r,net_supply_change_usd=None) if i==12 else r for i,r in enumerate(rows)]):
        assert assess_growth(broken,day,7)['state']=='Insufficient evidence'
    with pytest.raises(ValueError):assess_growth(rows,day,1)


def test_aum_decomposition_adds_to_endpoint_change_without_becoming_deposits():
    from backpack.growth import assess_growth
    rows=growth_history();rows[-1]=dict(rows[-1],reference_aum_usd=D(1440),components={1:dict(token_supply=D(120),underlying_price=D(12))})
    result=assess_growth(rows,rows[-1]['date'],7)
    assert result['supply_effect_usd']==200 and result['price_effect_usd']==240
    assert result['supply_effect_usd']+result['price_effect_usd']==rows[-1]['reference_aum_usd']-rows[-8]['reference_aum_usd']
