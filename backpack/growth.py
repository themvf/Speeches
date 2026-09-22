"""Adoption direction independent of trading coverage and BP price. Decimal, stored inputs only."""
from datetime import timedelta
from decimal import Decimal
from .metrics import number


def assess_growth(history, day, period, issuance_threshold=Decimal('0.1'), holder_threshold=Decimal('1'), slowdown_pp=Decimal('0.25')):
    if period not in (7,30,90) or any(not n.is_finite() or n<0 for n in (issuance_threshold,holder_threshold,slowdown_pp)):
        raise ValueError('Invalid growth assessment settings')
    result=dict(date=day,period_days=period,state='Insufficient evidence',momentum='Insufficient evidence',
        reason='Requires consecutive priced, complete holder observations for an unchanged security universe.',
        issuance_threshold_pct=issuance_threshold,holder_threshold_pct=holder_threshold,slowdown_threshold_pp=slowdown_pp,
        net_issuance_usd=None,previous_net_issuance_usd=None,issuance_aum_pct=None,previous_issuance_aum_pct=None,
        holder_growth_pct=None,previous_holder_growth_pct=None,aum_growth_pct=None,
        meaningful_holders=None,previous_meaningful_holders=None,multi_asset_adoption_pct=None,previous_multi_asset_adoption_pct=None,
        top_5_aum_pct=None,previous_top_5_aum_pct=None,significant_securities=None,previous_significant_securities=None,
        significance_threshold_usd=None,supply_effect_usd=None,price_effect_usd=None,
        methodology='growth-v1: adoption direction uses net issuance / starting reference AUM and meaningful-holder growth. Trading and BP price are excluded. Momentum compares adjacent equal-length windows. Endpoint AUM decomposition: (S1-S0)*P0 + S1*(P1-P0); price includes the interaction term.')
    indexed={r['date']:r for r in history}
    def window(end):
        rows=[indexed.get(end-timedelta(days=i)) for i in range(period+1)]
        if any(r is None for r in rows):return None
        cohort=rows[0].get('cohort')
        if not cohort or any(r.get('cohort')!=cohort or r.get('valuation_current') is not True for r in rows):return None
        if any(number(r.get(k)) is None for r in rows for k in ('reference_aum_usd','meaningful_holders')):return None
        if any(number(r.get('net_supply_change_usd')) is None for r in rows[:-1]):return None
        start=rows[-1];finish=rows[0]
        if number(start['reference_aum_usd'])<=0 or number(start['meaningful_holders'])<=0:return None
        issued=sum(number(r['net_supply_change_usd']) for r in rows[:-1])
        return dict(rows=rows,cohort=cohort,issued=issued,rate=issued/number(start['reference_aum_usd'])*100,
                    holders=(number(finish['meaningful_holders'])/number(start['meaningful_holders'])-1)*100)
    current=window(day)
    if current is None:return result
    now,before=current['rows'][0],current['rows'][-1]
    result.update(net_issuance_usd=current['issued'],issuance_aum_pct=current['rate'],holder_growth_pct=current['holders'],
        aum_growth_pct=(number(now['reference_aum_usd'])/number(before['reference_aum_usd'])-1)*100,
        meaningful_holders=now['meaningful_holders'],previous_meaningful_holders=before['meaningful_holders'])
    for field in ('multi_asset_adoption_pct','top_5_aum_pct','significant_securities'):
        result[field]=now.get(field);result['previous_'+field]=before.get(field)
    result['significance_threshold_usd']=now.get('significance_threshold_usd')
    # Require matching asset identities and endpoint prices for the decomposition.
    a,b=now.get('components',{}),before.get('components',{})
    if set(a)==set(b)==set(current['cohort']) and all(number(x.get(k)) is not None for x in list(a.values())+list(b.values()) for k in ('token_supply','underlying_price')):
        result['supply_effect_usd']=sum((number(a[k]['token_supply'])-number(b[k]['token_supply']))*number(b[k]['underlying_price']) for k in a)
        result['price_effect_usd']=sum(number(a[k]['token_supply'])*(number(a[k]['underlying_price'])-number(b[k]['underlying_price'])) for k in a)
    previous=window(day-timedelta(days=period))
    if previous and previous['cohort']==current['cohort']:
        result.update(previous_net_issuance_usd=previous['issued'],previous_issuance_aum_pct=previous['rate'],previous_holder_growth_pct=previous['holders'])
        di=current['rate']-previous['rate'];dh=current['holders']-previous['holders']
        result['momentum']='Slowing' if di < -slowdown_pp and dh < -slowdown_pp else 'Accelerating' if di>slowdown_pp and dh>slowdown_pp else 'Mixed / steady'
    if current['rate']>issuance_threshold and current['holders']>holder_threshold:
        result['state']='Growing, but slowing' if result['momentum']=='Slowing' else 'Growing'
        result['reason']='Net issuance and meaningful-holder growth exceed the disclosed thresholds.'
    elif current['rate'] < -issuance_threshold and current['holders'] < -holder_threshold:
        result.update(state='Declining',reason='Net issuance and meaningful-holder growth are both below their negative thresholds over this window.')
    else:
        result.update(state='Mixed / flat',reason='Issuance and holder growth disagree or do not both exceed the disclosed thresholds. Trading alone cannot change this assessment.')
    if result['state']=='Declining':
        result['momentum']={'Accelerating':'Contraction easing','Slowing':'Contraction deepening'}.get(result['momentum'],result['momentum'])
    return result
