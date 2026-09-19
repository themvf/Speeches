"""Resolution and outcome rules.

The outcome tests are the important ones: they pin the three-way status that keeps 'the horizon has
not elapsed' apart from 'we have no price' apart from 'it went down'. Collapsing those is how a
channel that posted an hour ago is made to look worse than one that posted last week.
"""
from datetime import datetime,timedelta,timezone
import pytest
import telegram_mentions as mentions

NOW=datetime(2026,9,19,18,0,tzinfo=timezone.utc)
MENTION=datetime(2026,9,19,12,0,tzinfo=timezone.utc)
FLEX='fvHLJUwsynVHJrssbZ8MLNyku9jt2izUspbBD4Spump'


def candles(count=200,start=MENTION,price=1.0,step=1.0,resolution='minute'):
    unit=timedelta(minutes=1) if resolution=='minute' else timedelta(hours=1)
    return [dict(minute=start+unit*i,resolution=resolution,open=price,high=price*1.5,low=price,
                 close=price*(step**i)) for i in range(count)]


def test_a_contract_is_resolved_whether_or_not_the_archive_holds_the_token():
    # Archive membership is a separate question. Conflating them makes "we never archived this
    # token" read as "the channel posted something unreadable".
    address,resolution,confidence=mentions.resolve_reference(
        dict(raw_reference=FLEX,reference_kind='contract'),lambda *_:[])
    assert (address,resolution,confidence)==(FLEX,'contract',1.0)


def test_a_ticker_resolves_only_when_exactly_one_archived_token_matches():
    one=lambda kind,value:[{'token_address':FLEX}]
    assert mentions.resolve_reference(dict(raw_reference='FLEX',reference_kind='ticker'),one)==(FLEX,'ticker_unique',0.7)


def test_an_ambiguous_ticker_is_never_silently_assigned():
    two=lambda kind,value:[{'token_address':FLEX},{'token_address':'X'*43}]
    address,resolution,_=mentions.resolve_reference(dict(raw_reference='FLEX',reference_kind='ticker'),two)
    # A wrong resolution is worse than an unresolved one: it puts another channel's token in this
    # channel's denominator.
    assert address is None and resolution=='unresolved_ambiguous'
    address,resolution,_=mentions.resolve_reference(dict(raw_reference='NOPE',reference_kind='ticker'),
                                                    lambda *_:[])
    assert address is None and resolution=='unresolved_not_in_archive'


def test_a_rung_whose_horizon_has_not_elapsed_is_pending_not_a_loss():
    result=mentions.mention_outcome(candles(20,step=1.0),MENTION,MENTION+timedelta(minutes=12))
    assert result['rungs'][10]['status']=='observed'
    assert result['rungs'][30]['status']=='pending' and result['rungs'][1440]['status']=='pending'
    # A pending rung carries no return at all, so it cannot be averaged into one by accident.
    assert 'return_pct' not in result['rungs'][30]


def test_an_elapsed_rung_with_no_candle_is_unobserved_with_its_reason():
    # A pool that stopped trading and a collection gap are indistinguishable here, so neither is
    # guessed and neither becomes a zero.
    result=mentions.mention_outcome(candles(15),MENTION,NOW)
    assert result['rungs'][10]['status']=='observed'
    assert result['rungs'][60]['status']=='unobserved'
    assert 'no minute candle' in result['rungs'][60]['unobserved_reason']


def test_a_loser_stays_in_the_denominator_with_a_measured_negative_return():
    result=mentions.mention_outcome(candles(60,price=1.0,step=0.9),MENTION,NOW)
    rung=result['rungs'][10]
    assert rung['status']=='observed' and rung['return_pct']<0
    # The brief's question: what a hypothetical $100 at the mention would be worth at this rung.
    assert rung['position_value']==pytest.approx(100*0.9**10,rel=1e-6)


def test_the_24h_rung_reads_hourly_candles_when_minute_candles_run_out():
    points=candles(60)+candles(30,price=3.0,step=1.0,resolution='hour')
    result=mentions.mention_outcome(points,MENTION,MENTION+timedelta(hours=30))
    assert result['rungs'][1440]['status']=='observed'
    assert result['rungs'][1440]['rung_price']==pytest.approx(3.0)


def test_without_a_price_at_the_mention_nothing_is_measured():
    # An unanchored return is not a conservative estimate, it is a different number.
    stale=candles(10,start=MENTION-timedelta(hours=5))
    result=mentions.mention_outcome(stale,MENTION,NOW)
    assert result['base_price'] is None
    # Elapsed rungs are unobserved; the 24h rung is still pending, which is a different fact.
    assert {result['rungs'][r]['status'] for r in (10,30,60,180)}=={'unobserved'}
    assert result['rungs'][1440]['status']=='pending'
    assert result['rungs'][10]['unobserved_reason']=='no price at the mention'


def test_peak_is_measured_from_the_mention_forward_only():
    # A high set before the channel posted is not something the channel called.
    before=candles(30,start=MENTION-timedelta(minutes=30));before[0]['high']=1000.0
    result=mentions.mention_outcome(before+candles(60,price=1.0,step=1.01),MENTION,NOW)
    assert result['peak']['peak_multiple']<10
    assert result['peak']['peak_at']>=MENTION


def test_ohlcv_parsing_survives_the_rows_the_api_actually_returns():
    payload={'data':{'attributes':{'ohlcv_list':[[1758283200,'1','2','0.5','1.5','900'],[],None,
                                                 ['bad',1,2,3,4,5]]}}}
    rows=mentions.parse_ohlcv(payload)
    assert len(rows)==1 and rows[0]['high']==2.0 and rows[0]['volume_usd']==900.0
    assert rows[0]['minute'].tzinfo is timezone.utc


ORIGINAL=('New call: FLEX just graduated, liquidity looks deep and the chart is clean, '
          'contract in the next message, size accordingly')
REPOST=('New call: FLEX just graduated, liquidity looks deep and the chart is clean, '
        'contract in the next message, size accordingly ⚡ via our partners')
INDEPENDENT=('FLEX holding above its graduation price two hours later, buyers still coming in, '
             'this one might actually have legs unlike the rest today')


def test_a_telegram_forward_is_a_relay_by_its_header():
    assert mentions.classify_origin('anything',True,[])[:2]==('forward',None)


def test_a_copy_paste_repost_is_attributed_to_the_post_it_copied():
    # The commoner shape in call channels, and the one with no header to announce it. Left
    # unattributed it becomes a second independent sighting of the same discovery.
    origin,relay_of,reason=mentions.classify_origin(REPOST,False,[(11,ORIGINAL)])
    assert origin=='repost' and relay_of==11 and '5-gram overlap' in reason


def test_two_channels_saying_different_things_are_independent():
    origin,relay_of,_=mentions.classify_origin(INDEPENDENT,False,[(11,ORIGINAL)])
    assert origin=='original' and relay_of is None


def test_a_bare_contract_posted_twice_is_two_posts_not_a_copy():
    # These channels legitimately post an address alone. Below the word floor only an exact copy
    # counts, or every terse channel would be labelled a relay of every other terse channel.
    assert mentions.classify_origin(FLEX,False,[(11,'ape '+FLEX)])[0]=='original'
    assert mentions.classify_origin(FLEX,False,[(11,FLEX)])[:2]==('repost',11)


def test_similarity_is_symmetric_and_bounded():
    assert mentions.similarity(ORIGINAL,ORIGINAL)==1.0
    assert mentions.similarity(ORIGINAL,INDEPENDENT)<mentions.REPOST_SIMILARITY
    assert mentions.similarity('',ORIGINAL)==0.0


def test_relay_reasons_state_what_was_measured_not_what_was_intended():
    """A relay label is an attribution of sequence, not a claim about intent.

    Channels share call-bot templates, quote the same launch announcement and reuse their own
    boilerplate - all of which produce high overlap with no copying involved. The consequence of the
    label is narrow and defensible on the observation alone (the later post takes no sequence
    position), so the reason string must stay a measurement. Same discipline as the Macro tab's
    assertion that no condition string contains a forecast verb.
    """
    forbidden=('copied','copying','stole','plagiar','deliberate','intentional','coordinat',
               'colluded','faked','shill','pump')
    reasons=[mentions.classify_origin(REPOST,False,[(11,ORIGINAL)])[2],
             mentions.classify_origin(FLEX,False,[(11,FLEX)])[2],
             mentions.classify_origin('anything',True,[])[2]]
    for reason in reasons:
        assert reason is None or not any(word in reason.lower() for word in forbidden),reason
