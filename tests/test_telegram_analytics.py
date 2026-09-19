"""Typology rules and the denominators they rest on.

These are the statistics a reader will quote, so what is pinned here is mostly what the numbers are
NOT allowed to do: label a channel off four calls, treat a forward as a discovery, or report a rate
without the three-way outcome denominator that produced it.
"""
import telegram_analytics as analytics


def stats(**over):
    base=dict(messages=200,mentions=60,forwarded_mentions=0,unresolved_mentions=0,
              wallet_alert_messages=0,tokens_distinct=50,graduated_tokens=40,pre_graduation=30,
              first_among_monitored=20,median_seconds_after_first=None,tight_pair_share=0.0)
    base.update(over)
    return base


def test_no_channel_is_labelled_off_a_handful_of_calls():
    label,reason=analytics.typology(stats(graduated_tokens=4,first_among_monitored=4))
    # Four-for-four is the most impressive-looking record there is, and it means nothing.
    assert label is None and 'below the' in reason


def test_an_originator_leads_the_monitored_set_before_graduation():
    label,reason=analytics.typology(stats(first_among_monitored=20,pre_graduation=30))
    assert label=='originator' and '20/40' in reason


def test_leading_after_graduation_is_not_being_early():
    # Posting first among the monitored channels, but only after the token already graduated, is a
    # different behaviour and must not inherit the originator label.
    label,_=analytics.typology(stats(first_among_monitored=20,pre_graduation=2,
                                     median_seconds_after_first=3600))
    assert label=='late_promoter'


def test_an_early_amplifier_is_close_behind_without_leading():
    label,reason=analytics.typology(stats(first_among_monitored=2,median_seconds_after_first=120))
    assert label=='early_amplifier' and '120s behind' in reason


def test_a_momentum_follower_arrives_long_after_the_first_mention():
    label,_=analytics.typology(stats(first_among_monitored=2,pre_graduation=30,
                                     median_seconds_after_first=5400))
    assert label=='momentum_follower'


def test_coordination_outranks_being_early():
    # A channel posting in lockstep with another is not independently early, however early it is.
    label,reason=analytics.typology(stats(first_among_monitored=20,tight_pair_share=0.8))
    assert label=='coordinated_cluster' and '120s' in reason


def test_a_channel_that_mostly_forwards_is_labelled_a_relay():
    label,reason=analytics.typology(stats(mentions=60,forwarded_mentions=40,
                                          median_seconds_after_first=60))
    assert label=='relay' and '40/60' in reason


def test_wallet_alert_shape_is_measured_from_the_messages_not_the_name():
    label,reason=analytics.typology(stats(wallet_alert_messages=120,median_seconds_after_first=60))
    assert label=='smart_wallet_relay' and '120/200' in reason


def test_a_measured_channel_matching_no_rule_says_so():
    label,reason=analytics.typology(stats(first_among_monitored=2,pre_graduation=30,
                                          median_seconds_after_first=None))
    assert label=='unclassified' and 'matches none' in reason


def test_every_rung_carries_all_three_statuses():
    # A rate whose denominator is not visible is the easiest way to publish a number nobody can
    # check - and here the denominator has three parts, not one.
    rows=[(1,10,'observed',7,12.5),(1,10,'pending',2,None),(1,1440,'unobserved',5,None)]
    block=analytics.rung_block(rows,1)
    assert block['10']==dict(observed=7,pending=2,unobserved=0,median_return_pct=12.5)
    assert block['1440']==dict(observed=0,pending=0,unobserved=5,median_return_pct=None)
    assert analytics.rung_block(rows,2)=={}


class Row(dict):
    """A propagation hop, shaped as analytics.propagation returns one."""


def test_acceptance_needs_a_price_at_the_mention_not_a_market_cap(monkeypatch):
    """The user's acceptance decision, pinned.

    Price at the mention is answerable for every mentioned graduate (measure_pool is set outside the
    25% ladder cohort). Market cap is not. So price gates acceptance and market cap rides along when
    directly observed - never derived, because a modelling assumption about constant supply does not
    belong inside a verification step.
    """
    hop=dict(mention_id=1,channel_id=1,channel='alpha',mentioned_at='2026-09-19T12:00:00Z',
             is_forward=False,origin='original',relay_of_channel_id=None,relay_reason=None,
             forward_source=None,resolution='contract',seconds_to_graduation=360,sequence=1,
             seconds_after_first=0,market_cap=None,base_price=0.0012,base_price_age_seconds=20,
             peak_multiple=3.0,claimed_multiple=None,
             outcomes={r:dict(status='observed',return_pct=5.0,hundred_dollars=105.0)
                       for r in ('30','60','180','1440')})
    monkeypatch.setattr(analytics,'propagation',lambda *a,**k:dict(
        network='solana',token_address='X',token=dict(graduated_at='2026-09-19T12:06:00Z'),
        hops=[hop,dict(hop,mention_id=2,channel_id=2,channel='beta',sequence=2,
                       mentioned_at='2026-09-19T12:04:00Z')],
        monitored_channels=2,original_posts=2,forwards=0,reposts=0))
    items={i['item']:i for i in analytics.case_study(None,'X')['acceptance']}
    price=items['price at the first original mention']
    assert price['satisfied'] is True
    assert price['detail']['price_at_mention']==0.0012
    # Absent market cap is explained rather than left looking like a collection failure.
    assert 'not observed' in price['detail']['market_cap_note'] and 'cohort' in price['detail']['market_cap_note']
