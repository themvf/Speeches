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
