"""Channel analytics, propagation graph, case study and health report over the Telegram layer.

Read-only. Everything here is derived from telegram_token_mentions + telegram_mention_outcomes and
can be recomputed from scratch, so a rule change never costs data.

Two things this module refuses to do, both deliberate:

  * It does not score channels. There is no weighted "channel quality" number, because every weight
    in one would be a guess made before there is enough labelled history to justify it - the same
    reason the graduation archive sets no thresholds in V1. What it produces is measured typology:
    where a channel sits in the order of posting, and what happened afterwards, each with its own
    denominator.
  * It does not read a channel's claims as results. `claimed_multiple` is carried alongside the
    measured return so the two can be compared, never substituted.

Spec: docs/telegram-osint-spec.md.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
import statistics

RUNGS=(10,30,60,180,1440)
TIGHT_SECONDS=120          # two channels posting the same token inside this window is the coordination signal
MIN_TYPOLOGY_SAMPLE=10     # resolved, graduated mentions before a channel is labelled at all
EARLY_SECONDS=600          # "shortly after another source": within ten minutes of the first mention
LATE_SECONDS=1800
FIRST_SHARE=0.4            # share of tokens where this channel led the monitored set
PRE_GRADUATION_SHARE=0.5
WALLET_ALERT_SHARE=0.4
FORWARD_SHARE=0.5


def _median(values):
    # float() because psycopg2 hands back Decimal for numeric aggregates, and a Decimal in a JSON
    # payload is a serialisation failure at the far end rather than here.
    values=[float(v) for v in values if v is not None]
    return statistics.median(values) if values else None


def channel_rows(conn,network='solana',window_days=30,now=None):
    """Per-channel raw material for the statistics: one row per resolved mention."""
    now=now or datetime.now(timezone.utc);since=now-timedelta(days=window_days)
    with conn,conn.cursor() as cur:
        cur.execute('''SELECT m.channel_id,m.id,m.token_address,m.resolution,m.mention_origin,m.graduated,
                              m.seconds_to_graduation,m.is_first_monitored_mention,m.monitored_sequence,
                              m.seconds_after_first_mention,s.market_cap_at_mention,s.peak_multiple,
                              m.claimed_multiple
                       FROM telegram_token_mentions m
                       LEFT JOIN telegram_mention_summary s ON s.mention_id=m.id
                       WHERE m.network=%s AND m.mentioned_at>=%s''',(network,since))
        columns=['channel_id','id','token_address','resolution','origin','graduated',
                 'seconds_to_graduation','is_first','sequence','seconds_after_first',
                 'market_cap','peak_multiple','claimed_multiple']
        rows=[dict(zip(columns,r)) for r in cur.fetchall()]
        cur.execute('''SELECT channel_id,count(*),count(*) FILTER (WHERE is_forward)
                       FROM telegram_messages WHERE posted_at>=%s GROUP BY channel_id''',(since,))
        messages={r[0]:(r[1],r[2]) for r in cur.fetchall()}
        # Wallet-alert shape: a message carrying a 32-byte base58 address that is NOT a token we
        # resolved. That is what a smart-money relay looks like in the raw text, and it is measured
        # rather than asserted from a channel's name.
        cur.execute('''SELECT m.channel_id,count(DISTINCT m.message_id)
                       FROM telegram_messages m
                       JOIN telegram_token_mentions r ON r.channel_id=m.channel_id AND r.message_id=m.message_id
                       WHERE m.posted_at>=%s AND r.resolution='unresolved_not_in_archive'
                       GROUP BY m.channel_id''',(since,))
        wallet_alerts={r[0]:r[1] for r in cur.fetchall()}
        cur.execute('''SELECT m.channel_id,o.rung_minutes,o.status,count(*),
                              percentile_disc(0.5) WITHIN GROUP (ORDER BY o.return_pct)
                       FROM telegram_mention_outcomes o
                       JOIN telegram_token_mentions m ON m.id=o.mention_id
                       WHERE m.network=%s AND m.mentioned_at>=%s
                       GROUP BY m.channel_id,o.rung_minutes,o.status''',(network,since))
        outcomes=cur.fetchall()
    return rows,messages,wallet_alerts,outcomes


def rung_block(outcomes,channel_id):
    """{rung: {observed, pending, unobserved, median_return_pct, up_share}} for one channel.

    All three statuses are always present, so no consumer can read a rate without the denominator
    that produced it - a pending rung is not a loss, and an unobserved one is not a zero.
    """
    block={}
    for cid,rung,status,count,median in outcomes:
        if cid!=channel_id:continue
        entry=block.setdefault(str(rung),dict(observed=0,pending=0,unobserved=0,median_return_pct=None))
        entry[status]=count
        if status=='observed':entry['median_return_pct']=float(median) if median is not None else None
    return block


def typology(stats):
    """(label, reason) from measured behaviour. (None, reason) while the sample is too small.

    Precedence matters and is fixed here rather than left to whoever reads the row: coordination
    first, because a channel posting in lockstep with another is not independently early even when
    it is early; then relay shape; then ordering. A channel that meets no rule is 'unclassified',
    which is a real answer and better than stretching a rule to cover it.
    """
    sample=stats['graduated_tokens']
    if sample<MIN_TYPOLOGY_SAMPLE:
        return None,f'sample of {sample} graduated mentions is below the {MIN_TYPOLOGY_SAMPLE} needed to label a channel'
    first_share=stats['first_among_monitored']/sample
    pre_share=stats['pre_graduation']/sample
    lag=stats['median_seconds_after_first']
    if stats.get('tight_pair_share',0)>=FIRST_SHARE:
        return 'coordinated_cluster',(f"{stats['tight_pair_share']:.0%} of its tokens were also posted by another "
                                      f'monitored channel within {TIGHT_SECONDS}s')
    if stats['messages'] and stats['wallet_alert_messages']/max(stats['messages'],1)>=WALLET_ALERT_SHARE:
        return 'smart_wallet_relay',(f"{stats['wallet_alert_messages']}/{stats['messages']} messages carry addresses "
                                     'that are not archived tokens - wallet/on-chain alert shape')
    relayed=stats['forwarded_mentions']+stats.get('reposted_mentions',0)
    if relayed/max(stats['mentions'],1)>=FORWARD_SHARE:
        return 'relay',(f"{relayed}/{stats['mentions']} mentions relay another channel "
                        f"({stats['forwarded_mentions']} forwarded, {stats.get('reposted_mentions',0)} reposted "
                        'without attribution), so most of what it posts is re-transmission rather than discovery')
    if first_share>=FIRST_SHARE and pre_share>=PRE_GRADUATION_SHARE:
        return 'originator',(f'first among monitored channels on {stats["first_among_monitored"]}/{sample} tokens, '
                             f'{pre_share:.0%} of them before graduation')
    if lag is not None and lag<=EARLY_SECONDS:
        return 'early_amplifier',(f'median {int(lag)}s behind the first monitored mention, leading on only '
                                  f'{stats["first_among_monitored"]}/{sample}')
    if lag is not None and lag>LATE_SECONDS and pre_share<0.25:
        return 'late_promoter',(f'median {int(lag)}s behind the first monitored mention and only {pre_share:.0%} '
                                'of its mentions precede graduation')
    if lag is not None and lag>EARLY_SECONDS:
        return 'momentum_follower',f'median {int(lag)}s behind the first monitored mention'
    return 'unclassified','measured, but matches none of the behavioural rules'


def channel_stats(conn,network='solana',window_days=30,now=None,store=False):
    now=now or datetime.now(timezone.utc)
    rows,messages,wallet_alerts,outcomes=channel_rows(conn,network,window_days,now)
    tight=tight_pair_shares(conn,network,window_days,now)
    out=[]
    by_channel={}
    for row in rows:by_channel.setdefault(row['channel_id'],[]).append(row)
    for channel_id,mentions in by_channel.items():
        resolved=[m for m in mentions if m['token_address']]
        # Discovery is an original post. A forward and an unattributed repost are both relays of
        # someone else's post, and neither is this channel finding a token.
        originals=[m for m in resolved if m['origin']=='original']
        graduated=[m for m in originals if m['graduated']]
        counts={}
        for m in resolved:counts[m['token_address']]=counts.get(m['token_address'],0)+1
        stats=dict(channel_id=channel_id,network=network,window_days=window_days,computed_at=now,
                   messages=messages.get(channel_id,(0,0))[0],
                   mentions=len(mentions),
                   forwarded_mentions=sum(1 for m in mentions if m['origin']=='forward'),
                   reposted_mentions=sum(1 for m in mentions if m['origin']=='repost'),
                   unresolved_mentions=sum(1 for m in mentions if not m['token_address']),
                   wallet_alert_messages=wallet_alerts.get(channel_id,0),
                   tokens_distinct=len(counts),
                   repeat_mentions_median=_median(list(counts.values())),
                   graduated_tokens=len(graduated),
                   pre_graduation=sum(1 for m in graduated if (m['seconds_to_graduation'] or 0)>0),
                   median_seconds_to_graduation=_median([m['seconds_to_graduation'] for m in graduated]),
                   first_among_monitored=sum(1 for m in graduated if m['is_first']),
                   median_sequence=_median([m['sequence'] for m in graduated]),
                   median_seconds_after_first=_median([m['seconds_after_first'] for m in graduated]),
                   median_market_cap_at_mention=_median([m['market_cap'] for m in resolved]),
                   median_peak_multiple=_median([m['peak_multiple'] for m in resolved]),
                   peak_sample=sum(1 for m in resolved if m['peak_multiple'] is not None),
                   median_claimed_multiple=_median([m['claimed_multiple'] for m in mentions]),
                   tight_pair_share=tight.get(channel_id,0.0),
                   rungs=rung_block(outcomes,channel_id))
        stats['typology'],stats['typology_reason']=typology(stats)
        stats['typology_sample']=stats['graduated_tokens']
        out.append(stats)
    out.sort(key=lambda s:(-s['first_among_monitored'],-s['graduated_tokens']))
    if store:store_channel_stats(conn,out)
    return out


def store_channel_stats(conn,stats):
    from psycopg2.extras import Json
    with conn,conn.cursor() as cur:
        for row in stats:
            cur.execute('''INSERT INTO telegram_channel_stats
                  (channel_id,network,window_days,computed_at,messages,mentions,forwarded_mentions,
                   unresolved_mentions,tokens_distinct,repeat_mentions_median,graduated_tokens,
                   pre_graduation,median_seconds_to_graduation,first_among_monitored,median_sequence,
                   median_market_cap_at_mention,rungs,median_peak_multiple,peak_sample,typology,
                   typology_reason,typology_sample,wallet_alert_messages,median_seconds_after_first,
                   tight_pair_share,median_claimed_multiple,reposted_mentions)
                  VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                  ON CONFLICT DO NOTHING''',
                        (row['channel_id'],row['network'],row['window_days'],row['computed_at'],
                         row['messages'],row['mentions'],row['forwarded_mentions'],
                         row['unresolved_mentions'],row['tokens_distinct'],row['repeat_mentions_median'],
                         row['graduated_tokens'],row['pre_graduation'],row['median_seconds_to_graduation'],
                         row['first_among_monitored'],row['median_sequence'],
                         row['median_market_cap_at_mention'],Json(row['rungs']),row['median_peak_multiple'],
                         row['peak_sample'],row['typology'],row['typology_reason'],row['typology_sample'],
                         row['wallet_alert_messages'],row['median_seconds_after_first'],
                         row['tight_pair_share'],row['median_claimed_multiple'],row['reposted_mentions']))
    return len(stats)


def pairs(conn,network='solana',window_days=30,now=None,store=False):
    """Channel pairs that posted the same token, with how tightly and who led.

    Forward-involving pairs are counted separately: a token that reached channel B by B forwarding
    A's post is one source seen twice, and treating it as two independent sightings is how a
    coordinated cluster comes to look like consensus.
    """
    now=now or datetime.now(timezone.utc);since=now-timedelta(days=window_days)
    with conn,conn.cursor() as cur:
        cur.execute('''SELECT a.channel_id,b.channel_id,a.token_address,
                              extract(epoch FROM b.mentioned_at-a.mentioned_at),
                              (a.mention_origin<>'original' OR b.mention_origin<>'original')
                       FROM telegram_token_mentions a
                       JOIN telegram_token_mentions b
                         ON b.network=a.network AND b.token_address=a.token_address
                        AND b.channel_id>a.channel_id
                       WHERE a.network=%s AND a.token_address IS NOT NULL AND a.mentioned_at>=%s''',
                    (network,since))
        rows=cur.fetchall()
    edges={}
    for a,b,token,delta,forwarded in rows:
        edge=edges.setdefault((a,b),dict(channel_a=a,channel_b=b,network=network,window_days=window_days,
                                         computed_at=now,tokens=set(),gaps=[],tight_pairs=0,
                                         a_first=0,b_first=0,forward_pairs=0))
        edge['tokens'].add(token);edge['gaps'].append(abs(delta or 0))
        if abs(delta or 0)<=TIGHT_SECONDS:edge['tight_pairs']+=1
        if (delta or 0)>0:edge['a_first']+=1
        elif (delta or 0)<0:edge['b_first']+=1
        if forwarded:edge['forward_pairs']+=1
    out=[]
    for edge in edges.values():
        edge['shared_tokens']=len(edge.pop('tokens'))
        edge['median_gap_seconds']=_median(edge.pop('gaps'))
        out.append(edge)
    out.sort(key=lambda e:(-e['tight_pairs'],-e['shared_tokens']))
    if store:
        with conn,conn.cursor() as cur:
            for edge in out:
                cur.execute('''INSERT INTO telegram_channel_pairs
                      (channel_a,channel_b,network,window_days,computed_at,shared_tokens,tight_pairs,
                       median_gap_seconds,a_first,b_first,forward_pairs)
                      VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING''',
                            (edge['channel_a'],edge['channel_b'],network,window_days,now,
                             edge['shared_tokens'],edge['tight_pairs'],edge['median_gap_seconds'],
                             edge['a_first'],edge['b_first'],edge['forward_pairs']))
    return out


def tight_pair_shares(conn,network='solana',window_days=30,now=None):
    """Per channel: the share of its tokens that another monitored channel also posted within
    TIGHT_SECONDS. This is the input to the coordinated-cluster typology, and it is a share of the
    channel's own tokens so a prolific channel cannot inherit the label by volume alone."""
    now=now or datetime.now(timezone.utc);since=now-timedelta(days=window_days)
    with conn,conn.cursor() as cur:
        cur.execute('''SELECT a.channel_id,
                              count(DISTINCT a.token_address) FILTER (WHERE b.id IS NOT NULL),
                              count(DISTINCT a.token_address)
                       FROM telegram_token_mentions a
                       LEFT JOIN telegram_token_mentions b
                         ON b.network=a.network AND b.token_address=a.token_address
                        AND b.channel_id<>a.channel_id
                        AND abs(extract(epoch FROM b.mentioned_at-a.mentioned_at))<=%s
                       WHERE a.network=%s AND a.token_address IS NOT NULL AND a.mentioned_at>=%s
                       GROUP BY a.channel_id''',(TIGHT_SECONDS,network,since))
        return {r[0]:(r[1]/r[2] if r[2] else 0.0) for r in cur.fetchall()}


def propagation(conn,token_address,network='solana'):
    """One token's path through the monitored channels, with the archive's own milestones.

    Forwards appear in the sequence flagged rather than removed: seeing which hop was a copy is the
    point of the graph.
    """
    with conn,conn.cursor() as cur:
        cur.execute('''SELECT symbol,name,launchpad_family,graduated,graduated_at,first_seen_at,
                              measure_pool,measure_pool_timing
                       FROM launchpad_tokens WHERE network=%s AND token_address=%s''',(network,token_address))
        row=cur.fetchone()
        token=dict(zip(['symbol','name','launchpad_family','graduated','graduated_at','first_seen_at',
                        'measure_pool','measure_pool_timing'],row)) if row else {'in_archive':False}
        cur.execute('''SELECT m.id,m.channel_id,coalesce(c.username,c.title,m.channel_id::text),
                              m.mentioned_at,m.is_forward,m.mention_origin,m.relay_of_channel_id,
                              m.relay_reason,m.forward_source,m.resolution,
                              m.seconds_to_graduation,m.monitored_sequence,m.seconds_after_first_mention,
                              s.market_cap_at_mention,s.base_price,s.peak_multiple,m.claimed_multiple
                       FROM (SELECT m.*,g.forward_from_name AS forward_source
                             FROM telegram_token_mentions m
                             JOIN telegram_messages g
                               ON g.channel_id=m.channel_id AND g.message_id=m.message_id) m
                       LEFT JOIN telegram_channels c ON c.channel_id=m.channel_id
                       LEFT JOIN telegram_mention_summary s ON s.mention_id=m.id
                       WHERE m.network=%s AND m.token_address=%s ORDER BY m.mentioned_at''',
                    (network,token_address))
        columns=['mention_id','channel_id','channel','mentioned_at','is_forward','origin',
                 'relay_of_channel_id','relay_reason','forward_source',
                 'resolution','seconds_to_graduation','sequence','seconds_after_first','market_cap',
                 'base_price','peak_multiple','claimed_multiple']
        hops=[dict(zip(columns,r)) for r in cur.fetchall()]
        cur.execute('''SELECT o.mention_id,o.rung_minutes,o.status,o.return_pct,o.position_value
                       FROM telegram_mention_outcomes o
                       JOIN telegram_token_mentions m ON m.id=o.mention_id
                       WHERE m.network=%s AND m.token_address=%s ORDER BY o.mention_id,o.rung_minutes''',
                    (network,token_address))
        for mention_id,rung,status,ret,value in cur.fetchall():
            for hop in hops:
                if hop['mention_id']==mention_id:
                    hop.setdefault('outcomes',{})[str(rung)]=dict(status=status,return_pct=ret,
                                                                  hundred_dollars=value)
    return dict(network=network,token_address=token_address,token=token,hops=hops,
                monitored_channels=len({h['channel_id'] for h in hops}),
                original_posts=sum(1 for h in hops if h['origin']=='original'),
                forwards=sum(1 for h in hops if h['origin']=='forward'),
                reposts=sum(1 for h in hops if h['origin']=='repost'))


FLEX='fvHLJUwsynVHJrssbZ8MLNyku9jt2izUspbBD4Spump'


def case_study(conn,token_address=FLEX,network='solana'):
    """The acceptance case: one token reconstructed end to end, item by item.

    This is a gate, not a report. Each item is answered from stored data or explicitly reported as
    missing with what is missing, because a case study that quietly skips the parts it cannot
    answer is exactly how a pipeline gets declared working before it is.
    """
    graph=propagation(conn,token_address,network)
    originals=[h for h in graph['hops'] if h['origin']=='original']
    first=originals[0] if originals else None
    token=graph['token']
    items=[]
    def item(name,ok,detail):items.append(dict(item=name,satisfied=bool(ok),detail=detail))
    item('first observed Telegram mention',first,
         f"{first['channel']} at {first['mentioned_at']}" if first else
         'no monitored channel posted this token in the collected window')
    item('market cap / price at that mention',first and (first.get('market_cap') or first.get('base_price')),
         f"market cap {first.get('market_cap')}, price at mention {first.get('base_price')}" if first else None)
    item('graduation timing',token.get('graduated_at'),
         (f"graduated {token.get('graduated_at')}, first mention "
          f"{'before' if (first or {}).get('seconds_to_graduation',0) and first['seconds_to_graduation']>0 else 'after'} it"
          f" by {abs(first['seconds_to_graduation']):.0f}s" if first and first.get('seconds_to_graduation') is not None
          else 'the archive holds no graduation for this token')
         if token.get('graduated_at') else 'token is not in the graduation archive')
    # One channel posting twice is not propagation, so the chain is distinct channels in the order
    # they first posted - which is also what the propagation question actually asks.
    chain=[];seen=set()
    for hop in originals:
        if hop['channel_id'] in seen:continue
        seen.add(hop['channel_id']);chain.append(hop)
    relays=[h for h in graph['hops'] if h['origin']!='original']
    item('relays told apart from independent mentions',
         not relays or all(h['relay_reason'] or h['origin']=='forward' for h in relays),
         [f"{h['channel']}: {h['origin']}"+(f" of {h['relay_of_channel_id']} ({h['relay_reason']})"
          if h['relay_of_channel_id'] else '') for h in relays] or 'no relayed mentions of this token')
    item('subsequent channels',len(chain)>1,
         ' -> '.join(f"{h['channel']} (+{h['seconds_after_first']:.0f}s)" if h.get('seconds_after_first')
                     else h['channel'] for h in chain) or 'no original posts collected')
    rungs=(first or {}).get('outcomes') or {}
    measured={k:v for k,v in rungs.items() if v['status']=='observed'}
    item('+30m / +1h / +3h / +24h performance from the first mention',
         all(rungs.get(r,{}).get('status')=='observed' for r in ('30','60','180','1440')),
         {r:dict(status=v['status'],return_pct=v.get('return_pct'),hundred_dollars=v.get('hundred_dollars'))
          for r,v in rungs.items()} or 'no outcome rows: the mention has no price history yet')
    return dict(token_address=token_address,network=network,token=token,
                hops=graph['hops'],original_posts=graph['original_posts'],forwards=graph['forwards'],
                reposts=graph['reposts'],
                measured_rungs=len(measured),acceptance=items,
                passed=all(i['satisfied'] for i in items),
                note=('Every item must be satisfied from stored data before the collector is scaled '
                      'past the pilot channel set. A pending rung is not a failure of the pipeline '
                      'but it is also not an acceptance: re-run once the horizon has elapsed.'))


def health(conn,network='solana',hours=48,now=None):
    """Is the Telegram layer continuous, and is what it produced measurable?

    Collection health and derivation health are reported apart, because a perfectly collected
    channel whose tokens the archive never saw produces zero measurable calls, and that is a
    coverage fact about the archive, not a collection failure.
    """
    now=now or datetime.now(timezone.utc);since=now-timedelta(hours=hours)
    out=dict(as_of=now.isoformat(),window_hours=hours)
    with conn,conn.cursor() as cur:
        cur.execute('''SELECT count(*),count(*) FILTER (WHERE NOT complete),coalesce(sum(messages_new),0),
                              coalesce(sum(flood_waits),0),min(started_at),max(started_at)
                       FROM telegram_collection_runs WHERE started_at>=%s''',(since,))
        runs=cur.fetchone()
        cur.execute('''SELECT coalesce(max(delta),0) FROM (
                         SELECT extract(epoch FROM started_at-lag(started_at) OVER (ORDER BY started_at)) AS delta
                         FROM telegram_collection_runs WHERE started_at>=%s) g''',(since,))
        longest=cur.fetchone()[0] or 0
        out['runs']=dict(recorded=runs[0],incomplete=runs[1],messages_new=runs[2],flood_waits=runs[3],
                         first=runs[4].isoformat() if runs[4] else None,
                         last=runs[5].isoformat() if runs[5] else None,
                         longest_interval_seconds=int(longest))
        cur.execute('''SELECT channel_id,coalesce(username,title,channel_id::text),access_state,
                              consecutive_failures,last_message_at,last_success_at,backfill_complete
                       FROM telegram_channels WHERE active ORDER BY last_success_at NULLS FIRST''')
        channels=[dict(channel_id=r[0],channel=r[1],access_state=r[2],consecutive_failures=r[3],
                       last_message_at=r[4].isoformat() if r[4] else None,
                       silent_seconds=int((now-r[4]).total_seconds()) if r[4] else None,
                       last_success_at=r[5].isoformat() if r[5] else None,
                       backfill_complete=r[6]) for r in cur.fetchall()]
        out['channels']=channels
        out['channels_losing_access']=[c['channel'] for c in channels
                                       if c['access_state'] in ('forbidden','not_found')
                                       or c['consecutive_failures']>=3]
        # Duplicate message ids are impossible by primary key, so the honest health signal is the
        # opposite one: gaps in a channel's id sequence, which are deletions or windows we missed.
        cur.execute('''SELECT channel_id,max(message_id)-min(message_id)+1-count(*),count(*)
                       FROM telegram_messages GROUP BY channel_id''')
        out['id_gaps']=[dict(channel_id=r[0],missing_ids=int(r[1] or 0),stored=r[2]) for r in cur.fetchall()]
        cur.execute('''SELECT count(*) FILTER (WHERE edited_at IS NOT NULL),
                              count(*) FILTER (WHERE deleted_detected_at IS NOT NULL),count(*)
                       FROM telegram_messages WHERE posted_at>=%s''',(since,))
        edited,deleted,total=cur.fetchone()
        out['messages']=dict(total=total,edited=edited,deleted_detected=deleted)
        cur.execute('''SELECT resolution,count(*) FROM telegram_token_mentions
                       WHERE network=%s AND mentioned_at>=%s GROUP BY resolution''',(network,since))
        out['resolution']={r[0]:r[1] for r in cur.fetchall()}
        cur.execute('''SELECT count(*) FILTER (WHERE m.token_address IS NOT NULL),
                              count(*) FILTER (WHERE t.token_address IS NOT NULL),
                              count(*) FILTER (WHERE t.measure_pool IS NOT NULL),
                              count(DISTINCT m.token_address) FILTER (WHERE s.mention_id IS NOT NULL)
                       FROM telegram_token_mentions m
                       LEFT JOIN launchpad_tokens t ON t.network=m.network AND t.token_address=m.token_address
                       LEFT JOIN telegram_mention_summary s ON s.mention_id=m.id
                       WHERE m.network=%s AND m.mentioned_at>=%s''',(network,since))
        resolved,in_archive,with_pool,measured=cur.fetchone()
        out['coverage']=dict(resolved_mentions=resolved,in_archive=in_archive,
                             with_measurement_pool=with_pool,tokens_with_prices=measured,
                             archive_share=round(in_archive/resolved,3) if resolved else None)
        cur.execute('''SELECT status,count(*) FROM telegram_mention_outcomes o
                       JOIN telegram_token_mentions m ON m.id=o.mention_id
                       WHERE m.network=%s AND m.mentioned_at>=%s GROUP BY status''',(network,since))
        out['outcome_status']={r[0]:r[1] for r in cur.fetchall()}
    gaps=[c for c in out['channels'] if c['silent_seconds'] and c['silent_seconds']>hours*3600]
    out['silent_channels']=[c['channel'] for c in gaps]
    out['verdict']=('collecting' if out['runs']['recorded'] and not out['runs']['incomplete']
                    and not out['channels_losing_access'] else 'DEGRADED: '+', '.join(filter(None,[
                        'no runs recorded' if not out['runs']['recorded'] else '',
                        f"{out['runs']['incomplete']} incomplete runs" if out['runs']['incomplete'] else '',
                        f"access lost on {len(out['channels_losing_access'])} channels"
                        if out['channels_losing_access'] else ''])))
    # Coverage is reported beside the verdict rather than folded into it: a low archive share is a
    # fact about which tokens these channels talk about, not a collection defect.
    if out['coverage']['archive_share'] is not None and out['coverage']['archive_share']<0.2:
        out['coverage']['note']=('under a fifth of resolved mentions are tokens the graduation archive holds; '
                                 'the archive is a graduate archive and samples a fraction of them, so most '
                                 'mentions are expected to be unmeasurable rather than wrong')
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channels',action='store_true',help='channel statistics + typologies')
    parser.add_argument('--pairs',action='store_true',help='co-mention edges between channels')
    parser.add_argument('--token',help='propagation graph for one token address')
    parser.add_argument('--health',action='store_true',help='collection + derivation health')
    parser.add_argument('--case-study',nargs='?',const=FLEX,help='acceptance reconstruction for one token (default FLEX)')
    parser.add_argument('--store',action='store_true',help='persist the rollups as well as printing them')
    parser.add_argument('--network',default='solana')
    parser.add_argument('--days',type=int,default=30)
    parser.add_argument('--hours',type=int,default=48)
    args=parser.parse_args()
    if not (args.channels or args.pairs or args.token or args.health or args.case_study):
        print(json.dumps(dict(mode='plan_only',network=args.network,reads_only=True,
                              network_requests=0,llm_calls=0)));return
    import psycopg2
    conn=psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
    try:
        out={}
        if args.channels:out['channels']=channel_stats(conn,args.network,args.days,store=args.store)
        if args.pairs:out['pairs']=pairs(conn,args.network,args.days,store=args.store)
        if args.token:out['propagation']=propagation(conn,args.token,args.network)
        if args.health:out['health']=health(conn,args.network,args.hours)
        if args.case_study:out['case_study']=case_study(conn,args.case_study,args.network)
        print(json.dumps(out,indent=1,default=str))
    finally:conn.close()


if __name__=='__main__':main()
