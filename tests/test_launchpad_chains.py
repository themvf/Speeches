from datetime import datetime,timedelta,timezone
import pytest
from launchpad_chains import (CHAINS,ROBINHOOD,SOLANA,choose_measure_pool,classify,in_cohort,
                              parse_extended_info,parse_pool_list,parse_trades,summarize_trades)

DEEP='ALPZYXZBTvbmT1cyHxwXUgvFLCyMVuAXaJv9nLYDpieq'
EMPTY='FramDv5MyadCwonKaShAqNQToV8kKBMmxh6S4QzJtTeb'


def test_each_chain_keeps_its_own_dex_identity():
    assert classify('pons-v2',ROBINHOOD)=='curve' and classify('pons-v2-dex',ROBINHOOD)=='graduate'
    assert classify('pump-fun',SOLANA)=='curve' and classify('pumpswap',SOLANA)=='graduate'
    # A chain must never classify another chain's DEXes.
    assert classify('pump-fun',ROBINHOOD)=='other' and classify('pons-v2',SOLANA)=='other'
    for chain in CHAINS.values():
        assert not (chain.curve_dexes & chain.graduate_dexes)


def test_solana_measures_the_deepest_pool_not_the_documented_destination():
    # Measured live: the launchpad's own destination field pointed at a pool holding $0.43 across 3
    # trades while the token's real pool held $30,562 across 1,964. Trusting the field would have
    # measured a dead pool at every rung and shown that every graduate dies instantly.
    pools=[(DEEP,'pumpswap',30562.7),(EMPTY,'meteora-damm-v2',0.0),('9ACiZK8','meteora-dbc',0.0)]
    address,reason=choose_measure_pool(SOLANA,pools,destination=EMPTY)
    assert address==DEEP
    assert 'deepest' in reason and 'destination field disagreed' in reason


def test_robinhood_still_follows_its_destination_field():
    # The Pons pairing was verified to agree, so nothing changes for the chain already collecting.
    address,reason=choose_measure_pool(ROBINHOOD,[],destination='0xdead')
    assert address=='0xdead' and reason=='launchpad destination'


def test_a_token_with_no_liquid_pool_is_recorded_rather_than_dropped():
    # Dropping these would bias every survival rate upward: they are outcomes, not missing data.
    address,reason=choose_measure_pool(SOLANA,[(EMPTY,'pumpswap',0.0)],destination=EMPTY)
    assert address is None and reason=='no liquid pool'
    assert choose_measure_pool(SOLANA,[],destination=None)==(None,'no pools listed')


def test_pool_list_parsing():
    payload={'data':[{'attributes':{'address':DEEP,'reserve_in_usd':'31356.0'},
                      'relationships':{'dex':{'data':{'id':'pumpswap'}}}},
                     {'attributes':{'address':EMPTY,'reserve_in_usd':'0.0'},
                      'relationships':{'dex':{'data':{'id':'meteora-damm-v2'}}}}]}
    assert parse_pool_list(payload)==[(DEEP,'pumpswap',31356.0),(EMPTY,'meteora-damm-v2',0.0)]


def test_extended_info_captures_the_creator_layer():
    payload={'data':{'attributes':{'developer_address':'7H7SkM44aaLZp2Y4CdxFWa3dmK5bTPYfL2pqywFAmMLq',
        'developer_holding_percentage':'19.99','is_honeypot':'unknown','mint_authority':'no',
        'freeze_authority':'no','telegram_handle':'someone','websites':['https://example.com'],
        'description':'a coin','categories':['meme'],'gt_score':75.87}}}
    info=parse_extended_info(payload)
    assert info['developer_address'].startswith('7H7SkM44')
    assert info['developer_holding']==pytest.approx(19.99)
    assert info['website']=='https://example.com' and info['categories']==['meme']
    # The raw payload is kept so a later question can be asked without re-collecting.
    assert info['info_raw']['gt_score']==75.87


def test_extended_info_tolerates_an_empty_record():
    info=parse_extended_info({'data':{'attributes':{}}})
    assert info['developer_address'] is None and info['categories'] is None and info['website'] is None


def trade(wallet,second,kind='buy',tx=None):
    return {'attributes':{'tx_from_address':wallet,'block_timestamp':f'2026-09-19T19:25:{second:02d}Z',
                          'kind':kind,'to_token_amount':'100','from_token_amount':'1',
                          'volume_in_usd':'250.5','tx_hash':tx or f'tx{wallet}{second}','block_number':1}}


def test_trade_parsing_keeps_wallet_identity():
    rows=parse_trades({'data':[trade('walletA',9),trade('walletB',12,'sell')]},DEEP)
    assert [r['wallet'] for r in rows]==['walletA','walletB']
    assert [r['kind'] for r in rows]==['buy','sell']
    assert rows[0]['usd']==pytest.approx(250.5) and rows[0]['pool']==DEEP


def test_trade_summary_describes_the_opening_cohort():
    rows=parse_trades({'data':[trade('a',9),trade('a',10),trade('b',11),trade('c',12,'sell')]},DEEP)
    s=summarize_trades(rows)
    assert s['trades']==4 and s['wallets']==3 and s['buyers']==2 and s['sellers']==1
    assert s['repeat_wallets']==1 and s['top_wallet_share']==pytest.approx(0.5)
    assert s['earliest'].endswith('19:25:09Z') and s['latest'].endswith('19:25:12Z')


def test_trade_summary_of_nothing_is_not_an_error():
    s=summarize_trades([])
    assert s['trades']==0 and s['wallets']==0 and s['top_wallet_share'] is None


def test_cohort_draw_is_stable_and_independent_of_token_properties():
    # Same token, same answer on a re-run - the cohort is a property of the data, not of when the
    # code happened to run.
    assert in_cohort(SOLANA,'abc')==in_cohort(SOLANA,'abc')
    assert all(in_cohort(ROBINHOOD,t) for t in ('abc','def','ghi'))   # fraction 1.0 takes everything
    drawn=sum(in_cohort(SOLANA,f'token{i}') for i in range(4000))
    # Hashing the mint gives roughly the configured fraction and cannot correlate with liquidity,
    # buyers, or anything we might later treat as an outcome.
    assert 0.20 < drawn/4000 < 0.30


def test_solana_addresses_are_never_case_folded():
    from launchpad_archive import parse_multi,parse_pool
    # Solana base58 is case-sensitive. A live sweep folded every address to lower case and then
    # 404'd on every enrichment call; EVM hex folds safely, base58 does not.
    assert ROBINHOOD.lowercase_addresses and not SOLANA.lowercase_addresses
    mixed='CgDpum9wUpdLebGb6vD6F2sk4qaGLstDKArn296fpump'
    entry={'attributes':{'address':'PoolAddr','name':'X / SOL','pool_created_at':'2026-09-19T19:19:08Z'},
           'relationships':{'dex':{'data':{'id':'pump-fun'}}},
           'base_token':{'data':{'id':'solana_'+mixed}}}
    entry['relationships']['base_token']={'data':{'id':'solana_'+mixed}}
    assert parse_pool(entry,SOLANA)['token']==mixed          # untouched
    assert parse_pool(entry,ROBINHOOD)['token']==mixed.lower()
    payload={'data':[{'attributes':{'address':mixed,'launchpad_details':{'graduation_percentage':1.0}}}]}
    assert mixed in parse_multi(payload,SOLANA)
    assert mixed.lower() in parse_multi(payload,ROBINHOOD)
