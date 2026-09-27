"""BP holder intelligence: cohort hysteresis, portfolio reads, the tiered classifier and alert rules.

Transaction fixtures use the raw RPC shape (jsonParsed unless stated) that getTransactionsForAddress returns.
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
import pytest
from backpack import alerts, cohorts, events as ev, portfolio as pf
from backpack.metrics import BP_MINT, USDC

UTC = timezone.utc
T0 = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
RENT = 2_039_280
OWNER, OTHER, SENDER = 'OwnerWallet', 'OtherWallet', 'SenderWallet'
TOKEN, TOKA, TOKB, TOKC, TOKD, LP, FEE, PD = 'TokenMint', 'TokA', 'TokB', 'TokC', 'TokD', 'LpMint', 'FeeMint', 'PdMint'
JUP, RAY, ORCA = 'JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4', '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8', 'whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc'
ATA = 'ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL'


def token(account, mint, owner, pre, post, decimals=6, lam_pre=None, lam_post=None):
    return dict(account=account, mint=mint, owner=owner, pre=pre, post=post, decimals=decimals,
                lam_pre=lam_pre if lam_pre is not None else (RENT + (pre if mint == pf.WSOL else 0) if pre is not None else 0),
                lam_post=lam_post if lam_post is not None else (RENT + (post if mint == pf.WSOL else 0) if post is not None else 0))


def make_tx(sig='sig-1', signers=(OWNER,), natives=None, tokens=(), fee=5000, err=None, programs=(), atas=(), slot=100,
            block_time=T0, extra_keys=(), json_encoding=False, creates=(), closes=()):
    """natives maps wallet -> (pre, post) lamports; the first signer pays the fee."""
    natives = natives or {OWNER: (10**10, 10**10 - fee)}
    keys = list(dict.fromkeys(list(signers) + list(natives) + list(extra_keys)))
    static_count = len(keys)
    lam = {k: natives.get(k, (0, 0)) for k in keys}
    for t in tokens:
        keys.append(t['account'])
        lam[t['account']] = (t['lam_pre'], t['lam_post'])
    keys += [p for p in programs if p not in keys]
    for p in programs: lam.setdefault(p, (1, 1))
    index = {k: i for i, k in enumerate(keys)}
    balance = lambda t, side: dict(accountIndex=index[t['account']], mint=t['mint'], owner=t['owner'],
                                   uiTokenAmount=dict(amount=str(t[side]), decimals=t['decimals'],
                                                      uiAmountString=str(D(t[side]) / D(10) ** t['decimals'])))
    instructions = [dict(programId=p, accounts=[], data='') for p in programs]
    instructions += [dict(program='spl-associated-token-account', programId=ATA,
                          parsed=dict(type='create', info=dict(wallet=w, mint=m, account=f'{w}-{m}-ata'))) for w, m in atas]
    instructions += [dict(program='system', programId='11111111111111111111111111111111',
                          parsed=dict(type='createAccount', info=dict(source=src, newAccount=acct, lamports=RENT))) for acct, src in creates]
    instructions += [dict(program='spl-token', programId=pf.TOKEN_PROGRAM,
                          parsed=dict(type='closeAccount', info=dict(account=acct, destination=dst))) for acct, dst in closes]
    message = dict(accountKeys=[dict(pubkey=k, signer=k in signers, writable=True, source='transaction') for k in keys],
                   instructions=instructions)
    meta = dict(err=err, fee=fee, preBalances=[lam[k][0] for k in keys], postBalances=[lam[k][1] for k in keys],
                preTokenBalances=[balance(t, 'pre') for t in tokens if t['pre'] is not None],
                postTokenBalances=[balance(t, 'post') for t in tokens if t['post'] is not None], innerInstructions=[])
    if json_encoding:
        static = keys[:static_count]
        message = dict(accountKeys=static, header=dict(numRequiredSignatures=len(signers)),
                       instructions=[dict(programIdIndex=index[p], accounts=[], data='') for p in programs])
        meta['loadedAddresses'] = dict(writable=keys[len(static):], readonly=[])
    return dict(slot=slot, blockTime=int(block_time.timestamp()), transaction=dict(signatures=[sig], message=message), meta=meta)


def enhanced_swap(owner, inputs, outputs, kind='SWAP', source='JUPITER'):
    return dict(type=kind, source=source, transactionError=None, events=dict(swap=dict(
        tokenInputs=[dict(userAccount=owner, mint=m) for m in inputs],
        tokenOutputs=[dict(userAccount=owner, mint=m) for m in outputs])))


def only(events):
    assert len(events) == 1, events
    return events[0]


# Classifier ---------------------------------------------------------------------------------------------------

def test_parsed_direct_swap_values_stable_leg_and_flags_new_position():
    tx = make_tx(tokens=[token('usdc-ata', USDC, OWNER, 1000_000000, 900_000000), token('tok-ata', TOKEN, OWNER, 0, 5_000000)],
                 programs=[JUP])
    e = only(ev.classify(tx, {OWNER}, enhanced=enhanced_swap(OWNER, [USDC], [TOKEN])))
    assert (e['kind'], e['tier'], e['venue']) == ('swap', 'parsed_swap', 'JUPITER')
    assert (e['input_mint'], e['input_raw'], e['output_mint'], e['output_raw']) == (USDC, 100_000000, TOKEN, 5_000000)
    assert e['usd_value'] == D(100) and e['valuation_status'] == 'estimated'
    assert e['new_position'] is True and e['pre_output_raw'] == 0 and e['post_output_raw'] == 5_000000
    assert e['ata_created'] is False and e['pre_balance_source'] == 'transaction'


def test_multi_hop_route_counts_economic_input_and_final_output_only():
    tx = make_tx(tokens=[token('usdc-ata', USDC, OWNER, 500_000000, 400_000000), token('mid-ata', TOKA, OWNER, 0, 0),
                         token('tok-ata', TOKEN, OWNER, 7, 107)], programs=[JUP, RAY, ORCA])
    e = only(ev.classify(tx, {OWNER}))
    assert (e['tier'], e['input_mint'], e['output_mint'], e['output_raw']) == ('inferred_swap', USDC, TOKEN, 100)
    assert 'Jupiter Aggregator v6' in e['venue'] and e['new_position'] is False


def test_aggregator_route_without_events_swap_is_inferred_and_labelled():
    tx = make_tx(tokens=[token('a', TOKA, OWNER, 100, 0), token('b', TOKB, OWNER, 0, 50)], programs=[JUP])
    e = only(ev.classify(tx, {OWNER}, enhanced=dict(type='UNKNOWN', source='JUPITER', events={})))
    assert e['kind'] == 'swap' and e['tier'] == 'inferred_swap'
    assert 'not a parsed swap' in e['detail']
    assert e['usd_value'] is None and e['valuation_status'] == 'unpriced'  # token-to-token: no stable or SOL leg
    assert (e['pre_input_raw'], e['post_input_raw']) == (100, 0)


def test_one_out_one_in_without_dex_evidence_is_unclassified_not_a_purchase():
    e = only(ev.classify(make_tx(tokens=[token('a', TOKA, OWNER, 100, 0), token('b', TOKB, OWNER, 0, 50)]), {OWNER}))
    assert e['kind'] == 'unclassified' and e['tier'] == 'unclassified'


def test_sol_swap_nets_fee_rent_and_temporary_wrapped_account():
    fee = 5000
    tx = make_tx(natives={OWNER: (10**10, 10**10 - fee - 10**9 - RENT)}, fee=fee, programs=[JUP],
                 tokens=[token('tok-ata', TOKEN, OWNER, None, 500_000000)], atas=[(OWNER, TOKEN)])
    price = lambda at: (D(150), at - timedelta(minutes=5))
    e = only(ev.classify(tx, {OWNER}, sol_price=price))
    assert (e['input_mint'], e['input_raw']) == (pf.NATIVE, 10**9)  # exactly 1 SOL; fee and ATA rent removed
    assert e['usd_value'] == D(150) and 'Jupiter SOL price' in e['valuation_source']
    assert e['ata_created'] is True and e['new_position'] is True


def test_sol_leg_without_nearby_price_stays_unpriced():
    tx = make_tx(natives={OWNER: (10**10, 10**10 - 5000 - 10**9)}, programs=[JUP], tokens=[token('t', TOKEN, OWNER, 0, 5)])
    e = only(ev.classify(tx, {OWNER}, sol_price=lambda at: None))
    assert e['usd_value'] is None and e['valuation_status'] == 'unpriced'


def test_incoming_transfer_and_airdrop_are_never_purchases():
    transfer = make_tx(signers=(SENDER,), natives={SENDER: (10**9, 10**9 - 5000)}, tokens=[token('o-ata', TOKEN, OWNER, 50, 150)])
    e = only(ev.classify(transfer, {OWNER}))
    assert e['kind'] == 'transfer_in' and e['tier'] == 'not_applicable' and not e['owner_signed']
    assert 'possible airdrop' in e['detail'] and e['new_position'] is False
    airdrop = make_tx(signers=(SENDER,), natives={SENDER: (10**9, 10**9 - 5000 - RENT)}, atas=[(OWNER, TOKEN)],
                      tokens=[token('o-ata', TOKEN, OWNER, None, 100)])
    e = only(ev.classify(airdrop, {OWNER}))
    assert e['kind'] == 'transfer_in' and e['new_position'] is True and e['ata_created'] is True
    assert 'incidental' not in e['detail']  # rent a sender parks in the owner's new account is not income


def test_rent_is_attributed_from_instructions_when_present():
    fee = 5000
    # The wallet pays 1 SOL for a swap while a sender funds a new account for the wallet in the same transaction.
    mixed = make_tx(natives={OWNER: (10**10, 10**10 - fee - 10**9), SENDER: (10**9, 10**9 - RENT)}, fee=fee, programs=[JUP],
                    tokens=[token('recv', TOKEN, OWNER, 0, 500_000000), token('newacct', TOKA, OWNER, None, 0)],
                    creates=[('newacct', SENDER)])
    assert only(ev.classify(mixed, {OWNER}))['input_raw'] == 10**9
    # Same swap where the wallet funds its own new output account: that rent still nets out exactly.
    own = make_tx(natives={OWNER: (10**10, 10**10 - fee - 10**9 - RENT)}, fee=fee, programs=[JUP],
                  tokens=[token('recv', TOKEN, OWNER, None, 500_000000)], creates=[('recv', OWNER)])
    assert only(ev.classify(own, {OWNER}))['input_raw'] == 10**9
    # Selling for SOL while closing an emptied account: refund to the wallet nets out, refund elsewhere is ignored.
    for destination, wallet_gain in ((OWNER, 10**9 + RENT), (SENDER, 10**9)):
        sell = make_tx(natives={OWNER: (10**10, 10**10 - fee + wallet_gain)}, fee=fee, programs=[JUP],
                       tokens=[token('t', TOKEN, OWNER, 500, None)], closes=[('t', destination)])
        e = only(ev.classify(sell, {OWNER}))
        assert (e['input_mint'], e['output_mint'], e['output_raw']) == (TOKEN, pf.NATIVE, 10**9), destination


def test_third_party_rent_never_becomes_owner_sol():
    # Two sender-funded accounts (above the incidental limit together) and a large Token-2022 account.
    for tokens in ([token('a', TOKEN, OWNER, None, 5), token('b', TOKA, OWNER, None, 7)],
                   [token('c', TOKB, OWNER, None, 9, lam_post=5_000_000)]):
        found = ev.classify(make_tx(signers=(SENDER,), natives={SENDER: (10**10, 10**10 - 10**7)}, tokens=tokens), {OWNER})
        assert {e['kind'] for e in found} == {'transfer_in'} and pf.NATIVE not in {e['output_mint'] for e in found}
    # An account the owner's delegate closes to someone else: the refund is not an owner sale of SOL.
    closed = make_tx(signers=(SENDER,), natives={SENDER: (10**9, 10**9 + RENT - 5000)}, tokens=[token('d', TOKEN, OWNER, 0, None)])
    assert ev.classify(closed, {OWNER}) == []
    # Owner-funded rent still nets out exactly while the owner is also paying SOL.
    buy = make_tx(natives={OWNER: (10**10, 10**10 - 5000 - 2 * 10**9 - 2 * RENT)}, programs=[JUP],
                  tokens=[token('x', TOKEN, OWNER, None, 1), token('y', TOKA, OWNER, None, 0)])
    assert only(ev.classify(buy, {OWNER}))['input_raw'] == 2 * 10**9


def test_wrap_and_unwrap_change_form_not_exposure():
    wrap = make_tx(natives={OWNER: (5 * 10**9, 5 * 10**9 - 5000 - 10**9 - RENT)}, tokens=[token('w', pf.WSOL, OWNER, None, 10**9, 9)])
    e = only(ev.classify(wrap, {OWNER}))
    assert (e['kind'], e['input_mint'], e['output_mint'], e['input_raw']) == ('wrap', pf.NATIVE, pf.WSOL, 10**9)
    unwrap = make_tx(natives={OWNER: (10**9, 10**9 - 5000 + RENT + 10**9)}, tokens=[token('w', pf.WSOL, OWNER, 10**9, None, 9)])
    assert only(ev.classify(unwrap, {OWNER}))['kind'] == 'unwrap'


def test_account_creation_and_closure_alone_produce_no_events():
    create = make_tx(natives={OWNER: (10**9, 10**9 - 5000 - RENT)}, tokens=[token('new', TOKEN, OWNER, None, 0)], atas=[(OWNER, TOKEN)])
    close = make_tx(natives={OWNER: (10**9, 10**9 - 5000 + RENT)}, tokens=[token('old', TOKEN, OWNER, 0, None)])
    memo = make_tx(programs=['MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr'])
    assert ev.classify(create, {OWNER}) == ev.classify(close, {OWNER}) == ev.classify(memo, {OWNER}) == []


def test_liquidity_operations_are_not_swaps():
    tx = make_tx(natives={OWNER: (10**10, 10**10 - 5000 - 10**9)}, programs=[RAY],
                 tokens=[token('u', USDC, OWNER, 100_000000, 0), token('lp', LP, OWNER, 0, 1000)])
    found = ev.classify(tx, {OWNER}, assets={LP: dict(asset_class='position')})
    assert {e['kind'] for e in found} == {'liquidity'} and len(found) == 3
    assert not any(e['tier'] in ev.ALERT_TIERS for e in found)
    # Without a position label, two-out-one-in through a DEX program is still liquidity, not a purchase.
    assert {e['kind'] for e in ev.classify(tx, {OWNER})} == {'liquidity'}


def test_failed_transactions_only_record_the_signer():
    failed = make_tx(err={'InstructionError': [2, {'Custom': 6001}]}, tokens=[token('t', TOKEN, OWNER, 5, 5)])
    e = only(ev.classify(failed, {OWNER, OTHER}))
    assert e['kind'] == 'failed' and e['wallet_address'] == OWNER
    receiver = make_tx(signers=(SENDER,), natives={SENDER: (10**9, 10**9)}, err={'x': 1}, tokens=[token('t', TOKEN, OWNER, 5, 5)])
    assert ev.classify(receiver, {OWNER}) == []


def test_several_tracked_wallets_in_one_transaction_get_separate_events():
    tx = make_tx(tokens=[token('a', TOKEN, OWNER, 100, 90), token('b', TOKEN, OTHER, 0, 10)])
    found = ev.classify(tx, {OWNER, OTHER})
    assert {(e['wallet_address'], e['kind'], e['event_index']) for e in found} == {(OWNER, 'transfer_out', 0), (OTHER, 'transfer_in', 0)}


def test_several_swaps_in_one_transaction_are_preserved_as_unpaired_legs():
    tx = make_tx(programs=[JUP], tokens=[token('a', TOKA, OWNER, 10, 0), token('b', TOKB, OWNER, 0, 5),
                                         token('c', TOKC, OWNER, 20, 0), token('d', TOKD, OWNER, 0, 7)])
    found = ev.classify(tx, {OWNER})
    assert len(found) == 4 and {e['kind'] for e in found} == {'unclassified'}
    assert [e['event_index'] for e in found] == [0, 1, 2, 3]
    assert 'cannot be paired' in found[0]['detail']


def test_transfer_fee_and_hook_mints_have_no_phantom_sale_leg():
    sell = make_tx(programs=[JUP, 'HookProgram1111111111111111111111111111111'],
                   tokens=[token('f', FEE, OWNER, 1000, 0), token('u', USDC, OWNER, 0, 50_000000)])
    e = only(ev.classify(sell, {OWNER}))
    assert e['kind'] == 'swap' and e['input_raw'] == 1000  # the fee withheld downstream is not the owner's leg
    received = make_tx(signers=(SENDER,), natives={SENDER: (10**9, 10**9 - 5000)}, tokens=[token('f', FEE, OWNER, 0, 990)])
    assert only(ev.classify(received, {OWNER}))['output_raw'] == 990


def test_permanent_delegate_transfer_is_never_a_sale():
    tx = make_tx(signers=('DelegateAuthority',), natives={'DelegateAuthority': (10**9, 10**9 - 5000)},
                 tokens=[token('p', PD, OWNER, 100, 0)])
    e = only(ev.classify(tx, {OWNER}, assets={PD: dict(extension_flags=['permanent_delegate'])}))
    assert e['kind'] == 'delegate_transfer' and 'never a sale' in e['detail']
    assert only(ev.classify(tx, {OWNER}))['kind'] == 'transfer_out'


def test_classification_is_deterministic_and_encoding_independent():
    tokens = [token('a', TOKA, OWNER, 10, 0), token('b', TOKB, OWNER, 0, 5), token('c', TOKC, OWNER, 20, 0), token('d', TOKD, OWNER, 0, 7)]
    first = ev.classify(make_tx(tokens=tokens, programs=[JUP]), {OWNER})
    again = ev.classify(make_tx(tokens=list(reversed(tokens)), programs=[JUP]), {OWNER})
    key = lambda e: (e['event_index'], e['kind'], e['input_mint'], e['output_mint'])
    assert list(map(key, first)) == list(map(key, again))
    parsed = ev.classify(make_tx(tokens=[token('a', TOKA, OWNER, 100, 0), token('b', TOKB, OWNER, 0, 50)], programs=[RAY]), {OWNER})
    raw = ev.classify(make_tx(tokens=[token('a', TOKA, OWNER, 100, 0), token('b', TOKB, OWNER, 0, 50)], programs=[RAY],
                              json_encoding=True), {OWNER})
    assert [(e['tier'], e['input_raw'], e['output_raw']) for e in parsed] == [(e['tier'], e['input_raw'], e['output_raw']) for e in raw]


def test_pre_balance_falls_back_to_snapshot_only_for_accounts_outside_the_transaction():
    tx = make_tx(tokens=[token('u', USDC, OWNER, 100_000000, 0), token('ata-1', TOKEN, OWNER, 0, 40)], programs=[JUP])
    snapshot = lambda owner, mint, before: [dict(address='ata-1', amount='0'), dict(address='ata-2', amount='500')] if mint == TOKEN else None
    e = only(ev.classify(tx, {OWNER}, snapshot=snapshot))
    assert e['pre_balance_source'] == 'snapshot' and e['pre_output_raw'] == 500 and e['post_output_raw'] == 540
    assert e['new_position'] is False


def test_swap_candidates_prefilter():
    assert ev.swap_candidates(make_tx(tokens=[token('a', TOKA, OWNER, 1, 0), token('b', TOKB, OWNER, 0, 1)]), {OWNER})
    assert not ev.swap_candidates(make_tx(tokens=[token('a', TOKA, OWNER, 1, 2)]), {OWNER})


# Portfolio ----------------------------------------------------------------------------------------------------

def account(address, mint, owner, amount, decimals=6, state='initialized', extensions=None, ui=None):
    info = dict(owner=owner, mint=mint, state=state, tokenAmount=dict(amount=str(amount), decimals=decimals,
                uiAmountString=ui if ui is not None else str(D(amount) / D(10) ** decimals)))
    if extensions: info['extensions'] = [dict(extension=x) for x in extensions]
    return dict(pubkey=address, account=dict(data=dict(parsed=dict(info=info))))


def test_owner_accounts_aggregate_exactly_and_anomalies_are_reported():
    rows = [account('a1', TOKEN, OWNER, 10**18 + 1, 9), account('a2', TOKEN, OWNER, 2, 9), account('x', TOKEN, OTHER, 5, 9),
            account('z', TOKA, OWNER, 0), account('bad', TOKB, OWNER, 3, 6), account('bad2', TOKB, OWNER, 3, 9)]
    holdings, notes = pf.parse_accounts(OWNER, rows, pf.TOKEN_PROGRAM)
    assert holdings[TOKEN]['raw'] == 10**18 + 3 and len(holdings[TOKEN]['accounts']) == 2
    assert TOKA not in holdings  # zero public balance with no confidential extension
    assert any('differs from requested owner' in n for n in notes) and any('disagree on decimals' in n for n in notes)


def test_token2022_confidential_and_interest_bearing_balances():
    rows = [account('c', TOKEN, OWNER, 0, extensions=['confidentialTransferAccount']),
            account('i', TOKA, OWNER, 1_000000, ui='1.05')]
    holdings, _ = pf.parse_accounts(OWNER, rows, pf.TOKEN_2022)
    assert holdings[TOKEN]['visibility'] == 'partial' and holdings[TOKEN]['raw'] == 0  # never reported as a zero holding
    assert holdings[TOKA]['raw'] == 1_000000 and holdings[TOKA]['ui'] == D('1.05')  # display amount kept separately


def test_wallet_read_status_never_treats_failed_reads_as_absent():
    sol = (dict(context=dict(slot=10), value=5 * 10**9), None)
    ok = lambda rows: (dict(context=dict(slot=11), value=rows), None)
    failed = (None, 'rpc_error')
    complete = pf.wallet_read(OWNER, sol, ok([account('a', TOKEN, OWNER, 5)]), ok([]))
    assert complete['status'] == 'complete' and set(complete['holdings']) == {TOKEN, pf.NATIVE}
    assert pf.wallet_read(OWNER, sol, ok([]), failed)['status'] == 'partial'
    assert pf.wallet_read(OWNER, failed, ok([]), ok([]))['status'] == 'partial'
    assert pf.wallet_read(OWNER, sol, failed, failed)['status'] == 'unavailable'
    oversized = pf.wallet_read(OWNER, sol, ok([account(str(i), f'm{i}', OWNER, 1) for i in range(3)]), ok([]), max_accounts=2)
    assert oversized['status'] == 'oversized' and oversized['holdings'] == {}


def test_asset_classification_records_reasons():
    nft = pf.classify_asset('nft', dict(interface='ProgrammableNFT', content=dict(metadata=dict(name='Art'))))
    assert nft['asset_class'] == 'nft'
    lp = pf.classify_asset('lp', dict(interface='FungibleToken', token_info=dict(symbol='SOL-USDC LP', decimals=6)))
    assert lp['asset_class'] == 'position' and 'liquidity' in lp['class_reason']
    spam = pf.classify_asset('spam', dict(interface='FungibleToken', content=dict(metadata=dict(name='Claim at bonus.xyz'))))
    assert spam['spam_class'] == 'suspected_spam' and 'advertises a link or claim' in spam['spam_reason']
    t22 = pf.classify_asset('t22', dict(interface='FungibleToken', mint_extensions=dict(permanentDelegate={}, transfer_fee_config={})))
    assert t22['extension_flags'] == ['permanent_delegate', 'transfer_fee_config']
    assert pf.classify_asset(USDC, dict(interface='FungibleToken', content=dict(metadata=dict(name='USD Coin'))))['is_stable']
    assert pf.classify_asset(BP_MINT)['is_bp'] and pf.classify_asset('x', None, decimals=0)['asset_class'] == 'unknown'


def test_balance_rows_keep_unknown_prices_unknown():
    now = T0
    read = pf.wallet_read(OWNER, (dict(context=dict(slot=1), value=2 * 10**9), None),
                          (dict(context=dict(slot=1), value=[account('a', TOKEN, OWNER, 3_000000), account('b', TOKA, OWNER, 1),
                                                             account('c', TOKB, OWNER, 5), account('n', 'NftMint', OWNER, 1, 0)]), None),
                          (dict(context=dict(slot=1), value=[]), None))
    assets = {'NftMint': pf.classify_asset('NftMint', dict(interface='V1_NFT'))}
    prices = {pf.WSOL: dict(price=D(150), price_at=now - timedelta(minutes=2), source='Jupiter Price V3'),
              TOKEN: dict(price=D('0.1'), price_at=now - timedelta(minutes=2), source='Jupiter Price V3'),
              TOKA: dict(price=D(9), price_at=now - timedelta(hours=3), source='Jupiter Price V3'),
              'NftMint': dict(price=D(99), price_at=now, source='Jupiter Price V3')}
    rows, total, unpriced = pf.balance_rows(read, assets, prices, now)
    by = {r['mint']: r for r in rows}
    assert by[pf.NATIVE]['value_usd'] == D(300) and by[pf.NATIVE]['price'] == D(150)  # native SOL priced from wrapped SOL
    assert by[TOKEN]['value_usd'] == D('0.3') and by[TOKEN]['dust'] is True
    assert by[TOKA]['pricing_status'] == 'stale' and by[TOKA]['value_usd'] is None and by[TOKA]['dust'] is None
    assert by[TOKB]['pricing_status'] == 'unpriced' and by[TOKB]['value_usd'] is None
    assert by['NftMint']['value_usd'] is None and 'NFT' in by['NftMint']['classification_reason']
    assert total == D('300.3') and unpriced == 2


# Cohorts ------------------------------------------------------------------------------------------------------

CFG = dict(size=3, exit_rank=4, exit_runs=2, entrant_cap=2, history_days=30)


def ranking(*wallets):
    return [dict(wallet_address=w, rank=i + 1, raw_balance=1000 - i) for i, w in enumerate(wallets)]


def previous(rows):
    return {r['wallet_address']: r for r in rows if r['member'] or r['event'] == 'queued'}


def test_ranking_uses_exact_raw_units_and_deterministic_ties():
    rows = [dict(wallet_address='b', balance_tokens=D('1.5'), excluded=False), dict(wallet_address='a', balance_tokens=D('1.5'), excluded=False),
            dict(wallet_address='sys', balance_tokens=D(9), excluded=True), dict(wallet_address='z', balance_tokens=D(0), excluded=False)]
    raw, filtered = cohorts.rank_owners(rows, 9)
    assert [(r['wallet_address'], r['raw_balance']) for r in raw] == [('sys', 9 * 10**9), ('a', 1_500_000_000), ('b', 1_500_000_000)]
    assert [(r['wallet_address'], r['rank']) for r in filtered] == [('a', 1), ('b', 2)]
    with pytest.raises(ValueError): cohorts.rank_owners([dict(wallet_address='x', balance_tokens=D('0.0000000001'), excluded=False)], 9)
    assert cohorts.exclusion_fingerprint(rows) == cohorts.exclusion_fingerprint(list(reversed(rows)))


def test_bootstrap_hysteresis_and_immediate_exclusion():
    rows, flags = cohorts.next_membership(ranking('a', 'b', 'c', 'd', 'e'), set(), None, CFG, T0)
    assert flags['bootstrap'] and [r['wallet_address'] for r in rows if r['member']] == ['a', 'b', 'c']
    # c falls to rank 4 (inside exit_rank): stays. b falls to 5: first strike only.
    rows, _ = cohorts.next_membership(ranking('a', 'x', 'y', 'c', 'b'), set(), previous(rows), dict(CFG, entrant_cap=0 or 1), T0)
    state = {r['wallet_address']: r for r in rows}
    assert state['c']['member'] and state['c']['event'] == 'rank_changed' and state['c']['below_exit_runs'] == 0
    assert state['b']['member'] and state['b']['below_exit_runs'] == 1
    assert state['x']['event'] == 'entered' and state['y']['event'] == 'queued' and not state['y']['member']
    # Second consecutive run below exit_rank: b leaves; the labelled system wallet a leaves immediately.
    rows, _ = cohorts.next_membership(ranking('x', 'y', 'c', 'q', 'b'), {'a'}, previous(rows), CFG, T0 + timedelta(days=1))
    state = {r['wallet_address']: r for r in rows}
    assert state['b']['event'] == 'left' and state['b']['exit_reason'] == 'below_exit_rank'
    assert state['a']['event'] == 'left' and state['a']['exit_reason'] == 'excluded_by_label'
    assert state['y']['event'] == 'entered'  # the queued wallet keeps its place


def test_recovery_resets_strikes_and_sold_out_wallets_leave_after_two_runs():
    rows, _ = cohorts.next_membership(ranking('a', 'b', 'c'), set(), None, CFG, T0)
    rows, _ = cohorts.next_membership(ranking('b', 'c'), set(), previous(rows), CFG, T0)  # a sold everything
    assert {r['wallet_address']: r['below_exit_runs'] for r in rows}['a'] == 1
    rows, _ = cohorts.next_membership(ranking('a', 'b', 'c'), set(), previous(rows), CFG, T0)
    assert {r['wallet_address']: r['below_exit_runs'] for r in rows}['a'] == 0
    rows, _ = cohorts.next_membership(ranking('b', 'c'), set(), previous(rows), CFG, T0)
    rows, _ = cohorts.next_membership(ranking('b', 'c'), set(), previous(rows), CFG, T0)
    assert {r['wallet_address']: r['exit_reason'] for r in rows}['a'] == 'no_positive_balance'


def test_entrant_cap_binds_and_is_recorded():
    rows, _ = cohorts.next_membership(ranking('a'), set(), None, dict(CFG, size=5, exit_rank=6), T0)
    rows, flags = cohorts.next_membership(ranking('a', 'b', 'c', 'd', 'e'), set(), previous(rows), dict(CFG, size=5, exit_rank=6), T0)
    assert flags['entrant_cap_bound'] and cohorts.summarize(rows) == dict(size=3, entered=2, left_count=0, rank_changed=0, queued=2)


# Alerts -------------------------------------------------------------------------------------------------------

def swap(wallet, minutes, usd=300, mint=TOKEN, tier='parsed_swap', new=True, sig=None, **extra):
    base = dict(signature=sig or f'{wallet}-{minutes}-{mint}', event_index=0, wallet_address=wallet, kind='swap', tier=tier,
                block_time=T0 + timedelta(minutes=minutes), input_mint=USDC if usd is not None else TOKA, input_raw=int(usd * 10**6) if usd is not None else 5, input_decimals=6,
                output_mint=mint, output_raw=1000, output_decimals=6, usd_value=D(usd) if usd is not None else None,
                pre_input_raw=10**12, new_position=new, pre_membership=False, finality='finalized')
    return dict(base, **extra)


RULES = {r: dict(enabled=True, params=p) for r, p in dict(
    new_position=dict(min_usd=250, window_minutes=60),
    multiple_buyers=dict(min_wallets=3, window_minutes=60, min_usd_per_wallet=250, exclude_mints=[USDC, pf.NATIVE, pf.WSOL]),
    accumulation=dict(min_wallets=5, window_minutes=1440, min_usd_per_wallet=250, exclude_mints=[USDC, pf.NATIVE, pf.WSOL]),
    major_sale=dict(min_fraction=0.5, window_minutes=60, exclude_mints=[USDC, pf.NATIVE, pf.WSOL])).items()}
MEMBERS = {'w1', 'w2', 'w3', 'w4', 'w5', 'w6'}


def rules(*names):
    return {k: v for k, v in RULES.items() if k in names}


def test_new_position_threshold_needs_a_valuation():
    found = alerts.evaluate([swap('w1', 0), swap('w2', 5, usd=200), swap('w3', 10, usd=None)], rules('new_position'), 7, MEMBERS)
    assert len(found) == 1 and found[0]['wallet_count'] == 1 and found[0]['alert_key'] == f'new_position|{TOKEN}|{T0.isoformat()}|7'


def test_multiple_buyers_counts_each_wallet_once_and_states_inferred_buyers():
    events = [swap('w1', 0), swap('w1', 1), swap('w2', 20, tier='inferred_swap'), swap('w3', 50, usd=150), swap('w3', 55, usd=150)]
    found = alerts.evaluate(events, rules('multiple_buyers'), 1, MEMBERS)
    assert len(found) == 1
    a = found[0]
    assert a['wallet_count'] == 3 and a['inferred_wallets'] == 1 and a['lowest_tier'] == 'inferred_swap'
    assert a['valuation_status'] == 'priced' and a['usd_value'] == D(1200)
    assert not alerts.evaluate([swap('w1', 0), swap('w2', 30), swap('w3', 61)], rules('multiple_buyers'), 1, MEMBERS)


def test_accumulation_requires_positive_net_flow():
    buyers = [swap(f'w{i}', i * 60) for i in range(1, 6)]
    assert len(alerts.evaluate(buyers, rules('accumulation'), 1, MEMBERS)) == 1
    dump = swap('w6', 100, mint=USDC, input_raw=10**6, output_raw=10, new=False)
    dump.update(input_mint=TOKEN)
    assert not alerts.evaluate(buyers + [dump], rules('accumulation'), 1, MEMBERS)


def test_major_sale_uses_pre_sale_holding_and_skips_funding_legs():
    sale = swap('w1', 0, mint=USDC, new=False)
    sale.update(input_mint=TOKEN, input_raw=600, pre_input_raw=1000)
    small = dict(sale, signature='small', wallet_address='w2', input_raw=400)
    funding = swap('w3', 0, new=False)  # spends all its USDC: a funding leg, not a sale signal
    funding.update(input_raw=10**9, pre_input_raw=10**9)
    found = alerts.evaluate([sale, small, funding], rules('major_sale'), 1, MEMBERS)
    assert [(a['mint'], a['wallet_count']) for a in found] == [(TOKEN, 1)]


def test_membership_tiers_and_transfers_gate_alerts():
    gated = [swap('w1', 0, pre_membership=True), swap('outsider', 1), swap('w2', 2, tier='unclassified'),
             swap('w3', 3, kind='transfer_in'), swap('w4', 4, finality='confirmed')]
    assert not alerts.evaluate(gated, RULES, 1, MEMBERS)
    transfers = ev.classify(make_tx(signers=(SENDER,), natives={SENDER: (10**9, 10**9)},
                                    tokens=[token(f'{w}-ata', TOKEN, w, 0, 10**9) for w in ('w1', 'w2', 'w3', 'w4', 'w5')]), MEMBERS)
    stored = [dict(e, pre_membership=False, usd_value=D(10**6)) for e in transfers]
    assert len(stored) == 5 and not alerts.evaluate(stored, RULES, 1, MEMBERS)


def test_rpc_batch_fallback_counts_each_call_once():
    from backpack.providers import Providers
    class Reply:
        def __init__(self, body): self.body, self.status_code, self.ok = body, 200, True
        def json(self): return self.body
    class Http:  # rejects JSON-RPC batches, answers single calls
        def request(self, method, url, timeout, json=None, **kwargs):
            return Reply({'error': 'batch unsupported'} if isinstance(json, list) else {'jsonrpc': '2.0', 'id': 'backpack', 'result': 7})
    p = Providers({'HELIUS_API_KEY': 'k'}, Http())
    assert p.rpc_many([('getSlot', []), ('getSlot', [])]) == [(7, None), (7, None)]
    assert p.calls['Helius RPC'] == 2 and p.usage['Helius RPC'] == 3  # one rejected batch + two single calls


def test_price_block_ages_only_look_up_blocks_near_the_hour_boundary():
    ref = T0
    estimated, exact = pf.price_block_ages([1_000_000, 1_000_000 - 7_000, 1_000_000 - 9_000, 1_000_000 - 20_000, 1_000_050, None],
                                           1_000_000, ref)
    assert exact == [991_000]  # ~60 minutes: could be either side, so it gets an exact getBlockTime
    assert estimated[1_000_000] == ref and estimated[1_000_050] == ref  # at or past the reference slot
    assert estimated[993_000] == ref - timedelta(seconds=2800)  # fresh at even the slowest slot time
    assert ref - estimated[980_000] > timedelta(hours=1)  # stale at even the fastest slot time


def test_rpc_pacing_counts_every_call_in_a_batch(monkeypatch):
    from backpack import providers
    clock, slept = [100.0], []
    monkeypatch.setattr(providers.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(providers.time, 'sleep', lambda s: (slept.append(s), clock.__setitem__(0, clock[0] + s)))
    p = providers.Providers({'BP_RPC_CALLS_PER_SECOND': '10'})
    p._pace(3)
    p._pace(1)
    assert slept == [pytest.approx(0.3)]  # the second request waits for the first batch's three calls
