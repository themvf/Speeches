"""Extraction rules. Every one of these is a rule about what counts as a mention, so each is pinned
on a string rather than discovered later from a statistic that quietly moved."""
import telegram_extract as extract

FLEX='fvHLJUwsynVHJrssbZ8MLNyku9jt2izUspbBD4Spump'
OTHER='ALPZYXZBTvbmT1cyHxwXUgvFLCyMVuAXaJv9nLYDpieq'


def test_a_solana_mint_is_identified_by_decoding_it_not_by_its_shape():
    # A 43-character word in base58's alphabet is not necessarily an address, and the suffix 'pump'
    # is a Pump.fun convention rather than a validity test - filtering on it would silently make
    # this a Pump.fun-only collector while the archive holds Meteora, Raydium and Boop graduates too.
    assert extract.is_solana_address(FLEX) and extract.is_solana_address(OTHER)
    assert not extract.is_solana_address('notanaddress')
    assert not extract.is_solana_address('1'*20)
    # A transaction signature is 64 bytes, so it must not be collected as a mint.
    assert not extract.is_solana_address('5'*88)


def test_addresses_inside_links_are_collected():
    # A Dexscreener or Birdeye link is one of the commonest ways a channel posts a contract. Dropping
    # link-borne addresses would lose exactly the messages that are most clearly about one token.
    text=f'new call https://dexscreener.com/solana/{FLEX} ape now'
    assert extract.solana_addresses(text)==[FLEX]


def test_addresses_are_deduplicated_in_order_of_appearance():
    assert extract.solana_addresses(f'{OTHER} then {FLEX} then {OTHER}')==[OTHER,FLEX]


def test_cashtags_drop_currencies_and_majors():
    assert extract.cashtags('$FLEX paired with $SOL, paid in $usdc')==['FLEX']


def test_a_message_with_a_contract_drops_its_cashtags():
    # The contract is what the message is about. A ticker beside it resolving to some other token is
    # how a channel gets credited with a call it never made.
    import telegram_mentions
    rows=telegram_mentions.message_references(f'$FLEX {FLEX} send it')
    assert [r['reference_kind'] for r in rows]==['contract']
    assert rows[0]['raw_reference']==FLEX


def test_a_message_with_only_a_ticker_keeps_it_as_a_candidate():
    import telegram_mentions
    rows=telegram_mentions.message_references('$FLEX looking strong')
    assert rows==[dict(raw_reference='FLEX',reference_kind='ticker',confidence=None)]


def test_claims_are_captured_but_are_claims():
    # Stored so a channel's claims can be compared against measured outcomes; never an input to one.
    assert extract.claims('already did 10x from $450K mcap')=={'claimed_multiple':10.0,
                                                               'claimed_market_cap':450000.0}
    assert extract.claims('MC: $1.2M entry')['claimed_market_cap']==1200000.0
    assert extract.claims('x5 incoming')['claimed_multiple']==5.0
    # A dollar figure that is not market-cap language is not a market cap.
    assert extract.claims('a $100 position')['claimed_market_cap'] is None
    assert extract.claims('no numbers here')=={'claimed_multiple':None,'claimed_market_cap':None}


def test_an_invisible_character_inside_an_address_defeats_extraction():
    """Not a bug to fix blindly - a mint with a zero-width joiner in it is not that mint, and
    'repairing' it would invent an address the message did not contain. It is a thing to SEE, which
    is why --inspect flags invisible characters beside the raw repr rather than stripping them."""
    broken=FLEX[:20]+'​'+FLEX[20:]
    assert extract.solana_addresses(broken)==[]
    assert extract.solana_addresses(FLEX)==[FLEX]


def test_a_contract_split_across_lines_is_not_reassembled():
    # Same reasoning: two halves on two lines are two strings, and neither decodes to 32 bytes.
    assert extract.solana_addresses(FLEX[:20]+'\n'+FLEX[20:])==[]


def test_decorative_punctuation_around_a_cashtag_still_resolves():
    assert extract.cashtags('🚀🚀 $FLEX 🚀🚀')==['FLEX']
    assert extract.cashtags('**$FLEX**')==['FLEX']
