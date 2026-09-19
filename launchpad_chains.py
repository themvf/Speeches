"""Per-chain adapters for the Graduation Archive. Chain-specific logic lives here, never in the sweep.

The engine in launchpad_archive.py is chain-agnostic: it pages a feed, detects graduations from
graduate-DEX arrivals, batches curve state, enriches graduates and fills a rung ladder. Everything
that differs between chains - which DEXes are curves, how often to sweep, which pool to measure,
what enrichment is worth storing, whether to capture opening trades - is a field on a Chain.

Measured 2026-09-19; see docs/solana-pumpfun-archive-spec.md.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional


def number(value):
    """Floats that are actually numbers; None for anything else. Duplicated deliberately: this
    module must not import the engine, or the dependency runs the wrong way."""
    try:
        import math
        out=float(value)
        return out if math.isfinite(out) else None
    except (TypeError,ValueError):return None


@dataclass(frozen=True)
class Chain:
    network: str
    curve_dexes: frozenset
    graduate_dexes: frozenset
    sweep_minutes: int
    rungs: tuple
    # Solana's migrated_destination_pool_address points at a pool that may hold almost nothing while
    # the real trading happens elsewhere, so each chain says how to pick what the ladder measures.
    deepest_pool_wins: bool = False
    # Opening trade capture: wallet-level rows at the moment of graduation. Perishable - see below.
    capture_trades: bool = False
    trade_pages: int = 2
    # Fraction of graduates drawn into the ladder cohort. The draw must never depend on a token's
    # properties, or the cohort stops being a random sample of graduates.
    cohort_fraction: float = 1.0
    extended_info: bool = False
    # EVM hex addresses are case-insensitive, so folding them to lower case is a safe way to make a
    # key. Solana's base58 is case-SENSITIVE: folding it produces an address that resolves to
    # nothing. A live sweep proved this - every enrichment call 404'd until it was switched off.
    lowercase_addresses: bool = True


ROBINHOOD=Chain(
    network='robinhood',
    curve_dexes=frozenset({'pons-v2','pons-v2-dex-curve','hoodit','o1-launchpad-robinhood','bankr-robinhood',
                           'clanker-robinhood','virtuals-robinhood','easya-kickstart-robinhood','mint-club-robinhood'}),
    graduate_dexes=frozenset({'pons-v2-dex'}),
    sweep_minutes=5,
    rungs=(10,30,60,180,360,720,1440,2880,10080),
)

# Solana: 25 new pools/minute and a feed only 5.8 minutes deep across all ten pages, so the cadence
# is two minutes. The +5m rung exists because Pump.fun's post-migration mechanism buys automatically
# in the opening minutes - that rung is structural, not a market reading.
SOLANA=Chain(
    network='solana',
    curve_dexes=frozenset({'pump-fun','meteora-dbc','raydium-launchlab','boop-fun','moonshot'}),
    graduate_dexes=frozenset({'pumpswap','meteora-damm-v2'}),
    sweep_minutes=2,
    rungs=(5,10,30,60,180,360,720,1440,2880,10080),
    deepest_pool_wins=True,
    capture_trades=True,
    cohort_fraction=0.25,
    extended_info=True,
    lowercase_addresses=False,
)

CHAINS={c.network:c for c in (ROBINHOOD,SOLANA)}


def classify(dex,chain=ROBINHOOD):
    return 'graduate' if dex in chain.graduate_dexes else 'curve' if dex in chain.curve_dexes else 'other'


def choose_measure_pool(chain,pools,destination=None):
    """Which pool the ladder reads, and why.

    Measured on Solana: a graduate's `migrated_destination_pool_address` held $0.43 across 3 trades
    while the token's deepest pool held $30,562 across 1,964 trades from 1,088 buyers in five
    minutes. Following the documented field would have measured a dead pool at every rung and shown
    that essentially every graduate dies instantly - confidently, and wrongly.

    `pools` is [(address, dex, liquidity)]. Returns (address, reason); (None, reason) is a real
    outcome for a token whose pools are all empty and must be recorded, not dropped, or every
    survival rate is biased upward.
    """
    if not chain.deepest_pool_wins:
        return (destination,'launchpad destination') if destination else (None,'no destination reported')
    graduate=[p for p in pools if p[1] in chain.graduate_dexes]
    candidates=graduate or list(pools)
    liquid=[p for p in candidates if (number(p[2]) or 0)>0]
    if not liquid:
        return (None,'no liquid pool') if candidates else (None,'no pools listed')
    best=max(liquid,key=lambda p:number(p[2]) or 0)
    reason='deepest graduate pool' if graduate else 'deepest pool, no graduate-dex pool listed'
    if destination and best[0]!=destination:
        # Recorded rather than silently preferred: the disagreement is itself a finding about the field.
        reason+=' (destination field disagreed)'
    return best[0],reason


def parse_pool_list(payload):
    """tokens/<address>/pools -> [(address, dex, liquidity)]."""
    out=[]
    for entry in (payload or {}).get('data') or []:
        attributes=entry.get('attributes') or {}
        dex=(((entry.get('relationships') or {}).get('dex') or {}).get('data') or {}).get('id')
        address=attributes.get('address')
        if address and dex:out.append((address,dex,number(attributes.get('reserve_in_usd'))))
    return out


def parse_extended_info(payload):
    """The rest of tokens/<address>/info: creator, authorities, socials, description.

    All free, in a call the archive already makes for every graduate. Once developer_address is
    stored, creator history - prior launches, graduation rate, survivor rate - is a self-join on our
    own archive rather than an external lookup.
    """
    attributes=((payload or {}).get('data') or {}).get('attributes') or {}
    websites=attributes.get('websites') or []
    return dict(developer_address=attributes.get('developer_address') or None,
                developer_holding=number(attributes.get('developer_holding_percentage')),
                is_honeypot=attributes.get('is_honeypot') or None,
                mint_authority=attributes.get('mint_authority') or None,
                freeze_authority=attributes.get('freeze_authority') or None,
                telegram_handle=attributes.get('telegram_handle') or None,
                website=(websites[0] if websites else None),
                description=attributes.get('description') or None,
                categories=list(attributes.get('categories') or []) or None,
                gt_score=number(attributes.get('gt_score')),
                # Raw payload kept so a later question can be asked of data already collected,
                # rather than needing a re-collection that is impossible for perishable fields.
                info_raw=attributes)


def parse_trades(payload,pool):
    """pools/<pool>/trades -> wallet-level rows.

    Deliberately NOT called "the first N trades": the endpoint returns a recent window whose
    relationship to the graduation moment is not guaranteed, and paging returned an overlapping,
    partly newer window rather than an older one. The capture records its own boundaries so how much
    of the opening window was caught is a measurement, not an assumption.
    """
    rows=[]
    for entry in (payload or {}).get('data') or []:
        a=entry.get('attributes') or {}
        timestamp=a.get('block_timestamp')
        if not timestamp:continue
        kind=(a.get('kind') or '').lower()
        rows.append(dict(pool=pool,wallet=a.get('tx_from_address'),traded_at=timestamp,
                         kind='buy' if kind=='buy' else 'sell' if kind=='sell' else kind or None,
                         token_amount=number(a.get('to_token_amount') if kind=='buy' else a.get('from_token_amount')),
                         usd=number(a.get('volume_in_usd')),tx_hash=a.get('tx_hash'),
                         block_number=a.get('block_number')))
    return rows


def summarize_trades(rows):
    """The small summary stored beside the raw rows. The raw rows stay: which features matter is
    exactly what we do not know yet."""
    from collections import Counter
    wallets=[r['wallet'] for r in rows if r['wallet']]
    buyers={r['wallet'] for r in rows if r['wallet'] and r['kind']=='buy'}
    sellers={r['wallet'] for r in rows if r['wallet'] and r['kind']=='sell'}
    counts=Counter(wallets)
    times=sorted(r['traded_at'] for r in rows if r['traded_at'])
    top=counts.most_common(1)[0][1] if counts else 0
    return dict(trades=len(rows),wallets=len(set(wallets)),buyers=len(buyers),sellers=len(sellers),
                top_wallet_share=(top/len(wallets)) if wallets else None,
                repeat_wallets=sum(1 for _,n in counts.items() if n>1),
                earliest=times[0] if times else None,latest=times[-1] if times else None)


def in_cohort(chain,token_address):
    """Deterministic, property-independent draw into the ladder cohort.

    Hashing the mint address gives a stable answer (a re-run picks the same tokens) that cannot
    correlate with liquidity, buyers, or anything else we might later treat as an outcome.
    """
    if chain.cohort_fraction>=1:return True
    import hashlib
    digest=hashlib.sha256(token_address.encode()).digest()
    return int.from_bytes(digest[:4],'big')/2**32 < chain.cohort_fraction
