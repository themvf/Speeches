"""Materialize source-neutral claims and independently observed events from saved archives.

This job performs no provider calls.  Its default mode prints a plan; ``--execute`` applies the
additive schema and derives facts from existing X and launchpad rows.  Telegram is intentionally a
contract-only adapter until a collector is configured.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

from crypto_coins import COINS, config, has_contract, mentions

MATERIALIZER_VERSION = "fusion-materializer-v1"
EXTRACTION_VERSION = "claim-extract-v1"
ENTITY_MATCH_VERSION = "asset-match-v1"
VERIFICATION_VERSION = "launchpad-verify-v1"
OUTCOME_VERSION = "claim-outcome-v1"
SOURCES = ("x", "telegram", "onchain", "exchange", "official_web", "github")

GRADUATION_RE = re.compile(r"\b(graduat(?:e[ds]?|ion)|migrat(?:e[ds]?|ion))\b", re.I)
LISTING_RE = re.compile(r"\b(will\s+list|lists?|listed|listing)\b", re.I)
POOL_RE = re.compile(r"\b(pool|pair|market)\b.*\b(created|opened|launched|live)\b|\bnow\s+on\s+(pumpswap|raydium|uniswap|pancakeswap)\b", re.I)
TRANSFER_RE = re.compile(r"\b(transferr?ed|sent|moved)\b", re.I)
AMOUNT_RE = re.compile(r"\b([0-9]+(?:\.[0-9]+)?)\s*([KkMmBb])?\s*([A-Z][A-Z0-9]{1,9})?\b")
TX_RE = re.compile(r"\b(?:tx|transaction)(?:\s+hash)?[:#\s]+([A-Za-z0-9]{20,100})\b", re.I)
VENUES = ("coinbase", "binance", "kraken", "robinhood", "okx", "bybit", "pumpswap", "raydium", "uniswap", "pancakeswap")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).hexdigest()[:32]
    return f"{prefix}_{digest}"


def normalize_address(network: str, address: str) -> str:
    return address.lower() if address.startswith("0x") else address


def asset_identity_key(network: str, address: str | None, symbol: str) -> str:
    if address:
        return f"asset:{network}:{normalize_address(network, address)}"
    return f"asset:{network}:native:{symbol.upper()}"


@dataclass(frozen=True)
class AssetIdentity:
    symbol: str
    name: str
    network: str
    address: str | None

    @property
    def identity_key(self) -> str:
        return asset_identity_key(self.network, self.address, self.symbol)

    @property
    def entity_id(self) -> str:
        return stable_id("ent", self.identity_key)


@dataclass(frozen=True)
class ObservationInput:
    source: str
    source_record_id: str
    content: str | None
    published_at: datetime | None
    observed_at: datetime
    source_actor_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    revision: int = 1

    @property
    def id(self) -> str:
        return stable_id("obs", self.source, self.source_record_id, self.revision)

    @property
    def raw_hash(self) -> str:
        return hashlib.sha256(canonical_json({"content": self.content, "metadata": self.metadata}).encode()).hexdigest()


@dataclass(frozen=True)
class ClaimCandidate:
    claim_type: str
    predicate: str
    object: dict[str, Any]
    verifiability: str
    effective_at: datetime | None = None
    confidence: float = 0.9

    def fingerprint(self, asset: AssetIdentity) -> str:
        body = {"subject": asset.identity_key, "type": self.claim_type, "predicate": self.predicate,
                "object": self.object, "effective_at": self.effective_at.isoformat() if self.effective_at else None}
        return hashlib.sha256(canonical_json(body).encode()).hexdigest()


def _venue(text: str) -> str | None:
    lower = text.lower()
    return next((venue for venue in VENUES if re.search(rf"\b{re.escape(venue)}\b", lower)), None)


def extract_claims(text: str, asset: AssetIdentity) -> list[ClaimCandidate]:
    """Conservative deterministic V1 extractor; a single observation may yield many claims."""
    claims: list[ClaimCandidate] = []
    if GRADUATION_RE.search(text):
        claims.append(ClaimCandidate("graduation", "asset_graduated", {}, "objective"))
    if LISTING_RE.search(text):
        venue = _venue(text)
        claims.append(ClaimCandidate("listing", "asset_listed", {"venue": venue}, "conditional"))
    if POOL_RE.search(text):
        claims.append(ClaimCandidate("pool_creation", "pool_created", {"venue": _venue(text)}, "objective"))
    if TRANSFER_RE.search(text):
        transfer_match = TRANSFER_RE.search(text)
        amount_match, tx_match = AMOUNT_RE.search(text[transfer_match.end():]), TX_RE.search(text)
        amount = None
        if amount_match:
            scale = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get((amount_match.group(2) or "").lower(), 1)
            amount = float(amount_match.group(1)) * scale
        claims.append(ClaimCandidate("transfer", "asset_transferred", {
            "amount": amount, "unit": amount_match.group(3) if amount_match else None,
            "tx_hash": tx_match.group(1) if tx_match else None,
        }, "objective", confidence=0.86 if (amount or tx_match) else 0.7))
    return claims


def evidence_relationship(text: str, propagation_observed: bool = False) -> str:
    lower = text.lower()
    if re.search(r"\b(retract|withdraw)\b", lower): return "retracts"
    if re.search(r"\b(correction|correcting|previously stated)\b", lower): return "corrects"
    if re.search(r"\b(false|fake|incorrect|not true|did not)\b", lower): return "disputes"
    return "repeats" if propagation_observed else "asserts"


class TelegramPlaceholderAdapter:
    status = "not_configured"

    def observations(self, _asset: AssetIdentity) -> list[ObservationInput]:
        return []


def _table_exists(cur, name: str) -> bool:
    cur.execute("SELECT to_regclass(%s)", (f"public.{name}",))
    return bool(cur.fetchone()[0])


def _insert_asset(cur, asset: AssetIdentity) -> None:
    cur.execute("""INSERT INTO intelligence_entities(id,kind,label,metadata) VALUES(%s,'asset',%s,%s)
                   ON CONFLICT DO NOTHING""", (asset.entity_id, f"{asset.name} ({asset.symbol})", json.dumps({"symbol": asset.symbol})))
    cur.execute("""INSERT INTO intelligence_asset_identities(entity_id,network,contract_address,native_symbol,identity_key)
                   VALUES(%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                (asset.entity_id, asset.network, asset.address, None if asset.address else asset.symbol, asset.identity_key))


def _insert_observation(cur, observation: ObservationInput, run_id: str) -> None:
    cur.execute("SELECT raw_hash FROM intelligence_observations WHERE source=%s AND source_record_id=%s AND revision=%s",
                (observation.source, observation.source_record_id, observation.revision))
    existing = cur.fetchone()
    if existing and existing[0] != observation.raw_hash:
        raise ValueError(f"{observation.source}:{observation.source_record_id} changed without a new revision")
    cur.execute("""INSERT INTO intelligence_observations
       (id,source,source_record_id,revision,source_actor_id,content,published_at,observed_at,raw_hash,collection_run_id,metadata)
       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
       (observation.id, observation.source, observation.source_record_id, observation.revision,
        observation.source_actor_id, observation.content, observation.published_at, observation.observed_at,
        observation.raw_hash, run_id, json.dumps(observation.metadata)))


def _link_observation_asset(cur, observation_id: str, asset: AssetIdentity, method: str, provisional: bool) -> None:
    cur.execute("""INSERT INTO intelligence_observation_entities
       (observation_id,entity_id,relationship,match_method,provisional,derivation_version)
       VALUES(%s,%s,'subject',%s,%s,%s) ON CONFLICT DO NOTHING""",
       (observation_id, asset.entity_id, method, provisional, ENTITY_MATCH_VERSION))


def _insert_claims(cur, observation: ObservationInput, asset: AssetIdentity, propagation_observed: bool = False) -> int:
    inserted = 0
    for candidate in extract_claims(observation.content or "", asset):
        fingerprint = candidate.fingerprint(asset)
        claim_id = stable_id("clm", EXTRACTION_VERSION, fingerprint)
        created = observation.published_at or observation.observed_at
        cur.execute("""INSERT INTO intelligence_claims
          (id,subject_entity_id,claim_type,predicate,object,effective_at,verifiability,canonical_fingerprint,extraction_version,created_at)
          VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
          (claim_id, asset.entity_id, candidate.claim_type, candidate.predicate, json.dumps(candidate.object),
           candidate.effective_at, candidate.verifiability, fingerprint, EXTRACTION_VERSION, created))
        relationship = evidence_relationship(observation.content or "", propagation_observed)
        cur.execute("""INSERT INTO intelligence_observation_claims
          (observation_id,claim_id,relationship,confidence,derivation_version,evidence)
          VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
          (observation.id, claim_id, relationship, candidate.confidence, EXTRACTION_VERSION,
           json.dumps({"extractor": "deterministic", "independence": "not_assessed"})))
        inserted += 1
    return inserted


def _materialize_x(cur, asset: AssetIdentity, run_id: str) -> tuple[int, int]:
    if not all(_table_exists(cur, table) for table in ("crypto_social_posts", "crypto_social_accounts", "crypto_social_matches", "crypto_social_windows")):
        return 0, 0
    cur.execute("""SELECT DISTINCT p.id,p.author_id,a.handle,a.name,p.text,p.posted_at,p.first_seen_at,p.kind,p.url
      FROM crypto_social_posts p JOIN crypto_social_accounts a ON a.id=p.author_id
      JOIN crypto_social_matches m ON m.post_id=p.id JOIN crypto_social_windows w ON w.id=m.window_id
      WHERE w.coin=%s ORDER BY p.posted_at,p.id""", (asset.symbol,))
    observations = claims = 0
    for row in cur.fetchall():
        post_id, author_id, handle, display_name, text, posted_at, first_seen_at, kind, url = row
        if not mentions(text, asset.symbol):
            continue
        actor_id = stable_id("actor", "x", author_id)
        cur.execute("""INSERT INTO intelligence_source_actors(id,source,source_actor_id,handle,display_name,first_observed_at,metadata)
          VALUES(%s,'x',%s,%s,%s,%s,'{}') ON CONFLICT DO NOTHING""",
          (actor_id, author_id, handle, display_name, first_seen_at))
        observation = ObservationInput("x", str(post_id), text, posted_at, first_seen_at, actor_id,
                                       {"url": url, "kind": kind})
        _insert_observation(cur, observation, run_id)
        exact = bool(asset.address and has_contract(text, asset.symbol))
        _link_observation_asset(cur, observation.id, asset, "exact_contract" if exact else "registry_text_match", not exact)
        claims += _insert_claims(cur, observation, asset, str(kind).lower() == "repost")
        observations += 1
    return observations, claims


def _event(cur, asset: AssetIdentity, observation: ObservationInput, event_type: str,
           occurred_at: datetime, attributes: dict[str, Any]) -> str:
    event_id = stable_id("evt", VERIFICATION_VERSION, event_type, asset.identity_key, canonical_json(attributes))
    cur.execute("""INSERT INTO intelligence_events(id,event_type,occurred_at,observed_at,verification_policy,verification_version,attributes)
      VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
      (event_id, event_type, occurred_at, observation.observed_at, "stored_launchpad_archive", VERIFICATION_VERSION, json.dumps(attributes)))
    cur.execute("INSERT INTO intelligence_event_entities(event_id,entity_id,relationship) VALUES(%s,%s,'subject') ON CONFLICT DO NOTHING",
                (event_id, asset.entity_id))
    cur.execute("INSERT INTO intelligence_event_observations(event_id,observation_id,relationship) VALUES(%s,%s,'records') ON CONFLICT DO NOTHING",
                (event_id, observation.id))
    return event_id


def _materialize_opening_trades(cur, asset: AssetIdentity, run_id: str) -> tuple[int, int]:
    if not asset.address or not all(_table_exists(cur, table) for table in ("launchpad_trade_captures", "launchpad_trades")):
        return 0, 0
    comparator = "lower(c.token_address)=lower(%s)" if asset.address.startswith("0x") else "c.token_address=%s"
    cur.execute(f"""SELECT c.id,t.sequence,c.pool,t.wallet,t.traded_at,t.kind,t.token_amount,t.usd,t.tx_hash,t.block_number
      FROM launchpad_trade_captures c JOIN launchpad_trades t ON t.capture_id=c.id
      WHERE c.network=%s AND {comparator} AND t.tx_hash IS NOT NULL ORDER BY t.traded_at,c.id,t.sequence""",
      (asset.network, asset.address))
    observations = events = 0
    for capture_id, sequence, pool, wallet, traded_at, kind, amount, usd, tx_hash, block_number in cur.fetchall():
        observation = ObservationInput("onchain", f"launchpad-trade:{capture_id}:{sequence}", None, traded_at, traded_at,
          metadata={"tx_hash": tx_hash, "wallet": wallet, "kind": kind, "token_amount": amount, "usd": usd,
                    "pool": pool, "block_number": block_number})
        _insert_observation(cur, observation, run_id); _link_observation_asset(cur, observation.id, asset, "exact_contract", False)
        _event(cur, asset, observation, "transfer", traded_at,
               {"tx_hash": tx_hash, "wallet": wallet, "kind": kind, "token_amount": amount, "usd": usd,
                "pool": pool, "block_number": block_number, "context": "opening_trade_capture"})
        observations += 1; events += 1
    return observations, events


def _materialize_launchpad(cur, asset: AssetIdentity, run_id: str) -> tuple[int, int, int]:
    if not asset.address or not _table_exists(cur, "launchpad_tokens"):
        return 0, 0, 0
    comparator = "lower(token_address)=lower(%s)" if asset.address.startswith("0x") else "token_address=%s"
    cur.execute(f"""SELECT network,token_address,symbol,name,graduated,graduated_at,graduated_detected_at,
      graduation_pool,measure_pool,dex,first_seen_at FROM launchpad_tokens WHERE network=%s AND {comparator}""",
      (asset.network, asset.address))
    token = cur.fetchone()
    observations = events = measurements = 0
    if token and token[4] and token[5]:
        network, address, symbol, name, _, graduated_at, detected_at, graduation_pool, measure_pool, dex, first_seen = token
        observation = ObservationInput("onchain", f"launchpad-token:{network}:{address}:graduation",
          f"Stored launchpad archive recorded graduation for {symbol or asset.symbol}.", graduated_at,
          detected_at or first_seen, metadata={"network": network, "token_address": address, "graduation_pool": graduation_pool})
        _insert_observation(cur, observation, run_id); _link_observation_asset(cur, observation.id, asset, "exact_contract", False)
        _event(cur, asset, observation, "graduation", graduated_at, {"pool": graduation_pool, "dex": dex})
        observations += 1; events += 1
        if graduation_pool:
            _event(cur, asset, observation, "pool_creation", graduated_at,
                   {"pool": graduation_pool, "measure_pool": measure_pool, "dex": dex})
            events += 1
    if _table_exists(cur, "launchpad_observations"):
        cur.execute(f"""SELECT observed_at,phase,rung_minutes,price_usd,liquidity_usd,volume_h1
          FROM launchpad_observations WHERE network=%s AND {comparator} ORDER BY observed_at""", (asset.network, asset.address))
        for observed_at, phase, rung, price, liquidity, volume in cur.fetchall():
            source_record = f"{asset.network}:{asset.address}:{observed_at.isoformat()}"
            measurement_id = stable_id("mkt", OUTCOME_VERSION, source_record)
            cur.execute("""INSERT INTO intelligence_market_measurements
              (id,entity_id,venue,pool,quote_currency,measured_at,price_usd,liquidity_usd,volume_usd,volume_window,source,methodology_version,source_record_id,metadata)
              VALUES(%s,%s,%s,%s,'USD',%s,%s,%s,%s,%s,'launchpad_archive',%s,%s,%s) ON CONFLICT DO NOTHING""",
              (measurement_id, asset.entity_id, None, None, observed_at, price, liquidity, volume,
               # The archive column is volume_h1, so the window is an hour. Never leave it implied.
               'h1' if volume is not None else None, OUTCOME_VERSION,
               source_record, json.dumps({"phase": phase, "rung_minutes": rung})))
            measurements += 1
    trade_observations, trade_events = _materialize_opening_trades(cur, asset, run_id)
    observations += trade_observations; events += trade_events
    return observations, events, measurements


def _assess_claims(cur, asset: AssetIdentity) -> int:
    cur.execute("""SELECT c.id,c.claim_type,e.id,e.event_type,e.occurred_at,e.observed_at,eo.observation_id
      FROM intelligence_claims c JOIN intelligence_event_entities ee ON ee.entity_id=c.subject_entity_id AND ee.relationship='subject'
      JOIN intelligence_events e ON e.id=ee.event_id JOIN intelligence_event_observations eo ON eo.event_id=e.id
      WHERE c.subject_entity_id=%s AND ((c.claim_type='graduation' AND e.event_type='graduation') OR
        (c.claim_type='pool_creation' AND e.event_type='pool_creation'))""", (asset.entity_id,))
    count = 0
    for claim_id, claim_type, event_id, event_type, occurred, observed, evidence_id in cur.fetchall():
        assessment_id = stable_id("asm", VERIFICATION_VERSION, claim_id, event_id)
        cur.execute("""INSERT INTO intelligence_claim_event_assessments
          (id,claim_id,event_id,status,assessed_at,policy,version,rationale,evidence_observation_ids)
          VALUES(%s,%s,%s,'confirmed',%s,'same_asset_same_event_type',%s,%s,%s) ON CONFLICT DO NOTHING""",
          (assessment_id, claim_id, event_id, max(occurred, observed), VERIFICATION_VERSION,
           f"Stored {event_type} event confirms the {claim_type} claim for the same canonical asset.", [evidence_id]))
        count += 1
    cur.execute("""SELECT c.id,e.id,e.occurred_at,e.observed_at,eo.observation_id
      FROM intelligence_claims c JOIN intelligence_event_entities ee ON ee.entity_id=c.subject_entity_id AND ee.relationship='subject'
      JOIN intelligence_events e ON e.id=ee.event_id AND e.event_type='transfer'
      JOIN intelligence_event_observations eo ON eo.event_id=e.id
      WHERE c.subject_entity_id=%s AND c.claim_type='transfer'
        AND nullif(c.object->>'tx_hash','') IS NOT NULL AND c.object->>'tx_hash'=e.attributes->>'tx_hash'""", (asset.entity_id,))
    for claim_id, event_id, occurred, observed, evidence_id in cur.fetchall():
        assessment_id = stable_id("asm", VERIFICATION_VERSION, claim_id, event_id)
        cur.execute("""INSERT INTO intelligence_claim_event_assessments
          (id,claim_id,event_id,status,assessed_at,policy,version,rationale,evidence_observation_ids)
          VALUES(%s,%s,%s,'confirmed',%s,'exact_transaction_hash',%s,%s,%s) ON CONFLICT DO NOTHING""",
          (assessment_id, claim_id, event_id, max(occurred, observed), VERIFICATION_VERSION,
           "The saved on-chain opening-trade record has the exact transaction hash asserted by the claim.", [evidence_id]))
        count += 1
    return count


def _link_outcomes(cur, asset: AssetIdentity) -> int:
    """Pin measurement timestamps to explicit anchors; this stores association, not causation."""
    cur.execute("""SELECT c.id,m.id,m.measured_at,anchor.anchor_type,anchor.anchor_at
      FROM intelligence_claims c
      JOIN LATERAL (
        SELECT 'first_social_observation'::text AS anchor_type,min(COALESCE(o.published_at,o.observed_at)) AS anchor_at
          FROM intelligence_observation_claims oc JOIN intelligence_observations o ON o.id=oc.observation_id
          WHERE oc.claim_id=c.id AND o.source IN ('x','telegram')
        UNION ALL
        SELECT 'event_occurred',min(e.occurred_at) FROM intelligence_claim_event_assessments a
          JOIN intelligence_events e ON e.id=a.event_id WHERE a.claim_id=c.id
        UNION ALL
        SELECT 'verified_at',min(a.assessed_at) FROM intelligence_claim_event_assessments a WHERE a.claim_id=c.id
      ) anchor ON anchor.anchor_at IS NOT NULL
      JOIN intelligence_market_measurements m ON m.entity_id=c.subject_entity_id AND m.measured_at>=anchor.anchor_at
      WHERE c.subject_entity_id=%s""", (asset.entity_id,))
    count = 0
    for claim_id, measurement_id, measured_at, anchor_type, anchor_at in cur.fetchall():
        horizon = max(0, int((measured_at - anchor_at).total_seconds()))
        outcome_id = stable_id("out", OUTCOME_VERSION, claim_id, measurement_id, anchor_type)
        cur.execute("""INSERT INTO intelligence_claim_outcomes
          (id,claim_id,measurement_id,anchor_type,anchor_at,horizon_seconds,methodology_version)
          VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
          (outcome_id, claim_id, measurement_id, anchor_type, anchor_at, horizon, OUTCOME_VERSION))
        count += 1
    return count


def materialize(database_url: str, symbols: Iterable[str]) -> dict[str, Any]:
    import psycopg2
    run_started = utc_now()
    run_id = stable_id("run", MATERIALIZER_VERSION, run_started.isoformat())
    totals = {"assets": 0, "observations": 0, "claims": 0, "events": 0, "measurements": 0, "assessments": 0, "outcomes": 0}
    conn = psycopg2.connect(database_url)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(Path(__file__).with_name("sql").joinpath("intelligence_fusion.sql").read_text())
            cur.execute("""INSERT INTO intelligence_collection_runs(id,kind,version,started_at,status,metadata)
              VALUES(%s,'archive_materialization',%s,%s,'running',%s)""",
              (run_id, MATERIALIZER_VERSION, run_started, json.dumps({"network_calls": 0, "telegram": "not_configured"})))
        for symbol in symbols:
            cfg = config(symbol)
            asset = AssetIdentity(symbol, cfg["name"], cfg["network"], cfg["address"])
            with conn, conn.cursor() as cur:
                _insert_asset(cur, asset)
                x_obs, claims = _materialize_x(cur, asset, run_id)
                chain_obs, events, measurements = _materialize_launchpad(cur, asset, run_id)
                assessments = _assess_claims(cur, asset)
                outcomes = _link_outcomes(cur, asset)
                totals["assets"] += 1; totals["observations"] += x_obs + chain_obs; totals["claims"] += claims
                totals["events"] += events; totals["measurements"] += measurements; totals["assessments"] += assessments
                totals["outcomes"] += outcomes
        with conn, conn.cursor() as cur:
            cur.execute("UPDATE intelligence_collection_runs SET finished_at=%s,status='complete',metadata=metadata||%s::jsonb WHERE id=%s",
                        (utc_now(), json.dumps(totals), run_id))
    except Exception:
        conn.rollback()
        try:
            with conn, conn.cursor() as cur:
                cur.execute("UPDATE intelligence_collection_runs SET finished_at=%s,status='failed' WHERE id=%s", (utc_now(), run_id))
        except Exception:
            pass
        raise
    finally:
        conn.close()
    return {"run_id": run_id, "version": MATERIALIZER_VERSION, "telegram": "not_configured", **totals}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="write derived rows from saved archives")
    parser.add_argument("--coin", action="append", choices=sorted(COINS), help="limit to one or more registry symbols")
    args = parser.parse_args()
    symbols = args.coin or list(COINS)
    plan = {"mode": "execute" if args.execute else "plan", "coins": symbols, "network_calls": 0,
            "sources": {"x": "saved_archive", "onchain": "saved_launchpad_archive", "telegram": "not_configured"},
            "versions": {"materializer": MATERIALIZER_VERSION, "extraction": EXTRACTION_VERSION,
                         "verification": VERIFICATION_VERSION}}
    if not args.execute:
        print(json.dumps(plan, indent=2)); return 0
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required with --execute")
    print(json.dumps(materialize(database_url, symbols), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
