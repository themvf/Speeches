from datetime import datetime, timezone
import os
from pathlib import Path

import pytest
from crypto_coins import config as coin_config

from intelligence_fusion import (
    AssetIdentity,
    ObservationInput,
    TelegramPlaceholderAdapter,
    asset_identity_key,
    evidence_relationship,
    extract_claims,
    materialize,
)


NOW = datetime(2026, 9, 20, tzinfo=timezone.utc)


def asset():
    return AssetIdentity("ABC", "Alpha Beta Coin", "solana", "AbC123")


def test_contract_identity_is_chain_scoped_and_preserves_non_evm_case():
    assert asset_identity_key("solana", "AbC123", "ABC") == "asset:solana:AbC123"
    assert asset_identity_key("base", "0xAbC", "ABC") == "asset:base:0xabc"
    assert asset_identity_key("ethereum", None, "ETH") == "asset:ethereum:native:ETH"


def test_one_observation_can_extract_multiple_supported_claims():
    claims = extract_claims(
        "ABC graduated and its pool is now live on PumpSwap; wallet transferred 3M ABC tx: 12345678901234567890",
        asset(),
    )
    assert [claim.claim_type for claim in claims] == ["graduation", "pool_creation", "transfer"]
    transfer = claims[-1]
    assert transfer.object == {"amount": 3_000_000.0, "unit": "ABC", "tx_hash": "12345678901234567890"}
    assert transfer.verifiability == "objective"


def test_equivalent_claims_cluster_across_sources():
    first = extract_claims("ABC graduated", asset())[0]
    second = extract_claims("Breaking: ABC graduation confirmed", asset())[0]
    assert first.fingerprint(asset()) == second.fingerprint(asset())


def test_relationship_is_conservative_about_corroboration():
    assert evidence_relationship("ABC graduated", False) == "asserts"
    assert evidence_relationship("ABC graduated", True) == "repeats"
    assert evidence_relationship("Another source says ABC graduated", False) == "asserts"
    assert evidence_relationship("Correction: ABC did not graduate", True) == "corrects"
    assert evidence_relationship("That ABC claim is false", True) == "disputes"


def test_observations_are_deterministic_and_revision_aware():
    one = ObservationInput("x", "42", "ABC graduated", NOW, NOW)
    retry = ObservationInput("x", "42", "ABC graduated", NOW, NOW)
    revision = ObservationInput("x", "42", "ABC graduated (edited)", NOW, NOW, revision=2)
    assert one.id == retry.id
    assert one.raw_hash == retry.raw_hash
    assert revision.id != one.id
    assert revision.raw_hash != one.raw_hash


def test_telegram_is_an_explicit_non_collecting_placeholder():
    adapter = TelegramPlaceholderAdapter()
    assert adapter.status == "not_configured"
    assert adapter.observations(asset()) == []


@pytest.mark.skipif(not os.environ.get("CRYPTO_SOCIAL_TEST_DATABASE_URL"), reason="test Postgres not configured")
def test_saved_x_and_chain_rows_materialize_without_provider_calls():
    import psycopg2

    url = os.environ["CRYPTO_SOCIAL_TEST_DATABASE_URL"]
    root = Path(__file__).parents[1]
    cfg = coin_config("PONS")
    conn = psycopg2.connect(url)
    with conn, conn.cursor() as cur:
        cur.execute(root.joinpath("sql", "crypto_social.sql").read_text())
        cur.execute(root.joinpath("sql", "launchpad_archive.sql").read_text())
        cur.execute("""INSERT INTO crypto_social_coins(symbol,name,query,address,identity_status)
          VALUES('PONS','Pons','test',%s,'contract') ON CONFLICT DO NOTHING""", (cfg["address"],))
        cur.execute("""INSERT INTO crypto_social_accounts(id,handle,name,observed_at)
          VALUES('fusion-test-actor','fusion_test','Fusion Test','2040-01-01T00:01:00Z') ON CONFLICT DO NOTHING""")
        cur.execute("""INSERT INTO crypto_social_posts(id,author_id,text,posted_at,kind,url,first_seen_at)
          VALUES('fusion-test-post','fusion-test-actor','PONS graduated and its pool is live on PumpSwap',
          '2040-01-01T00:00:00Z','post','https://x.example/fusion-test','2040-01-01T00:01:00Z') ON CONFLICT DO NOTHING""")
        cur.execute("""INSERT INTO crypto_social_windows(coin,start_at,end_at,query,status)
          VALUES('PONS','2040-01-01T00:00:00Z','2040-01-02T00:00:00Z','fusion test','search_exhausted')
          ON CONFLICT(coin,start_at,end_at) DO UPDATE SET query=EXCLUDED.query RETURNING id""")
        window_id = cur.fetchone()[0]
        cur.execute("INSERT INTO crypto_social_matches(post_id,window_id) VALUES('fusion-test-post',%s) ON CONFLICT DO NOTHING", (window_id,))
        cur.execute("""INSERT INTO launchpad_tokens(network,token_address,symbol,name,dex,first_seen_at,graduated,
          graduated_at,graduated_detected_at,graduation_pool,last_seen_at,state)
          VALUES('robinhood',%s,'PONS','Pons','test-dex','2040-01-01T00:00:30Z',true,
          '2040-01-01T00:02:00Z','2040-01-01T00:02:10Z','pool-test','2040-01-01T00:02:10Z','graduated')
          ON CONFLICT DO NOTHING""", (cfg["address"],))
        cur.execute("""INSERT INTO launchpad_observations(network,token_address,observed_at,phase,rung_minutes,price_usd,liquidity_usd)
          VALUES('robinhood',%s,'2040-01-01T00:12:00Z','post',10,0.01,10000) ON CONFLICT DO NOTHING""", (cfg["address"],))
    conn.close()

    result = materialize(url, ["PONS"])
    assert result["telegram"] == "not_configured"
    assert result["observations"] >= 2
    assert result["claims"] >= 2
    assert result["events"] >= 2

    conn = psycopg2.connect(url)
    with conn, conn.cursor() as cur:
        cur.execute("""SELECT count(*) FROM intelligence_claim_event_assessments a
          JOIN intelligence_claims c ON c.id=a.claim_id WHERE c.claim_type IN ('graduation','pool_creation')""")
        assert cur.fetchone()[0] >= 2
        cur.execute("SELECT count(*) FROM intelligence_observations WHERE source='telegram'")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM intelligence_claim_outcomes")
        assert cur.fetchone()[0] >= 1
    conn.close()
