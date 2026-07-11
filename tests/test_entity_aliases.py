"""Tests for the entity alias map (see CLAUDE.md "Entity normalization /
alias map"): the shared JSON config, the Python resolver (entity_aliases.py),
its wiring into _normalize_enrichment_payload, and the
backfill_entity_mentions.py driver script.

The normalization fixture pairs here are mirrored in
apps/web/lib/server/entity-aliases.test.ts - the two suites pin the TS and
Python implementations to identical behavior. If you change a pair here,
change it there too.

No real Postgres connection is used for the backfill tests - psycopg2 is
mocked, matching this repo's existing pattern (see
test_neon_documents_backfill.py).
"""

import json
from unittest.mock import MagicMock, patch

import backfill_entity_mentions as backfill
import entity_aliases
import run_financial_news_pipeline as core


# ─── shared config integrity ────────────────────────────────────────────────

def test_alias_config_file_loads_and_is_well_formed():
    payload = json.loads(entity_aliases.ALIAS_CONFIG_PATH.read_text(encoding="utf-8"))
    entries = payload["entities"]
    assert entries, "alias config must not be empty"
    for entry in entries:
        assert str(entry["canonical"]).strip(), f"empty canonical in {entry}"
        assert isinstance(entry["aliases"], list)


def test_alias_config_has_no_duplicate_normalized_keys():
    """Two canonical entities claiming the same normalized alias would make
    resolution order-dependent (first entry silently wins on both sides)."""
    payload = json.loads(entity_aliases.ALIAS_CONFIG_PATH.read_text(encoding="utf-8"))
    seen = {}
    for entry in payload["entities"]:
        for alias in [entry["canonical"], *entry["aliases"]]:
            key = entity_aliases.normalize_mention_value(alias)
            assert key, f"alias {alias!r} normalizes to empty"
            assert key not in seen or seen[key] == entry["canonical"], (
                f"normalized alias {key!r} claimed by both {seen.get(key)!r} and {entry['canonical']!r}"
            )
            seen[key] = entry["canonical"]


# ─── normalization parity fixtures (mirrored in entity-aliases.test.ts) ─────

NORMALIZATION_FIXTURES = [
    ("SEC", "sec"),
    ("  The U.S. Securities & Exchange Commission  ", "the u s securities exchange commission"),
    ("O'Brien-Smith", "obrien smith"),
    ("“Smart” quotes aren’t kept", "smart quotes arent kept"),
    ("J.P. Morgan", "j p morgan"),
    ("", ""),
]


def test_normalize_mention_value_matches_ts_fixtures():
    for raw, expected in NORMALIZATION_FIXTURES:
        assert entity_aliases.normalize_mention_value(raw) == expected


# ─── alias resolution (mirrored in entity-aliases.test.ts) ──────────────────

def test_known_alias_pairs_collapse_to_same_canonical():
    for variant in [
        "SEC",
        "sec",
        "Securities and Exchange Commission",
        "U.S. Securities and Exchange Commission",
        "Securities & Exchange Commission",
        "S.E.C.",
        "the Commission",
    ]:
        assert entity_aliases.canonical_entity_label(variant) == "SEC", variant
        assert entity_aliases.canonical_normalized_entity_value(variant) == "sec", variant


def test_unknown_entities_pass_through_unchanged():
    assert entity_aliases.canonical_entity_label("Acme Widgets LLC") == "Acme Widgets LLC"
    assert entity_aliases.canonical_normalized_entity_value("Acme Widgets LLC") == "acme widgets llc"
    assert entity_aliases.canonical_entity_label("") == ""


def test_entity_alias_pairs_excludes_identity_mappings():
    pairs = entity_aliases.entity_alias_pairs()
    assert pairs, "expected at least one non-identity alias pair"
    for alias_norm, label, canonical_norm in pairs:
        assert alias_norm != canonical_norm
        assert entity_aliases.normalize_mention_value(label) == canonical_norm
    # canonical self-mappings must not be in the work list
    assert all(alias_norm != "sec" for alias_norm, _, _ in pairs)


# ─── wiring into _normalize_enrichment_payload ──────────────────────────────

def test_enrichment_payload_merges_alias_variant_entities():
    payload = {
        "entities": [
            {"name": "SEC", "type": "ORG", "mentions": 2},
            {"name": "Securities and Exchange Commission", "type": "ORG", "mentions": 3},
            {"name": "the Commission", "type": "OTHER", "mentions": 1},
            {"name": "Acme Widgets LLC", "type": "ORG", "mentions": 1},
        ]
    }
    result = core._normalize_enrichment_payload(payload)
    entities = {e["name"]: e for e in result["entities"]}
    assert set(entities) == {"SEC", "Acme Widgets LLC"}
    assert entities["SEC"]["mentions"] == 6
    assert entities["SEC"]["type"] == "ORG"


def test_enrichment_payload_canonicalizes_single_alias_name():
    result = core._normalize_enrichment_payload(
        {"entities": [{"name": "Financial Industry Regulatory Authority", "type": "ORG", "mentions": 4}]}
    )
    assert result["entities"] == [{"name": "FINRA", "type": "ORG", "mentions": 4}]


def test_enrichment_payload_merge_upgrades_other_type():
    result = core._normalize_enrichment_payload(
        {
            "entities": [
                {"name": "the Commission", "type": "", "mentions": 1},
                {"name": "SEC", "type": "ORG", "mentions": 1},
            ]
        }
    )
    assert result["entities"] == [{"name": "SEC", "type": "ORG", "mentions": 2}]


# ─── backfill_entity_mentions.py ────────────────────────────────────────────

def _mock_conn(alias_row_counts):
    """alias_row_counts: dict normalized-alias -> count returned for the
    COUNT query; every other execute is recorded but returns rowcount 1."""
    cursor = MagicMock()
    executed = []

    def _execute(sql, params=None):
        executed.append((" ".join(sql.split()), params))
        if "COUNT(*)" in sql:
            cursor.fetchone.return_value = {"alias_rows": alias_row_counts.get(params["alias"], 0)}
        cursor.rowcount = 1

    cursor.execute.side_effect = _execute
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn, cursor, executed


def test_backfill_dry_run_counts_but_never_writes():
    conn, cursor, executed = _mock_conn({"securities and exchange commission": 7})
    with patch.object(backfill.neon_feeds, "_get_conn", return_value=conn):
        summary = backfill._run(dry_run=True)

    assert summary["ok"] is True
    assert summary["dry_run"] is True
    assert summary["rows_rewritten"] == 0
    assert summary["rows_merged_and_deleted"] == 0
    pairs = {d["alias_normalized"]: d for d in summary["pairs_with_rows"]}
    assert pairs["securities and exchange commission"]["alias_rows"] == 7
    sql_run = " ".join(sql for sql, _ in executed)
    assert "UPDATE" not in sql_run and "DELETE" not in sql_run
    conn.rollback.assert_called_once()


def test_backfill_real_run_merges_deletes_then_rewrites_in_order():
    conn, cursor, executed = _mock_conn({"the commission": 3})
    with patch.object(backfill.neon_feeds, "_get_conn", return_value=conn):
        summary = backfill._run(dry_run=False)

    assert summary["ok"] is True
    assert summary["rows_rewritten"] >= 1
    per_alias = [
        (sql, params) for sql, params in executed if params and params["alias"] == "the commission"
    ]
    kinds = [
        "count" if "COUNT(*)" in sql else "merge" if "GREATEST" in sql else "delete" if sql.startswith("DELETE") else "rewrite"
        for sql, _ in per_alias
    ]
    assert kinds == ["count", "merge", "delete", "rewrite"]
    # every write is scoped to entity mentions and carries the canonical label
    for sql, params in per_alias[1:]:
        assert "mention_type = 'entity'" in sql
        assert params["canonical"] == "sec"
        assert params["label"] == "SEC"
    conn.rollback.assert_not_called()


def test_backfill_skips_pairs_with_no_rows():
    conn, cursor, executed = _mock_conn({})  # every alias counts 0
    with patch.object(backfill.neon_feeds, "_get_conn", return_value=conn):
        summary = backfill._run(dry_run=False)

    assert summary["ok"] is True
    assert summary["pairs_with_rows"] == []
    sql_run = " ".join(sql for sql, _ in executed)
    assert "UPDATE" not in sql_run and "DELETE" not in sql_run


def test_backfill_main_reports_failure_as_json():
    with patch.object(backfill.neon_feeds, "_get_conn", side_effect=RuntimeError("DATABASE_URL is not set")):
        exit_code = backfill.main([])
    assert exit_code == 1
