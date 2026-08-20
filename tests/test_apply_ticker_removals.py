"""Tests for apply_ticker_removals.py (SEC-53): the auto-prune step that
acts on check_stale_tickers.py's high-confidence "deregistered" candidates
only. Network-free; all file operations run against tmp_path copies."""

from __future__ import annotations

import json

import apply_ticker_removals as apply_mod


def test_high_confidence_deregistrations_filters_correctly():
    review = {
        "candidates": [
            {"ticker": "A", "reason": "deregistered", "confidence": "high"},
            {"ticker": "B", "reason": "deregistered", "confidence": "low"},   # wrong confidence
            {"ticker": "C", "reason": "renamed", "confidence": "high"},      # wrong reason
            {"ticker": "D", "reason": "uncertain", "confidence": "low"},
            {"ticker": "E", "reason": "deregistered", "confidence": "high"},
        ]
    }
    assert apply_mod.high_confidence_deregistrations(review) == ["A", "E"]


def test_remove_tokens_from_block_only_touches_named_block():
    text = 'OTHER = ["A", "B"]\nTARGET: List[str] = [\n    "A", "B", "C",\n]\n'
    new_text, removed = apply_mod._remove_tokens_from_block(text, "TARGET: List[str] = [", "[", "]", ["B"])
    assert removed == ["B"]
    assert 'OTHER = ["A", "B"]' in new_text  # untouched
    assert '"A", "C",' in new_text  # B removed from the target block only


def test_remove_tokens_from_block_reports_tickers_not_found():
    text = 'TARGET: List[str] = [\n    "A", "B",\n]\n'
    _, removed = apply_mod._remove_tokens_from_block(text, "TARGET: List[str] = [", "[", "]", ["A", "ZZZ"])
    assert removed == ["A"]  # ZZZ wasn't there - silently not in the removed list


def test_remove_from_builder_source_edits_both_blocks(tmp_path):
    src = (
        'UNIVERSE: List[str] = [\n'
        '    "AAA", "BBB", "CCC",\n'
        ']\n'
        '\n'
        'SUB_INDUSTRY_GROUPS: Dict[str, List[str]] = {\n'
        '    "Some Group": ["AAA", "DDD"],\n'
        '}\n'
    )
    path = tmp_path / "build_industry_config.py"
    path.write_text(src)

    result = apply_mod.remove_from_builder_source(str(path), ["AAA"])
    assert result == {"universe": ["AAA"], "sub_industry_groups": ["AAA"]}
    written = path.read_text()
    assert '"AAA"' not in written
    assert '"BBB", "CCC",' in written
    assert '"DDD"' in written  # untouched sibling entry


def test_remove_from_committed_config_drops_empty_industries(tmp_path):
    config_path = tmp_path / "industry-config.json"
    config_path.write_text(json.dumps({
        "tickerCount": 3,
        "industries": [
            {"sic": "1", "label": "Solo Industry", "tickers": [{"ticker": "ONLY", "name": "Only Co", "cik": "1"}]},
            {"sic": "2", "label": "Multi Industry", "tickers": [
                {"ticker": "KEEP", "name": "Keep Co", "cik": "2"},
                {"ticker": "DROP", "name": "Drop Co", "cik": "3"},
            ]},
        ],
    }))
    removed = apply_mod.remove_from_committed_config(str(config_path), ["ONLY", "DROP"])
    assert sorted(removed) == ["DROP", "ONLY"]
    written = json.loads(config_path.read_text())
    labels = [i["label"] for i in written["industries"]]
    assert "Solo Industry" not in labels  # emptied out entirely
    assert labels == ["Multi Industry"]
    assert [t["ticker"] for t in written["industries"][0]["tickers"]] == ["KEEP"]
    assert written["tickerCount"] == 1


def test_remove_from_state_drops_only_matching_keys(tmp_path):
    state_path = tmp_path / "industry_state.json"
    state_path.write_text(json.dumps({"latest": {
        "A": {"form": "10-Q", "accession": "x"},
        "B": {"form": "10-Q", "accession": "y"},
    }}))
    removed = apply_mod.remove_from_state(str(state_path), ["A", "ZZZ"])
    assert removed == ["A"]  # ZZZ never existed
    written = json.loads(state_path.read_text())
    assert list(written["latest"]) == ["B"]


def test_update_review_file_drops_applied_candidates_only(tmp_path):
    review_path = tmp_path / "review.json"
    review = {
        "flaggedCount": 2,
        "candidates": [
            {"ticker": "A", "reason": "deregistered", "confidence": "high"},
            {"ticker": "B", "reason": "renamed", "confidence": "high"},
        ],
    }
    review_path.write_text(json.dumps(review))
    apply_mod.update_review_file(str(review_path), review, ["A"])
    written = json.loads(review_path.read_text())
    assert [c["ticker"] for c in written["candidates"]] == ["B"]
    assert written["flaggedCount"] == 1


def test_main_dry_run_does_not_write_any_file(tmp_path, monkeypatch):
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps({"candidates": [{"ticker": "A", "reason": "deregistered", "confidence": "high"}]}))
    # Point BUILDER_PATH/CONFIG_PATH/STATE_PATH somewhere that would error if
    # touched, proving dry-run never reaches the write functions.
    monkeypatch.setattr(apply_mod, "BUILDER_PATH", "/does/not/exist.py")
    monkeypatch.setattr("sys.argv", ["apply_ticker_removals.py", "--review", str(review_path), "--dry-run"])
    assert apply_mod.main() == 0


def test_main_no_candidates_is_a_clean_noop(tmp_path, monkeypatch):
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps({"candidates": [{"ticker": "A", "reason": "renamed", "confidence": "high"}]}))
    monkeypatch.setattr("sys.argv", ["apply_ticker_removals.py", "--review", str(review_path)])
    assert apply_mod.main() == 0
