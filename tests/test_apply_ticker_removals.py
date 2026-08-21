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


# ---------------------------------------------------------------------------
# Safety guards. These matter more than the happy path: the workflow runs this
# unattended and pushes to main, and build_industry_config.py is also what the
# hourly filing watch rebuilds from, so a bad edit has blast radius beyond the
# weekly job.
# ---------------------------------------------------------------------------

BUILDER_SRC = (
    'UNIVERSE: List[str] = [\n'
    '    "AAA", "BBB", "CCC",\n'
    ']\n'
    '\n'
    'SUB_INDUSTRY_GROUPS: Dict[str, List[str]] = {\n'
    '    "Some Group": ["AAA", "DDD"],\n'
    '}\n'
)


def _fixture(tmp_path, candidates):
    builder = tmp_path / "build_industry_config.py"
    builder.write_text(BUILDER_SRC)
    config = tmp_path / "industry-config.json"
    config.write_text(json.dumps({
        "tickerCount": 3,
        "industries": [{"sic": "1", "label": "L", "tickers": [
            {"ticker": "AAA", "name": "A", "cik": "1"},
            {"ticker": "BBB", "name": "B", "cik": "2"},
            {"ticker": "CCC", "name": "C", "cik": "3"},
        ]}],
    }))
    state = tmp_path / "industry_state.json"
    state.write_text(json.dumps({"latest": {"AAA": {"form": "10-Q", "accession": "x"}}}))
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"flaggedCount": len(candidates), "candidates": candidates}))
    return builder, config, state, review


def _argv(builder, config, state, review):
    return ["apply_ticker_removals.py", "--review", str(review), "--builder", str(builder),
            "--config", str(config), "--state", str(state)]


def test_extract_universe_tickers_parses_list_via_ast():
    assert apply_mod.extract_universe_tickers(BUILDER_SRC) == ["AAA", "BBB", "CCC"]


def test_extract_universe_tickers_raises_on_corrupted_source():
    import pytest
    with pytest.raises(SyntaxError):
        apply_mod.extract_universe_tickers('UNIVERSE: List[str] = [\n    "AAA",\n')  # unclosed


def test_mass_removal_over_cap_applies_nothing(tmp_path, monkeypatch):
    many = [{"ticker": f"T{i}", "reason": "deregistered", "confidence": "high"} for i in range(30)]
    builder, config, state, review = _fixture(tmp_path, many)
    before = builder.read_text(), config.read_text(), state.read_text(), review.read_text()

    monkeypatch.setattr(apply_mod, "MAX_AUTO_REMOVALS", 25)
    monkeypatch.setattr("sys.argv", _argv(builder, config, state, review))
    assert apply_mod.main() == 0  # degrades to flag-only, does not fail the workflow

    # Nothing touched - the whole batch stays flagged for a human.
    assert (builder.read_text(), config.read_text(), state.read_text(), review.read_text()) == before


def test_validation_failure_restores_every_file(tmp_path, monkeypatch):
    candidates = [{"ticker": "AAA", "reason": "deregistered", "confidence": "high"}]
    builder, config, state, review = _fixture(tmp_path, candidates)
    before = builder.read_text(), config.read_text(), state.read_text(), review.read_text()

    # Simulate exactly the failure mode the List[str] bracket bug would have
    # produced: the edit silently corrupts the builder source.
    def corrupting_edit(path, tickers):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("UNIVERSE: List[str] = [\n    'AAA',\n")  # unclosed -> unparseable
        return {"universe": ["AAA"], "sub_industry_groups": []}

    monkeypatch.setattr(apply_mod, "remove_from_builder_source", corrupting_edit)
    monkeypatch.setattr("sys.argv", _argv(builder, config, state, review))

    assert apply_mod.main() == 1  # non-zero so the workflow step fails loudly
    # Every file byte-for-byte as it started - no partial application.
    assert (builder.read_text(), config.read_text(), state.read_text(), review.read_text()) == before


def test_validation_catches_collateral_ticker_removal(tmp_path, monkeypatch):
    """A greedy pattern that removes MORE than intended must be caught, not
    committed - the review file said one ticker, so exactly one must go."""
    candidates = [{"ticker": "AAA", "reason": "deregistered", "confidence": "high"}]
    builder, config, state, review = _fixture(tmp_path, candidates)
    before_builder = builder.read_text()

    def over_removing_edit(path, tickers):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('UNIVERSE: List[str] = [\n    "CCC",\n]\n')  # also ate BBB
        return {"universe": ["AAA"], "sub_industry_groups": []}

    monkeypatch.setattr(apply_mod, "remove_from_builder_source", over_removing_edit)
    monkeypatch.setattr("sys.argv", _argv(builder, config, state, review))

    assert apply_mod.main() == 1
    assert builder.read_text() == before_builder


def test_happy_path_still_applies_and_validates(tmp_path, monkeypatch):
    candidates = [{"ticker": "AAA", "reason": "deregistered", "confidence": "high"}]
    builder, config, state, review = _fixture(tmp_path, candidates)
    monkeypatch.setattr("sys.argv", _argv(builder, config, state, review))

    assert apply_mod.main() == 0
    assert apply_mod.extract_universe_tickers(builder.read_text()) == ["BBB", "CCC"]
    written_config = json.loads(config.read_text())
    assert written_config["tickerCount"] == 2
    assert "AAA" not in json.loads(state.read_text())["latest"]
    assert json.loads(review.read_text())["candidates"] == []
