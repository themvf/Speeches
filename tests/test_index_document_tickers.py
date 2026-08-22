"""Tests for index_document_tickers.py.

The attribution rules are the point of these tests: on a public page a wrong
ticker chip beside an enforcement action reads as an accusation about a real
company, so "does a mention become a stored row, and at what confidence" is
the behaviour worth pinning down.
"""

from __future__ import annotations

import argparse
from unittest import mock

import pytest

import index_document_tickers as indexer


def _doc(document_id="doc-1", title="", full_text="", source_kind="sec_speech"):
    return {
        "document_id": document_id,
        "title": title,
        "full_text": full_text,
        "source_kind": source_kind,
        "url": f"https://example.test/{document_id}",
        "published_date": "2026-08-20",
    }


def _args(**overrides):
    defaults = dict(
        backfill=True, since_days=3, dry_run=False, limit=0, batch_size=200, sample_size=15,
        include_body=False, no_scope=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ── the attribution rules ────────────────────────────────────────────────────

def test_title_match_outranks_a_body_match():
    resolved = indexer.resolve_document("SEC charges NVDA with disclosure failures", "")
    assert resolved == {"NVDA": indexer.TITLE_CONFIDENCE}


def test_unambiguous_body_match_is_recorded_at_the_lower_confidence():
    resolved = indexer.resolve_document(
        "Quarterly market commentary", "Shares of $NVDA moved.", include_body=True
    )
    assert resolved == {"NVDA": indexer.BODY_CONFIDENCE}


def test_the_body_tier_is_off_unless_asked_for():
    """The first dry run showed it was ~90% of rows and mostly acronyms that
    are also tickers - DAO, RSI, ASIC, ASA, ADV. Opt-in only."""
    assert indexer.resolve_document("Quarterly market commentary", "Shares of $NVDA moved.") == {}


def test_a_title_hit_is_not_downgraded_by_also_appearing_in_the_body():
    resolved = indexer.resolve_document("NVDA under review", "More about $NVDA here.", include_body=True)
    assert resolved == {"NVDA": indexer.TITLE_CONFIDENCE}


def test_company_name_tier_never_qualifies_on_body_alone():
    """The 0.7 curated-name tier is what fires on ordinary regulatory prose,
    so it must not create a chip from a body mention."""
    with mock.patch.object(indexer.ticker_resolver, "resolve_tickers") as resolve:
        # Title resolves to nothing; body resolves at the name tier only.
        resolve.side_effect = [{}, {"AAPL": 0.7}]
        assert indexer.resolve_document(
            "A speech about competition policy", "... Apple ...", include_body=True
        ) == {}


def test_body_is_truncated_before_resolution():
    long_body = "x" * (indexer.MAX_BODY_CHARS + 5_000)
    with mock.patch.object(indexer.ticker_resolver, "resolve_tickers", return_value={}) as resolve:
        indexer.resolve_document("t", long_body, include_body=True)
    assert len(resolve.call_args_list[1].args[0]) == indexer.MAX_BODY_CHARS


def test_prose_with_no_ticker_produces_no_rows():
    document = _doc(title="Remarks on monetary policy and financial stability",
                    full_text="The Committee reviewed conditions across all sectors.")
    assert indexer.build_mention_rows(document) == []


def test_rows_carry_the_document_source_type_and_id():
    rows = indexer.build_mention_rows(_doc(document_id="d-42", title="MSFT earnings review"))
    assert rows == [{
        "source_type": "document",
        "source_id": "d-42",
        "mention_type": "ticker",
        "value": "MSFT",
        "normalized_value": "MSFT",
        "confidence": indexer.TITLE_CONFIDENCE,
    }]


def test_a_document_with_no_id_is_skipped():
    assert indexer.build_mention_rows(_doc(document_id="  ", title="NVDA results")) == []


# ── run behaviour ────────────────────────────────────────────────────────────

def test_dry_run_resolves_but_never_writes():
    with mock.patch.object(indexer.neon_feeds, "iter_documents_for_ticker_index",
                           return_value=iter([[_doc(title="NVDA earnings")]])), \
         mock.patch.object(indexer.neon_feeds, "insert_ticker_mentions") as insert:
        summary = indexer._run(_args(dry_run=True))

    insert.assert_not_called()
    assert summary["mention_rows"] == 1
    assert summary["inserted"] == 0
    assert summary["ok"] is True


def test_incremental_mode_passes_a_since_window_and_backfill_does_not():
    with mock.patch.object(indexer.neon_feeds, "iter_documents_for_ticker_index",
                           return_value=iter([])) as it:
        indexer._run(_args(backfill=False, since_days=3))
    assert it.call_args.kwargs["since"] is not None

    with mock.patch.object(indexer.neon_feeds, "iter_documents_for_ticker_index",
                           return_value=iter([])) as it:
        indexer._run(_args(backfill=True))
    assert it.call_args.kwargs["since"] is None


def test_a_failing_write_is_reported_without_aborting_the_run():
    batches = iter([[_doc("d-1", title="NVDA up")], [_doc("d-2", title="MSFT up")]])
    with mock.patch.object(indexer.neon_feeds, "iter_documents_for_ticker_index", return_value=batches), \
         mock.patch.object(indexer.neon_feeds, "insert_ticker_mentions",
                           side_effect=[RuntimeError("boom"), 1]):
        summary = indexer._run(_args())

    assert summary["ok"] is False
    assert summary["failed_batches"][0]["error"] == "boom"
    # The second batch still ran and still counted.
    assert summary["inserted"] == 1
    assert summary["documents_scanned"] == 2


def test_samples_are_capped_for_eyeballing():
    docs = [[_doc(f"d-{i}", title="NVDA moves") for i in range(10)]]
    with mock.patch.object(indexer.neon_feeds, "iter_documents_for_ticker_index", return_value=iter(docs)), \
         mock.patch.object(indexer.neon_feeds, "insert_ticker_mentions", return_value=10):
        summary = indexer._run(_args(sample_size=3))

    assert len(summary["samples"]) == 3
    assert summary["documents_with_tickers"] == 10


@pytest.mark.parametrize("mode,expected", [(True, "backfill"), (False, "incremental")])
def test_summary_names_the_mode(mode, expected):
    with mock.patch.object(indexer.neon_feeds, "iter_documents_for_ticker_index", return_value=iter([])):
        assert indexer._run(_args(backfill=mode))["mode"] == expected


# ── tracked-universe scoping ─────────────────────────────────────────────────

def test_scoping_drops_tickers_no_board_can_show():
    """Chips only ever render for tickers on the Movers or Industries boards,
    so anything outside that set is noise that can never be displayed."""
    with mock.patch.object(indexer.ticker_resolver, "resolve_tickers",
                           return_value={"NVDA": 1.0, "DAO": 1.0, "ASIC": 1.0}):
        resolved = indexer.resolve_document("headline", "", tracked={"NVDA", "AAPL"})
    assert resolved == {"NVDA": indexer.TITLE_CONFIDENCE}


def test_an_empty_tracked_set_means_no_scoping_not_no_results():
    """A failed config read must degrade to the previous behaviour, never
    silently index nothing."""
    with mock.patch.object(indexer.ticker_resolver, "resolve_tickers", return_value={"NVDA": 1.0}):
        assert indexer.resolve_document("headline", "", tracked=set()) == {"NVDA": 1.0}
        assert indexer.resolve_document("headline", "", tracked=None) == {"NVDA": 1.0}


def test_the_real_tracked_universe_covers_boards_and_excludes_the_known_noise():
    tracked = indexer.load_tracked_tickers()
    assert len(tracked) > 500
    for symbol in ["NVDA", "AAPL", "MSFT", "MARA"]:
        assert symbol in tracked, f"{symbol} should be indexable"
    # Every false positive the first dry run surfaced, except RSI - which is a
    # real tracked ticker (Rush Street Interactive) and therefore genuine
    # ambiguity rather than a resolver bug.
    for symbol in ["DAO", "ASIC", "WSE", "ASA", "ADV", "SBS", "FUFU", "LITS"]:
        assert symbol not in tracked, f"{symbol} was a known false positive"


def test_the_summary_reports_the_scope_it_ran_under():
    with mock.patch.object(indexer.neon_feeds, "iter_documents_for_ticker_index", return_value=iter([])):
        unscoped = indexer._run(_args(no_scope=True))
        scoped = indexer._run(_args(no_scope=False))
    assert unscoped["tracked_universe"] == "unscoped"
    assert unscoped["include_body"] is False
    assert isinstance(scoped["tracked_universe"], int) and scoped["tracked_universe"] > 500


def test_full_text_is_only_fetched_when_the_body_tier_will_use_it():
    """full_text is the largest column in the documents table. With the body
    tier off, a full-corpus pass would drag tens of thousands of document
    bodies out of Neon and resolve against none of them."""
    with mock.patch.object(indexer.neon_feeds, "iter_documents_for_ticker_index",
                           return_value=iter([])) as it:
        indexer._run(_args(include_body=False))
    assert it.call_args.kwargs["include_full_text"] is False

    with mock.patch.object(indexer.neon_feeds, "iter_documents_for_ticker_index",
                           return_value=iter([])) as it:
        indexer._run(_args(include_body=True))
    assert it.call_args.kwargs["include_full_text"] is True
