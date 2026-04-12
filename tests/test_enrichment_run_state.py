import unittest

from enrichment_run_state import (
    abort_enrichment_run_state,
    advance_enrichment_run_state,
    create_enrichment_run_state,
    next_enrichment_run_doc_ids,
    normalize_enrichment_run_state,
)


class EnrichmentRunStateTests(unittest.TestCase):
    def test_create_run_state_deduplicates_doc_ids_and_starts_running(self):
        state = create_enrichment_run_state(
            run_type="batch",
            doc_ids=["doc-1", "doc-2", "doc-1", "", "doc-3"],
            scope_key="sec",
            scope_label="SEC",
            model_name="gpt-4o-mini",
            mode="only_missing_or_failed",
        )

        self.assertEqual(state["status"], "running")
        self.assertEqual(state["total_selected"], 3)
        self.assertEqual(state["remaining_doc_ids"], ["doc-1", "doc-2", "doc-3"])
        self.assertEqual(next_enrichment_run_doc_ids(state, max_docs=2), ["doc-1", "doc-2"])

    def test_advance_run_state_tracks_progress_and_completes(self):
        state = create_enrichment_run_state(
            run_type="batch",
            doc_ids=["doc-1", "doc-2"],
            scope_key="__all__",
            scope_label="All Organizations",
            model_name="gpt-4o-mini",
            mode="re_enrich_all",
        )

        state = advance_enrichment_run_state(
            state,
            processed_doc_ids=["doc-1"],
            fallback_count=1,
            skipped_count=0,
            last_doc_id="doc-1",
            last_doc_title="First Doc",
        )
        self.assertEqual(state["status"], "running")
        self.assertEqual(state["processed"], 1)
        self.assertEqual(state["fallback_count"], 1)
        self.assertEqual(state["remaining_doc_ids"], ["doc-2"])

        state = advance_enrichment_run_state(
            state,
            processed_doc_ids=["doc-2"],
            skipped_count=1,
            last_doc_id="doc-2",
            last_doc_title="Second Doc",
        )
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["processed"], 2)
        self.assertEqual(state["skipped_count"], 1)
        self.assertEqual(state["remaining_doc_ids"], [])
        self.assertTrue(state["finished_at"].endswith("Z"))

    def test_abort_run_state_sets_error_and_preserves_progress(self):
        state = create_enrichment_run_state(
            run_type="batch",
            doc_ids=["doc-1", "doc-2", "doc-3"],
            scope_key="sec",
            scope_label="SEC",
            model_name="gpt-4o-mini",
            mode="only_missing_or_failed",
        )
        state = advance_enrichment_run_state(state, processed_doc_ids=["doc-1"])
        aborted = abort_enrichment_run_state(state, "GCS save failed")

        self.assertEqual(aborted["status"], "aborted")
        self.assertEqual(aborted["processed"], 1)
        self.assertEqual(aborted["remaining_doc_ids"], ["doc-2", "doc-3"])
        self.assertEqual(aborted["error"], "GCS save failed")
        self.assertEqual(aborted["last_error"], "GCS save failed")

    def test_normalize_returns_empty_dict_for_invalid_state(self):
        self.assertEqual(normalize_enrichment_run_state(None), {})
        self.assertEqual(normalize_enrichment_run_state({"status": "running"}), {})


if __name__ == "__main__":
    unittest.main()
