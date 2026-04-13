import unittest

from enrichment_model_fallback import (
    build_model_attempt_order,
    is_model_access_error,
    is_transient_error,
    run_with_model_fallback,
)


class EnrichmentModelFallbackTests(unittest.TestCase):
    def test_build_model_attempt_order_deduplicates_and_prefers_preferred_model(self):
        ordered = build_model_attempt_order(
            preferred_model="gpt-4o-mini",
            accessible_models=["gpt-5-mini", "gpt-4o-mini", "gpt-4.1"],
        )

        self.assertEqual(ordered[:3], ["gpt-4o-mini", "gpt-5-mini", "gpt-4.1"])
        self.assertEqual(len(ordered), len(set(ordered)))

    def test_model_access_error_detection(self):
        self.assertTrue(is_model_access_error(RuntimeError("model_not_found for project key")))
        self.assertFalse(is_model_access_error(RuntimeError("request timed out")))

    def test_transient_error_detection(self):
        self.assertTrue(is_transient_error(RuntimeError("Rate limit exceeded")))
        self.assertTrue(is_transient_error(RuntimeError("Gateway timeout from provider")))
        self.assertFalse(is_transient_error(RuntimeError("Model did not return parseable JSON")))

    def test_run_with_model_fallback_retries_transient_error_on_same_model(self):
        calls = []

        def run_agent(_client, _doc, model_name):
            calls.append(model_name)
            if len(calls) == 1:
                raise RuntimeError("Rate limit exceeded")
            return {"summary": "ok", "model": model_name}

        outcome = run_with_model_fallback(
            client=object(),
            doc={"doc_id": "doc-1"},
            preferred_model="gpt-5-mini",
            accessible_models=["gpt-5-mini"],
            run_agent=run_agent,
            sleep_fn=lambda _seconds: None,
        )

        self.assertEqual(outcome["model_used"], "gpt-5-mini")
        self.assertEqual(calls, ["gpt-5-mini", "gpt-5-mini"])

    def test_run_with_model_fallback_moves_to_next_model_after_failure(self):
        calls = []

        def run_agent(_client, _doc, model_name):
            calls.append(model_name)
            if model_name == "gpt-5-mini":
                raise RuntimeError("Model did not return parseable JSON")
            return {"summary": "ok", "model": model_name}

        outcome = run_with_model_fallback(
            client=object(),
            doc={"doc_id": "doc-1"},
            preferred_model="gpt-5-mini",
            accessible_models=["gpt-5-mini", "gpt-4o-mini"],
            run_agent=run_agent,
            sleep_fn=lambda _seconds: None,
        )

        self.assertEqual(outcome["model_used"], "gpt-4o-mini")
        self.assertEqual(calls, ["gpt-5-mini", "gpt-4o-mini"])


if __name__ == "__main__":
    unittest.main()
