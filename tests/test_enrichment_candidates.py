import unittest

from enrichment_candidates import build_enrichment_candidates


class BuildEnrichmentCandidatesTests(unittest.TestCase):
    def test_finra_notice_and_comment_are_preserved_for_enrichment(self):
        knowledge_data = {
            "speeches": [
                {
                    "metadata": {
                        "document_id": "notice-1",
                        "organization": "FINRA",
                        "title": "Regulatory Notice 26-01",
                        "doc_type": "Regulatory Notice",
                        "source_kind": "finra_regulatory_notice",
                        "date": "2026-01-10",
                        "url": "https://www.finra.org/rules-guidance/notices/26-01",
                        "word_count": 1200,
                    },
                    "content": {"full_text": "Notice body text"},
                },
                {
                    "metadata": {
                        "document_id": "comment-1",
                        "organization": "FINRA",
                        "title": "Comment Letter from Trade Association",
                        "doc_type": "Comment Letter",
                        "source_kind": "finra_comment_letter",
                        "date": "2026-01-20",
                        "url": "https://www.finra.org/rules-guidance/notices/26-01/comment-letter-1",
                        "word_count": 800,
                    },
                    "content": {"full_text": "We support the proposal with several targeted changes."},
                },
                {
                    "metadata": {
                        "document_id": "sec-1",
                        "organization": "SEC",
                        "title": "Unrelated SEC Speech",
                        "doc_type": "Speech",
                        "source_kind": "sec_speech",
                        "date": "2026-01-21",
                        "url": "https://www.sec.gov/newsroom/speeches-statements/example",
                        "word_count": 600,
                    },
                    "content": {"full_text": "SEC speech body"},
                },
            ]
        }

        candidates = build_enrichment_candidates(knowledge_data, org_key="finra")
        by_id = {item["doc_id"]: item for item in candidates}

        self.assertEqual(set(by_id.keys()), {"notice-1", "comment-1"})
        self.assertEqual(by_id["notice-1"]["source_kind"], "finra_regulatory_notice")
        self.assertEqual(by_id["comment-1"]["source_kind"], "finra_comment_letter")

    def test_blank_full_text_is_not_enrichment_candidate(self):
        knowledge_data = {
            "speeches": [
                {
                    "metadata": {
                        "document_id": "comment-blank",
                        "organization": "FINRA",
                        "title": "Empty Comment",
                        "doc_type": "Comment Letter",
                        "source_kind": "finra_comment_letter",
                    },
                    "content": {"full_text": ""},
                }
            ]
        }

        candidates = build_enrichment_candidates(knowledge_data, org_key="finra")
        self.assertEqual(candidates, [])

    def test_sec_rule_release_and_comment_are_preserved_for_enrichment(self):
        knowledge_data = {
            "speeches": [
                {
                    "metadata": {
                        "document_id": "sec-rule-1",
                        "organization": "SEC",
                        "title": "Application of the Federal Securities Laws to Certain Types of Crypto Assets",
                        "doc_type": "Interpretive Release",
                        "source_kind": "sec_rule_release",
                        "date": "2026-03-17",
                        "url": "https://www.sec.gov/rules-regulations/2026/03/s7-2026-09",
                        "word_count": 2400,
                    },
                    "content": {"full_text": "Interpretive release body text"},
                },
                {
                    "metadata": {
                        "document_id": "sec-comment-1",
                        "organization": "SEC",
                        "title": "Comment from Example Commenter",
                        "doc_type": "Public Comment",
                        "source_kind": "sec_rule_comment",
                        "date": "2026-03-20",
                        "url": "https://www.sec.gov/comments/s7-2026-09/example.pdf",
                        "word_count": 900,
                    },
                    "content": {"full_text": "We support the interpretive approach but recommend targeted clarification."},
                },
            ]
        }

        candidates = build_enrichment_candidates(knowledge_data, org_key="sec")
        by_id = {item["doc_id"]: item for item in candidates}

        self.assertEqual(set(by_id.keys()), {"sec-rule-1", "sec-comment-1"})
        self.assertEqual(by_id["sec-rule-1"]["source_kind"], "sec_rule_release")
        self.assertEqual(by_id["sec-comment-1"]["source_kind"], "sec_rule_comment")

    def test_lightweight_mode_can_filter_by_doc_id_without_copying_full_text(self):
        knowledge_data = {
            "speeches": [
                {
                    "metadata": {
                        "document_id": "doc-1",
                        "organization": "SEC",
                        "title": "Doc 1",
                        "source_kind": "sec_speech",
                    },
                    "content": {"full_text": "First body"},
                },
                {
                    "metadata": {
                        "document_id": "doc-2",
                        "organization": "SEC",
                        "title": "Doc 2",
                        "source_kind": "sec_speech",
                    },
                    "content": {"full_text": "Second body"},
                },
            ]
        }

        candidates = build_enrichment_candidates(
            knowledge_data,
            org_key="sec",
            include_full_text=False,
            allowed_doc_ids=["doc-2"],
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["doc_id"], "doc-2")
        self.assertNotIn("full_text", candidates[0])


if __name__ == "__main__":
    unittest.main()
