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


if __name__ == "__main__":
    unittest.main()
