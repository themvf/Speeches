import unittest

from comment_position import infer_comment_position, is_comment_position_document


class CommentPositionInferenceTests(unittest.TestCase):
    def test_regulations_gov_public_comment_is_eligible(self):
        doc = {"source_kind": "regulations_gov_comment", "doc_type": "Public Comment"}
        self.assertTrue(is_comment_position_document(doc))

    def test_supportive_public_comment_can_still_note_limited_objections(self):
        doc = {"source_kind": "regulations_gov_comment", "doc_type": "Public Comment"}
        text = (
            "MX Technologies strongly urges the CFPB to preserve and properly implement the PFDR Rule. "
            "MX supports the current prohibition on data access fees, recommends targeted adjustments to "
            "avoid service disruption, and opposes any reinterpretation that would limit authorized third parties."
        )
        result = infer_comment_position(doc, text)
        self.assertEqual(result["label"], "supportive")
        self.assertGreaterEqual(result["confidence"], 0.8)

    def test_opposed_public_comment_catches_narrow_or_abandon_language(self):
        doc = {"source_kind": "regulations_gov_comment", "doc_type": "Public Comment"}
        text = (
            "The Cato Institute urges the CFPB to significantly narrow or abandon the proposal. "
            "The comment is critical of the rule and argues that the Bureau's interpretation exceeds statutory authority."
        )
        result = infer_comment_position(doc, text)
        self.assertEqual(result["label"], "opposed")
        self.assertGreaterEqual(result["confidence"], 0.8)

    def test_mixed_public_comment_detects_support_with_material_opposition(self):
        doc = {"source_kind": "regulations_gov_comment", "doc_type": "Public Comment"}
        text = (
            "We support the proposal's portability goals, but we oppose the mandatory fee restrictions "
            "and recommend significant revisions before the rule is finalized."
        )
        result = infer_comment_position(doc, text)
        self.assertEqual(result["label"], "mixed")

    def test_non_comment_document_remains_not_applicable(self):
        doc = {"source_kind": "sec_speech", "doc_type": "Speech"}
        result = infer_comment_position(doc, "We support better disclosure rules.")
        self.assertEqual(result["label"], "not_applicable")
        self.assertEqual(result["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
