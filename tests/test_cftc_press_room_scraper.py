import unittest

from cftc_press_room_scraper import (
    _infer_public_statement_doc_type,
    _is_cftc_detail_url,
    _speaker_from_title,
)


class CFTCPressRoomScraperHelpersTests(unittest.TestCase):
    def test_detail_url_detection(self):
        self.assertTrue(
            _is_cftc_detail_url(
                "https://www.cftc.gov/PressRoom/PressReleases/9191-26",
                "cftc_press_release",
            )
        )
        self.assertTrue(
            _is_cftc_detail_url(
                "https://www.cftc.gov/PressRoom/SpeechesTestimony/oparomero13",
                "cftc_public_statement_remark",
            )
        )
        self.assertFalse(
            _is_cftc_detail_url(
                "https://www.cftc.gov/PressRoom/PressReleases",
                "cftc_press_release",
            )
        )
        self.assertFalse(
            _is_cftc_detail_url(
                "https://www.cftc.gov/PressRoom/SpeechesTestimony/index.htm",
                "cftc_public_statement_remark",
            )
        )

    def test_public_statement_doc_type_inference(self):
        self.assertEqual(
            _infer_public_statement_doc_type(
                "Testimony of Chairman Rostin Behnam before the Senate Agriculture Committee"
            ),
            "Testimony",
        )
        self.assertEqual(
            _infer_public_statement_doc_type(
                "Remarks of Commissioner Christy Goldsmith Romero at FIA Boca"
            ),
            "Remarks",
        )
        self.assertEqual(
            _infer_public_statement_doc_type(
                "Statement of Commissioner Kristin N. Johnson on Digital Asset Policy"
            ),
            "Statement",
        )

    def test_speaker_inference_from_title(self):
        self.assertEqual(
            _speaker_from_title(
                "Statement of Commissioner Christy Goldsmith Romero on AI and Financial Stability"
            ),
            "Commissioner Christy Goldsmith Romero",
        )
        self.assertEqual(
            _speaker_from_title(
                "Remarks of Chairman Rostin Behnam at the International Swaps and Derivatives Association"
            ),
            "Chairman Rostin Behnam",
        )


if __name__ == "__main__":
    unittest.main()
