import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import unittest
from src.data.claim_parser import PatentClaimParser
from src.data.preprocess import PatentPreprocessor

class TestClaimParserAndPreprocessor(unittest.TestCase):
    def test_claim_parser_independent_and_dependent(self):
        sample_claims = (
            "1. An apparatus comprising a processor and a memory configured to store data.\n"
            "2. The apparatus of claim 1, further comprising a sensor.\n"
            "3. A method for dynamic routing comprising transmitting packets over a network."
        )
        parsed = PatentClaimParser.parse_claims(sample_claims)
        self.assertEqual(len(parsed["independent"]), 2)
        self.assertEqual(len(parsed["dependent"]), 1)
        self.assertEqual(parsed["independent"][0]["claim_number"], "1")
        self.assertEqual(parsed["independent"][1]["claim_number"], "3")
        self.assertEqual(parsed["dependent"][0]["claim_number"], "2")

    def test_preprocessor_preserves_technical_units(self):
        raw = "Title: <b>High Speed RAM</b>\nAbstract: Operating at 100MHz &amp; 3.5V."
        rec = {"title": "High Speed RAM", "abstract": "Operating at 100MHz & 3.5V.", "claims": "1. A memory unit."}
        proc = PatentPreprocessor.process_patent_record(rec)
        self.assertIn("100MHz", proc["abstract"])
        self.assertIn("3.5V", proc["abstract"])
        self.assertNotIn("<b>", proc["title"])
        self.assertIn("Title:", proc["combined_text_transformer"])

if __name__ == "__main__":
    unittest.main()
