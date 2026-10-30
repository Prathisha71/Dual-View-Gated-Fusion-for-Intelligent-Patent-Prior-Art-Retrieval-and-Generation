import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import unittest
import pandas as pd
from src.classification.baseline_classifier import TfidfBaselineClassifier
from src.retrieval.tfidf_retriever import TfidfRetriever

class TestTFIDFBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_df = pd.read_parquet("data/splits/train.parquet")
        cls.val_df = pd.read_parquet("data/splits/val.parquet")

    def test_tfidf_classifier_training_and_eval(self):
        clf = TfidfBaselineClassifier()
        metrics = clf.train(self.train_df.head(200), self.val_df.head(50))
        self.assertIn("micro_f1", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertTrue(0.0 <= metrics["micro_f1"] <= 1.0)

    def test_tfidf_retriever_temporal_safety(self):
        retriever = TfidfRetriever()
        retriever.index_corpus(self.train_df)
        query = self.val_df.iloc[0].to_dict()
        results = retriever.retrieve(query, top_k=10)
        
        q_date = pd.to_datetime(query["grant_date"])
        for r in results:
            cand_date = pd.to_datetime(r["grant_date"])
            self.assertLessEqual(cand_date, q_date)
            self.assertNotEqual(r["patent_id"], query["patent_id"])

if __name__ == "__main__":
    unittest.main()
