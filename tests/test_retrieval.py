import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import unittest
import pandas as pd
from src.retrieval.bm25_retriever import BM25Retriever

class TestBM25Retrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_df = pd.read_parquet("data/splits/train.parquet")
        cls.val_df = pd.read_parquet("data/splits/val.parquet")

    def test_bm25_retriever_temporal_safety_and_eval(self):
        retriever = BM25Retriever()
        retriever.index_corpus(self.train_df)
        
        query = self.val_df.iloc[0].to_dict()
        results = retriever.retrieve(query, top_k=10)
        
        q_date = pd.to_datetime(query["grant_date"])
        for r in results:
            cand_date = pd.to_datetime(r["grant_date"])
            self.assertLessEqual(cand_date, q_date)
            self.assertNotEqual(r["patent_id"], query["patent_id"])
            
        metrics = retriever.evaluate_queries(self.val_df.head(20), ks=[5, 10])
        self.assertIn("recall@10", metrics)
        self.assertIn("map@10", metrics)

if __name__ == "__main__":
    unittest.main()
