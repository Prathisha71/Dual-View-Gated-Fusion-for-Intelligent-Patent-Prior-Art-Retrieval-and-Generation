import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import unittest
import pandas as pd

class TestTemporalLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train_path = "data/splits/train.parquet"
        val_path = "data/splits/val.parquet"
        test_path = "data/splits/test.parquet"
        
        if not os.path.exists(train_path):
            from src.data.build_dataset import build_and_save_master_dataset
            from src.data.create_temporal_split import create_temporal_split
            df = build_and_save_master_dataset()
            create_temporal_split(df)
            
        cls.train_df = pd.read_parquet(train_path)
        cls.val_df = pd.read_parquet(val_path)
        cls.test_df = pd.read_parquet(test_path)

    def test_disjoint_split_ids(self):
        train_ids = set(self.train_df["patent_id"])
        val_ids = set(self.val_df["patent_id"])
        test_ids = set(self.test_df["patent_id"])
        
        self.assertEqual(len(train_ids.intersection(val_ids)), 0)
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)
        self.assertEqual(len(val_ids.intersection(test_ids)), 0)

    def test_temporal_ordering_of_splits(self):
        train_max_year = pd.to_datetime(self.train_df["grant_date"]).dt.year.max()
        val_min_year = pd.to_datetime(self.val_df["grant_date"]).dt.year.min()
        val_max_year = pd.to_datetime(self.val_df["grant_date"]).dt.year.max()
        test_min_year = pd.to_datetime(self.test_df["grant_date"]).dt.year.min()
        
        self.assertLessEqual(train_max_year, 2022)
        self.assertGreaterEqual(val_min_year, 2023)
        self.assertLessEqual(val_max_year, 2024)
        self.assertEqual(test_min_year, 2025)

    def test_citation_ground_truth_temporal_validity(self):
        """Verifies all cited ground truth patents have grant_date <= query patent grant_date."""
        all_df = pd.concat([self.train_df, self.val_df, self.test_df]).set_index("patent_id")
        date_map = pd.to_datetime(all_df["grant_date"]).to_dict()
        
        violations = 0
        for pid, row in all_df.iterrows():
            cites = [c.strip() for c in str(row.get("cited_patent_ids", "")).split(",") if c.strip()]
            q_date = date_map[pid]
            for cite_id in cites:
                if cite_id in date_map:
                    c_date = date_map[cite_id]
                    if c_date > q_date:
                        violations += 1
                        
        self.assertEqual(violations, 0, f"Detected {violations} future citation temporal leakage violations!")

if __name__ == "__main__":
    unittest.main()
