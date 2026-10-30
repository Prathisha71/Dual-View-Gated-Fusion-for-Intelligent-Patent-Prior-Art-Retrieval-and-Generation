import sys
import os
import unittest

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.utils.seed import set_seed
from src.utils.metrics import compute_classification_metrics, compute_retrieval_metrics
import numpy as np

class TestPhase1(unittest.TestCase):
    def test_seed_reproducibility(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        self.assertTrue(np.allclose(a, b))

    def test_classification_metrics(self):
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[1, 0, 0], [0, 1, 0]])
        metrics = compute_classification_metrics(y_true, y_pred)
        self.assertIn("micro_f1", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertTrue(0.0 <= metrics["micro_f1"] <= 1.0)

    def test_retrieval_metrics(self):
        rankings = [["p1", "p2", "p3"], ["p4", "p5", "p6"]]
        positives = [{"p1"}, {"p6"}]
        metrics = compute_retrieval_metrics(rankings, positives, ks=[1, 3])
        self.assertIn("recall@3", metrics)
        self.assertIn("map@3", metrics)
        self.assertIn("mrr", metrics)
        self.assertGreater(metrics["mrr"], 0.0)

if __name__ == "__main__":
    unittest.main()
