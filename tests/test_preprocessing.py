import pytest
from src.utils.seed import set_seed
from src.utils.metrics import compute_classification_metrics, compute_retrieval_metrics
import numpy as np

def test_seed_reproducibility():
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    assert np.allclose(a, b)

def test_classification_metrics():
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred = np.array([[1, 0, 0], [0, 1, 0]])
    metrics = compute_classification_metrics(y_true, y_pred)
    assert "micro_f1" in metrics
    assert "macro_f1" in metrics
    assert 0.0 <= metrics["micro_f1"] <= 1.0

def test_retrieval_metrics():
    rankings = [["p1", "p2", "p3"], ["p4", "p5", "p6"]]
    positives = [{"p1"}, {"p6"}]
    metrics = compute_retrieval_metrics(rankings, positives, ks=[1, 3])
    assert "recall@3" in metrics
    assert "map@3" in metrics
    assert "mrr" in metrics
    assert metrics["mrr"] > 0.0
