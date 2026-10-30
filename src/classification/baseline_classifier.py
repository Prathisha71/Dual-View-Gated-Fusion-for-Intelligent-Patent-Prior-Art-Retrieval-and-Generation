import os
import joblib
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Dict, Any, List, Tuple
from src.features.tfidf_features import PatentTfidfExtractor
from src.utils.metrics import compute_classification_metrics
from src.utils.logging import setup_logger

logger = setup_logger("BaselineTfidfClassifier")

class TfidfBaselineClassifier:
    """TF-IDF + OneVsRest Logistic Regression Multi-Label Classifier Baseline."""
    
    def __init__(self, cpc_classes: List[str] = None):
        self.cpc_classes = cpc_classes
        self.mlb = MultiLabelBinarizer(classes=cpc_classes)
        self.clf = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=500, solver="liblinear"))
        self.extractor = PatentTfidfExtractor()
        self.is_fitted = False

    def _prepare_labels(self, cpc_series: pd.Series) -> np.ndarray:
        label_lists = [[c.strip() for c in str(x).split(",") if c.strip()] for x in cpc_series]
        if not self.is_fitted:
            return self.mlb.fit_transform(label_lists)
        return self.mlb.transform(label_lists)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None) -> Dict[str, float]:
        logger.info("Training TF-IDF + Logistic Regression baseline classifier...")
        X_train = self.extractor.fit_transform(train_df["combined_text_lexical"].tolist())
        Y_train = self._prepare_labels(train_df["cpc_subclasses"])
        
        self.clf.fit(X_train, Y_train)
        self.is_fitted = True
        
        train_probs = self.clf.predict_proba(X_train)
        train_metrics = compute_classification_metrics(Y_train, train_probs > 0.5)
        logger.info(f"TF-IDF Baseline Train Micro F1: {train_metrics['micro_f1']:.4f}")
        
        if val_df is not None:
            val_metrics = self.evaluate(val_df)
            logger.info(f"TF-IDF Baseline Val Micro F1: {val_metrics['micro_f1']:.4f}")
            return val_metrics
            
        return train_metrics

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.extractor.transform(texts)
        return self.clf.predict_proba(X)

    def evaluate(self, df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
        X = self.extractor.transform(df["combined_text_lexical"].tolist())
        Y_true = self._prepare_labels(df["cpc_subclasses"])
        probs = self.clf.predict_proba(X)
        preds = (probs >= threshold).astype(int)
        return compute_classification_metrics(Y_true, preds)

    def save(self, output_path: str = "models/classification/tfidf_baseline_model.pkl"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump({"clf": self.clf, "mlb": self.mlb, "extractor": self.extractor}, output_path)
        logger.info(f"Saved TF-IDF baseline classifier to {output_path}")

    def load(self, input_path: str = "models/classification/tfidf_baseline_model.pkl"):
        data = joblib.load(input_path)
        self.clf = data["clf"]
        self.mlb = data["mlb"]
        self.extractor = data["extractor"]
        self.is_fitted = True
        logger.info(f"Loaded TF-IDF baseline classifier from {input_path}")
