import os
import joblib
import yaml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Optional
from src.utils.logging import setup_logger

logger = setup_logger("TfidfFeatures")

class PatentTfidfExtractor:
    """TF-IDF Feature Extractor for Patent Classification and Retrieval."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)["tfidf"]
            
        self.max_features = cfg.get("max_features", 25000)
        self.ngram_range = tuple(cfg.get("ngram_range", [1, 2]))
        self.min_df = cfg.get("min_df", 3)
        self.max_df = cfg.get("max_df", 0.9)
        self.sublinear_tf = cfg.get("sublinear_tf", True)
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            stop_words="english"
        )
        self.is_fitted = False

    def fit_transform(self, texts: list) -> pd.DataFrame:
        logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} texts...")
        matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return matrix

    def transform(self, texts: list):
        if not self.is_fitted:
            raise ValueError("TF-IDF Vectorizer is not fitted yet!")
        return self.vectorizer.transform(texts)

    def save(self, output_path: str = "models/classification/tfidf_vectorizer.pkl"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self.vectorizer, output_path)
        logger.info(f"Saved fitted TF-IDF vectorizer to {output_path}")

    def load(self, input_path: str = "models/classification/tfidf_vectorizer.pkl"):
        self.vectorizer = joblib.load(input_path)
        self.is_fitted = True
        logger.info(f"Loaded TF-IDF vectorizer from {input_path}")
