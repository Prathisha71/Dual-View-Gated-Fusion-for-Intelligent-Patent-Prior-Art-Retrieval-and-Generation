import os
import re
import math
import joblib
import yaml
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from collections import Counter
from src.utils.logging import setup_logger

logger = setup_logger("BM25Index")

class NativeBM25Okapi:
    """High-performance, pure Python/NumPy BM25Okapi implementation."""
    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_len = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 1.0
        
        self.doc_freqs = []
        self.idf = {}
        self.nd = {}
        
        # Calculate term frequencies per document and IDF
        df_counter = Counter()
        for doc in corpus_tokens:
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            for word in frequencies.keys():
                df_counter[word] += 1
                
        for word, freq in df_counter.items():
            # Standard Lucene/BM25Okapi IDF formula
            idf_val = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[word] = max(idf_val, 0.0)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        q_freqs = Counter(query_tokens)
        
        for word in q_freqs.keys():
            if word not in self.idf:
                continue
            idf_val = self.idf[word]
            for idx, doc_freq in enumerate(self.doc_freqs):
                tf = doc_freq.get(word, 0)
                if tf > 0:
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_len[idx] / self.avgdl))
                    scores[idx] += idf_val * (tf * (self.k1 + 1.0) / denom)
                    
        return scores.tolist()


class PatentBM25Index:
    """BM25 Lexical Index for Patent Text Retrieval."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)["bm25"]
        self.k1 = cfg.get("k1", 1.5)
        self.b = cfg.get("b", 0.75)
        self.index = None
        self.corpus_tokens = []
        self.is_fitted = False

    @staticmethod
    def tokenize(text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []
        return re.findall(r'\b\w+\b', text.lower())

    def fit(self, texts: List[str]):
        logger.info(f"Tokenizing {len(texts)} texts for BM25 indexing...")
        self.corpus_tokens = [self.tokenize(t) for t in texts]
        logger.info(f"Building NativeBM25Okapi index (k1={self.k1}, b={self.b})...")
        self.index = NativeBM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)
        self.is_fitted = True

    def get_scores(self, query_text: str) -> List[float]:
        if not self.is_fitted:
            raise ValueError("BM25 Index is not fitted!")
        query_tokens = self.tokenize(query_text)
        return self.index.get_scores(query_tokens)

    def save(self, output_path: str = "models/retrieval/bm25_index.pkl"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump({"index": self.index, "corpus_tokens": self.corpus_tokens}, output_path)
        logger.info(f"Saved BM25 index to {output_path}")

    def load(self, input_path: str = "models/retrieval/bm25_index.pkl"):
        data = joblib.load(input_path)
        self.index = data["index"]
        self.corpus_tokens = data["corpus_tokens"]
        self.is_fitted = True
        logger.info(f"Loaded BM25 index from {input_path}")
