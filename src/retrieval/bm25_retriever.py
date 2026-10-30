import numpy as np
import pandas as pd
from typing import List, Dict, Any
from src.features.bm25_index import PatentBM25Index
from src.utils.metrics import compute_retrieval_metrics
from src.utils.logging import setup_logger

logger = setup_logger("BM25Retriever")

class BM25Retriever:
    """BM25 Lexical Retriever with Strict Temporal Filtering."""
    
    def __init__(self, index: PatentBM25Index = None):
        self.index = index or PatentBM25Index()
        self.corpus_df = None
        self.corpus_dates = None

    def index_corpus(self, corpus_df: pd.DataFrame):
        logger.info(f"Indexing candidate retrieval corpus of {len(corpus_df)} patents with BM25...")
        self.corpus_df = corpus_df.reset_index(drop=True)
        self.index.fit(self.corpus_df["combined_text_lexical"].tolist())
        self.corpus_dates = pd.to_datetime(self.corpus_df["grant_date"]).values

    def retrieve(self, query_record: Dict[str, Any], top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves prior art candidates using BM25 scoring.
        Applies strict temporal filtering: candidate grant_date <= query grant_date.
        Excludes query patent itself.
        """
        query_text = query_record["combined_text_lexical"]
        query_id = query_record["patent_id"]
        query_date = pd.to_datetime(query_record["grant_date"])
        
        scores = np.array(self.index.get_scores(query_text))
        
        # Temporal mask: candidate_date <= query_date AND candidate_id != query_id
        valid_mask = (self.corpus_dates <= query_date.to_datetime64()) & (self.corpus_df["patent_id"].values != query_id)
        
        filtered_scores = np.where(valid_mask, scores, -1.0)
        top_indices = np.argsort(filtered_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(filtered_scores[idx])
            if score < 0.0:
                break
            cand = self.corpus_df.iloc[idx].to_dict()
            cand["similarity_score"] = score
            results.append(cand)
            
        return results

    def evaluate_queries(self, test_df: pd.DataFrame, ks: List[int] = [10, 50, 100]) -> Dict[str, float]:
        """Evaluates BM25 retrieval performance across test queries against citation ground truth."""
        logger.info(f"Evaluating BM25 retriever across {len(test_df)} test queries...")
        rankings = []
        positives = []
        
        for record in test_df.to_dict(orient="records"):
            cites = set([c.strip() for c in str(record.get("cited_patent_ids", "")).split(",") if c.strip()])
            if not cites:
                continue
                
            retrieved = self.retrieve(record, top_k=max(ks))
            retrieved_ids = [r["patent_id"] for r in retrieved]
            
            rankings.append(retrieved_ids)
            positives.append(cites)
            
        metrics = compute_retrieval_metrics(rankings, positives, ks=ks)
        logger.info(f"BM25 Retrieval Recall@10: {metrics.get('recall@10', 0.0):.4f}, Recall@50: {metrics.get('recall@50', 0.0):.4f}, MAP@10: {metrics.get('map@10', 0.0):.4f}")
        return metrics
