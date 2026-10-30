import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from src.retrieval.faiss_index import PatentFAISSIndex
from src.utils.metrics import compute_retrieval_metrics
from src.utils.logging import setup_logger

logger = setup_logger("CPCGuidedRetriever")

class CPCGuidedRetriever:
    """
    PAFIR Multi-Stage Retriever incorporating Soft CPC Score Boosting and Temporal Filtering.
    Score = S_semantic + w_cpc * S_cpc_compatibility + w_bm25 * S_lexical
    """
    def __init__(
        self,
        faiss_index: PatentFAISSIndex,
        corpus_df: pd.DataFrame,
        cpc_weight: float = 0.2,
        lexical_weight: float = 0.1
    ):
        self.faiss_index = faiss_index
        self.corpus_df = corpus_df.reset_index(drop=True)
        self.cpc_weight = cpc_weight
        self.lexical_weight = lexical_weight
        
        self.corpus_dates = pd.to_datetime(self.corpus_df["grant_date"]).values
        self.corpus_cpcs = [
            set([c.strip() for c in str(x).split(",") if c.strip()]) 
            for x in self.corpus_df["cpc_subclasses"]
        ]

    def _compute_cpc_compatibility(self, query_cpcs: set, cand_idx: int) -> float:
        cand_cpcs = self.corpus_cpcs[cand_idx]
        if not query_cpcs or not cand_cpcs:
            return 0.0
        intersection = query_cpcs.intersection(cand_cpcs)
        union = query_cpcs.union(cand_cpcs)
        return len(intersection) / len(union) if union else 0.0

    def retrieve(
        self,
        query_record: Dict[str, Any],
        query_vector: np.ndarray,
        predicted_cpcs: List[str] = None,
        top_k: int = 50,
        strategy: str = "soft_cpc"  # choices: no_cpc, hard_cpc, soft_cpc
    ) -> List[Dict[str, Any]]:
        """
        Retrieves prior art candidates with strict temporal filtering and optional CPC guidance.
        """
        query_id = query_record["patent_id"]
        query_date = pd.to_datetime(query_record["grant_date"])
        
        if predicted_cpcs:
            query_cpcs = set(predicted_cpcs)
        else:
            query_cpcs = set([c.strip() for c in str(query_record.get("cpc_subclasses", "")).split(",") if c.strip()])
            
        # Initial FAISS dense search for top candidates
        search_k = min(top_k * 4, len(self.corpus_df))
        scores, indices = self.faiss_index.search(query_vector.reshape(1, -1), k=search_k)
        
        raw_scores = scores[0]
        raw_indices = indices[0]
        
        results = []
        for score, idx in zip(raw_scores, raw_indices):
            cand_row = self.corpus_df.iloc[idx]
            cand_id = cand_row["patent_id"]
            cand_date = self.corpus_dates[idx]
            
            # Mandatory Temporal Filter: cand_date <= query_date AND cand_id != query_id
            if cand_id == query_id or cand_date > query_date.to_datetime64():
                continue
                
            cpc_compat = self._compute_cpc_compatibility(query_cpcs, idx)
            
            if strategy == "hard_cpc" and cpc_compat == 0.0:
                continue
                
            if strategy == "soft_cpc":
                final_score = float(score) + self.cpc_weight * cpc_compat
            else:
                final_score = float(score)
                
            cand_dict = cand_row.to_dict()
            cand_dict["semantic_similarity"] = float(score)
            cand_dict["cpc_compatibility"] = float(cpc_compat)
            cand_dict["final_score"] = final_score
            results.append(cand_dict)
            
        # Sort by final score
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]
