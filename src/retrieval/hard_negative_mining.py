import random
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Set
from src.utils.logging import setup_logger

logger = setup_logger("HardNegativeMining")

class HardNegativeMiner:
    """
    Mines hard negatives for citation contrastive learning.
    Hard negatives must:
    - Share the same primary CPC subclass
    - Have grant_date <= query grant_date
    - NOT be in the cited positive set
    - Be semantically distinct yet challenging
    """
    def __init__(self, corpus_df: pd.DataFrame):
        self.corpus_df = corpus_df.reset_index(drop=True)
        self.cpc_to_indices = {}
        
        for idx, row in self.corpus_df.iterrows():
            cpc = str(row["primary_cpc"])
            if cpc not in self.cpc_to_indices:
                self.cpc_to_indices[cpc] = []
            self.cpc_to_indices[cpc].append(idx)
            
        self.id_to_idx = {row["patent_id"]: idx for idx, row in self.corpus_df.iterrows()}
        self.dates = pd.to_datetime(self.corpus_df["grant_date"]).values

    def mine_hard_negatives(
        self,
        query_id: str,
        n_negatives: int = 3
    ) -> List[str]:
        if query_id not in self.id_to_idx:
            return []
            
        q_idx = self.id_to_idx[query_id]
        q_row = self.corpus_df.iloc[q_idx]
        q_cpc = str(q_row["primary_cpc"])
        q_date = pd.to_datetime(q_row["grant_date"])
        pos_cites = set([c.strip() for c in str(q_row.get("cited_patent_ids", "")).split(",") if c.strip()])
        
        same_cpc_cand_indices = self.cpc_to_indices.get(q_cpc, [])
        valid_negs = []
        
        for idx in same_cpc_cand_indices:
            cand_id = self.corpus_df.iloc[idx]["patent_id"]
            cand_date = self.dates[idx]
            
            if cand_id != query_id and cand_id not in pos_cites and cand_date <= q_date.to_datetime64():
                valid_negs.append(cand_id)
                
        if len(valid_negs) >= n_negatives:
            return random.sample(valid_negs, k=n_negatives)
        elif valid_negs:
            return valid_negs
            
        # Fallback to general earlier patents if same CPC has insufficient count
        earlier_indices = [idx for idx in range(len(self.corpus_df)) if self.dates[idx] <= q_date.to_datetime64() and self.corpus_df.iloc[idx]["patent_id"] not in pos_cites and idx != q_idx]
        if len(earlier_indices) >= n_negatives:
            return [self.corpus_df.iloc[i]["patent_id"] for i in random.sample(earlier_indices, k=n_negatives)]
        return [self.corpus_df.iloc[i]["patent_id"] for i in earlier_indices]
