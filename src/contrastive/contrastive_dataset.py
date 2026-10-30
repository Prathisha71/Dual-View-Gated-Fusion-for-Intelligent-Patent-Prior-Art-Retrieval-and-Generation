import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

class ContrastivePatentDataset(Dataset):
    """
    Dataset loader for citation contrastive learning (InfoNCE / MultipleNegativesRankingLoss).
    Each item returns (query_fused_vec, positive_fused_vec, hard_negative_fused_vecs).
    """
    def __init__(
        self,
        pairs_df: pd.DataFrame,
        id_to_embeddings: Dict[str, np.ndarray],
        id_to_negatives: Dict[str, List[str]] = None
    ):
        self.pairs = pairs_df.to_dict(orient="records")
        self.id_to_embeddings = id_to_embeddings
        self.id_to_negatives = id_to_negatives or {}
        
        # Filter valid pairs present in embedding map
        self.valid_pairs = [
            p for p in self.pairs 
            if p["query_id"] in id_to_embeddings and p["positive_id"] in id_to_embeddings
        ]

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = self.valid_pairs[idx]
        q_id = pair["query_id"]
        p_id = pair["positive_id"]
        
        q_vec = torch.tensor(self.id_to_embeddings[q_id], dtype=torch.float32)
        p_vec = torch.tensor(self.id_to_embeddings[p_id], dtype=torch.float32)
        
        neg_ids = self.id_to_negatives.get(q_id, [])
        if neg_ids and neg_ids[0] in self.id_to_embeddings:
            n_vec = torch.tensor(self.id_to_embeddings[neg_ids[0]], dtype=torch.float32)
        else:
            # Fallback zero or random noise if no hard negative
            n_vec = torch.tensor(np.random.randn(q_vec.shape[0]), dtype=torch.float32)
            
        return q_vec, p_vec, n_vec
