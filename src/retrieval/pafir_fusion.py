import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from typing import Tuple, Dict, Any, Union
from src.utils.logging import setup_logger

logger = setup_logger("PAFIRFusion")

class PAFIRFusionModule(nn.Module):
    """
    PAFIR: Patent Aware Fusion for Intelligent Retrieval.
    Combines Context View (Title + Abstract) and Claim View (Independent Claims) via
    Learnable Gated Fusion, Weighted Fusion, or Projection.
    """
    def __init__(
        self,
        embedding_dim: int = 384,
        fusion_type: str = "gated",
        context_weight: float = 0.5
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fusion_type = fusion_type
        self.context_weight = context_weight
        
        if self.fusion_type == "gated":
            # Gating network takes concatenated [v_context; v_claim] -> scalar gate per feature dimension
            self.gate_fc = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.Sigmoid()
            )
        elif self.fusion_type == "projection":
            self.proj_fc = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )

    def forward(
        self,
        context_vecs: torch.Tensor,
        claim_vecs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing fused patent representation and gate weights.
        Returns:
            fused_vecs: Normalized fused tensor (batch_size, embedding_dim)
            gate_weights: Gate tensor (batch_size, embedding_dim) or None
        """
        if self.fusion_type == "simple_avg":
            fused = 0.5 * (context_vecs + claim_vecs)
            gates = None
            
        elif self.fusion_type == "weighted_avg":
            w = self.context_weight
            fused = w * context_vecs + (1.0 - w) * claim_vecs
            gates = None
            
        elif self.fusion_type == "projection":
            concat = torch.cat([context_vecs, claim_vecs], dim=-1)
            fused = self.proj_fc(concat)
            gates = None
            
        elif self.fusion_type == "gated":
            concat = torch.cat([context_vecs, claim_vecs], dim=-1)
            gates = self.gate_fc(concat)
            fused = gates * context_vecs + (1.0 - gates) * claim_vecs
            
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
            
        # L2 Normalization
        fused_norm = F.normalize(fused, p=2, dim=-1)
        return fused_norm, gates

    def fuse_numpy(
        self,
        context_arr: np.ndarray,
        claim_arr: np.ndarray
    ) -> np.ndarray:
        """Helper to fuse NumPy array representations."""
        self.eval()
        with torch.no_grad():
            c_tensor = torch.tensor(context_arr, dtype=torch.float32)
            k_tensor = torch.tensor(claim_arr, dtype=torch.float32)
            fused, _ = self.forward(c_tensor, k_tensor)
            return fused.numpy()

    def save(self, output_path: str = "models/retrieval/pafir_fusion_module.pt"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(self.state_dict(), output_path)
        logger.info(f"Saved PAFIR fusion module to {output_path}")

    def load(self, input_path: str = "models/retrieval/pafir_fusion_module.pt"):
        self.load_state_dict(torch.load(input_path, map_location="cpu"))
        self.eval()
        logger.info(f"Loaded PAFIR fusion module from {input_path}")
