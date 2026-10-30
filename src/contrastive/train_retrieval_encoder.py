import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from src.contrastive.contrastive_dataset import ContrastivePatentDataset
from src.retrieval.pafir_fusion import PAFIRFusionModule
from src.utils.device import get_device
from src.utils.logging import setup_logger

logger = setup_logger("TrainRetrievalEncoder")

class InfoNCELoss(nn.Module):
    """InfoNCE Loss for citation-based contrastive representation learning."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, q_vecs: torch.Tensor, p_vecs: torch.Tensor, n_vecs: torch.Tensor) -> torch.Tensor:
        # Cosine similarities
        q_norm = F.normalize(q_vecs, dim=-1)
        p_norm = F.normalize(p_vecs, dim=-1)
        n_norm = F.normalize(n_vecs, dim=-1)
        
        pos_sim = torch.sum(q_norm * p_norm, dim=-1) / self.temperature
        neg_sim = torch.sum(q_norm * n_norm, dim=-1) / self.temperature
        
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(len(q_vecs), dtype=torch.long, device=q_vecs.device)
        
        return F.cross_entropy(logits, labels)


def fine_tune_pafir_encoder(
    pafir_fusion: PAFIRFusionModule,
    train_pairs_df: pd.DataFrame,
    id_to_embeddings: Dict[str, np.ndarray],
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    output_dir: str = "models/retrieval"
) -> PAFIRFusionModule:
    """Fine-tunes the PAFIR Learnable Fusion Module using InfoNCE Citation Contrastive Learning."""
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    
    pafir_fusion = pafir_fusion.to(device)
    dataset = ContrastivePatentDataset(train_pairs_df, id_to_embeddings)
    if len(dataset) == 0:
        logger.warning("No valid contrastive pairs found for fine-tuning. Skipping fine-tuning pass.")
        return pafir_fusion
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = InfoNCELoss(temperature=0.07)
    optimizer = optim.AdamW(pafir_fusion.parameters(), lr=lr)
    
    logger.info(f"Fine-tuning PAFIR encoder on {len(dataset)} positive citation pairs...")
    for epoch in range(1, epochs + 1):
        pafir_fusion.train()
        total_loss = 0.0
        for q_b, p_b, n_b in loader:
            q_b, p_b, n_b = q_b.to(device), p_b.to(device), n_b.to(device)
            optimizer.zero_grad()
            loss = criterion(q_b, p_b, n_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(q_b)
            
        avg_loss = total_loss / len(dataset)
        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Contrastive InfoNCE Loss: {avg_loss:.4f}")
        
    pafir_fusion.save(os.path.join(output_dir, "pafir_fusion_fine_tuned.pt"))
    return pafir_fusion
