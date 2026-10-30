import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
from src.utils.device import get_device
from src.utils.logging import setup_logger

logger = setup_logger("MultiLabelClassifier")

class CPCMultiLabelClassifier(nn.Module):
    """
    CPC Multi-Label Neural Classifier Head.
    Architecture: Fused Embedding -> Dense Layer (BatchNorm/ReLU) -> Dropout -> Sigmoid Output
    """
    def __init__(
        self,
        input_dim: int = 384,
        num_classes: int = 15,
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw unnormalized logits for BCE/Focal loss calculation."""
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probabilities for multi-label classification."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def save(self, output_path: str = "models/classification/cpc_classifier.pt"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "num_classes": self.num_classes
        }, output_path)
        logger.info(f"Saved CPC multi-label classifier head to {output_path}")

    @classmethod
    def load(cls, input_path: str = "models/classification/cpc_classifier.pt") -> "CPCMultiLabelClassifier":
        checkpoint = torch.load(input_path, map_location="cpu")
        model = cls(input_dim=checkpoint["input_dim"], num_classes=checkpoint["num_classes"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        logger.info(f"Loaded CPC multi-label classifier head from {input_path}")
        return model
