import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss for Multi-Label Patent CPC Subclass Imbalance.
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_factor * ((1 - p_t) ** self.gamma)
        
        loss = focal_weight * bce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
