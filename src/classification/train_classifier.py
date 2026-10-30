import os
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import MultiLabelBinarizer

from src.classification.multilabel_classifier import CPCMultiLabelClassifier
from src.classification.losses import ClassBalancedFocalLoss
from src.features.embedding_generator import PatentEmbeddingGenerator
from src.utils.metrics import compute_classification_metrics
from src.utils.device import get_device
from src.utils.logging import setup_logger

logger = setup_logger("TrainClassifier")

DEFAULT_CPC_CLASSES = [
    "A61B", "A61K", "B29C", "B60L", "B65G",
    "C07C", "C07D", "C08G", "G01N", "G06F",
    "G06N", "G06Q", "H01L", "H02J", "H04L"
]

def train_cpc_classifier(
    embeddings_train: np.ndarray,
    cpc_series_train: pd.Series,
    embeddings_val: np.ndarray,
    cpc_series_val: pd.Series,
    cpc_classes: List[str],
    loss_type: str = "focal",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 0.001,
    output_dir: str = "models/classification"
) -> Tuple[CPCMultiLabelClassifier, List[str], Dict[str, float]]:
    """
    Trains the CPC Multi-Label Neural Classifier Head on PAFIR fused embeddings.
    Exports atomic cpc_classifier.pt, cpc_mlb.pkl, and cpc_threshold.json in a single run.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    
    # 1. Fit binarizer strictly on cpc_classes
    mlb = MultiLabelBinarizer(classes=cpc_classes)
    mlb.fit([[c] for c in cpc_classes])
    
    cpc_set = set(cpc_classes)
    y_train_list = [[sub.strip()[:4] for sub in str(x).split(",") if sub.strip()[:4] in cpc_set] for x in cpc_series_train]
    y_val_list = [[sub.strip()[:4] for sub in str(x).split(",") if sub.strip()[:4] in cpc_set] for x in cpc_series_val]
    
    Y_train = mlb.transform(y_train_list).astype(np.float32)
    Y_val = mlb.transform(y_val_list).astype(np.float32)
    
    num_classes = len(cpc_classes)
    input_dim = embeddings_train.shape[1]
    
    model = CPCMultiLabelClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    
    if loss_type == "focal":
        criterion = ClassBalancedFocalLoss(alpha=0.25, gamma=2.0)
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_dataset = TensorDataset(torch.tensor(embeddings_train, dtype=torch.float32), torch.tensor(Y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_val_f1 = 0.0
    best_threshold = 0.40
    best_val_metrics = {}
    
    logger.info(f"Training CPC Classifier ({loss_type} loss) for {epochs} epochs on device: {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x_b)
            
        scheduler.step()
        avg_loss = total_loss / len(embeddings_train)
        
        # Validation Evaluation & Threshold Tuning
        model.eval()
        with torch.no_grad():
            val_x = torch.tensor(embeddings_val, dtype=torch.float32).to(device)
            val_probs = model.predict_proba(val_x).cpu().numpy()
            
            # Tune threshold on validation set
            for thresh in np.arange(0.20, 0.60, 0.05):
                metrics = compute_classification_metrics(Y_val, val_probs >= thresh)
                if metrics["micro_f1"] > best_val_f1:
                    best_val_f1 = metrics["micro_f1"]
                    best_threshold = float(thresh)
                    best_val_metrics = metrics
                    
                    # Save best atomic checkpoint
                    model.save(os.path.join(output_dir, "cpc_classifier.pt"))
                    
                    with open(os.path.join(output_dir, "cpc_mlb.pkl"), "wb") as f:
                        pickle.dump(cpc_classes, f)
                        
                    with open(os.path.join(output_dir, "cpc_threshold.json"), "w", encoding="utf-8") as f:
                        json.dump({
                            "optimal_threshold": best_threshold,
                            "validation_micro_f1": best_val_f1,
                            "metrics": metrics
                        }, f, indent=2)
                    
        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f} | Val Micro F1: {best_val_f1:.4f} (best tau: {best_threshold:.2f})")
            
    logger.info(f"Training completed. Best Val Micro F1: {best_val_f1:.4f} at tau = {best_threshold:.2f}")
    return model, cpc_classes, best_val_metrics


def main():
    logger.info("Starting Scientific CPC Classifier Training Pipeline...")
    
    train_path = "data/splits/train.parquet"
    val_path = "data/splits/val.parquet"
    
    if not os.path.exists(train_path):
        train_path = "data/processed/master_patents.parquet"
        val_path = "data/processed/master_patents.parquet"
        
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    
    generator = PatentEmbeddingGenerator()
    
    logger.info("Encoding training set context & claims...")
    ctx_train = generator.encode_texts(df_train["combined_text_transformer"].tolist())
    clm_train = generator.encode_texts(df_train["independent_claims"].tolist())
    embs_train = 0.6 * ctx_train + 0.4 * clm_train
    norms_tr = np.linalg.norm(embs_train, axis=1, keepdims=True) + 1e-10
    embs_train = (embs_train / norms_tr).astype(np.float32)
    
    logger.info("Encoding validation set context & claims...")
    ctx_val = generator.encode_texts(df_val["combined_text_transformer"].tolist())
    clm_val = generator.encode_texts(df_val["independent_claims"].tolist())
    embs_val = 0.6 * ctx_val + 0.4 * clm_val
    norms_va = np.linalg.norm(embs_val, axis=1, keepdims=True) + 1e-10
    embs_val = (embs_val / norms_va).astype(np.float32)
    
    cpc_series_tr = df_train["cpc_subclasses"]
    cpc_series_va = df_val["cpc_subclasses"]
    
    train_cpc_classifier(
        embeddings_train=embs_train,
        cpc_series_train=cpc_series_tr,
        embeddings_val=embs_val,
        cpc_series_val=cpc_series_va,
        cpc_classes=DEFAULT_CPC_CLASSES,
        loss_type="focal",
        epochs=25,
        batch_size=32,
        lr=0.001
    )

if __name__ == "__main__":
    main()
