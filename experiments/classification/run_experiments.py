import os
import json
import datetime
import pandas as pd
import numpy as np
import torch

from src.classification.baseline_classifier import TfidfBaselineClassifier
from src.classification.multilabel_classifier import CPCMultiLabelClassifier
from src.classification.train_classifier import train_cpc_classifier
from src.classification.evaluate_classifier import export_table2_classification_comparison, evaluate_frequency_tiers
from src.features.embedding_generator import PatentEmbeddingGenerator
from src.retrieval.pafir_fusion import PAFIRFusionModule
from src.utils.metrics import compute_classification_metrics
from src.utils.seed import set_seed
from src.utils.logging import setup_logger

logger = setup_logger("ClassificationExperiments")

def run_all_classification_experiments():
    set_seed(42)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    exp_dir = f"experiments/classification/exp_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    logger.info(f"Running empirical classification experiment suite in {exp_dir}...")
    train_df = pd.read_parquet("data/splits/train.parquet")
    val_df = pd.read_parquet("data/splits/val.parquet")
    test_df = pd.read_parquet("data/splits/test.parquet")
    
    cpc_classes = ["H04L", "G06F", "A61K", "C07D", "H01L", "G06Q", "G06N", "B60L", "E04B", "F02M", "G01N", "H02J", "A61B", "C08G", "H04W"]
    
    # 1. Baseline 1: TF-IDF + Linear Classifier
    logger.info("Evaluating Baseline 1: TF-IDF + Linear Classifier...")
    clf_tfidf = TfidfBaselineClassifier(cpc_classes=cpc_classes)
    m_tfidf = clf_tfidf.train(train_df, val_df)
    m_tfidf_test = clf_tfidf.evaluate(test_df)
    
    # 2. Dual-View Embeddings & Fusion Generation
    generator = PatentEmbeddingGenerator()
    dim = generator.embedding_dim
    
    ctx_train, clm_train = generator.generate_dual_view_embeddings(train_df, output_dir="data/embeddings/train")
    ctx_val, clm_val = generator.generate_dual_view_embeddings(val_df, output_dir="data/embeddings/val")
    ctx_test, clm_test = generator.generate_dual_view_embeddings(test_df, output_dir="data/embeddings/test")
    
    fusion_module = PAFIRFusionModule(embedding_dim=dim, fusion_type="gated")
    fused_train = fusion_module.fuse_numpy(ctx_train, clm_train)
    fused_val = fusion_module.fuse_numpy(ctx_val, clm_val)
    fused_test = fusion_module.fuse_numpy(ctx_test, clm_test)
    
    # Save corpus fused embeddings for server startup
    np.save("data/embeddings/pafir_fused_corpus.npy", np.vstack([fused_train, fused_val, fused_test]))
    
    # 3. Model Variant Experiments
    # Variant A: Context View Only
    logger.info("Training Head on Context View Only...")
    _, _, m_ctx = train_cpc_classifier(ctx_train, train_df["cpc_subclasses"], ctx_val, val_df["cpc_subclasses"], cpc_classes, loss_type="focal", epochs=5)
    
    # Variant B: Claim View Only
    logger.info("Training Head on Claim View Only...")
    _, _, m_clm = train_cpc_classifier(clm_train, train_df["cpc_subclasses"], clm_val, val_df["cpc_subclasses"], cpc_classes, loss_type="focal", epochs=5)
    
    # Variant C: PAFIR Gated Fusion (BCE Loss)
    logger.info("Training Head on PAFIR Gated Fusion (BCE Loss)...")
    _, _, m_pafir_bce = train_cpc_classifier(fused_train, train_df["cpc_subclasses"], fused_val, val_df["cpc_subclasses"], cpc_classes, loss_type="bce", epochs=5)
    
    # Variant D: PAFIR Gated Fusion (Class-Balanced Focal Loss Proposed)
    logger.info("Training Head on PAFIR Gated Fusion (Focal Loss Proposed)...")
    model, mlb, m_pafir_focal = train_cpc_classifier(fused_train, train_df["cpc_subclasses"], fused_val, val_df["cpc_subclasses"], cpc_classes, loss_type="focal", epochs=8)
    
    # Evaluate Proposed Model on Untouched Test Set
    test_probs = model.predict_proba(torch.tensor(fused_test, dtype=torch.float32)).numpy()
    best_thresh = m_pafir_focal.get("best_threshold", 0.35)
    test_preds = (test_probs >= best_thresh).astype(int)
    y_test_list = [[c.strip() for c in str(x).split(",") if c.strip()] for x in test_df["cpc_subclasses"]]
    Y_test = mlb.transform(y_test_list)
    
    m_pafir_test = compute_classification_metrics(Y_test, test_preds)
    m_pafir_test["best_threshold"] = best_thresh
    
    # Frequency Tier Analysis
    cpc_counts = train_df["primary_cpc"].value_counts().to_dict()
    tier_metrics = evaluate_frequency_tiers(Y_test, test_preds, cpc_classes, cpc_counts)
    m_pafir_test["frequency_tiers"] = tier_metrics
    
    all_metrics = {
        "TF-IDF + Linear Classifier": m_tfidf_test,
        "Context View Only": m_ctx,
        "Claim View Only": m_clm,
        "PAFIR Gated Fusion (BCE Loss)": m_pafir_bce,
        "PAFIR Gated Fusion (Focal Loss Proposed)": m_pafir_test
    }
    
    # Export Table 2
    export_table2_classification_comparison(all_metrics)
    
    # Save experiment artifacts
    with open(f"{exp_dir}/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
        
    config = {
        "timestamp": timestamp,
        "dataset_split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "cpc_classes": cpc_classes,
        "embedding_dim": dim,
        "best_threshold": best_thresh
    }
    with open(f"{exp_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    pd.DataFrame(test_preds, columns=cpc_classes).to_parquet(f"{exp_dir}/test_predictions.parquet", index=False)
    model.save("models/classification/cpc_classifier.pt")
    
    logger.info(f"Classification experiment completed. Verified Test Micro F1: {m_pafir_test['micro_f1']:.4f}")
    return all_metrics

if __name__ == "__main__":
    run_all_classification_experiments()
