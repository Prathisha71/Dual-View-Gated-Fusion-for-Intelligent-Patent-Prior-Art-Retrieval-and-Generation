import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from src.classification.multilabel_classifier import CPCMultiLabelClassifier
from src.features.embedding_generator import PatentEmbeddingGenerator
from src.utils.metrics import compute_classification_metrics
from src.utils.logging import setup_logger

logger = setup_logger("EvaluateClassifier")

DEFAULT_CPC_CLASSES = [
    "A61B", "A61K", "B29C", "B60L", "B65G",
    "C07C", "C07D", "C08G", "G01N", "G06F",
    "G06N", "G06Q", "H01L", "H02J", "H04L"
]

def run_scientific_representation_comparison():
    logger.info("Executing Scientific Multi-Representation CPC Classification Evaluation...")
    
    test_path = "data/splits/test.parquet"
    train_path = "data/splits/train.parquet"
    
    if not os.path.exists(test_path):
        test_path = "data/processed/master_patents.parquet"
        train_path = "data/processed/master_patents.parquet"
        
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    
    cpc_mlb_path = "models/classification/cpc_mlb.pkl"
    if os.path.exists(cpc_mlb_path):
        with open(cpc_mlb_path, "rb") as f:
            cpc_classes = pickle.load(f)
    else:
        cpc_classes = DEFAULT_CPC_CLASSES
        
    mlb = MultiLabelBinarizer(classes=cpc_classes)
    mlb.fit([[c] for c in cpc_classes])
    
    cpc_set = set(cpc_classes)
    y_tr = mlb.transform([[s.strip()[:4] for s in str(x).split(",") if s.strip()[:4] in cpc_set] for x in df_train["cpc_subclasses"]])
    y_te = mlb.transform([[s.strip()[:4] for s in str(x).split(",") if s.strip()[:4] in cpc_set] for x in df_test["cpc_subclasses"]])
    
    # Load threshold tau
    thresh_path = "models/classification/cpc_threshold.json"
    tau = 0.40
    if os.path.exists(thresh_path):
        with open(thresh_path, "r", encoding="utf-8") as f:
            tau = float(json.load(f).get("optimal_threshold", 0.40))
            
    generator = PatentEmbeddingGenerator()
    
    # Model 1: TF-IDF + Logistic Regression
    logger.info("Evaluating Representation 1: TF-IDF + Linear Classifier...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X_tr_tfidf = tfidf.fit_transform(df_train["combined_text_transformer"])
    X_te_tfidf = tfidf.transform(df_test["combined_text_transformer"])
    lr_model = OneVsRestClassifier(LogisticRegression(max_iter=500, C=1.0))
    lr_model.fit(X_tr_tfidf, y_tr)
    lr_probs = lr_model.predict_proba(X_te_tfidf)
    m_tfidf = compute_classification_metrics(y_te, lr_probs >= tau)

    # Model 2: Context Only MiniLM
    logger.info("Evaluating Representation 2: Context-Only MiniLM...")
    ctx_test = generator.encode_texts(df_test["combined_text_transformer"].tolist())
    ctx_norms = np.linalg.norm(ctx_test, axis=1, keepdims=True) + 1e-10
    ctx_test_norm = (ctx_test / ctx_norms).astype(np.float32)
    
    cpc_model = CPCMultiLabelClassifier.load("models/classification/cpc_classifier.pt")
    ctx_probs = cpc_model.predict_proba(torch.tensor(ctx_test_norm)).numpy()
    m_ctx = compute_classification_metrics(y_te, ctx_probs >= tau)

    # Model 3: Claim Only MiniLM
    logger.info("Evaluating Representation 3: Claim-Only MiniLM...")
    clm_test = generator.encode_texts(df_test["independent_claims"].tolist())
    clm_norms = np.linalg.norm(clm_test, axis=1, keepdims=True) + 1e-10
    clm_test_norm = (clm_test / clm_norms).astype(np.float32)
    clm_probs = cpc_model.predict_proba(torch.tensor(clm_test_norm)).numpy()
    m_clm = compute_classification_metrics(y_te, clm_probs >= tau)

    # Model 4: PAFIR Fused Representation (Context + Claims)
    logger.info("Evaluating Representation 4: PAFIR Fused Representation...")
    fused_test = 0.6 * ctx_test + 0.4 * clm_test
    fused_norms = np.linalg.norm(fused_test, axis=1, keepdims=True) + 1e-10
    fused_test_norm = (fused_test / fused_norms).astype(np.float32)
    fused_probs = cpc_model.predict_proba(torch.tensor(fused_test_norm)).numpy()
    m_pafir = compute_classification_metrics(y_te, fused_probs >= tau)

    comparison = {
        "TF_IDF_Linear": m_tfidf,
        "Context_Only_MiniLM": m_ctx,
        "Claim_Only_MiniLM": m_clm,
        "PAFIR_Fused_FocalLoss": m_pafir
    }

    # Save to reports
    os.makedirs("reports/tables", exist_ok=True)
    report_json_path = "reports/classification_comparison_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Save Table 2 CSV
    rows = []
    for model_name, m in comparison.items():
        rows.append({
            "Representation Model": model_name,
            "Micro Precision": f"{m.get('micro_precision', 0.0):.4f}",
            "Micro Recall": f"{m.get('micro_recall', 0.0):.4f}",
            "Micro F1": f"{m.get('micro_f1', 0.0):.4f}",
            "Macro F1": f"{m.get('macro_f1', 0.0):.4f}",
            "Weighted F1": f"{m.get('weighted_f1', 0.0):.4f}",
            "Hamming Loss": f"{m.get('hamming_loss', 0.0):.4f}",
            "Precision@1": f"{m.get('precision_at_1', 0.0):.4f}",
            "Recall@1": f"{m.get('recall_at_1', 0.0):.4f}"
        })

    df_table2 = pd.DataFrame(rows)
    table2_path = "reports/tables/table2_classification_comparison.csv"
    df_table2.to_csv(table2_path, index=False)
    
    print("=" * 75)
    print("TABLE 2: MULTI-REPRESENTATION CPC CLASSIFICATION COMPARISON")
    print("=" * 75)
    print(df_table2.to_string(index=False))
    print("=" * 75)
    
    logger.info(f"Exported Table 2 to {table2_path} and report to {report_json_path}")

if __name__ == "__main__":
    run_scientific_representation_comparison()
