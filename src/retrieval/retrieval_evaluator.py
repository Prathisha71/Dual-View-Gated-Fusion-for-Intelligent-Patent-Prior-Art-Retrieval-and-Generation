import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.utils.metrics import compute_retrieval_metrics
from src.utils.logging import setup_logger

logger = setup_logger("RetrievalEvaluator")

def export_table3_retrieval_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    output_tables_dir: str = "reports/tables"
) -> pd.DataFrame:
    """
    Exports Table 3 & Table 48: Retrieval Method Benchmarking across TF-IDF, BM25, Generic BGE, PAFIR, PAFIR+CPC, PAFIR+CPC+Reranker.
    """
    os.makedirs(output_tables_dir, exist_ok=True)
    rows = []
    
    for method_name, m in metrics_dict.items():
        rows.append({
            "Retrieval Method / Strategy": method_name,
            "Precision@10": f"{m.get('precision@10', 0.0):.4f}",
            "Recall@10": f"{m.get('recall@10', 0.0):.4f}",
            "Recall@50": f"{m.get('recall@50', 0.0):.4f}",
            "MRR": f"{m.get('mrr', 0.0):.4f}",
            "MAP@10": f"{m.get('map@10', 0.0):.4f}",
            "nDCG@10": f"{m.get('ndcg@10', 0.0):.4f}",
            "Latency (ms)": f"{m.get('latency_ms', 0.0):.2f}"
        })
        
    df = pd.DataFrame(rows)
    output_path = os.path.join(output_tables_dir, "table3_retrieval_comparison.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Table 3 exported cleanly to {output_path}")
    return df
