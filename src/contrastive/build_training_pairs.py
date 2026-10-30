import os
import random
import pandas as pd
from typing import List, Dict, Any, Tuple
from src.utils.logging import setup_logger

logger = setup_logger("BuildTrainingPairs")

def build_citation_pairs(
    df: pd.DataFrame,
    output_dir: str = "data/splits"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds positive citation pairs and query sets for training, validation, and testing.
    Ensures zero temporal leakage.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Extracting citation pairs from dataset of {len(df)} patents...")
    
    id_to_row = df.set_index("patent_id").to_dict(orient="index")
    pairs = []
    
    for _, row in df.iterrows():
        query_id = row["patent_id"]
        cites = [c.strip() for c in str(row.get("cited_patent_ids", "")).split(",") if c.strip()]
        q_date = pd.to_datetime(row["grant_date"])
        q_cpc = str(row["primary_cpc"])
        split = row.get("split", "train")
        
        for pos_id in cites:
            if pos_id in id_to_row:
                pos_row = id_to_row[pos_id]
                pos_date = pd.to_datetime(pos_row["grant_date"])
                
                # Assert temporal safety: pos_date <= q_date
                if pos_date <= q_date:
                    pairs.append({
                        "query_id": query_id,
                        "positive_id": pos_id,
                        "query_date": str(row["grant_date"]),
                        "positive_date": str(pos_row["grant_date"]),
                        "primary_cpc": q_cpc,
                        "split": split
                    })
                    
    pairs_df = pd.DataFrame(pairs)
    train_pairs = pairs_df[pairs_df["split"] == "train"].reset_index(drop=True)
    val_pairs = pairs_df[pairs_df["split"] == "val"].reset_index(drop=True)
    
    train_pairs.to_parquet(os.path.join(output_dir, "train_positive_pairs.parquet"), index=False)
    val_pairs.to_parquet(os.path.join(output_dir, "validation_pairs.parquet"), index=False)
    
    logger.info(f"Citation pairs constructed. Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    return train_pairs, val_pairs

if __name__ == "__main__":
    train_df = pd.read_parquet("data/splits/train.parquet")
    val_df = pd.read_parquet("data/splits/val.parquet")
    all_df = pd.concat([train_df, val_df])
    build_citation_pairs(all_df)
