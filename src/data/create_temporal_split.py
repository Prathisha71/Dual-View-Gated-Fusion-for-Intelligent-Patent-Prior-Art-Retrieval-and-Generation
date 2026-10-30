import os
import yaml
import pandas as pd
from typing import Tuple, Dict, Any
from src.utils.logging import setup_logger

logger = setup_logger("TemporalSplitter")

def create_temporal_split(
    df: pd.DataFrame,
    output_dir: str = "data/splits",
    config_path: str = "config/data_config.yaml"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates strict temporal split partitions:
    - Train: 2010 to 2022
    - Validation: 2023 to 2024
    - Test: 2025
    Verifies zero overlap between patent IDs.
    """
    logger.info("Creating temporal train/validation/test splits...")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    train_years = cfg["temporal_split"]["train_years"]
    val_years = cfg["temporal_split"]["val_years"]
    test_years = cfg["temporal_split"]["test_years"]
    
    df["grant_date_dt"] = pd.to_datetime(df["grant_date"])
    df["grant_year"] = df["grant_date_dt"].dt.year
    
    train_df = df[(df["grant_year"] >= train_years[0]) & (df["grant_year"] <= train_years[1])].copy()
    val_df = df[(df["grant_year"] >= val_years[0]) & (df["grant_year"] <= val_years[1])].copy()
    test_df = df[(df["grant_year"] >= test_years[0]) & (df["grant_year"] <= test_years[1])].copy()
    
    # Assign split column
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    # Verify disjoint IDs
    train_ids = set(train_df["patent_id"])
    val_ids = set(val_df["patent_id"])
    test_ids = set(test_df["patent_id"])
    
    intersection_tv = train_ids.intersection(val_ids)
    intersection_tt = train_ids.intersection(test_ids)
    intersection_vt = val_ids.intersection(test_ids)
    
    if intersection_tv or intersection_tt or intersection_vt:
        raise ValueError(
            f"TEMPORAL LEAKAGE DETECTED! ID Overlaps: Train-Val={len(intersection_tv)}, "
            f"Train-Test={len(intersection_tt)}, Val-Test={len(intersection_vt)}"
        )
        
    logger.info(f"Splits successfully created cleanly:")
    logger.info(f"  Train ({train_years[0]}-{train_years[1]}): {len(train_df)} patents")
    logger.info(f"  Val   ({val_years[0]}-{val_years[1]}): {len(val_df)} patents")
    logger.info(f"  Test  ({test_years[0]}-{test_years[1]}): {len(test_df)} patents")
    
    # Save splits
    train_df.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(output_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    master_path = "data/processed/master_patents.parquet"
    if os.path.exists(master_path):
        df = pd.read_parquet(master_path)
        create_temporal_split(df)
    else:
        from src.data.build_dataset import build_and_save_master_dataset
        df = build_and_save_master_dataset()
        create_temporal_split(df)
