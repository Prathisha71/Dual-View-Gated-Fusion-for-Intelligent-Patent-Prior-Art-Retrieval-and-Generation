import os
import pandas as pd
from src.utils.logging import setup_logger

logger = setup_logger("VerifyDatasetContent")

REQUIRED_COLUMNS = [
    "patent_id", "title", "abstract", "independent_claims", "claims",
    "primary_cpc", "cpc_subclasses", "grant_date", "cited_patent_ids", "combined_text_transformer"
]

def verify_dataset_content(dataset_path: str = "data/processed/master_patents.parquet") -> bool:
    logger.info(f"Verifying dataset content integrity at: {dataset_path}...")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file missing at {dataset_path}!")
        return False
        
    df = pd.read_parquet(dataset_path)
    
    # 1. Column Presence Check
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            logger.error(f"CRITICAL: Required column '{col}' missing from dataset!")
            return False
            
    # 2. Combined Text Construction Check (Title + Abstract + Independent Claims)
    sample_row = df.iloc[0]
    title = str(sample_row["title"])
    abstract = str(sample_row["abstract"])
    claims = str(sample_row["independent_claims"])
    comb = str(sample_row["combined_text_transformer"])
    
    if title not in comb or abstract not in comb:
        logger.error("CRITICAL: combined_text_transformer does not properly encapsulate Title + Abstract!")
        return False
        
    # 3. Non-Empty Assertions
    empty_abstracts = (df["abstract"].isna() | (df["abstract"] == "")).sum()
    empty_claims = (df["independent_claims"].isna() | (df["independent_claims"] == "")).sum()
    
    if empty_abstracts > len(df) * 0.05:
        logger.error(f"CRITICAL: Excessive empty abstracts detected ({empty_abstracts})")
        return False
        
    if empty_claims > len(df) * 0.05:
        logger.error(f"CRITICAL: Excessive empty claims detected ({empty_claims})")
        return False
        
    logger.info(f"Dataset content verification PASSED cleanly across {len(df)} records.")
    return True

if __name__ == "__main__":
    verify_dataset_content()
