import pandas as pd
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def merge_patent_tables(
    patents_df: pd.DataFrame,
    claims_df: pd.DataFrame,
    cpc_df: pd.DataFrame,
    citations_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Consolidates raw PatentsView relational tables into a single master Parquet dataset.
    Schema:
    patent_id, title, abstract, independent_claims, all_claims, primary_cpc,
    cpc_subclasses, grant_date, cited_patent_ids, combined_text
    """
    logger.info("Merging patent metadata, claims, CPC classifications, and citation tables...")
    
    # Ensure patent_id strings
    for df in [patents_df, claims_df, cpc_df, citations_df]:
        if "patent_id" in df.columns:
            df["patent_id"] = df["patent_id"].astype(str)
            
    # Merge claims
    merged = pd.merge(patents_df, claims_df, on="patent_id", how="left")
    
    # Merge CPC
    merged = pd.merge(merged, cpc_df, on="patent_id", how="left")
    
    # Merge Citations
    merged = pd.merge(merged, citations_df, on="patent_id", how="left")
    
    logger.info(f"Merged dataset shape: {merged.shape}")
    return merged
