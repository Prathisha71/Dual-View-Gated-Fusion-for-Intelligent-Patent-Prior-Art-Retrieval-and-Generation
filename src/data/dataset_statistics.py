import os
import json
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import plotly.express as px
except ImportError:
    px = None

from typing import Dict, Any
from src.utils.logging import setup_logger

logger = setup_logger("DatasetStatistics")

def generate_dataset_statistics(
    df: pd.DataFrame,
    output_figures_dir: str = "reports/figures",
    output_tables_dir: str = "reports/tables"
) -> Dict[str, Any]:
    """
    Computes dataset quality metrics, generates distribution plots, and validates data sanity.
    """
    logger.info("Computing dataset quality statistics...")
    os.makedirs(output_figures_dir, exist_ok=True)
    os.makedirs(output_tables_dir, exist_ok=True)
    
    n_patents = len(df)
    dates = pd.to_datetime(df["grant_date"])
    date_min = str(dates.min().strftime("%Y-%m-%d"))
    date_max = str(dates.max().strftime("%Y-%m-%d"))
    
    # CPC subclasses statistics
    all_cpc_list = []
    for cpcs in df["cpc_subclasses"].dropna():
        all_cpc_list.extend([c.strip() for c in str(cpcs).split(",") if c.strip()])
        
    cpc_series = pd.Series(all_cpc_list)
    unique_cpcs = cpc_series.nunique()
    cpc_counts = cpc_series.value_counts()
    
    # Lengths
    abstract_lens = df["abstract"].str.len().fillna(0)
    claim_lens = df["independent_claims"].str.len().fillna(0)
    
    # Missing percentages
    missing_abstract_pct = (df["abstract"].isna() | (df["abstract"] == "")).mean() * 100.0
    missing_claim_pct = (df["independent_claims"].isna() | (df["independent_claims"] == "")).mean() * 100.0
    
    # Citation statistics
    citation_counts = df["cited_patent_ids"].apply(lambda x: len([c for c in str(x).split(",") if c.strip()]) if pd.notna(x) else 0)
    avg_citations = float(citation_counts.mean())
    max_citations = int(citation_counts.max())
    
    # Patents per year
    df["grant_year"] = dates.dt.year
    patents_per_year = df["grant_year"].value_counts().sort_index().to_dict()
    
    stats = {
        "n_patents": n_patents,
        "date_range": [date_min, date_max],
        "unique_cpc_subclasses": int(unique_cpcs),
        "avg_abstract_length": float(abstract_lens.mean()),
        "avg_independent_claim_length": float(claim_lens.mean()),
        "missing_abstract_percentage": float(missing_abstract_pct),
        "missing_claim_percentage": float(missing_claim_pct),
        "avg_citations_per_patent": avg_citations,
        "max_citations_per_patent": max_citations,
        "patents_per_year": patents_per_year,
        "top_cpc_subclasses": cpc_counts.head(10).to_dict()
    }
    
    # Quality assertion check
    if n_patents == 0 or missing_abstract_pct > 10.0 or missing_claim_pct > 10.0:
        raise ValueError(f"Dataset quality check FAILED: missing abstract={missing_abstract_pct:.2f}%, missing claims={missing_claim_pct:.2f}%")
        
    logger.info("Dataset quality assertion PASSED cleanly.")
    
    # Generate Plots if matplotlib is available
    if plt is not None:
        # 1. CPC Subclass Distribution (Bar plot)
        plt.figure(figsize=(10, 5))
        cpc_counts.head(15).plot(kind="bar", color="#2b5c8f")
        plt.title("Top 15 CPC Subclass Class Distribution")
        plt.xlabel("CPC Subclass")
        plt.ylabel("Number of Patents")
        plt.tight_layout()
        plt.savefig(os.path.join(output_figures_dir, "cpc_class_distribution.png"), dpi=300)
        plt.savefig(os.path.join(output_figures_dir, "cpc_class_distribution.svg"))
        plt.close()
        
        # 2. Patents Per Year (Line plot)
        plt.figure(figsize=(8, 4))
        years = list(patents_per_year.keys())
        counts = list(patents_per_year.values())
        plt.plot(years, counts, marker="o", color="#10b981", linewidth=2)
        plt.title("Patents Granted Per Year (2010-2025)")
        plt.xlabel("Year")
        plt.ylabel("Patent Count")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_figures_dir, "patents_per_year.png"), dpi=300)
        plt.savefig(os.path.join(output_figures_dir, "patents_per_year.svg"))
        plt.close()
    
    # Export Table 1: Dataset Statistics
    table1_df = pd.DataFrame([
        {"Metric": "Total Utility Patents", "Value": str(n_patents)},
        {"Metric": "Date Coverage", "Value": f"{date_min} to {date_max}"},
        {"Metric": "Unique CPC Subclasses", "Value": str(unique_cpcs)},
        {"Metric": "Mean Abstract Char Length", "Value": f"{abstract_lens.mean():.1f}"},
        {"Metric": "Mean Independent Claim Char Length", "Value": f"{claim_lens.mean():.1f}"},
        {"Metric": "Mean Prior Art Citations / Patent", "Value": f"{avg_citations:.2f}"},
        {"Metric": "Missing Abstract Rate (%)", "Value": f"{missing_abstract_pct:.2f}%"},
        {"Metric": "Missing Claim Rate (%)", "Value": f"{missing_claim_pct:.2f}%"}
    ])
    
    table1_csv = os.path.join(output_tables_dir, "table1_dataset_statistics.csv")
    table1_df.to_csv(table1_csv, index=False)
    logger.info(f"Table 1 saved to: {table1_csv}")
    
    return stats

if __name__ == "__main__":
    from src.data.build_dataset import build_and_save_master_dataset
    master_df = build_and_save_master_dataset()
    generate_dataset_statistics(master_df)
