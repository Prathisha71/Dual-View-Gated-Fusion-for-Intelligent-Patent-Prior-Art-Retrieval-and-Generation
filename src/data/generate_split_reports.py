import os
import json
import pandas as pd
from typing import Dict, Any
from src.utils.logging import setup_logger

logger = setup_logger("GenerateSplitReports")

def generate_split_reports(
    splits_dir: str = "data/splits",
    reports_dir: str = "reports"
):
    os.makedirs(reports_dir, exist_ok=True)
    logger.info("Generating split reports for Train, Validation, and Test partitions...")
    
    split_files = {
        "train": os.path.join(splits_dir, "train.parquet"),
        "val": os.path.join(splits_dir, "val.parquet"),
        "test": os.path.join(splits_dir, "test.parquet")
    }
    
    report_data = {}
    rows = []
    
    for split_name, file_path in split_files.items():
        if not os.path.exists(file_path):
            logger.warning(f"Split file missing for '{split_name}' at {file_path}")
            continue
            
        df = pd.read_parquet(file_path)
        
        n_patents = len(df)
        dates = pd.to_datetime(df["grant_date"])
        d_min = str(dates.min().strftime("%Y-%m-%d"))
        d_max = str(dates.max().strftime("%Y-%m-%d"))
        
        # Unique CPC labels
        cpcs = []
        for x in df["cpc_subclasses"].dropna():
            cpcs.extend([c.strip() for c in str(x).split(",") if c.strip()])
        cpc_series = pd.Series(cpcs)
        unique_cpcs = int(cpc_series.nunique())
        top_cpcs = cpc_series.value_counts().head(5).to_dict()
        
        # Citations
        citations = df["cited_patent_ids"].apply(lambda x: len([c for c in str(x).split(",") if c.strip()]) if pd.notna(x) else 0)
        total_citations = int(citations.sum())
        avg_citations = float(citations.mean())
        
        split_stats = {
            "split_name": split_name,
            "number_of_patents": n_patents,
            "date_range": [d_min, d_max],
            "number_of_cpc_labels": unique_cpcs,
            "total_citations": total_citations,
            "avg_citations_per_patent": avg_citations,
            "top_cpc_distribution": top_cpcs
        }
        report_data[split_name] = split_stats
        
        rows.append({
            "Split": split_name.upper(),
            "Number of Patents": n_patents,
            "Date Range": f"{d_min} to {d_max}",
            "Unique CPC Labels": unique_cpcs,
            "Total Citations": total_citations,
            "Avg Citations/Patent": f"{avg_citations:.2f}"
        })
        
    # Save JSON and CSV
    json_path = os.path.join(reports_dir, "split_report.json")
    csv_path = os.path.join(reports_dir, "split_report.csv")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
        
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"Split reports generated: {json_path} and {csv_path}")
    return report_data

if __name__ == "__main__":
    generate_split_reports()
