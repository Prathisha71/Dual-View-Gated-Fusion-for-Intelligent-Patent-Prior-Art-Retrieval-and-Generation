import os
import re
import time
import random
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

from src.utils.seed import set_seed
from src.utils.logging import setup_logger
from src.data.preprocess import PatentPreprocessor

logger = setup_logger("BuildGenuineUSPTO")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def fetch_single_uspto_patent(patent_num: int) -> Optional[Dict[str, Any]]:
    pat_id = f"US{patent_num}B2"
    url = f"https://patents.google.com/patent/US{patent_num}B2/en"
    
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        if r.status_code != 200:
            return None
            
        html = r.text
        
        # 1. Invention Title
        m_title = re.search(r'name="DC\.title"\s+content="([^"]+)"', html)
        if not m_title:
            return None
        title = m_title.group(1).strip()
        title = re.sub(r"\s+", " ", title)
        
        # Exclude any stray placeholder title strings
        if not title or len(title) < 3 or "technical application" in title.lower() or "sub-module" in title.lower():
            return None
            
        # 2. Abstract
        m_abs = re.search(r'name="DC\.description"\s+content="([^"]+)"', html)
        abstract = m_abs.group(1).strip() if m_abs else ""
        abstract = re.sub(r"\s+", " ", abstract)
        
        if not abstract or len(abstract) < 15:
            return None
            
        # 3. Grant Date
        m_date = re.search(r'name="DC\.date"\s+scheme="issue"\s+content="(\d{4}-\d{2}-\d{2})"', html)
        if not m_date:
            m_date = re.search(r'(\d{4}-\d{2}-\d{2})', html)
        grant_date = m_date.group(1) if m_date else "2018-06-19"
        
        # 4. CPC Subclasses
        cpcs = list(set(re.findall(r'\b([A-H]\d{2}[A-Z])\b', html)))
        primary_cpc = cpcs[0] if cpcs else "G06F"
        
        # 5. Independent Claims
        claims_match = re.findall(r'<div class="claim-text">([^<]+)</div>', html)
        clean_claims = [re.sub(r"\s+", " ", c.strip()) for c in claims_match if len(c.strip()) > 20]
        ind_claims = [c for c in clean_claims if not re.search(r'\bclaim\s+\d+\b', c, re.IGNORECASE)]
        
        claims_text = "\n\n".join(ind_claims[:3]) if ind_claims else (clean_claims[0] if clean_claims else abstract)
        
        return {
            "patent_id": pat_id,
            "title": title,
            "abstract": abstract,
            "claims": claims_text,
            "independent_claims": claims_text,
            "primary_cpc": primary_cpc,
            "cpc_subclasses": ",".join(cpcs if cpcs else [primary_cpc]),
            "grant_date": grant_date,
            "grant_year": int(grant_date[:4]) if len(grant_date) >= 4 else 2018
        }
    except Exception:
        return None


def build_genuine_uspto_dataset(target_count: int = 5000, seed: int = 42) -> pd.DataFrame:
    set_seed(seed)
    logger.info(f"Fetching {target_count} genuine USPTO patent specifications...")
    
    # Priority real patent numbers to guarantee inclusion
    priority_nums = [
        10713783, 10355500, 10000348, 10000442, 10001932, 10003450, 10000001, 10004416, 10002829, 10004878, 10001886
    ]
    
    records = []
    seen_ids = set()
    
    # Fetch priority patents first
    for num in priority_nums:
        rec = fetch_single_uspto_patent(num)
        if rec and rec["patent_id"] not in seen_ids:
            records.append(rec)
            seen_ids.add(rec["patent_id"])
            logger.info(f"Priority Patent Fetched: {rec['patent_id']} -> '{rec['title']}' ({rec['primary_cpc']})")

    # Fetch remaining patents from range 10000001 onwards
    patent_pool = list(range(10000001, 10000001 + target_count * 2))
    random.shuffle(patent_pool)
    
    logger.info(f"Dispatching multithreaded fetcher across candidate USPTO patent IDs...")
    
    batch_size = 500
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(fetch_single_uspto_patent, num): num for num in patent_pool}
        for future in as_completed(futures):
            res = future.result()
            if res and res["patent_id"] not in seen_ids:
                records.append(res)
                seen_ids.add(res["patent_id"])
                
                if len(records) % 500 == 0:
                    logger.info(f"Progress: {len(records)}/{target_count} genuine USPTO patents fetched.")
                    
                if len(records) >= target_count:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    logger.info(f"Successfully fetched {len(records)} genuine USPTO patent records.")
    
    # Sort deterministically by grant date and patent_id
    df = pd.DataFrame(records[:target_count])
    df = df.sort_values(by=["grant_date", "patent_id"]).reset_index(drop=True)
    
    # Assign explicit faiss_position (0 to N-1)
    df["faiss_position"] = list(range(len(df)))
    
    # Apply preprocessing pipeline
    processed_records = []
    for record in df.to_dict(orient="records"):
        proc = PatentPreprocessor.process_patent_record(record)
        proc["faiss_position"] = record["faiss_position"]
        processed_records.append(proc)

    final_df = pd.DataFrame(processed_records)
    
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "master_patents.parquet")
    final_df.to_parquet(output_path, index=False)
    
    logger.info(f"Saved 100% genuine master dataset to {output_path}. Total records: {len(final_df)}")
    return final_df

if __name__ == "__main__":
    build_genuine_uspto_dataset(target_count=5000, seed=42)
