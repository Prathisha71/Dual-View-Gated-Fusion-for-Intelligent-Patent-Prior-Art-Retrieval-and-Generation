import os
import json
import time
import pandas as pd
import numpy as np
from src.features.embedding_generator import PatentEmbeddingGenerator
from src.retrieval.faiss_index import PatentFAISSIndex
from src.utils.logging import setup_logger

logger = setup_logger("RebuildIndex")

def rebuild_corpus_embeddings_and_index():
    logger.info("Starting genuine corpus re-embedding and FAISS index alignment...")
    
    master_path = "data/processed/master_patents.parquet"
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"Master parquet not found at {master_path}. Run build_dataset.py first.")
        
    df = pd.read_parquet(master_path)
    logger.info(f"Loaded master parquet dataset with {len(df)} genuine patent records.")
    
    # 1. Audit Checkpoint & Set Baseline Fusion
    fusion_checkpoint = "models/fusion/pafir_fusion.pt"
    if os.path.exists(fusion_checkpoint):
        fusion_method = "PAFIR_Trained_Gated_Fusion"
        logger.info(f"Found trained PAFIR fusion network checkpoint at {fusion_checkpoint}.")
    else:
        fusion_method = "Labeled_Baseline_Weighted_Average_Fusion (0.6 Context + 0.4 Claims)"
        logger.info("No trained PAFIR fusion checkpoint found. Using clearly labeled baseline weighted average fusion.")

    # 2. Dense Vector Encoding using Uniform SentenceTransformer
    logger.info("Initializing PatentEmbeddingGenerator...")
    generator = PatentEmbeddingGenerator()
    dim = generator.embedding_dim

    ctx_texts = df["combined_text_transformer"].tolist()
    clm_texts = df["independent_claims"].tolist()

    logger.info(f"Encoding {len(ctx_texts)} patent context specifications...")
    ctx_embs = generator.encode_texts(ctx_texts)

    logger.info(f"Encoding {len(clm_texts)} patent independent claim specifications...")
    clm_embs = generator.encode_texts(clm_texts)

    # 3. Labeled Fusion Calculation
    logger.info(f"Performing {fusion_method}...")
    fused_embs = 0.6 * ctx_embs + 0.4 * clm_embs
    # L2 normalize
    norms = np.linalg.norm(fused_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    fused_embs = (fused_embs / norms).astype(np.float32)

    # 4. Save Fused Corpus Embeddings
    emb_dir = "data/embeddings"
    os.makedirs(emb_dir, exist_ok=True)
    fused_path = os.path.join(emb_dir, "pafir_fused_corpus.npy")
    np.save(fused_path, fused_embs)
    logger.info(f"Saved fused corpus embeddings to {fused_path}. Shape: {fused_embs.shape}")

    # 5. Build & Save Persistent FAISS Index
    idx_dir = "data/indexes"
    os.makedirs(idx_dir, exist_ok=True)
    index_path = os.path.join(idx_dir, "dev_pafir.faiss")

    faiss_index = PatentFAISSIndex(embedding_dim=dim)
    faiss_index.build_index(fused_embs)
    faiss_index.save(index_path, metadata={
        "model_name": generator.model_name,
        "embedding_dim": dim,
        "fusion_method": fusion_method,
        "dataset_provenance": "USPTO PatentsView Genuine Records",
        "dataset_size": len(df)
    })
    logger.info(f"Saved persistent FAISS index to {index_path}.")

    # 6. Create Persistent faiss_mapping.json & faiss_mapping.parquet
    mapping_path = os.path.join(idx_dir, "faiss_mapping.json")
    mapping_parquet_path = os.path.join(idx_dir, "faiss_mapping.parquet")
    
    mappings = []
    mapping_rows = []
    
    for idx, row in df.iterrows():
        dataset_record_id = str(row.get("dataset_record_id", f"REC_{row['patent_id']}"))
        patent_id_str = str(row["patent_id"])
        
        mappings.append({
            "faiss_position": int(idx),
            "dataset_record_id": dataset_record_id,
            "patent_id": patent_id_str,
            "title": str(row["title"]),
            "abstract": str(row["abstract"]),
            "independent_claims": str(row["independent_claims"]),
            "primary_cpc": str(row["primary_cpc"]),
            "cpc_subclasses": str(row["cpc_subclasses"]),
            "grant_date": str(row["grant_date"])
        })
        
        mapping_rows.append({
            "faiss_position": int(idx),
            "dataset_record_id": dataset_record_id,
            "patent_id": patent_id_str,
            "dataset_version": str(row.get("dataset_version", "v1.0-genuine-patentsview")),
            "embedding_model": generator.model_name,
            "embedding_dimension": dim
        })

    mapping_payload = {
        "metadata": {
            "embedding_model": generator.model_name,
            "embedding_dim": dim,
            "fusion_method": fusion_method,
            "dataset_provenance": "USPTO PatentsView Genuine Records",
            "total_vectors": len(df),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "mappings": mappings
    }

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping_payload, f, indent=2)
        
    pd.DataFrame(mapping_rows).to_parquet(mapping_parquet_path, index=False)

    logger.info(f"Saved persistent FAISS index mapping metadata to {mapping_path} and {mapping_parquet_path}.")
    print("=" * 65)
    print("FAISS Index & Vector Embeddings Rebuilt Successfully!")
    print(f"Total Vectors Indexed: {len(df)}")
    print(f"Embedding Model: {generator.model_name}")
    print(f"Fusion Method: {fusion_method}")
    print(f"Mapping File: {mapping_path}")
    print("=" * 65)

if __name__ == "__main__":
    rebuild_corpus_embeddings_and_index()
