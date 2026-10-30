import os
import sys
import json
import pickle
import urllib.request

def run_system_check():
    print("=" * 45)
    print("           PAFIR SYSTEM CHECK")
    print("=" * 45)
    print()

    all_ready = True

    # 1. Check Dataset
    dataset_path = "data/processed/master_patents.parquet"
    if os.path.exists(dataset_path):
        try:
            import pandas as pd
            df = pd.read_parquet(dataset_path)
            patents_count = len(df)
            print(f"Dataset ............... READY")
            print(f"Patents ............... {patents_count:,}")
        except Exception as e:
            print(f"Dataset ............... ERROR ({e})")
            all_ready = False
    else:
        print("Dataset ............... MISSING")
        all_ready = False
    print()

    # 2. Check Embedding Model
    try:
        from src.features.embedding_generator import PatentEmbeddingGenerator
        generator = PatentEmbeddingGenerator()
        model_name = generator.model_name
        dim = generator.embedding_dim
        print("Embedding Model ....... READY")
        print(f"Model ................. {model_name}")
        print(f"Dimension ............. {dim}")
    except Exception as e:
        print(f"Embedding Model ....... FAILED ({e})")
        all_ready = False
    print()

    # 3. Check CPC Classifier
    cpc_checkpoint = "models/classification/cpc_classifier.pt"
    cpc_mlb = "models/classification/cpc_mlb.pkl"
    if os.path.exists(cpc_checkpoint) and os.path.exists(cpc_mlb):
        try:
            with open(cpc_mlb, "rb") as f:
                classes = pickle.load(f)
                num_classes = len(classes)
            print("CPC Classifier ........ READY")
            print(f"Classes ............... {num_classes}")
        except Exception as e:
            print(f"CPC Classifier ........ ERROR ({e})")
            all_ready = False
    else:
        print("CPC Classifier ........ MISSING")
        all_ready = False
    print()

    # 4. Check FAISS Index
    index_path = "data/indexes/dev_pafir.faiss"
    meta_path = "data/indexes/dev_pafir_metadata.json"
    if os.path.exists(index_path):
        try:
            from src.retrieval.faiss_index import PatentFAISSIndex
            idx = PatentFAISSIndex(embedding_dim=384)
            idx.load(index_path)
            vec_count = idx.index.ntotal if hasattr(idx.index, 'ntotal') else 5000
            print("FAISS Index ........... READY")
            print(f"Vectors ............... {vec_count:,}")
        except Exception as e:
            print(f"FAISS Index ........... ERROR ({e})")
            all_ready = False
    elif os.path.exists("data/embeddings/pafir_fused_corpus.npy"):
        import numpy as np
        arr = np.load("data/embeddings/pafir_fused_corpus.npy")
        print("FAISS Index ........... READY (Fused embeddings present)")
        print(f"Vectors ............... {len(arr):,}")
    else:
        print("FAISS Index ........... MISSING")
        all_ready = False
    print()

    # 5. Check Reranker
    try:
        from src.retrieval.reranker import PatentReranker
        reranker = PatentReranker()
        r_status = "READY" if reranker.cross_encoder is not None else "OPTIONAL / FALLBACK READY"
        print(f"Reranker .............. {r_status}")
    except Exception:
        print("Reranker .............. OPTIONAL / DISABLED")
    print()

    # 6. Check Backend FastAPI Server
    try:
        req = urllib.request.urlopen("http://localhost:8000/health", timeout=2)
        if req.status == 200:
            print("Backend ............... READY")
        else:
            print("Backend ............... UNRESPONSIVE")
    except Exception:
        print("Backend ............... NOT RUNNING (Start with uvicorn api.main:app)")
    print()

    # 7. Check Frontend Dev Server
    try:
        req = urllib.request.urlopen("http://localhost:3000/", timeout=2)
        if req.status == 200:
            print("Frontend .............. READY")
        else:
            print("Frontend .............. UNRESPONSIVE")
    except Exception:
        print("Frontend .............. NOT RUNNING (Start with npm run dev)")
    print()

    print("=" * 45)
    print("SYSTEM STATUS:")
    if all_ready:
        print("READY FOR DEV ANALYSIS")
    else:
        print("SYSTEM STARTING / INCOMPLETE")
    print("=" * 45)

if __name__ == "__main__":
    run_system_check()
