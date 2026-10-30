import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from src.utils.logging import setup_logger

logger = setup_logger("FAISSIndex")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("FAISS library not found. Initializing NumPyInnerProductIndex fallback.")


class NumPyInnerProductIndex:
    """NumPy-accelerated inner product index fallback for cosine similarity search."""
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = None
        self.ntotal = 0

    def add(self, x: np.ndarray):
        x = np.ascontiguousarray(x.astype(np.float32))
        if self.vectors is None:
            self.vectors = x
        else:
            self.vectors = np.vstack([self.vectors, x])
        self.ntotal = len(self.vectors)

    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        x = np.ascontiguousarray(x.astype(np.float32))
        sims = np.dot(x, self.vectors.T)  # (n_queries, n_corpus)
        
        top_indices = np.argsort(-sims, axis=1)[:, :k]
        top_distances = np.take_along_axis(sims, top_indices, axis=1)
        return top_distances.astype(np.float32), top_indices.astype(np.int64)


class PatentFAISSIndex:
    """
    FAISS Index Manager supporting IndexFlatIP, HNSW, IVF, and fallback NumPy indexing.
    Persists index state to disk and loads fast on server startup.
    """
    def __init__(self, embedding_dim: int = 384, index_type: str = "IndexFlatIP"):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.is_built = False

    def build_index(self, embeddings: np.ndarray):
        logger.info(f"Building vector index '{self.index_type}' for {len(embeddings)} vectors (dim={self.embedding_dim})...")
        # Ensure L2 normalization for inner product cosine equivalence
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        norm_embeddings = (embeddings / norms).astype(np.float32)
        
        if HAS_FAISS:
            if self.index_type == "IndexFlatIP":
                self.index = faiss.IndexFlatIP(self.embedding_dim)
            elif self.index_type == "HNSW":
                self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            elif self.index_type == "IVF":
                nlist = min(100, max(1, len(embeddings) // 10))
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
                self.index.train(norm_embeddings)
            else:
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                
            self.index.add(norm_embeddings)
        else:
            self.index = NumPyInnerProductIndex(self.embedding_dim)
            self.index.add(norm_embeddings)
            
        self.is_built = True
        logger.info(f"Vector index constructed cleanly. Total indexed vectors: {len(embeddings)}")

    def search(self, query_vectors: np.ndarray, k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_built:
            raise ValueError("Vector index is not built yet!")
            
        norms = np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-12
        norm_queries = (query_vectors / norms).astype(np.float32)
        
        scores, indices = self.index.search(norm_queries, k)
        return scores, indices

    def save(self, output_path: str = "data/indexes/dev_pafir.faiss", metadata: Optional[Dict[str, Any]] = None):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if HAS_FAISS and hasattr(faiss, "write_index"):
            faiss.write_index(self.index, output_path)
        else:
            joblib.dump({"index": self.index, "dim": self.embedding_dim, "type": self.index_type}, output_path)
        logger.info(f"Saved vector index to {output_path}")

        meta_path = os.path.splitext(output_path)[0] + "_metadata.json"
        import json, time
        meta_content = {
            "model": metadata.get("model", "all MiniLM L6 v2") if metadata else "all MiniLM L6 v2",
            "embedding_dimension": self.embedding_dim,
            "number_of_vectors": self.index.ntotal if hasattr(self.index, 'ntotal') else 0,
            "dataset_size": metadata.get("dataset_size", 5000) if metadata else 5000,
            "creation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_version": metadata.get("dataset_version", "DEV_MODE v1.0") if metadata else "DEV_MODE v1.0"
        }
        with open(meta_path, "w") as f:
            json.dump(meta_content, f, indent=2)
        logger.info(f"Saved vector index metadata to {meta_path}")

    def load(self, input_path: str = "data/indexes/dev_pafir.faiss"):
        if HAS_FAISS and hasattr(faiss, "read_index") and not input_path.endswith(".pkl"):
            self.index = faiss.read_index(input_path)
            self.embedding_dim = self.index.d
        else:
            data = joblib.load(input_path)
            self.index = data["index"]
            self.embedding_dim = data["dim"]
            self.index_type = data["type"]
        self.is_built = True
        logger.info(f"Loaded vector index from {input_path}")
