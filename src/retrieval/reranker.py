import time
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from src.utils.logging import setup_logger

logger = setup_logger("PatentReranker")

class PatentReranker:
    """
    Cross-Encoder / Terminology-aware Reranker engine for fine-grained re-scoring of Top 50 prior art candidates down to Top 10.
    Supports fast CPU technical keyword overlap re-scoring (<15ms) and GPU Cross-Encoder with a strict 2.0s timeout guard.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-large", force_fast_cpu: bool = True):
        self.model_name = model_name
        self.cross_encoder = None
        self.force_fast_cpu = force_fast_cpu
        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_reranker()

    def _init_reranker(self):
        if self.device_name == "cpu" and self.force_fast_cpu:
            logger.info("CPU execution detected. Initializing fast technical term overlap re-scorer (<15ms)...")
            self.cross_encoder = None
            return

        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading Cross-Encoder reranker '{self.model_name}' on {self.device_name}...")
            self.cross_encoder = CrossEncoder(self.model_name, max_length=256, device=self.device_name)
            logger.info("Cross-Encoder loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize Cross-Encoder model '{self.model_name}': {e}")
            logger.info("Initializing fallback heuristic technical overlap re-scorer...")
            self.cross_encoder = None

    def rerank(
        self,
        query_record: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        top_k: int = 50,
        alpha_sim: float = 0.7,
        beta_cpc: float = 0.2,
        gamma_rerank: float = 0.1,
        timeout_sec: float = 2.0
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
            
        t0 = time.time()
        q_text = query_record.get("combined_text_transformer", query_record.get("combined_text", ""))
        pairs = []
        
        for cand in candidates:
            c_text = cand.get("combined_text_transformer", cand.get("combined_text", ""))
            pairs.append([q_text[:500], c_text[:500]])

        scores = None
        used_fallback = False

        if self.cross_encoder is not None:
            # Run predict with 2.0s timeout guard
            def _run_predict():
                return self.cross_encoder.predict(pairs)

            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_run_predict)
                    scores = future.result(timeout=timeout_sec)
            except FutureTimeoutError:
                logger.warning(f"Cross-Encoder reranking exceeded {timeout_sec}s timeout. Falling back to fast overlap re-scorer.")
                used_fallback = True
            except Exception as e:
                logger.warning(f"Cross-Encoder reranking failed ({e}). Falling back to fast overlap re-scorer.")
                used_fallback = True

        if scores is None or used_fallback:
            # TF-IDF / keyword overlap re-scorer (<15ms)
            scores = []
            q_words = set([w.lower() for w in q_text.split() if len(w) > 2])
            for pair in pairs:
                c_words = set([w.lower() for w in pair[1].split() if len(w) > 2])
                intersection = len(q_words.intersection(c_words))
                union = len(q_words.union(c_words)) or 1
                jaccard = intersection / union
                scores.append(jaccard)
                
        for idx, cand in enumerate(candidates):
            sc = float(scores[idx]) if idx < len(scores) else 0.0
            sem_sim = float(cand.get("semantic_similarity", 0.0))
            cpc_compat = float(cand.get("cpc_compatibility", 0.0))
            
            cand["rerank_score"] = sc
            cand["final_score"] = alpha_sim * sem_sim + beta_cpc * cpc_compat + gamma_rerank * sc
            
        reranked = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
        dt_ms = (time.time() - t0) * 1000.0
        logger.info(f"Reranking completed in {dt_ms:.1f}ms. Method: {'Cross-Encoder' if (self.cross_encoder and not used_fallback) else 'Fast Technical Overlap'}")
        return reranked[:top_k]

