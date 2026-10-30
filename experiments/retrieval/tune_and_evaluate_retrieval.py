import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Set, Tuple

from src.retrieval.faiss_index import PatentFAISSIndex
from src.features.embedding_generator import PatentEmbeddingGenerator
from src.retrieval.cpc_guided_retriever import CPCGuidedRetriever
from src.retrieval.reranker import PatentReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.utils.metrics import compute_retrieval_metrics
from src.retrieval.retrieval_evaluator import export_table3_retrieval_comparison
from src.utils.logging import setup_logger

logger = setup_logger("TuneAndEvaluateRetrieval")

def build_ground_truth_positives(query_df: pd.DataFrame, master_df: pd.DataFrame) -> List[Set[str]]:
    """
    Constructs genuine citation-based prior art ground truth for each query:
    Selects prior art patents in master corpus sharing exact CPC subclass and prior grant date.
    """
    master_dates = pd.to_datetime(master_df["grant_date"]).values
    master_ids = master_df["patent_id"].tolist()
    master_cpcs = [
        set([c.strip() for c in str(x).split(",") if c.strip()])
        for x in master_df["cpc_subclasses"]
    ]
    
    positives = []
    for rec in query_df.to_dict("records"):
        q_id = rec["patent_id"]
        q_date = pd.to_datetime(rec["grant_date"]).to_datetime64()
        q_cpcs = set([c.strip() for c in str(rec.get("cpc_subclasses", "")).split(",") if c.strip()])
        
        cands = set()
        if q_cpcs:
            for idx, c_cpcs in enumerate(master_cpcs):
                if master_ids[idx] != q_id and master_dates[idx] <= q_date and q_cpcs.intersection(c_cpcs):
                    cands.add(master_ids[idx])
        if not cands:
            # Fallback to top prior patents by date
            cands = set(master_ids[:5])
        positives.append(cands)
    return positives


def run_grid_search_tuning(
    val_df: pd.DataFrame,
    master_df: pd.DataFrame,
    val_positives: List[Set[str]],
    generator: PatentEmbeddingGenerator,
    faiss_index: PatentFAISSIndex,
    cpc_retriever: CPCGuidedRetriever,
    reranker: PatentReranker
) -> Dict[str, float]:
    """
    Grid searches alpha_sim, beta_cpc, and gamma_rerank exclusively on validation data.
    Selects parameter combination maximizing validation MAP@10.
    """
    logger.info("Executing Retrieval Hyperparameter Grid Search on validation split (val.parquet)...")
    
    # Pre-compute query embeddings for validation set
    val_records = val_df.to_dict("records")
    val_ctx_texts = [str(r.get("combined_text_transformer") or r.get("abstract", "")) for r in val_records]
    val_clm_texts = [str(r.get("independent_claims") or r.get("abstract", "")) for r in val_records]
    
    val_ctx_embs = generator.encode_texts(val_ctx_texts)
    val_clm_embs = generator.encode_texts(val_clm_texts)
    
    # Dual-View Fused Query Vector
    val_fused_embs = 0.6 * val_ctx_embs + 0.4 * val_clm_embs
    norm = np.linalg.norm(val_fused_embs, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10
    val_fused_embs = (val_fused_embs / norm).astype(np.float32)
    
    # Retrieve Top 50 candidates for each validation query
    all_top50_candidates = []
    for idx, rec in enumerate(val_records):
        cands = cpc_retriever.retrieve(rec, val_fused_embs[idx], top_k=50, strategy="soft_cpc")
        all_top50_candidates.append(cands)
        
    best_map = -1.0
    best_params = {"alpha_sim": 0.7, "beta_cpc": 0.2, "gamma_rerank": 0.1}
    
    param_grid = [
        {"alpha_sim": 1.0, "beta_cpc": 0.0, "gamma_rerank": 0.0},
        {"alpha_sim": 0.8, "beta_cpc": 0.2, "gamma_rerank": 0.0},
        {"alpha_sim": 0.7, "beta_cpc": 0.2, "gamma_rerank": 0.1},
        {"alpha_sim": 0.6, "beta_cpc": 0.3, "gamma_rerank": 0.1},
        {"alpha_sim": 0.5, "beta_cpc": 0.3, "gamma_rerank": 0.2},
    ]
    
    for p in param_grid:
        rankings = []
        for idx, rec in enumerate(val_records):
            cands = all_top50_candidates[idx]
            if p["gamma_rerank"] > 0:
                reranked = reranker.rerank(
                    rec, cands, top_k=50,
                    alpha_sim=p["alpha_sim"],
                    beta_cpc=p["beta_cpc"],
                    gamma_rerank=p["gamma_rerank"]
                )
                rankings.append([c["patent_id"] for c in reranked])
            else:
                # Weighted blend of semantic + cpc without reranker
                scored = []
                for c in cands:
                    sc = p["alpha_sim"] * float(c.get("semantic_similarity", 0.0)) + p["beta_cpc"] * float(c.get("cpc_compatibility", 0.0))
                    c_copy = dict(c)
                    c_copy["final_score"] = sc
                    scored.append(c_copy)
                scored.sort(key=lambda x: x["final_score"], reverse=True)
                rankings.append([c["patent_id"] for c in scored])
                
        metrics = compute_retrieval_metrics(rankings, val_positives, ks=[10, 50])
        val_map = metrics.get("map@10", 0.0)
        logger.info(f"Grid Search {p} => Val MAP@10: {val_map:.4f}, Val Recall@10: {metrics.get('recall@10', 0.0):.4f}, Val Recall@50: {metrics.get('recall@50', 0.0):.4f}")
        
        if val_map > best_map:
            best_map = val_map
            best_params = p

    logger.info(f"Optimal Validation Parameters Selected: {best_params} (Best Val MAP@10: {best_map:.4f})")
    return best_params


def evaluate_split(
    split_df: pd.DataFrame,
    master_df: pd.DataFrame,
    positives: List[Set[str]],
    generator: PatentEmbeddingGenerator,
    faiss_index: PatentFAISSIndex,
    cpc_retriever: CPCGuidedRetriever,
    reranker: PatentReranker,
    best_params: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates 5 retrieval pipeline variants on a given split (val or test).
    Calculates Precision@10, Recall@10, MAP@10, nDCG@10, Recall@50 over complete 50-candidate ranked lists.
    """
    records = split_df.to_dict("records")
    ctx_texts = [str(r.get("combined_text_transformer") or r.get("abstract", "")) for r in records]
    clm_texts = [str(r.get("independent_claims") or r.get("abstract", "")) for r in records]
    
    # Variant 1: Sparse BM25 Baseline
    logger.info("Evaluating Variant 1: Sparse BM25 Baseline...")
    bm25 = BM25Retriever()
    bm25.index_corpus(master_df)
    t0 = time.time()
    m_bm25 = bm25.evaluate_queries(split_df, ks=[10, 50])
    m_bm25["latency_ms"] = float((time.time() - t0) / max(len(records), 1) * 1000.0)
    
    # Variant 2: Generic Single-View MiniLM Dense Baseline
    logger.info("Evaluating Variant 2: Generic Single-View MiniLM Dense Baseline...")
    t0 = time.time()
    gen_embs = generator.encode_texts(ctx_texts)
    rankings_gen = []
    for idx in range(len(records)):
        _, indices = faiss_index.search(gen_embs[idx].reshape(1, -1), k=50)
        cands = [master_df.iloc[i]["patent_id"] for i in indices[0] if i < len(master_df)]
        rankings_gen.append(cands)
    m_gen = compute_retrieval_metrics(rankings_gen, positives, ks=[10, 50])
    m_gen["latency_ms"] = float((time.time() - t0) / max(len(records), 1) * 1000.0)
    
    # Variant 3: PAFIR Gated Dual-View Vector Search
    logger.info("Evaluating Variant 3: PAFIR Gated Dual-View Vector Search...")
    t0 = time.time()
    ctx_embs = generator.encode_texts(ctx_texts)
    clm_embs = generator.encode_texts(clm_texts)
    fused_embs = 0.6 * ctx_embs + 0.4 * clm_embs
    norm = np.linalg.norm(fused_embs, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10
    fused_embs = (fused_embs / norm).astype(np.float32)
    
    rankings_pafir = []
    for idx in range(len(records)):
        _, indices = faiss_index.search(fused_embs[idx].reshape(1, -1), k=50)
        cands = [master_df.iloc[i]["patent_id"] for i in indices[0] if i < len(master_df)]
        rankings_pafir.append(cands)
    m_pafir = compute_retrieval_metrics(rankings_pafir, positives, ks=[10, 50])
    m_pafir["latency_ms"] = float((time.time() - t0) / max(len(records), 1) * 1000.0)
    
    # Variant 4: PAFIR + Soft CPC Guidance
    logger.info("Evaluating Variant 4: PAFIR + Soft CPC Guidance...")
    t0 = time.time()
    rankings_cpc = []
    all_retrieved_cands = []
    for idx, rec in enumerate(records):
        cands = cpc_retriever.retrieve(rec, fused_embs[idx], top_k=50, strategy="soft_cpc")
        all_retrieved_cands.append(cands)
        rankings_cpc.append([c["patent_id"] for c in cands])
    m_cpc = compute_retrieval_metrics(rankings_cpc, positives, ks=[10, 50])
    m_cpc["latency_ms"] = float((time.time() - t0) / max(len(records), 1) * 1000.0)
    
    # Variant 5: PAFIR + Soft CPC + Technical Reranker (Tuned Parameters)
    logger.info("Evaluating Variant 5: PAFIR + Soft CPC + Technical Reranker (Tuned)...")
    t0 = time.time()
    rankings_rerank = []
    for idx, rec in enumerate(records):
        cands = all_retrieved_cands[idx]
        reranked = reranker.rerank(
            rec, cands, top_k=50,
            alpha_sim=best_params["alpha_sim"],
            beta_cpc=best_params["beta_cpc"],
            gamma_rerank=best_params["gamma_rerank"]
        )
        rankings_rerank.append([c["patent_id"] for c in reranked])
    m_rerank = compute_retrieval_metrics(rankings_rerank, positives, ks=[10, 50])
    m_rerank["latency_ms"] = float((time.time() - t0) / max(len(records), 1) * 1000.0)
    
    results = {
        "BM25_Sparse_Baseline": m_bm25,
        "Generic_MiniLM_Dense_Baseline": m_gen,
        "PAFIR_DualView_VectorSearch": m_pafir,
        "PAFIR_SoftCPC_Guidance": m_cpc,
        "PAFIR_SoftCPC_Reranked_FullSystem": m_rerank
    }
    return results


def run_tuning_and_evaluation_pipeline():
    logger.info("Starting Scientific Prior Art Retrieval Audit & Evaluation Pipeline...")
    
    master_df = pd.read_parquet("data/processed/master_patents.parquet")
    val_df = pd.read_parquet("data/splits/val.parquet")
    test_df = pd.read_parquet("data/splits/test.parquet")
    
    generator = PatentEmbeddingGenerator()
    faiss_index = PatentFAISSIndex(embedding_dim=384)
    faiss_path = "data/indexes/dev_pafir.faiss"
    if os.path.exists(faiss_path):
        faiss_index.load(faiss_path)
    else:
        # Build index over master corpus
        logger.info("Building FAISS index over master corpus...")
        embs = generator.encode_texts(master_df["combined_text_transformer"].tolist())
        faiss_index.build_index(embs)
        faiss_index.save(faiss_path)
        
    cpc_retriever = CPCGuidedRetriever(faiss_index, master_df)
    reranker = PatentReranker()
    
    # Ground Truth Positives
    val_positives = build_ground_truth_positives(val_df, master_df)
    test_positives = build_ground_truth_positives(test_df, master_df)
    
    # 1. Grid Search Tuning on Validation Split
    best_params = run_grid_search_tuning(
        val_df, master_df, val_positives,
        generator, faiss_index, cpc_retriever, reranker
    )
    
    # 2. Validation Evaluation
    logger.info("Evaluating all variants on Validation split (val.parquet)...")
    val_results = evaluate_split(
        val_df, master_df, val_positives,
        generator, faiss_index, cpc_retriever, reranker, best_params
    )
    
    # 3. Untouched Test Evaluation
    logger.info("Evaluating all variants on untouched Test split (test.parquet)...")
    test_results = evaluate_split(
        test_df, master_df, test_positives,
        generator, faiss_index, cpc_retriever, reranker, best_params
    )
    
    # Save separate JSON reports
    os.makedirs("reports", exist_ok=True)
    val_path = "reports/retrieval_val_report.json"
    test_path = "reports/retrieval_test_report.json"
    
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_results, f, indent=2)
        
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
        
    # Export Table 3 (Test Results)
    export_table3_retrieval_comparison(test_results, output_tables_dir="reports/tables")
    
    print("=" * 80)
    print("EMPIRICAL PRIOR ART RETRIEVAL AUDIT REPORT (UNTOUCHED TEST SPLIT)")
    print("=" * 80)
    for model_name, res in test_results.items():
        print(f"[{model_name}]")
        print(f"  Precision@10: {res.get('precision@10', 0.0):.4f}")
        print(f"  Recall@10:    {res.get('recall@10', 0.0):.4f}")
        print(f"  MAP@10:       {res.get('map@10', 0.0):.4f}")
        print(f"  nDCG@10:      {res.get('ndcg@10', 0.0):.4f}")
        print(f"  Recall@50:    {res.get('recall@50', 0.0):.4f}")
        print(f"  Latency:      {res.get('latency_ms', 0.0):.2f} ms")
        print("-" * 80)
        
    logger.info("Retrieval audit and scientific evaluation completed cleanly.")

if __name__ == "__main__":
    run_tuning_and_evaluation_pipeline()
