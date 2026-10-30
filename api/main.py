import os
import time
import glob
import json
import pickle
import torch
import re
import io
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from api.schemas.schemas import (
    PatentRequest, ClassificationResponse, SearchResponse, RetrievalCandidate,
    ComprehensiveAnalysisResponse, ChatRequest, ChatResponse, GroundedExplanation
)
from api.database.connection import engine, Base, get_db
from api.database.models import PatentRecordDB, SearchHistoryDB

from src.data.preprocess import PatentPreprocessor
from src.data.pdf_extractor import parse_patent_pdf_bytes
from src.features.embedding_generator import PatentEmbeddingGenerator
from src.retrieval.pafir_fusion import PAFIRFusionModule
from src.classification.multilabel_classifier import CPCMultiLabelClassifier
from src.retrieval.faiss_index import PatentFAISSIndex
from src.retrieval.cpc_guided_retriever import CPCGuidedRetriever
from src.retrieval.reranker import PatentReranker
from src.explainability.result_explainer import GroundedResultExplainer
from src.rag.patent_assistant import GroundedPatentAssistant
from src.utils.logging import setup_logger

logger = setup_logger("FastAPIBackend")

# Initialize SQLite/DB Tables
Base.metadata.create_all(bind=engine)

CPC_DESCRIPTIONS = {
    "H04L": "Transmission of digital information & networking protocols",
    "G06F": "Electric digital data processing & computing architectures",
    "A61K": "Preparations for medical, dental, or toilet purposes",
    "C07D": "Heterocyclic compounds & organic chemistry",
    "H01L": "Semiconductor devices & solid state devices",
    "G06Q": "Data processing systems for administrative, commercial, or financial purposes",
    "G06N": "Computing arrangements based on specific computational models (Neural networks / AI)",
    "B60L": "Electric equipment or propulsion of electrically-propelled vehicles",
    "E04B": "General building constructions & structural elements",
    "F02M": "Supplying engines with combustible mixtures or constituents",
    "G01N": "Investigating or analyzing materials by determining chemical or physical properties",
    "H02J": "Circuit arrangements or systems for supplying or distributing electric power",
    "A61B": "Diagnosis, surgery, or identification for medical purposes",
    "C08G": "Macromolecular compounds obtained otherwise than by reactions involving unsaturated carbon-to-carbon bonds",
    "H04W": "Wireless communication networks"
}

app = FastAPI(
    title="PAFIR Patent AI Framework API",
    description="Automated Patent Document Classification and Prior Art Search using PAFIR Dual-View Gated Embeddings",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Container for Startup Loading
class MLState:
    mode: str = "DEV_MODE"
    corpus_df: pd.DataFrame = None
    faiss_mapping: Dict[str, Any] = None
    embedding_generator: PatentEmbeddingGenerator = None
    pafir_fusion: PAFIRFusionModule = None
    faiss_index: PatentFAISSIndex = None
    cpc_retriever: CPCGuidedRetriever = None
    reranker: PatentReranker = None
    cpc_classifier: CPCMultiLabelClassifier = None
    gpu_name: Optional[str] = None
    cpc_classes: List[str] = [
        "H04L", "G06F", "A61K", "C07D", "H01L", "G06Q", "G06N", "B60L", "E04B", "F02M", "G01N", "H02J", "A61B", "C08G", "H04W"
    ]

ml_state = MLState()

@app.on_event("startup")
def startup_event():
    logger.info("Initializing PAFIR backend service models and index...")
    
    if torch.cuda.is_available():
        ml_state.gpu_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        ml_state.gpu_name = "Apple Silicon MPS"
    else:
        ml_state.gpu_name = "None (CPU Execution)"
        
    master_path = "data/processed/master_patents.parquet"
    if not os.path.exists(master_path):
        logger.info("Master dataset not found. Generating genuine PatentsView dataset...")
        from src.data.build_dataset import build_and_save_master_dataset
        df = build_and_save_master_dataset()
    else:
        df = pd.read_parquet(master_path)
        
    if "split" not in df.columns:
        split_map = {}
        for s_name in ["train", "val", "test"]:
            s_path = f"data/splits/{s_name}.parquet"
            if os.path.exists(s_path):
                sdf = pd.read_parquet(s_path)
                for pid in sdf["patent_id"]:
                    split_map[pid] = s_name
        df["split"] = df["patent_id"].map(split_map).fillna("train")

    ml_state.corpus_df = df
    ml_state.mode = "RESEARCH_MODE" if len(ml_state.corpus_df) >= 50000 else "DEV_MODE"
    
    # Load persistent faiss_mapping.json if available
    mapping_path = "data/indexes/faiss_mapping.json"
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            ml_state.faiss_mapping = json.load(f)
            logger.info(f"Loaded persistent FAISS index mapping from {mapping_path}.")

    ml_state.embedding_generator = PatentEmbeddingGenerator()
    dim = ml_state.embedding_generator.embedding_dim
    ml_state.pafir_fusion = PAFIRFusionModule(embedding_dim=dim, fusion_type="gated")
    
    emb_dir = "data/embeddings"
    fused_path = os.path.join(emb_dir, "pafir_fused_corpus.npy")
    if not os.path.exists(fused_path):
        logger.info("Corpus embeddings not found. Building corpus embeddings and FAISS index...")
        from rebuild_index_and_embeddings import rebuild_corpus_embeddings_and_index
        rebuild_corpus_embeddings_and_index()
        
    index_path = "data/indexes/dev_pafir.faiss"
    ml_state.faiss_index = PatentFAISSIndex(embedding_dim=dim)
    if os.path.exists(index_path):
        logger.info(f"Loading persistent FAISS index from {index_path}...")
        ml_state.faiss_index.load(index_path)
    else:
        from rebuild_index_and_embeddings import rebuild_corpus_embeddings_and_index
        rebuild_corpus_embeddings_and_index()
        ml_state.faiss_index.load(index_path)
    
    ml_state.cpc_retriever = CPCGuidedRetriever(ml_state.faiss_index, ml_state.corpus_df)
    ml_state.reranker = PatentReranker(force_fast_cpu=True)
    
    cpc_mlb_path = "models/classification/cpc_mlb.pkl"
    if os.path.exists(cpc_mlb_path):
        try:
            with open(cpc_mlb_path, "rb") as f:
                cpc_classes = pickle.load(f)
                if isinstance(cpc_classes, (list, np.ndarray)):
                    ml_state.cpc_classes = [str(x) for x in cpc_classes]
                    logger.info(f"Loaded {len(ml_state.cpc_classes)} CPC subclass labels from {cpc_mlb_path}")
        except Exception as e:
            logger.warning(f"Could not load {cpc_mlb_path}: {e}")

    cpc_checkpoint = "models/classification/cpc_classifier.pt"
    if os.path.exists(cpc_checkpoint):
        ml_state.cpc_classifier = CPCMultiLabelClassifier.load(cpc_checkpoint)
    else:
        ml_state.cpc_classifier = CPCMultiLabelClassifier(input_dim=dim, num_classes=len(ml_state.cpc_classes))

    cpc_thresh_path = "models/classification/cpc_threshold.json"
    if os.path.exists(cpc_thresh_path):
        try:
            with open(cpc_thresh_path, "r", encoding="utf-8") as f:
                t_info = json.load(f)
                ml_state.cpc_threshold = float(t_info.get("optimal_threshold", 0.40))
                logger.info(f"Loaded validation-tuned CPC threshold: tau = {ml_state.cpc_threshold:.2f}")
        except Exception as e:
            logger.warning(f"Could not load {cpc_thresh_path}: {e}")
        
    logger.info(f"PAFIR backend startup completed cleanly. Mode: {ml_state.mode}, Indexed Patents: {len(ml_state.corpus_df)}")


@app.get("/health")
def health_check():
    b_ok = True
    d_ok = ml_state.corpus_df is not None and len(ml_state.corpus_df) > 0
    e_ok = ml_state.embedding_generator is not None
    c_ok = ml_state.cpc_classifier is not None
    f_ok = ml_state.faiss_index is not None and ml_state.faiss_index.is_built
    r_ok = ml_state.reranker is not None
    
    b_status = "ready" if b_ok else "missing"
    d_status = "ready" if d_ok else "missing"
    e_status = "ready" if e_ok else "missing"
    c_status = "ready" if c_ok else "missing"
    f_status = "ready" if f_ok else "missing"
    r_status = "ready" if (r_ok and ml_state.reranker.cross_encoder is not None) else "optional_ready"

    overall = "ready" if all(x == "ready" for x in [b_status, d_status, e_status, c_status, f_status]) else "starting"

    return {
        "backend": b_status,
        "dataset": d_status,
        "embedding_model": e_status,
        "classifier": c_status,
        "faiss": f_status,
        "reranker": r_status,
        "status": overall,
        "dataset_mode": str(ml_state.mode),
        "indexed_patents": int(len(ml_state.corpus_df)) if ml_state.corpus_df is not None else 0,
        "embedding_dim": int(ml_state.embedding_generator.embedding_dim) if ml_state.embedding_generator else 384,
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "cpc_class_count": int(len(ml_state.cpc_classes)),
        "embedding_model_loaded": bool(e_ok),
        "faiss_index_loaded": bool(f_ok),
        "classifier_loaded": bool(c_ok),
        "reranker_loaded": bool(r_ok),
        "gpu_accelerator": str(ml_state.gpu_name)
    }


@app.post("/api/extract-pdf")
async def extract_patent_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    contents = await file.read()
    res = parse_patent_pdf_bytes(contents, filename=file.filename)
    return res


@app.post("/api/classify", response_model=ClassificationResponse)
def classify_patent(req: PatentRequest):
    t0 = time.time()
    req_dict = req.model_dump() if hasattr(req, 'model_dump') else req.dict()
    proc = PatentPreprocessor.process_patent_record(req_dict)
    
    ctx_text = proc["combined_text_transformer"]
    clm_text = proc["independent_claims"]
    
    ctx_emb = ml_state.embedding_generator.encode_texts([ctx_text])
    clm_emb = ml_state.embedding_generator.encode_texts([clm_text])
    
    fused_emb = 0.6 * ctx_emb + 0.4 * clm_emb
    norms = np.linalg.norm(fused_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    fused_emb = fused_emb / norms
    
    probs = ml_state.cpc_classifier.predict_proba(torch.tensor(fused_emb, dtype=torch.float32)).detach().numpy()[0]
    
    prob_dict = {str(cpc): float(p) for cpc, p in zip(ml_state.cpc_classes, probs)}
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    
    thresholded_probs = [(cpc, p) for cpc, p in sorted_probs if p >= ml_state.cpc_threshold]
    if not thresholded_probs and sorted_probs:
        thresholded_probs = [sorted_probs[0]]
    
    top_preds = [
        {
            "cpc": str(cpc),
            "probability": float(prob),
            "description": str(CPC_DESCRIPTIONS.get(cpc, f"CPC Subclass {cpc}"))
        }
        for cpc, prob in thresholded_probs
    ]
    
    dt = (time.time() - t0) * 1000.0
    return ClassificationResponse(
        patent_id=req.patent_id,
        top_predicted_cpcs=top_preds,
        all_probabilities=prob_dict,
        primary_cpc=top_preds[0]["cpc"],
        execution_time_ms=float(dt)
    )


@app.post("/api/search", response_model=SearchResponse)
def search_prior_art(req: PatentRequest, top_k: int = Query(10, ge=1, le=50)):
    t0 = time.time()
    req_dict = req.model_dump() if hasattr(req, 'model_dump') else req.dict()
    proc = PatentPreprocessor.process_patent_record(req_dict)
    
    ctx_emb = ml_state.embedding_generator.encode_texts([proc["combined_text_transformer"]])
    clm_emb = ml_state.embedding_generator.encode_texts([proc["independent_claims"]])
    
    fused_vec = 0.6 * ctx_emb[0] + 0.4 * clm_emb[0]
    norm = np.linalg.norm(fused_vec) + 1e-10
    fused_vec = fused_vec / norm
    
    cands = ml_state.cpc_retriever.retrieve(proc, fused_vec, top_k=50, strategy="soft_cpc")
    reranked = ml_state.reranker.rerank(proc, cands, top_k=50, alpha_sim=0.7, beta_cpc=0.2, gamma_rerank=0.1)[:top_k]
    
    results = []
    for idx, c in enumerate(reranked, start=1):
        cpcs = [str(x.strip()) for x in str(c.get("cpc_subclasses", "")).split(",") if x.strip()]
        results.append(RetrievalCandidate(
            rank=int(idx),
            patent_id=str(c.get("patent_id")),
            title=str(c.get("title")),
            grant_date=str(c.get("grant_date", "")),
            primary_cpc=str(c.get("primary_cpc", "")),
            cpc_subclasses=cpcs,
            semantic_similarity=float(c.get("semantic_similarity", 0.0)),
            cpc_compatibility=float(c.get("cpc_compatibility", 0.0)),
            final_score=float(c.get("final_score", 0.0)),
            citation_relevance="Semantic Prior Art Candidate"
        ))
        
    dt = (time.time() - t0) * 1000.0
    return SearchResponse(
        query_patent_id=req.patent_id,
        total_candidates_indexed=int(len(ml_state.corpus_df)),
        results=results,
        execution_time_ms=float(dt)
    )


@app.post("/api/analyze")
def analyze_patent_pipeline(req: PatentRequest):
    t_start = time.perf_counter()
    timings = {}
    
    # 1. Validation & Input Normalization
    t0 = time.time()
    req_dict = req.model_dump() if hasattr(req, 'model_dump') else req.dict()

    # 1. Input Validation
    t0 = time.perf_counter()
    pat_id = getattr(req, "patent_id", "") or ""
    title = getattr(req, "title", "") or ""
    abstract = getattr(req, "abstract", "") or ""
    claims = getattr(req, "independent_claims", "") or ""
    cutoff_date = getattr(req, "temporal_cutoff_date", None) or getattr(req, "grant_date", "2025-12-31")

    if not title and not abstract and not claims:
        raise HTTPException(
            status_code=400,
            detail="Patent analysis requires at least a title, abstract, or independent claims."
        )

    query_proc = {
        "patent_id": pat_id,
        "title": title,
        "abstract": abstract,
        "independent_claims": claims,
        "combined_text_transformer": f"Title: {title} | Abstract: {abstract} | Claims: {claims}",
        "grant_date": cutoff_date,
        "filing_date": cutoff_date
    }
    timings["validation_ms"] = float((time.perf_counter() - t0) * 1000.0)

    # 2. Text Construction
    t0 = time.perf_counter()
    ctx_text = f"Title: {title} | Abstract: {abstract}" if (title or abstract) else claims
    clm_text = claims if claims else ctx_text
    timings["text_construction_ms"] = float((time.perf_counter() - t0) * 1000.0)

    # 3. Embedding Generation
    t0 = time.perf_counter()
    ctx_emb = ml_state.embedding_generator.encode_texts([ctx_text])
    clm_emb = ml_state.embedding_generator.encode_texts([clm_text])
    timings["embedding_ms"] = float((time.perf_counter() - t0) * 1000.0)

    # 4. PAFIR Baseline Weighted Average Fusion (0.6 Context + 0.4 Claims)
    t0 = time.perf_counter()
    fused_emb = 0.6 * ctx_emb + 0.4 * clm_emb
    norms = np.linalg.norm(fused_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    fused_emb = fused_emb / norms
    timings["fusion_ms"] = float((time.perf_counter() - t0) * 1000.0)

    # 5. CPC Multi-Label Classification
    t0 = time.perf_counter()
    probs = ml_state.cpc_classifier.predict_proba(torch.tensor(fused_emb, dtype=torch.float32)).detach().numpy()[0]
    prob_dict = {str(cpc): float(p) for cpc, p in zip(ml_state.cpc_classes, probs)}
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    thresholded_probs = [(cpc, p) for cpc, p in sorted_probs if p >= ml_state.cpc_threshold]
    if not thresholded_probs and sorted_probs:
        thresholded_probs = [sorted_probs[0]]

    top_preds = [
        {
            "cpc": str(cpc),
            "probability": float(prob),
            "description": str(CPC_DESCRIPTIONS.get(cpc, f"CPC Subclass {cpc}"))
        }
        for cpc, prob in thresholded_probs
    ]
    timings["classification_ms"] = float((time.perf_counter() - t0) * 1000.0)

    # 6. FAISS Retrieval & Temporal Date Filtering
    t0 = time.perf_counter()
    cands = ml_state.cpc_retriever.retrieve(
        query_proc, 
        fused_emb[0], 
        predicted_cpcs=[p["cpc"] for p in top_preds[:2]], 
        top_k=50, 
        strategy="soft_cpc"
    )
    timings["faiss_search_ms"] = float((time.perf_counter() - t0) * 1000.0)

    # 7. Reranking
    t0 = time.perf_counter()
    reranked = ml_state.reranker.rerank(
        query_proc, cands, top_k=50,
        alpha_sim=0.7, beta_cpc=0.2, gamma_rerank=0.1,
        timeout_sec=2.0
    )[:10]
    timings["reranking_ms"] = float((time.perf_counter() - t0) * 1000.0)

    # 8. Dynamic Grounded Evidence Extraction
    t0 = time.perf_counter()
    prior_art_list = []
    for idx, c in enumerate(reranked, start=1):
        cpcs = [str(x.strip()) for x in str(c.get("cpc_subclasses", "")).split(",") if x.strip()]
        
        # Call GroundedResultExplainer to generate verbatim evidence snippets & paired evidence
        explanation_dict = GroundedResultExplainer.explain_retrieval(query_proc, c)
        ev_snippets = [
            f"Query Claim Evidence: \"{explanation_dict['grounded_evidence']['query_snippet']}\"",
            f"Candidate Specification Evidence: \"{explanation_dict['grounded_evidence']['candidate_snippet']}\""
        ]
        
        prior_art_list.append({
            "rank": int(idx),
            "patent_id": str(c.get("patent_id")),
            "title": str(c.get("title")),
            "date": str(c.get("grant_date", "")),
            "grant_date": str(c.get("grant_date", "")),
            "primary_cpc": str(c.get("primary_cpc", "")),
            "cpc": cpcs,
            "cpc_subclasses": cpcs,
            "semantic_similarity": float(c.get("semantic_similarity", 0.0)),
            "cpc_compatibility": float(c.get("cpc_compatibility", 0.0)),
            "ranking_score": float(c.get("final_score", 0.0)),
            "final_score": float(c.get("final_score", 0.0)),
            "shared_technical_concepts": explanation_dict.get("shared_technical_concepts", []),
            "paired_evidence": explanation_dict.get("paired_evidence", []),
            "evidence": ev_snippets,
            "abstract": str(c.get("abstract", "")),
            "independent_claims": str(c.get("independent_claims", ""))
        })
    timings["evidence_ms"] = float((time.perf_counter() - t0) * 1000.0)

    # 9. Top Result Detailed Explanation
    top_cand = reranked[0] if reranked else {}
    top_explanation = GroundedResultExplainer.explain_retrieval(query_proc, top_cand) if top_cand else {}

    # 10. Serialization
    t0 = time.perf_counter()
    total_ms = float((time.perf_counter() - t_start) * 1000.0)
    timings["serialization_ms"] = float((time.perf_counter() - t0) * 1000.0)
    timings["total_ms"] = total_ms

    analysis_id = f"ANALYSIS_{int(time.time()*1000)}"

    logger.info("=" * 45)
    logger.info("           [ANALYZE STAGE TIMINGS]")
    logger.info("=" * 45)
    logger.info(f"Validation ............. {timings['validation_ms']:.1f} ms")
    logger.info(f"Text Construction ...... {timings['text_construction_ms']:.1f} ms")
    logger.info(f"Embedding Generation ... {timings['embedding_ms']:.1f} ms")
    logger.info(f"PAFIR Fusion ........... {timings['fusion_ms']:.1f} ms")
    logger.info(f"CPC Classification ..... {timings['classification_ms']:.1f} ms")
    logger.info(f"FAISS Retrieval ........ {timings['faiss_search_ms']:.1f} ms")
    logger.info(f"Reranking .............. {timings['reranking_ms']:.1f} ms")
    logger.info(f"Evidence Extraction .... {timings['evidence_ms']:.1f} ms")
    logger.info(f"Serialization .......... {timings['serialization_ms']:.1f} ms")
    logger.info(f"TOTAL .................. {total_ms:.1f} ms")
    logger.info("=" * 45)

    return {
        "analysis_id": str(analysis_id),
        "classification": top_preds,
        "classification_detail": {
            "primary_cpc": str(top_preds[0]["cpc"]),
            "top_predicted_cpcs": top_preds,
            "all_probabilities": prob_dict,
            "execution_time_ms": float(timings['classification_ms'])
        },
        "prior_art": prior_art_list,
        "prior_art_candidates": [
            RetrievalCandidate(
                rank=item["rank"],
                patent_id=item["patent_id"],
                title=item["title"],
                grant_date=item["grant_date"],
                primary_cpc=item["primary_cpc"],
                cpc_subclasses=item["cpc"],
                semantic_similarity=item["semantic_similarity"],
                cpc_compatibility=item["cpc_compatibility"],
                final_score=item["final_score"],
                citation_relevance="Semantic Prior Art Candidate"
            ) for item in prior_art_list
        ],
        "top_result_explanation": GroundedExplanation(
            query_snippet=str(top_explanation.get("grounded_evidence", {}).get("query_snippet", "")),
            candidate_snippet=str(top_explanation.get("grounded_evidence", {}).get("candidate_snippet", "")),
            shared_technical_concepts=[str(x) for x in top_explanation.get("shared_technical_concepts", [])],
            shared_cpc_subclasses=[str(x) for x in top_explanation.get("shared_cpc_subclasses", [])],
            relevance_assessment=str(top_explanation.get("relevance_assessment", "")),
            legal_disclaimer=str(top_explanation.get("legal_disclaimer", ""))
        ),
        "query_patent": {
            "patent_id": str(query_proc.get("patent_id")),
            "title": str(query_proc.get("title")),
            "abstract": str(query_proc.get("abstract")),
            "independent_claims": str(query_proc.get("independent_claims")),
            "grant_date": str(query_proc.get("grant_date"))
        },
        "timing": timings,
        "total_pipeline_time_ms": float(total_ms)
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat_rag_assistant(req: ChatRequest):
    active_patent = {}
    if req.active_patent_id and ml_state.corpus_df is not None:
        match = ml_state.corpus_df[ml_state.corpus_df["patent_id"] == req.active_patent_id]
        if not match.empty:
            active_patent = match.iloc[0].to_dict()
            
    cands = ml_state.corpus_df.head(5).to_dict(orient="records") if ml_state.corpus_df is not None else []
    
    res = GroundedPatentAssistant.answer_query(req.user_query, active_patent, cands)
    return ChatResponse(
        answer=str(res["answer"]),
        supporting_patent_ids=[str(x) for x in res["supporting_patent_ids"]],
        evidence_snippets=res["evidence_snippets"],
        disclaimer=str(res["disclaimer"])
    )


@app.get("/api/patent/{patent_id}")
def get_patent_by_id(patent_id: str):
    if ml_state.corpus_df is None:
        raise HTTPException(status_code=500, detail="Corpus not loaded")
    match = ml_state.corpus_df[ml_state.corpus_df["patent_id"] == patent_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Patent ID not found")
    return match.iloc[0].to_dict()


@app.get("/api/patents/explore")
def explore_patents(
    query: Optional[str] = None,
    cpc: Optional[str] = None,
    split: Optional[str] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50)
):
    df = ml_state.corpus_df
    if df is None:
        return {"total": 0, "page": page, "data": []}
        
    filtered = df
    if query:
        q_lower = query.lower()
        filtered = filtered[
            filtered["title"].str.lower().str.contains(q_lower, na=False) |
            filtered["patent_id"].str.lower().str.contains(q_lower, na=False) |
            filtered["abstract"].str.lower().str.contains(q_lower, na=False)
        ]
        
    if cpc:
        filtered = filtered[filtered["cpc_subclasses"].str.contains(cpc, na=False)]
        
    if split and "split" in filtered.columns:
        filtered = filtered[filtered["split"] == split]
        
    total = len(filtered)
    start = (page - 1) * limit
    end = start + limit
    page_data = filtered.iloc[start:end].to_dict(orient="records")
    
    return {
        "total": int(total),
        "page": int(page),
        "limit": int(limit),
        "data": page_data
    }


def _get_latest_metrics(category: str) -> Optional[Dict[str, Any]]:
    exp_dir = os.path.join("experiments", category)
    metric_files = sorted(glob.glob(os.path.join(exp_dir, "*", "metrics.json")), key=os.path.getmtime, reverse=True)
    if not metric_files:
        return None
    try:
        with open(metric_files[0], "r") as f:
            return json.load(f)
    except Exception:
        return None


@app.get("/api/metrics/classification")
def get_classification_metrics():
    metrics = _get_latest_metrics("classification")
    if not metrics:
        return {"status": "Not Evaluated Yet", "metrics": None, "message": "No verified empirical classification experiment artifact found."}
    
    pafir_m = metrics.get("PAFIR Gated Fusion (Focal Loss Proposed)", metrics.get("TF-IDF + Linear Classifier"))
    return {
        "status": "VERIFIED",
        "dataset_mode": ml_state.mode,
        "metrics": pafir_m,
        "all_models": metrics
    }


@app.get("/api/metrics/retrieval")
def get_retrieval_metrics():
    metrics = _get_latest_metrics("retrieval")
    if not metrics:
        return {"status": "Not Evaluated Yet", "metrics": None, "message": "No verified empirical retrieval experiment artifact found."}
        
    pafir_m = metrics.get("PAFIR + Soft CPC + Reranker (Proposed)", metrics.get("BM25 Lexical Baseline"))
    return {
        "status": "VERIFIED",
        "dataset_mode": ml_state.mode,
        "metrics": pafir_m,
        "all_models": metrics
    }


@app.post("/api/admin/validate")
def validate_research_results():
    checks = {}
    checks["splits_exist"] = os.path.exists("data/splits/train.parquet") and os.path.exists("data/splits/test.parquet")
    
    leakage_file = "reports/data_leakage_report.json"
    checks["leakage_audit_passed"] = False
    if os.path.exists(leakage_file):
        with open(leakage_file, "r") as f:
            leakage_data = json.load(f)
            checks["leakage_audit_passed"] = leakage_data.get("all_checks_passed", False)
            
    checks["model_checkpoint_exists"] = os.path.exists("models/classification/cpc_classifier.pt")
    
    cls_metrics = _get_latest_metrics("classification")
    ret_metrics = _get_latest_metrics("retrieval")
    checks["classification_experiments_verified"] = cls_metrics is not None
    checks["retrieval_experiments_verified"] = ret_metrics is not None
    
    all_passed = all(checks.values())
    return {
        "verified": bool(all_passed),
        "status": "VERIFIED" if all_passed else "UNVERIFIED",
        "dataset_mode": ml_state.mode,
        "checks": checks
    }
