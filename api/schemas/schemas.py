from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PatentRequest(BaseModel):
    patent_id: Optional[str] = None
    title: str = Field(..., description="Patent Title")
    abstract: str = Field(..., description="Patent Abstract")
    independent_claims: Optional[str] = Field("", description="Independent Claims Text")
    claims: Optional[str] = Field("", description="Full Patent Claims Text")
    grant_date: Optional[str] = Field("2025-01-01", description="Grant Date (YYYY-MM-DD)")
    temporal_cutoff_date: Optional[str] = Field(None, description="Temporal Cutoff Date (YYYY-MM-DD)")

class ClassificationResponse(BaseModel):
    patent_id: Optional[str]
    top_predicted_cpcs: List[Dict[str, Any]]
    all_probabilities: Dict[str, float]
    primary_cpc: str
    execution_time_ms: float

class RetrievalCandidate(BaseModel):
    rank: int
    patent_id: str
    title: str
    grant_date: str
    primary_cpc: str
    cpc_subclasses: List[str]
    semantic_similarity: float
    cpc_compatibility: float
    final_score: float
    citation_relevance: Optional[str] = None

class SearchResponse(BaseModel):
    query_patent_id: Optional[str]
    total_candidates_indexed: int
    results: List[RetrievalCandidate]
    execution_time_ms: float

class GroundedExplanation(BaseModel):
    query_snippet: str
    candidate_snippet: str
    shared_technical_concepts: List[str]
    shared_cpc_subclasses: List[str]
    relevance_assessment: str
    legal_disclaimer: str

class ComprehensiveAnalysisResponse(BaseModel):
    query_patent: Dict[str, Any]
    classification: ClassificationResponse
    prior_art_candidates: List[RetrievalCandidate]
    top_result_explanation: GroundedExplanation
    total_pipeline_time_ms: float

class ChatRequest(BaseModel):
    active_patent_id: Optional[str] = None
    user_query: str
    conversation_history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    answer: str
    supporting_patent_ids: List[str]
    evidence_snippets: List[Dict[str, str]]
    disclaimer: str
