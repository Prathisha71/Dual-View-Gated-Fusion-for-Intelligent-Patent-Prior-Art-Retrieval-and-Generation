from typing import Dict, Any, List
from src.explainability.concept_extractor import TechnicalConceptExtractor
from src.explainability.evidence_matcher import EvidenceMatcher

LEGAL_DISCLAIMER = (
    "LEGAL NOTICE: This system provides automated patent similarity analysis for preliminary research purposes only. "
    "It does not constitute legal counsel, formal prior art examination, or a definitive determination of patent validity or novelty."
)

class GroundedResultExplainer:
    """Generates evidence-grounded technical explanations with paired concept-to-passage mappings for retrieved prior art patents."""
    
    @classmethod
    def explain_retrieval(
        cls,
        query_record: Dict[str, Any],
        candidate_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        q_text = str(query_record.get("independent_claims") or query_record.get("combined_text_transformer") or query_record.get("abstract", ""))
        c_text = str(candidate_record.get("abstract") or candidate_record.get("combined_text_transformer") or candidate_record.get("claims", ""))
        
        # Grounded paired evidence: concept -> query_passage -> candidate_passage
        paired_evidence = TechnicalConceptExtractor.extract_paired_evidence(q_text, c_text)
        shared_concepts = [item["concept"] for item in paired_evidence]
        
        # Exact matched evidence snippets
        q_snippet, c_snippet, match_score = EvidenceMatcher.match_evidence_snippets(q_text, c_text)
        
        # Shared CPC Subclasses - strictly intersect query and candidate CPC sets
        q_cpc_str = str(query_record.get("cpc_subclasses") or query_record.get("primary_cpc") or "")
        c_cpc_str = str(candidate_record.get("cpc_subclasses") or candidate_record.get("primary_cpc") or "")
        
        q_cpcs = set([c.strip() for c in q_cpc_str.split(",") if c.strip()])
        c_cpcs = set([c.strip() for c in c_cpc_str.split(",") if c.strip()])
        
        shared_cpcs = sorted(list(q_cpcs.intersection(c_cpcs)))
        
        if shared_cpcs:
            rel_msg = f"Potentially relevant prior art sharing CPC subclass(es) {', '.join(shared_cpcs)} and grounded technical concepts."
        else:
            rel_msg = "Potentially relevant prior art candidate based on dense semantic vector similarity."

        explanation = {
            "query_patent_id": query_record.get("patent_id"),
            "candidate_patent_id": candidate_record.get("patent_id"),
            "candidate_title": candidate_record.get("title"),
            "candidate_grant_date": candidate_record.get("grant_date"),
            "semantic_similarity": float(candidate_record.get("semantic_similarity", candidate_record.get("final_score", 0.0))),
            "cpc_compatibility": float(candidate_record.get("cpc_compatibility", 0.0)),
            "final_ranking_score": float(candidate_record.get("final_score", 0.0)),
            "shared_technical_concepts": shared_concepts,
            "shared_cpc_subclasses": shared_cpcs,
            "paired_evidence": paired_evidence,
            "grounded_evidence": {
                "query_snippet": q_snippet,
                "candidate_snippet": c_snippet,
                "snippet_match_fidelity": match_score
            },
            "relevance_assessment": rel_msg,
            "legal_disclaimer": LEGAL_DISCLAIMER
        }
        
        return explanation
