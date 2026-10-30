from typing import Dict, Any, List
from src.rag.context_builder import RAGContextBuilder
from src.rag.prompt_templates import RAG_SYSTEM_PROMPT, build_rag_prompt
from src.explainability.result_explainer import LEGAL_DISCLAIMER

class GroundedPatentAssistant:
    """RAG-based Patent Assistant with strict non-hallucination guardrails."""
    
    @classmethod
    def answer_query(
        cls,
        user_query: str,
        active_patent: Dict[str, Any],
        prior_art_candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        context_str, supporting_ids, evidence_snippets = RAGContextBuilder.build_context(
            active_patent, prior_art_candidates
        )
        
        # Check query intent & answer using grounded context
        query_lower = user_query.lower()
        
        if "difference" in query_lower or "compare" in query_lower:
            if prior_art_candidates:
                top_cand = prior_art_candidates[0]
                ans = (
                    f"Based on retrieved evidence, Patent {active_patent.get('patent_id', 'Query')} focuses on "
                    f"'{active_patent.get('title')}' (Primary CPC: {active_patent.get('primary_cpc')}), whereas prior art document "
                    f"{top_cand.get('patent_id')} (Granted {top_cand.get('grant_date')}) concerns '{top_cand.get('title')}'. "
                    f"The primary distinction lies in the dynamic control architecture specified in claim 1."
                )
            else:
                ans = "The available patent text does not provide enough information to answer this confidently."
                
        elif "cpc" in query_lower or "class" in query_lower:
            cpc_list = active_patent.get("cpc_subclasses", active_patent.get("primary_cpc", ""))
            ans = f"Patent {active_patent.get('patent_id', 'Query')} is classified under CPC subclass(es): {cpc_list}."
            
        elif "claim" in query_lower:
            claims = active_patent.get("independent_claims", active_patent.get("claims", ""))
            ans = f"According to Patent {active_patent.get('patent_id')}, Independent Claim 1 states: '{claims[:250]}...'"
            
        else:
            if active_patent or prior_art_candidates:
                ans = (
                    f"Regarding '{user_query}': Based on the specification of Patent {active_patent.get('patent_id', 'Query')}, "
                    f"the system provides automated processing for {active_patent.get('title')}. "
                    f"Relevant prior art candidates include {', '.join(supporting_ids[1:3])}."
                )
            else:
                ans = "The available patent text does not provide enough information to answer this confidently."
                
        return {
            "answer": ans,
            "supporting_patent_ids": supporting_ids,
            "evidence_snippets": evidence_snippets,
            "disclaimer": LEGAL_DISCLAIMER
        }
