from typing import List, Dict, Any, Tuple

class RAGContextBuilder:
    """Formats retrieved prior art patents and active patent details into grounded context blocks."""
    
    @staticmethod
    def build_context(
        active_patent: Dict[str, Any],
        prior_art_candidates: List[Dict[str, Any]]
    ) -> Tuple[str, List[str], List[Dict[str, str]]]:
        context_blocks = []
        supporting_ids = []
        evidence_snippets = []
        
        if active_patent:
            pid = active_patent.get("patent_id", "QUERY_PATENT")
            title = active_patent.get("title", "")
            abstract = active_patent.get("abstract", "")
            claims = active_patent.get("independent_claims", active_patent.get("claims", ""))
            
            supporting_ids.append(pid)
            block = f"--- ACTIVE PATENT [{pid}] ---\nTitle: {title}\nAbstract: {abstract}\nIndependent Claims: {claims}\n"
            context_blocks.append(block)
            evidence_snippets.append({"patent_id": pid, "snippet": f"{title}: {abstract[:150]}..."})
            
        for cand in prior_art_candidates[:3]:
            cid = cand.get("patent_id", "PRIOR_ART")
            ctitle = cand.get("title", "")
            cabs = cand.get("abstract", "")
            cdate = cand.get("grant_date", "")
            cpc = cand.get("primary_cpc", "")
            
            supporting_ids.append(cid)
            cblock = f"--- RETRIEVED PRIOR ART [{cid}] (Granted: {cdate}, CPC: {cpc}) ---\nTitle: {ctitle}\nAbstract: {cabs}\n"
            context_blocks.append(cblock)
            evidence_snippets.append({"patent_id": cid, "snippet": f"{ctitle}: {cabs[:150]}..."})
            
        context_str = "\n".join(context_blocks)
        return context_str, list(set(supporting_ids)), evidence_snippets
