import re
from typing import Dict, Any, Tuple
from src.explainability.concept_extractor import TechnicalConceptExtractor

class EvidenceMatcher:
    """Matches exact sentence and claim evidence snippets between query and candidate patents."""
    
    @staticmethod
    def _split_sentences(text: str) -> list:
        if not text:
            return []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 15]

    @classmethod
    def match_evidence_snippets(
        cls,
        query_text: str,
        cand_text: str
    ) -> Tuple[str, str, float]:
        q_sents = cls._split_sentences(query_text)
        c_sents = cls._split_sentences(cand_text)
        
        if not q_sents or not c_sents:
            return query_text[:200], cand_text[:200], 0.0
            
        best_q = q_sents[0]
        best_c = c_sents[0]
        best_jaccard = 0.0
        
        for q in q_sents:
            q_words = set(re.findall(r'\b\w+\b', q.lower()))
            for c in c_sents:
                c_words = set(re.findall(r'\b\w+\b', c.lower()))
                union = q_words.union(c_words)
                if union:
                    jaccard = len(q_words.intersection(c_words)) / len(union)
                    if jaccard > best_jaccard:
                        best_jaccard = jaccard
                        best_q = q
                        best_c = c
                        
        return best_q, best_c, float(best_jaccard)
